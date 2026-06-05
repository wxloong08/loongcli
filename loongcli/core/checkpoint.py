from __future__ import annotations

import logging
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

MODIFY_TOOLS = {"write_file", "edit_file", "shell"}
MAX_CHECKPOINTS = 50


@dataclass
class _Snapshot:
    ckpt_id: str
    backed_files: dict[str, Path] = field(default_factory=dict)  # rel_path -> backup_path
    has_stash: bool = False


class CheckpointManager:
    """Checkpoint manager with git stash + file backup dual safety.

    File backup is the ground truth — always runs. Git stash is an optional
    optimization layer on top. This ensures reliability regardless of git
    edge cases (untracked files, unchanged tracked files, etc.).
    """

    def __init__(self, cwd: Path | None = None):
        self.cwd = cwd or Path.cwd()
        self._git_available: bool | None = None
        self._snapshots: dict[str, _Snapshot] = {}
        self._backup_dir = Path.home() / ".loongcli" / "checkpoints"
        self._cleanup_orphans()

    @property
    def is_git_repo(self) -> bool:
        if self._git_available is None:
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--git-dir"],
                    cwd=self.cwd,
                    capture_output=True,
                    text=True,
                )
                self._git_available = result.returncode == 0
            except FileNotFoundError:
                self._git_available = False
        return self._git_available

    def save(self, files: list[str] | None = None) -> str | None:
        """Create a checkpoint. Returns ckpt_id or None if nothing to save.

        Always performs file backup (reliable). Additionally attempts git
        stash for git repos. Restore will prefer git stash but falls back
        to file backup if stash is unavailable.
        """
        if not files:
            return None

        existing = [f for f in files if (self.cwd / f).exists()]
        if not existing:
            return None

        ckpt_id = f"loongcli-ckpt-{uuid.uuid4().hex[:12]}"
        snap = _Snapshot(ckpt_id=ckpt_id)

        # Always perform file backup (ground truth)
        backup_root = self._backup_dir / ckpt_id
        backup_root.mkdir(parents=True, exist_ok=True)
        for f in existing:
            src = self.cwd / f
            dst = backup_root / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                if src.exists():
                    shutil.copy2(src, dst)
                    snap.backed_files[f] = dst
            except (PermissionError, OSError) as e:
                logger.debug("checkpoint copy skipped %s: %s", f, e)

        self._snapshots[ckpt_id] = snap
        self._cleanup_old()
        return ckpt_id

    def save_workdir(self) -> str | None:
        """Checkpoint the entire working directory via git stash (no specific files)."""
        if not self.is_git_repo:
            return None
        ckpt_id = f"loongcli-ckpt-{uuid.uuid4().hex[:12]}"
        snap = _Snapshot(ckpt_id=ckpt_id)
        try:
            result = subprocess.run(
                ["git", "stash", "push", "--include-untracked", "-m", ckpt_id],
                cwd=self.cwd, capture_output=True, text=True,
            )
            if result.returncode == 0 and "No local changes" not in result.stdout:
                snap.has_stash = True
                self._snapshots[ckpt_id] = snap
                self._cleanup_old()
                return ckpt_id
        except Exception:
            logger.debug("git stash (workdir) failed", exc_info=True)
        return None

    def _find_stash_index(self, ckpt_id: str) -> str | None:
        """Find stash index by checkpoint message. Returns 'stash@{N}' or None."""
        try:
            result = subprocess.run(
                ["git", "stash", "list"],
                cwd=self.cwd, capture_output=True, text=True,
            )
            for line in result.stdout.split("\n"):
                if ckpt_id in line:
                    return line.split(":")[0].strip()
        except Exception:
            logger.debug("git stash list failed", exc_info=True)
        return None

    def restore(self, ckpt_id: str) -> bool:
        """Restore files from a checkpoint. Returns True on success."""
        snap = self._snapshots.get(ckpt_id)
        if not snap:
            return False

        if snap.has_stash:
            stash_ref = self._find_stash_index(ckpt_id)
            if stash_ref:
                try:
                    subprocess.run(
                        ["git", "stash", "pop", stash_ref],
                        cwd=self.cwd, capture_output=True, text=True,
                    )
                except Exception:
                    logger.debug("git stash pop failed, using file backup", exc_info=True)

        # File backup restore (always works, fills gaps git stash missed)
        for rel_path, backup_path in snap.backed_files.items():
            dst = self.cwd / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if backup_path.exists():
                shutil.copy2(backup_path, dst)

        self._cleanup_snapshot(ckpt_id, snap)
        return True

    def discard(self, ckpt_id: str) -> bool:
        """Discard a checkpoint (verification passed)."""
        snap = self._snapshots.get(ckpt_id)
        if not snap:
            return False

        if snap.has_stash:
            stash_ref = self._find_stash_index(ckpt_id)
            if stash_ref:
                try:
                    subprocess.run(
                        ["git", "stash", "drop", stash_ref],
                        cwd=self.cwd, capture_output=True, text=True,
                    )
                except Exception:
                    logger.debug("git stash drop failed", exc_info=True)

        self._cleanup_snapshot(ckpt_id, snap)
        return True

    def list_checkpoints(self) -> list[str]:
        """List active checkpoint IDs, oldest first."""
        return list(self._snapshots.keys())

    def _cleanup_snapshot(self, ckpt_id: str, snap: _Snapshot) -> None:
        """Remove a snapshot and its backup files."""
        backup_root = self._backup_dir / ckpt_id
        if backup_root.exists():
            shutil.rmtree(backup_root, ignore_errors=True)
        self._snapshots.pop(ckpt_id, None)

    def _cleanup_old(self) -> None:
        """Remove checkpoints beyond MAX_CHECKPOINTS, oldest first."""
        while len(self._snapshots) > MAX_CHECKPOINTS:
            oldest_id = next(iter(self._snapshots))
            self.discard(oldest_id)

    def _cleanup_orphans(self, max_age_days: int = 7) -> None:
        """Remove orphaned backup dirs older than max_age_days on startup."""
        if not self._backup_dir.exists():
            return
        cutoff = datetime.now().timestamp() - max_age_days * 86400
        for d in self._backup_dir.iterdir():
            if d.is_dir() and d.name.startswith("loongcli-ckpt-"):
                try:
                    if d.stat().st_mtime < cutoff:
                        shutil.rmtree(d, ignore_errors=True)
                except OSError:
                    pass
