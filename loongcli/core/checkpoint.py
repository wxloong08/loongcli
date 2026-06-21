from __future__ import annotations

import logging
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

MODIFY_TOOLS = {"write_file", "edit_file"}
MAX_CHECKPOINTS = 50


@dataclass
class _Snapshot:
    ckpt_id: str
    backed_files: dict[str, Path] = field(default_factory=dict)


class CheckpointManager:
    """File-backup checkpoint manager. No git dependency.

    Copies target files before modification so they can be restored
    on verification failure. Pure file I/O — works in any directory,
    with or without git.
    """

    def __init__(self, cwd: Path | None = None):
        self.cwd = cwd or Path.cwd()
        self._snapshots: dict[str, _Snapshot] = {}
        self._backup_dir = Path.home() / ".loongcli" / "checkpoints"
        self._cleanup_orphans()

    def _resolve(self, f: str) -> Path:
        p = Path(f)
        return p if p.is_absolute() else self.cwd / p

    def save(self, files: list[str] | None = None) -> str | None:
        """Copy files to backup. Returns ckpt_id or None if nothing to save."""
        if not files:
            return None

        existing = [f for f in files if self._resolve(f).exists()]
        if not existing:
            return None

        ckpt_id = f"loongcli-ckpt-{uuid.uuid4().hex[:12]}"
        snap = _Snapshot(ckpt_id=ckpt_id)

        backup_root = self._backup_dir / ckpt_id
        backup_root.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(existing):
            src = self._resolve(f)
            # Index-prefix the backup name so two files sharing a basename
            # (e.g. pkg_a/__init__.py and pkg_b/__init__.py) don't clobber each
            # other in the backup dir and corrupt each other on restore.
            dst = backup_root / f"{i:03d}_{src.name}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dst)
                snap.backed_files[f] = dst
            except (PermissionError, OSError) as e:
                logger.debug("checkpoint copy skipped %s: %s", f, e)

        if not snap.backed_files:
            shutil.rmtree(backup_root, ignore_errors=True)
            return None

        self._snapshots[ckpt_id] = snap
        self._cleanup_old()
        return ckpt_id

    def restore(self, ckpt_id: str) -> bool:
        """Restore files from a checkpoint. Returns True on success."""
        snap = self._snapshots.get(ckpt_id)
        if not snap:
            return False

        for rel_path, backup_path in snap.backed_files.items():
            dst = self._resolve(rel_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if backup_path.exists():
                shutil.copy2(backup_path, dst)

        self._cleanup_snapshot(ckpt_id, snap)
        return True

    def discard(self, ckpt_id: str) -> bool:
        """Discard a checkpoint (tool succeeded, backup no longer needed)."""
        snap = self._snapshots.get(ckpt_id)
        if not snap:
            return False
        self._cleanup_snapshot(ckpt_id, snap)
        return True

    def list_checkpoints(self) -> list[str]:
        return list(self._snapshots.keys())

    def _cleanup_snapshot(self, ckpt_id: str, snap: _Snapshot) -> None:
        backup_root = self._backup_dir / ckpt_id
        if backup_root.exists():
            shutil.rmtree(backup_root, ignore_errors=True)
        self._snapshots.pop(ckpt_id, None)

    def _cleanup_old(self) -> None:
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
