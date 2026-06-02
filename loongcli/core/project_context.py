from __future__ import annotations
import os
from pathlib import Path

CONTEXT_FILENAME = "LOONG.md"


def find_project_context_files(start: Path | None = None) -> list[Path]:
    """Walk up from start dir to git root (or filesystem root), collecting LOONG.md files.
    Returns paths ordered from outermost (user-level) to innermost (closest to CWD).
    """
    found: list[Path] = []

    user_file = Path.home() / ".loongcli" / CONTEXT_FILENAME
    if user_file.is_file():
        found.append(user_file)

    start = start or Path.cwd()
    start = start.resolve()

    project_files: list[Path] = []
    current = start
    while True:
        candidate = current / CONTEXT_FILENAME
        if candidate.is_file():
            project_files.append(candidate)
        if (current / ".git").exists():
            break
        parent = current.parent
        if parent == current:
            break
        current = parent

    project_files.reverse()
    found.extend(project_files)
    return found


def load_project_context(start: Path | None = None) -> str:
    """Load and concatenate all LOONG.md files into a single context string."""
    files = find_project_context_files(start)
    if not files:
        return ""

    sections: list[str] = []
    for f in files:
        try:
            content = f.read_text(encoding="utf-8").strip()
            if content:
                rel = _describe_path(f)
                sections.append(f"<!-- {rel} -->\n{content}")
        except OSError:
            continue

    return "\n\n".join(sections)


def _describe_path(p: Path) -> str:
    home = Path.home()
    try:
        return f"~/{p.relative_to(home).as_posix()}"
    except ValueError:
        cwd = Path.cwd()
        try:
            return p.relative_to(cwd).as_posix()
        except ValueError:
            return str(p)
