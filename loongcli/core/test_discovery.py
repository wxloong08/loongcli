from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def detect_framework(cwd: Path) -> str | None:
    """Detect the project's test framework from config files.

    Returns the framework identifier or None if undetected.
    Priority order: pytest > npm test > go test > cargo test > make test
    """
    checks = [
        (["pyproject.toml", "pytest.ini", "setup.cfg"], _detect_pytest),
        (["package.json"], _detect_npm_test),
        (["go.mod"], lambda p: "go test"),
        (["Cargo.toml"], lambda p: "cargo test"),
        (["Makefile"], _detect_make_test),
    ]
    for filenames, detector in checks:
        for name in filenames:
            path = cwd / name
            if path.exists() and path.is_file():
                result = detector(path)
                if result:
                    return result
    return None


def _detect_pytest(path: Path) -> str | None:
    """Check if a file indicates pytest."""
    content = path.read_text(encoding="utf-8", errors="replace").lower()
    if path.name == "pyproject.toml":
        if "pytest" in content:
            return "pytest"
    elif path.name == "pytest.ini":
        return "pytest"
    elif path.name == "setup.cfg":
        if "pytest" in content:
            return "pytest"
    return None


def _detect_npm_test(path: Path) -> str | None:
    """Check if package.json has a test script."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "scripts" in data and "test" in data["scripts"]:
            return "npm test"
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _detect_make_test(path: Path) -> str | None:
    """Check if Makefile has a test target."""
    content = path.read_text(encoding="utf-8", errors="replace")
    for line in content.split("\n"):
        if line.startswith("test:"):
            return "make test"
    return None


def discover_test_command(
    cwd: Path | None = None,
    changed_files: list[str] | None = None,
) -> str | None:
    """Discover the project's test command.

    1. Check LOONG.md / CLAUDE.md for explicit ``test: ...`` declaration
    2. Auto-detect framework from project files
    3. Narrow scope if changed_files are provided and matching tests exist

    Returns the command string or None.
    """
    cwd = cwd or Path.cwd()

    # 1. Explicit declaration in project context files
    for context_file in ("LOONG.md", "CLAUDE.md"):
        path = cwd / context_file
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("test:"):
                        cmd = line[5:].strip()
                        if cmd:
                            return cmd
            except OSError:
                pass

    # 2. Auto-detect framework
    framework = detect_framework(cwd)
    if not framework:
        return None

    command = _framework_to_command(framework)

    # 3. Narrow scope if changed files given
    if changed_files:
        narrowed = _narrow_scope(cwd, command, changed_files)
        if narrowed:
            return narrowed

    return command


def _framework_to_command(framework: str) -> str:
    """Map framework identifier to default command."""
    commands = {
        "pytest": "pytest",
        "npm test": "npm test",
        "go test": "go test ./...",
        "cargo test": "cargo test",
        "make test": "make test",
    }
    return commands.get(framework, framework)


def _narrow_scope(cwd: Path, base_cmd: str, changed_files: list[str]) -> str | None:
    """Try to narrow the test command to target specific test files.

    Convention: src/foo.py → tests/test_foo.py or tests/foo_test.py
    """
    for src_file in changed_files:
        stem = Path(src_file).stem
        # Common test file naming conventions
        candidates = [
            f"tests/test_{stem}.py",
            f"tests/{stem}_test.py",
            f"test/test_{stem}.py",
            f"test/{stem}_test.py",
        ]
        for candidate in candidates:
            if (cwd / candidate).exists():
                if base_cmd.startswith("pytest"):
                    return f"pytest {candidate} -v"
                if base_cmd.startswith("go test"):
                    pkg = str(Path(candidate).parent).replace("\\", "/")
                    return f"go test ./{pkg}/..."
                return f"{base_cmd} {candidate}"
    return None
