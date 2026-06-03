from __future__ import annotations

import json
from pathlib import Path

import pytest

from loongcli.core.test_discovery import discover_test_command, detect_framework


def test_detect_pytest_via_pyproject(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\nminversion = \"7.0\"")
    assert detect_framework(tmp_path) == "pytest"


def test_detect_pytest_via_pytest_ini(tmp_path: Path):
    (tmp_path / "pytest.ini").write_text("[pytest]\ntestpaths = tests")
    assert detect_framework(tmp_path) == "pytest"


def test_detect_pytest_via_setup_cfg(tmp_path: Path):
    (tmp_path / "setup.cfg").write_text("[tool:pytest]\ntestpaths = tests")
    assert detect_framework(tmp_path) == "pytest"


def test_detect_npm_test(tmp_path: Path):
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "jest"}}))
    assert detect_framework(tmp_path) == "npm test"


def test_detect_go_test(tmp_path: Path):
    (tmp_path / "go.mod").write_text("module example\n\ngo 1.21")
    assert detect_framework(tmp_path) == "go test"


def test_detect_cargo_test(tmp_path: Path):
    (tmp_path / "Cargo.toml").write_text("[package]\nname = \"test\"")
    assert detect_framework(tmp_path) == "cargo test"


def test_detect_make_test(tmp_path: Path):
    (tmp_path / "Makefile").write_text("test:\n\tpytest\n\nlint:\n\truff check .")
    assert detect_framework(tmp_path) == "make test"


def test_detect_nothing(tmp_path: Path):
    assert detect_framework(tmp_path) is None


def test_discover_returns_command(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]")
    cmd = discover_test_command(tmp_path)
    assert cmd == "pytest"


def test_discover_none(tmp_path: Path):
    assert discover_test_command(tmp_path) is None


def test_loong_md_override(tmp_path: Path):
    """LOONG.md test: declaration overrides auto-detection."""
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]")
    (tmp_path / "LOONG.md").write_text("test: pytest -x --tb=short --coverage")
    cmd = discover_test_command(tmp_path)
    assert cmd == "pytest -x --tb=short --coverage"


def test_claude_md_override(tmp_path: Path):
    (tmp_path / "CLAUDE.md").write_text("test: npm run test -- --verbose")
    cmd = discover_test_command(tmp_path)
    assert cmd == "npm run test -- --verbose"


def test_loong_md_beats_claude_md(tmp_path: Path):
    """LOONG.md takes precedence over CLAUDE.md."""
    (tmp_path / "LOONG.md").write_text("test: pytest")
    (tmp_path / "CLAUDE.md").write_text("test: make test")
    assert discover_test_command(tmp_path) == "pytest"


def test_narrow_scope_with_changed_files(tmp_path: Path):
    """When changed_files includes src/foo.py, suggest targeted test if exists."""
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_foo.py").write_text("")
    cmd = discover_test_command(tmp_path, changed_files=["src/foo.py"])
    assert cmd is not None
    assert "test_foo.py" in cmd


def test_narrow_scope_no_match(tmp_path: Path):
    """When no matching test file exists, return generic command."""
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    cmd = discover_test_command(tmp_path, changed_files=["src/bar.py"])
    assert cmd == "pytest"  # no test_bar.py, fall back to generic


def test_framework_order_priority(tmp_path: Path):
    """Multiple frameworks present: pytest beats npm test beats go test."""
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "jest"}}))
    (tmp_path / "go.mod").write_text("module x")
    assert detect_framework(tmp_path) == "pytest"  # pytest has highest priority
