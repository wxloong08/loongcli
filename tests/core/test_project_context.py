import pytest
from pathlib import Path

from loongcli.core.project_context import find_project_context_files, load_project_context


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Point Path.home() at an empty temp dir so these tests don't pick up the
    real user-level ~/.loongcli/LOONG.md (which find_project_context_files loads
    unconditionally). Without this, exact-equality assertions fail on any machine
    that actually has a user-level LOONG.md."""
    fake_home = tmp_path / "_home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda *a, **k: fake_home)


def test_find_no_files(tmp_path):
    result = find_project_context_files(start=tmp_path)
    assert result == []


def test_find_in_current_dir(tmp_path):
    f = tmp_path / "LOONG.md"
    f.write_text("# Project Rules", encoding="utf-8")
    result = find_project_context_files(start=tmp_path)
    assert len(result) == 1
    assert result[0] == f


def test_find_walks_up_to_git_root(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "LOONG.md").write_text("root rules", encoding="utf-8")

    sub = root / "src" / "app"
    sub.mkdir(parents=True)
    (sub / "LOONG.md").write_text("app rules", encoding="utf-8")

    result = find_project_context_files(start=sub)
    assert len(result) == 2
    assert result[0] == root / "LOONG.md"
    assert result[1] == sub / "LOONG.md"


def test_find_stops_at_git_root(tmp_path):
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / "LOONG.md").write_text("outer", encoding="utf-8")

    repo = outer / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "LOONG.md").write_text("repo", encoding="utf-8")

    result = find_project_context_files(start=repo)
    paths = [p.name for p in result]
    assert all(p == "LOONG.md" for p in paths)
    contents = [p.read_text(encoding="utf-8") for p in result]
    assert "outer" not in contents


def test_load_empty(tmp_path):
    result = load_project_context(start=tmp_path)
    assert result == ""


def test_load_single_file(tmp_path):
    (tmp_path / "LOONG.md").write_text("use pytest for tests", encoding="utf-8")
    result = load_project_context(start=tmp_path)
    assert "use pytest for tests" in result


def test_load_multiple_files(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "LOONG.md").write_text("global rule", encoding="utf-8")

    sub = root / "pkg"
    sub.mkdir()
    (sub / "LOONG.md").write_text("pkg rule", encoding="utf-8")

    result = load_project_context(start=sub)
    assert "global rule" in result
    assert "pkg rule" in result
    assert result.index("global rule") < result.index("pkg rule")


def test_load_skips_empty_file(tmp_path):
    (tmp_path / "LOONG.md").write_text("", encoding="utf-8")
    result = load_project_context(start=tmp_path)
    assert result == ""


def test_load_utf8_chinese(tmp_path):
    (tmp_path / "LOONG.md").write_text("# 项目规则\n使用中文注释", encoding="utf-8")
    result = load_project_context(start=tmp_path)
    assert "使用中文注释" in result
