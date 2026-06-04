import pytest
from pathlib import Path
from loongcli.tools.glob_tool import GlobTool


@pytest.fixture
def tool():
    return GlobTool()


@pytest.fixture
def project_dir(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("main", encoding="utf-8")
    (tmp_path / "src" / "utils.py").write_text("utils", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("test", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    return str(tmp_path)


@pytest.mark.asyncio
async def test_glob_py_files(tool, project_dir):
    result = await tool.execute(pattern="**/*.py", path=project_dir)
    assert "main.py" in result
    assert "utils.py" in result
    assert "test_main.py" in result
    assert "README.md" not in result


@pytest.mark.asyncio
async def test_glob_specific_dir(tool, project_dir):
    result = await tool.execute(pattern="src/*.py", path=project_dir)
    assert "main.py" in result
    assert "test_main.py" not in result


@pytest.mark.asyncio
async def test_glob_no_matches(tool, project_dir):
    result = await tool.execute(pattern="**/*.rs", path=project_dir)
    assert "未找到" in result


@pytest.mark.asyncio
async def test_glob_default_cwd(tool):
    result = await tool.execute(pattern="*.toml")
    assert "pyproject.toml" in result


def test_tool_schema(tool):
    assert tool.name == "glob"
    assert "pattern" in tool.parameters["properties"]


@pytest.mark.asyncio
async def test_glob_scan_limit(tool, tmp_path, monkeypatch):
    """Scan limit stops early on huge directories."""
    import loongcli.tools.glob_tool as mod

    monkeypatch.setattr(mod, "_MAX_SCAN", 5)
    for i in range(50):
        (tmp_path / f"file_{i}.txt").write_text("", encoding="utf-8")
    result = await tool.execute(pattern="*.txt", path=str(tmp_path))
    assert "扫描已截断" in result


# ── zero-result diagnostics tests ──


@pytest.mark.asyncio
async def test_zero_result_shows_dir_contents(tool, tmp_path):
    """On zero results, show directory contents as hints."""
    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text("", encoding="utf-8")
    result = await tool.execute(pattern="**/*.rs", path=str(tmp_path))
    assert "未找到" in result
    assert "src" in result  # directory listing should mention it


@pytest.mark.asyncio
async def test_backslash_suggests_forward_slash(tool, tmp_path):
    """Backslash in pattern → diagnostic hint."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("", encoding="utf-8")
    result = await tool.execute(pattern="src\\*.rs", path=str(tmp_path))
    assert "未找到" in result
    assert "反斜杠" in result or "backslash" in result.lower()


@pytest.mark.asyncio
async def test_similar_dir_suggestion(tool, tmp_path):
    """Typo in directory name → suggest similar existing dir."""
    (tmp_path / "components").mkdir()
    (tmp_path / "components" / "Button.tsx").write_text("", encoding="utf-8")
    result = await tool.execute(pattern="componets/*", path=str(tmp_path))  # typo
    assert "未找到" in result
    assert "components" in result  # should suggest the correct name


@pytest.mark.asyncio
async def test_trailing_slash_warns_directory_match(tool, tmp_path):
    """Pattern ending with / suggests adding *."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("", encoding="utf-8")
    result = await tool.execute(pattern="src/", path=str(tmp_path))
    assert "未找到" in result
    assert "/" in result  # diagnostic mentions the trailing slash issue


@pytest.mark.asyncio
async def test_empty_directory_shown(tool, tmp_path):
    """Empty directory → report it as empty."""
    (tmp_path / "just_this.txt").write_text("", encoding="utf-8")
    result = await tool.execute(pattern="**/*.rs", path=str(tmp_path))
    assert "未找到" in result
    # Should show the one file that exists
    assert "just_this.txt" in result
