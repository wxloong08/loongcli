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
