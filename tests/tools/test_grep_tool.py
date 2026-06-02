import pytest
from pathlib import Path
from loongcli.tools.grep_tool import GrepTool


@pytest.fixture
def tool():
    return GrepTool()


@pytest.fixture
def project_dir(tmp_path):
    (tmp_path / "a.py").write_text("def hello():\n    print('hello')\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def world():\n    print('world')\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("no functions here\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.py").write_text("def nested():\n    pass\n", encoding="utf-8")
    return str(tmp_path)


@pytest.mark.asyncio
async def test_grep_finds_pattern(tool, project_dir):
    result = await tool.execute(pattern=r"def \w+", path=project_dir)
    assert "hello" in result
    assert "world" in result
    assert "nested" in result


@pytest.mark.asyncio
async def test_grep_with_glob_filter(tool, project_dir):
    result = await tool.execute(pattern=r"def \w+", path=project_dir, glob="*.py")
    assert "hello" in result
    assert "no functions" not in result


@pytest.mark.asyncio
async def test_grep_no_matches(tool, project_dir):
    result = await tool.execute(pattern=r"class \w+", path=project_dir)
    assert "未找到" in result


@pytest.mark.asyncio
async def test_grep_shows_line_numbers(tool, project_dir):
    result = await tool.execute(pattern="print", path=project_dir)
    assert ":2:" in result


@pytest.mark.asyncio
async def test_grep_case_insensitive(tool, project_dir):
    result = await tool.execute(pattern="HELLO", path=project_dir, case_insensitive=True)
    assert "hello" in result


def test_tool_schema(tool):
    assert tool.name == "grep"
    assert "pattern" in tool.parameters["properties"]
