import pytest
from loongcli.tools.read_file import ReadFileTool


@pytest.fixture
def tool():
    return ReadFileTool()


@pytest.fixture
def sample_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")
    return str(f)


@pytest.mark.asyncio
async def test_read_whole_file(tool, sample_file):
    result = await tool.execute(path=sample_file)
    assert "line1" in result
    assert "line5" in result


@pytest.mark.asyncio
async def test_read_with_offset_and_limit(tool, sample_file):
    result = await tool.execute(path=sample_file, offset=2, limit=2)
    assert "line2" in result
    assert "line3" in result
    assert "line1" not in result
    assert "line4" not in result


@pytest.mark.asyncio
async def test_read_nonexistent_file(tool):
    result = await tool.execute(path="/nonexistent/file.txt")
    assert "错误" in result or "Error" in result


@pytest.mark.asyncio
async def test_read_file_has_line_numbers(tool, sample_file):
    result = await tool.execute(path=sample_file)
    assert "1\t" in result


def test_tool_schema(tool):
    assert tool.name == "read_file"
    assert "path" in tool.parameters["properties"]
