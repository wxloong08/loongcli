import pytest
from pathlib import Path
from loongcli.tools.edit_file import EditFileTool


@pytest.fixture
def tool():
    return EditFileTool()


@pytest.fixture
def sample_file(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("def hello():\n    print('hello')\n    return True\n", encoding="utf-8")
    return str(f)


@pytest.mark.asyncio
async def test_edit_replaces_string(tool, sample_file):
    result = await tool.execute(path=sample_file, old_string="print('hello')", new_string="print('world')")
    assert "成功" in result
    content = Path(sample_file).read_text(encoding="utf-8")
    assert "print('world')" in content
    assert "print('hello')" not in content


@pytest.mark.asyncio
async def test_edit_old_string_not_found(tool, sample_file):
    result = await tool.execute(path=sample_file, old_string="nonexistent string", new_string="replacement")
    assert "未找到" in result


@pytest.mark.asyncio
async def test_edit_multiple_matches_fails(tool, tmp_path):
    f = tmp_path / "dup.txt"
    f.write_text("aaa\nbbb\naaa\n", encoding="utf-8")
    result = await tool.execute(path=str(f), old_string="aaa", new_string="ccc")
    assert "多处匹配" in result or "2" in result


@pytest.mark.asyncio
async def test_edit_nonexistent_file(tool):
    result = await tool.execute(path="/nonexistent/file.txt", old_string="a", new_string="b")
    assert "错误" in result


def test_tool_schema(tool):
    assert tool.name == "edit_file"
    assert "old_string" in tool.parameters["properties"]
    assert "new_string" in tool.parameters["properties"]
