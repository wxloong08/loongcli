import pytest
from pathlib import Path
from loongcli.tools.write_file import WriteFileTool


@pytest.fixture
def tool():
    return WriteFileTool()


@pytest.mark.asyncio
async def test_write_new_file(tool, tmp_path):
    target = str(tmp_path / "new.txt")
    result = await tool.execute(path=target, content="hello world")
    assert "成功" in result
    assert Path(target).read_text(encoding="utf-8") == "hello world"


@pytest.mark.asyncio
async def test_write_creates_parent_dirs(tool, tmp_path):
    target = str(tmp_path / "sub" / "deep" / "file.txt")
    result = await tool.execute(path=target, content="nested")
    assert Path(target).exists()
    assert Path(target).read_text(encoding="utf-8") == "nested"


@pytest.mark.asyncio
async def test_write_overwrites_existing(tool, tmp_path):
    target = tmp_path / "exist.txt"
    target.write_text("old", encoding="utf-8")
    result = await tool.execute(path=str(target), content="new")
    assert target.read_text(encoding="utf-8") == "new"


def test_tool_schema(tool):
    assert tool.name == "write_file"
    assert "path" in tool.parameters["properties"]
    assert "content" in tool.parameters["properties"]


@pytest.mark.asyncio
async def test_write_rejects_oversized_content(tool, tmp_path, monkeypatch):
    """Content exceeding _MAX_CONTENT_SIZE should be rejected."""
    import loongcli.tools.write_file as mod

    monkeypatch.setattr(mod, "_MAX_CONTENT_SIZE", 100)
    result = await tool.execute(
        path=str(tmp_path / "big.txt"),
        content="x" * 200,
    )
    assert "过大" in result or "错误" in result
    assert not (tmp_path / "big.txt").exists()
