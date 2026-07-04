"""memorize 工具溯源：模型主动写入的记忆同样带 source_session 案底。"""
import pytest

from loongcli.memory.markdown_store import MarkdownMemoryStore
from loongcli.tools.memorize import MemorizeTool


@pytest.mark.asyncio
async def test_memorize_writes_source_session(tmp_path):
    store = MarkdownMemoryStore(base_dir=tmp_path)
    tool = MemorizeTool(store, session_provider=lambda: "sess42")
    await tool.execute(operation="save", name="m1", description="d", content="c")
    raw = (tmp_path / "m1.md").read_text(encoding="utf-8")
    assert "source_session:" in raw and "memorize:sess42" in raw


@pytest.mark.asyncio
async def test_memorize_source_without_provider(tmp_path):
    store = MarkdownMemoryStore(base_dir=tmp_path)
    tool = MemorizeTool(store)
    await tool.execute(operation="save", name="m2", description="d", content="c")
    raw = (tmp_path / "m2.md").read_text(encoding="utf-8")
    assert "source_session: memorize" in raw
