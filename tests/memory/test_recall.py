import pytest
from loongcli.memory.markdown_store import MarkdownMemoryStore
from loongcli.tools.recall import RecallTool


@pytest.fixture
def recall(tmp_path):
    store = MarkdownMemoryStore(base_dir=tmp_path)
    store.save(name="user-role", description="Python developer", type="user", content="Senior Python dev")
    store.save(name="no-comments", description="Don't add comments", type="feedback", content="Stop commenting")
    store.save(name="db-choice", description="SQLite for simplicity", type="project", content="Using SQLite")
    return RecallTool(store)


@pytest.mark.asyncio
async def test_recall_all(recall):
    result = await recall.execute()
    assert "user-role" in result
    assert "no-comments" in result
    assert "db-choice" in result


@pytest.mark.asyncio
async def test_recall_by_name(recall):
    result = await recall.execute(name="user-role")
    assert "Python" in result
    assert "user" in result


@pytest.mark.asyncio
async def test_recall_by_type(recall):
    result = await recall.execute(type="user")
    assert "user-role" in result
    assert "no-comments" not in result


@pytest.mark.asyncio
async def test_recall_missing(recall):
    result = await recall.execute(name="nope")
    assert "未找到" in result


@pytest.mark.asyncio
async def test_recall_empty_type(tmp_path):
    store = MarkdownMemoryStore(base_dir=tmp_path)
    tool = RecallTool(store)
    result = await tool.execute(type="reference")
    assert "没有" in result


@pytest.mark.asyncio
async def test_recall_empty_store(tmp_path):
    store = MarkdownMemoryStore(base_dir=tmp_path)
    tool = RecallTool(store)
    result = await tool.execute()
    assert "暂无" in result
