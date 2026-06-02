import pytest
from loongcli.memory.store import MemoryStore
from loongcli.tools.recall import RecallTool


@pytest.fixture
def recall(tmp_path):
    mem = MemoryStore(path=tmp_path / "kv.json")
    mem.set("git", "repo", "loongcli")
    mem.set("git", "branch", "main")
    mem.set("deploy", "target", "prod")
    return RecallTool(mem)


@pytest.mark.asyncio
async def test_recall_all(recall):
    result = await recall.execute()
    assert "git" in result
    assert "deploy" in result
    assert "repo" in result


@pytest.mark.asyncio
async def test_recall_category(recall):
    result = await recall.execute(category="git")
    assert "repo: loongcli" in result
    assert "branch: main" in result


@pytest.mark.asyncio
async def test_recall_key(recall):
    result = await recall.execute(category="git", key="repo")
    assert "loongcli" in result


@pytest.mark.asyncio
async def test_recall_missing(recall):
    result = await recall.execute(category="nope", key="nope")
    assert "未找到" in result
