import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
from loongcli.memory.recall_engine import RecallEngine
from loongcli.memory.markdown_store import MarkdownMemoryStore


@pytest.fixture
def store(tmp_path):
    s = MarkdownMemoryStore(base_dir=tmp_path)
    s.save(name="git-workflow", description="dev/master branch strategy", type="feedback", content="Use dev for development, master for push")
    s.save(name="user-role", description="Python developer", type="user", content="Senior Python dev")
    s.save(name="api-docs", description="DeepSeek API documentation URL", type="reference", content="https://api-docs.deepseek.com")
    return s


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.chat = AsyncMock()
    return llm


def test_build_recall_prompt(store, mock_llm):
    engine = RecallEngine(memory=store, llm=mock_llm)
    prompt = engine._build_prompt("How should I push my code?")
    assert "git-workflow" in prompt
    assert "user-role" in prompt
    assert "api-docs" in prompt
    assert "How should I push my code?" in prompt


@pytest.mark.asyncio
async def test_recall_returns_relevant_memories(store, mock_llm):
    mock_llm.chat.return_value = "git-workflow"
    engine = RecallEngine(memory=store, llm=mock_llm)
    results = await engine.recall("How do I push code?")
    assert len(results) == 1
    assert results[0]["name"] == "git-workflow"


@pytest.mark.asyncio
async def test_recall_handles_multiple_names(store, mock_llm):
    mock_llm.chat.return_value = "git-workflow, user-role"
    engine = RecallEngine(memory=store, llm=mock_llm)
    results = await engine.recall("Tell me about this project")
    assert len(results) == 2


@pytest.mark.asyncio
async def test_recall_handles_llm_failure(store, mock_llm):
    mock_llm.chat.side_effect = Exception("API error")
    engine = RecallEngine(memory=store, llm=mock_llm)
    results = await engine.recall("anything")
    assert results == []


@pytest.mark.asyncio
async def test_recall_empty_store(tmp_path, mock_llm):
    store = MarkdownMemoryStore(base_dir=tmp_path)
    engine = RecallEngine(memory=store, llm=mock_llm)
    results = await engine.recall("anything")
    assert results == []
    mock_llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_recall_max_5(store, mock_llm):
    for i in range(10):
        store.save(name=f"mem-{i}", description=f"Memory {i}", type="project", content=f"Content {i}")
    mock_llm.chat.return_value = "mem-0, mem-1, mem-2, mem-3, mem-4, mem-5, mem-6"
    engine = RecallEngine(memory=store, llm=mock_llm)
    results = await engine.recall("everything")
    assert len(results) <= 5


@pytest.mark.asyncio
async def test_recall_ignores_nonexistent_names(store, mock_llm):
    mock_llm.chat.return_value = "git-workflow, nonexistent-memory"
    engine = RecallEngine(memory=store, llm=mock_llm)
    results = await engine.recall("test")
    assert len(results) == 1
    assert results[0]["name"] == "git-workflow"


def test_format_for_injection(store, mock_llm):
    engine = RecallEngine(memory=store, llm=mock_llm)
    memories = [store.load("git-workflow"), store.load("user-role")]
    text = engine.format_for_injection(memories)
    assert "# 相关记忆" in text
    assert "git-workflow" in text
    assert "user-role" in text
    assert "dev for development" in text


def test_format_for_injection_empty(store, mock_llm):
    engine = RecallEngine(memory=store, llm=mock_llm)
    assert engine.format_for_injection([]) == ""
