import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
from loongcli.memory.auto_extract import AutoExtractor
from loongcli.memory.markdown_store import MarkdownMemoryStore
from loongcli.core.agent import AgentLoop
from loongcli.core.llm import LLMClient
from loongcli.core.events import AgentDone
from loongcli.tools.base import ToolRegistry
from loongcli.security.permissions import PermissionChecker


def _make_chunk(content=None, finish_reason=None):
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    delta = MagicMock(spec=[])
    delta.content = content
    delta.tool_calls = None
    chunk.choices[0].delta = delta
    chunk.choices[0].finish_reason = finish_reason
    chunk.usage = None
    return chunk


# --- AutoExtractor tests ---

@pytest.fixture
def store(tmp_path):
    return MarkdownMemoryStore(base_dir=tmp_path)


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.chat = AsyncMock()
    return llm


def test_parse_memories_valid_json(store, mock_llm):
    extractor = AutoExtractor(store, mock_llm)
    result = extractor._parse_memories(
        '[{"name": "test", "description": "d", "type": "user", "content": "c"}]'
    )
    assert len(result) == 1
    assert result[0]["name"] == "test"


def test_parse_memories_filter_invalid(store, mock_llm):
    extractor = AutoExtractor(store, mock_llm)
    result = extractor._parse_memories(
        '[{"name": "ok", "content": "yes"}, {"bad": "missing content"}]'
    )
    assert len(result) == 1
    assert result[0]["name"] == "ok"


def test_parse_memories_empty(store, mock_llm):
    extractor = AutoExtractor(store, mock_llm)
    assert extractor._parse_memories("[]") == []


def test_parse_memories_garbage(store, mock_llm):
    extractor = AutoExtractor(store, mock_llm)
    assert extractor._parse_memories("just some text") == []


def test_build_prompt(store, mock_llm):
    extractor = AutoExtractor(store, mock_llm)
    messages = [
        {"role": "user", "content": "I prefer Python"},
        {"role": "assistant", "content": "Noted, you like Python."},
    ]
    prompt = extractor._build_prompt(messages)
    assert "I prefer Python" in prompt
    assert "Noted" in prompt


@pytest.mark.asyncio
async def test_extract_saves_memories(store, mock_llm):
    mock_llm.chat.return_value = json.dumps([
        {"name": "user-lang", "description": "Prefers Python", "type": "user",
         "content": "User writes Python. **Why:** main language. **How to apply:** suggest Python libraries."},
    ])
    extractor = AutoExtractor(store, mock_llm)
    count = await extractor.extract([
        {"role": "user", "content": "I write Python"},
        {"role": "assistant", "content": "Got it."},
    ])
    assert count == 1
    mem = store.load("user-lang")
    assert mem is not None
    assert "Python" in mem["content"]


@pytest.mark.asyncio
async def test_extract_handles_llm_failure(store, mock_llm):
    mock_llm.chat.side_effect = Exception("API down")
    extractor = AutoExtractor(store, mock_llm)
    count = await extractor.extract([{"role": "user", "content": "hi"}])
    assert count == 0


# --- AgentLoop integration test ---

@pytest.mark.asyncio
async def test_agent_fires_auto_extract():
    llm = LLMClient(api_key="test")

    async def mock_stream(**kwargs):
        yield _make_chunk(content="Hello!")
        yield _make_chunk(finish_reason="stop")

    llm.chat_stream = mock_stream

    auto_extractor = AsyncMock()
    auto_extractor.extract = AsyncMock(return_value=0)

    agent = AgentLoop(
        llm=llm,
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(),
        system_prompt="You are helpful.",
        auto_extractor=auto_extractor,
    )

    events = []
    async for event in agent.run_stream("hi"):
        events.append(event)

    assert any(isinstance(e, AgentDone) for e in events)
    auto_extractor.extract.assert_called_once()
