import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
from loongcli.memory.auto_extract import AutoExtractor
from loongcli.memory.markdown_store import MarkdownMemoryStore
from loongcli.core.agent import AgentLoop, AgentServices
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


def test_build_prompt_with_image_message_no_crash(store, mock_llm):
    """含图片的多模态 user 消息不能让提取 prompt 构造崩（'list' has no strip 回归）。"""
    ext = AutoExtractor(memory=store, llm=mock_llm)
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "这是什么界面"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,X"}},
        ]},
        {"role": "assistant", "content": "这是一个登录页"},
    ]
    prompt = ext._build_prompt(messages)  # 不抛异常
    assert "这是什么界面" in prompt
    assert "登录页" in prompt


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
    assert len(count) == 1 and count[0]["name"] == "user-lang"
    mem = store.load("user-lang")
    assert mem is not None
    assert "Python" in mem["content"]


@pytest.mark.asyncio
async def test_extract_drops_feedback_without_why(store, mock_llm):
    """闸 3: 缺 Why 的 feedback/project（一次性意见）被挡在门外。"""
    mock_llm.chat.return_value = json.dumps([
        {"name": "no-why", "description": "one-off opinion", "type": "feedback",
         "content": "用户说这段不通顺。"},
    ])
    extractor = AutoExtractor(store, mock_llm)
    count = await extractor.extract([{"role": "user", "content": "这段不通顺"}])
    assert count == []
    assert store.load("no-why") is None


@pytest.mark.asyncio
async def test_extract_keeps_user_without_why(store, mock_llm):
    """Why 闸只管 feedback/project，user/reference 不要求 Why，照常保存。"""
    mock_llm.chat.return_value = json.dumps([
        {"name": "u-role", "description": "backend engineer", "type": "user",
         "content": "用户是后端工程师。"},
    ])
    extractor = AutoExtractor(store, mock_llm)
    count = await extractor.extract([{"role": "user", "content": "我是后端"}])
    assert len(count) == 1
    assert count[0]["name"] == "u-role"
    assert store.load("u-role") is not None


@pytest.mark.asyncio
async def test_extract_handles_llm_failure(store, mock_llm):
    mock_llm.chat.side_effect = Exception("API down")
    extractor = AutoExtractor(store, mock_llm)
    count = await extractor.extract([{"role": "user", "content": "hi"}])
    assert count == []


# --- AgentLoop integration test ---

@pytest.mark.asyncio
async def test_agent_does_not_fire_auto_extract_per_turn():
    """回合内不再触发自动提取——提取已移到了会话退出时（main.py 的 finally 块）。

    会话级别的完整历史交给提取器，能自然区分"跨会话耐久的"和"过程性的 skill 产出"，
    不再把 jobhunter 类的扫描结论误记为记忆。
    """
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
        services=AgentServices(auto_extractor=auto_extractor),
    )

    events = []
    async for event in agent.run_stream("hi"):
        events.append(event)

    assert any(isinstance(e, AgentDone) for e in events)
    auto_extractor.extract.assert_not_called()  # 已移出 run_stream


# ── 闸 4：否定存在断言拦截（毒记忆回归：grep 故障产出 "cache_aware 不存在"） ──

@pytest.mark.asyncio
async def test_gate4_drops_negative_existence_claim(store, mock_llm):
    """"搜了没找到 → 不存在" 式 project 记忆必须被确定性拦下，即使带了像样的 Why。"""
    mock_llm.chat.return_value = json.dumps([{
        "name": "project-x-cache-aware-not-found",
        "description": "项目中无 cache_aware 标识符",
        "type": "project",
        "content": "grep 未找到任何匹配。**Why:** 搜索结果为空。**How to apply:** 视为不存在。",
    }])
    ext = AutoExtractor(memory=store, llm=mock_llm)
    saved = await ext.extract([{"role": "user", "content": "搜一下"}])
    assert saved == []
    assert store.list_all() == []


@pytest.mark.asyncio
async def test_gate4_allows_normal_project_memory(store, mock_llm):
    """正常的项目决策记忆不受闸 4 影响。"""
    mock_llm.chat.return_value = json.dumps([{
        "name": "project-uses-sqlite",
        "description": "项目选用 SQLite",
        "type": "project",
        "content": "存储层用 SQLite。**Why:** 单机场景要简单。**How to apply:** 新模块直接复用。",
    }])
    ext = AutoExtractor(memory=store, llm=mock_llm)
    saved = await ext.extract([{"role": "user", "content": "定一下存储"}])
    assert len(saved) == 1


# ── 溯源字段 + 写入可见化（毒记忆事故后的两项加固） ──

@pytest.mark.asyncio
async def test_extract_writes_source_session(store, mock_llm):
    """auto-extract 落库的记忆必须带 source_session: auto-extract:<会话id>。"""
    mock_llm.chat.return_value = json.dumps([{
        "name": "u-fact", "description": "d", "type": "user", "content": "事实",
    }])
    ext = AutoExtractor(memory=store, llm=mock_llm, session_provider=lambda: "abc123def456")
    saved = await ext.extract([{"role": "user", "content": "hi"}])
    assert len(saved) == 1
    raw = (store.base_dir / "u-fact.md").read_text(encoding="utf-8")
    assert "source_session:" in raw and "auto-extract:abc123def456" in raw


@pytest.mark.asyncio
async def test_extract_source_without_provider(store, mock_llm):
    """未接线 session_provider 时也要有写入者标记（auto-extract），不留空白溯源。"""
    mock_llm.chat.return_value = json.dumps([{
        "name": "u-fact2", "description": "d", "type": "user", "content": "事实",
    }])
    ext = AutoExtractor(memory=store, llm=mock_llm)
    await ext.extract([{"role": "user", "content": "hi"}])
    raw = (store.base_dir / "u-fact2.md").read_text(encoding="utf-8")
    assert "source_session: auto-extract" in raw


def test_store_save_without_source_no_field(store):
    """不传 source 的写入（如用户 /remember）不出现 source_session 行，老格式不变。"""
    store.save(name="plain", description="d", type="user", content="c")
    raw = (store.base_dir / "plain.md").read_text(encoding="utf-8")
    assert "source_session" not in raw
