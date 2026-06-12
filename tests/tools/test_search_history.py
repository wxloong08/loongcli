"""search_history 工具：检索完整历史（含归档段）、AND 匹配、重叠去重。"""
from __future__ import annotations

import pytest

from loongcli.memory.conversation import ConversationStore
from loongcli.tools.search_history import SearchHistoryTool


@pytest.fixture
def store(tmp_path):
    return ConversationStore(base_dir=tmp_path / "sessions")


@pytest.fixture
def tool(store):
    return SearchHistoryTool(store)


def msgs(*contents, role="user"):
    return [{"role": role, "content": c} for c in contents]


async def test_finds_content_in_archived_segment(store, tool):
    """核心场景：被 compact 归档的内容（当前上下文已看不到）仍可检索到。"""
    original = msgs("数据库连接报错 ConnectionRefusedError 端口 5432", "其他无关消息")
    store.save(original)
    store.archive_segment(original, reason="auto-compact")
    store.save(msgs("[对话历史摘要] 处理了数据库问题"))

    result = await tool.execute(query="ConnectionRefusedError")
    assert "ConnectionRefusedError" in result
    assert "5432" in result


async def test_and_semantics(store, tool):
    store.save(msgs("苹果 香蕉", "苹果 梨", "香蕉 梨"))
    result = await tool.execute(query="苹果 香蕉")
    assert "找到 1 条匹配" in result


async def test_no_match(store, tool):
    store.save(msgs("一些内容"))
    result = await tool.execute(query="不存在的词xyz")
    assert "没有匹配" in result


async def test_empty_history(store, tool):
    result = await tool.execute(query="任何词")
    assert "没有可检索的历史" in result


async def test_dedup_overlap_between_segment_and_current(store, tool):
    """compact 保留最近几轮 → 归档段与当前 messages 有重叠，不应重复返回。"""
    overlap = "这条消息同时存在于归档段和当前上下文 特殊标记词"
    original = msgs("旧消息", overlap)
    store.save(original)
    store.archive_segment(original)
    store.save(msgs("摘要", overlap))

    result = await tool.execute(query="特殊标记词")
    assert "找到 1 条匹配" in result


async def test_searches_tool_call_arguments(store, tool):
    store.save([{
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "t1", "function": {"name": "edit_file", "arguments": '{"path": "special_target.py"}'}}],
    }])
    result = await tool.execute(query="special_target")
    assert "special_target" in result


async def test_limit_returns_most_recent(store, tool):
    store.save(msgs(*[f"重复词 第{i}条" for i in range(10)]))
    result = await tool.execute(query="重复词", limit=2)
    assert "第9条" in result
    assert "第0条" not in result


async def test_blank_query(store, tool):
    result = await tool.execute(query="   ")
    assert "请提供" in result
