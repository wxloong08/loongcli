"""执行中输入排队 + 迭代边界注入的 agent 层语义测试。"""
import json
import pytest
from unittest.mock import MagicMock

from loongcli.core.agent import AgentLoop
from loongcli.core.llm import LLMClient
from loongcli.core.events import QueuedInputInjected, AgentDone
from loongcli.tools.base import Tool, ToolRegistry
from loongcli.security.permissions import PermissionChecker


def _text_chunks(texts):
    chunks = []
    for t in texts:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = t
        chunk.choices[0].delta.tool_calls = None
        chunk.choices[0].finish_reason = None
        chunk.usage = None
        chunks.append(chunk)
    if chunks:
        chunks[-1].choices[0].finish_reason = "stop"
    return chunks


def _tool_call_chunks(tool_id, tool_name, arguments):
    tc_delta = MagicMock()
    tc_delta.index = 0
    tc_delta.id = tool_id
    tc_delta.type = "function"
    tc_delta.function = MagicMock()
    tc_delta.function.name = tool_name
    tc_delta.function.arguments = json.dumps(arguments)

    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = None
    chunk1.choices[0].delta.tool_calls = [tc_delta]
    chunk1.choices[0].finish_reason = None
    chunk1.usage = None

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = None
    chunk2.choices[0].delta.tool_calls = None
    chunk2.choices[0].finish_reason = "tool_calls"
    chunk2.usage = None
    return [chunk1, chunk2]


def _make_agent(llm, registry=None) -> AgentLoop:
    return AgentLoop(
        llm=llm,
        tool_registry=registry or ToolRegistry(),
        permission_checker=PermissionChecker(),
    )


class TestQueueSemantics:
    def test_take_clears_queue(self):
        agent = _make_agent(LLMClient(api_key="test"))
        agent.queue_user_input("a")
        agent.queue_user_input("b")
        assert agent.take_queued_inputs() == ["a", "b"]
        assert agent.take_queued_inputs() == []


class TestInjection:
    @pytest.mark.asyncio
    async def test_prequeued_injected_and_event_yielded(self):
        llm = LLMClient(api_key="test")
        chunks = _text_chunks(["ok"])

        async def mock_stream(**kwargs):
            for c in chunks:
                yield c

        llm.chat_stream = mock_stream
        agent = _make_agent(llm)
        agent.queue_user_input("插话内容")

        events = []
        async for ev in agent.run_stream("hi"):
            events.append(ev)

        injected = [e for e in events if isinstance(e, QueuedInputInjected)]
        assert len(injected) == 1
        assert injected[0].text == "插话内容"
        # 消息序：user(hi) → user(插话) → assistant
        roles = [(m["role"], m.get("content")) for m in agent.messages]
        assert ("user", "插话内容") in roles
        idx_hi = next(i for i, m in enumerate(agent.messages) if m.get("content") == "hi")
        idx_q = next(i for i, m in enumerate(agent.messages) if m.get("content") == "插话内容")
        assert idx_hi < idx_q

    @pytest.mark.asyncio
    async def test_midturn_queue_injected_next_iteration(self):
        """工具执行中排队 → 下一迭代注入，且落在完整的 assistant→tool 序列之后。"""
        llm = LLMClient(api_key="test")
        tool_chunks = _tool_call_chunks("c1", "poke", {})
        text_chunks = _text_chunks(["Done"])
        call_count = 0

        async def mock_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            for c in (tool_chunks if call_count == 1 else text_chunks):
                yield c

        llm.chat_stream = mock_stream

        agent_holder: list[AgentLoop] = []

        class PokeTool(Tool):
            name = "poke"
            description = "queues a message during execution"
            parameters = {"type": "object", "properties": {}}

            async def execute(self) -> str:
                agent_holder[0].queue_user_input("执行中插话")
                return "poked"

        registry = ToolRegistry()
        registry.register(PokeTool())
        agent = _make_agent(llm, registry)
        agent_holder.append(agent)

        events = []
        async for ev in agent.run_stream("run poke"):
            events.append(ev)

        injected = [e for e in events if isinstance(e, QueuedInputInjected)]
        assert len(injected) == 1
        # 序列断言：assistant(tool_calls) → tool → user(插话)
        roles = [m["role"] for m in agent.messages]
        i_tool = roles.index("tool")
        i_q = next(i for i, m in enumerate(agent.messages)
                   if m["role"] == "user" and m.get("content") == "执行中插话")
        assert i_tool < i_q
        assert agent.messages[i_q - 1]["role"] == "tool"  # 紧跟在完整 tool 序列后

    @pytest.mark.asyncio
    async def test_no_queue_zero_events(self):
        """无排队 → 零 QueuedInputInjected：非交互模式/子代理行为不变的隔离证明。"""
        llm = LLMClient(api_key="test")
        chunks = _text_chunks(["ok"])

        async def mock_stream(**kwargs):
            for c in chunks:
                yield c

        llm.chat_stream = mock_stream
        agent = _make_agent(llm)

        events = []
        async for ev in agent.run_stream("hi"):
            events.append(ev)

        assert not any(isinstance(e, QueuedInputInjected) for e in events)
        assert any(isinstance(e, AgentDone) for e in events)

    @pytest.mark.asyncio
    async def test_multiple_queued_injected_in_order(self):
        llm = LLMClient(api_key="test")
        chunks = _text_chunks(["ok"])

        async def mock_stream(**kwargs):
            for c in chunks:
                yield c

        llm.chat_stream = mock_stream
        agent = _make_agent(llm)
        agent.queue_user_input("第一条")
        agent.queue_user_input("第二条")

        events = []
        async for ev in agent.run_stream("hi"):
            events.append(ev)

        injected = [e.text for e in events if isinstance(e, QueuedInputInjected)]
        assert injected == ["第一条", "第二条"]
