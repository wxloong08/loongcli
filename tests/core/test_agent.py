import pytest
import json
from unittest.mock import MagicMock
from loongcli.core.agent import AgentLoop, MAX_TOOL_CALLS_PER_TURN, LOOP_DETECT_THRESHOLD
from loongcli.core.llm import LLMClient
from loongcli.core.events import TextDelta, ToolCallStart, ToolCallResult, AgentDone, ConfirmRequest, BatchProgress
from loongcli.tools.base import Tool, ToolRegistry
from loongcli.security.permissions import PermissionChecker


class EchoTool(Tool):
    name = "echo"
    description = "Echoes input"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def execute(self, text: str) -> str:
        return f"echo: {text}"


def _make_text_chunks(texts: list[str]):
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


def _make_tool_call_chunks(tool_id: str, tool_name: str, arguments: dict):
    args_str = json.dumps(arguments)

    tc_delta = MagicMock()
    tc_delta.index = 0
    tc_delta.id = tool_id
    tc_delta.type = "function"
    tc_delta.function = MagicMock()
    tc_delta.function.name = tool_name
    tc_delta.function.arguments = args_str

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


@pytest.mark.asyncio
async def test_agent_text_only_response():
    llm = LLMClient(api_key="test")

    text_chunks = _make_text_chunks(["Hello", " World"])

    async def mock_stream(**kwargs):
        for c in text_chunks:
            yield c

    llm.chat_stream = mock_stream

    registry = ToolRegistry()
    checker = PermissionChecker()
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=checker)

    events = []
    async for event in agent.run_stream("hi"):
        events.append(event)

    text_events = [e for e in events if isinstance(e, TextDelta)]
    done_events = [e for e in events if isinstance(e, AgentDone)]
    assert len(text_events) == 2
    assert text_events[0].text == "Hello"
    assert len(done_events) == 1
    assert done_events[0].content == "Hello World"


@pytest.mark.asyncio
async def test_agent_tool_call_then_text():
    llm = LLMClient(api_key="test")

    tool_chunks = _make_tool_call_chunks("call_1", "echo", {"text": "world"})
    text_chunks = _make_text_chunks(["Done!"])

    call_count = 0

    async def mock_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        src = tool_chunks if call_count == 1 else text_chunks
        for c in src:
            yield c

    llm.chat_stream = mock_stream

    registry = ToolRegistry()
    registry.register(EchoTool())
    checker = PermissionChecker()
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=checker)

    events = []
    async for event in agent.run_stream("say world"):
        events.append(event)

    tool_start_events = [e for e in events if isinstance(e, ToolCallStart)]
    tool_result_events = [e for e in events if isinstance(e, ToolCallResult)]
    done_events = [e for e in events if isinstance(e, AgentDone)]

    assert len(tool_start_events) == 1
    assert tool_start_events[0].tool_name == "echo"
    assert len(tool_result_events) == 1
    assert tool_result_events[0].result == "echo: world"
    assert len(done_events) == 1
    assert done_events[0].content == "Done!"
    assert call_count == 2


@pytest.mark.asyncio
async def test_agent_max_iterations():
    llm = LLMClient(api_key="test")

    tool_chunks = _make_tool_call_chunks("call_1", "echo", {"text": "loop"})

    async def mock_stream(**kwargs):
        for c in tool_chunks:
            yield c

    llm.chat_stream = mock_stream

    registry = ToolRegistry()
    registry.register(EchoTool())
    checker = PermissionChecker()
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=checker, max_iterations=3)

    events = []
    async for event in agent.run_stream("loop forever"):
        events.append(event)

    done_events = [e for e in events if isinstance(e, AgentDone)]
    assert len(done_events) == 1
    assert "迭代上限" in done_events[0].content


class ShellTool(Tool):
    name = "shell"
    description = "Run a command"
    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }

    async def execute(self, command: str) -> str:
        return f"ran: {command}"


@pytest.mark.asyncio
async def test_agent_confirm_approved():
    llm = LLMClient(api_key="test")

    tool_chunks = _make_tool_call_chunks("call_1", "shell", {"command": "rm -rf /tmp/test"})
    text_chunks = _make_text_chunks(["Done"])

    call_count = 0

    async def mock_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        src = tool_chunks if call_count == 1 else text_chunks
        for c in src:
            yield c

    llm.chat_stream = mock_stream

    registry = ToolRegistry()
    registry.register(ShellTool())
    checker = PermissionChecker()
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=checker)

    events = []
    async for event in agent.run_stream("delete stuff"):
        if isinstance(event, ConfirmRequest):
            assert "shell" in event.risk_reason
            event.future.set_result(True)
        events.append(event)

    confirm_events = [e for e in events if isinstance(e, ConfirmRequest)]
    result_events = [e for e in events if isinstance(e, ToolCallResult)]
    assert len(confirm_events) == 1
    assert len(result_events) == 1
    assert "ran:" in result_events[0].result


@pytest.mark.asyncio
async def test_agent_confirm_denied():
    llm = LLMClient(api_key="test")

    tool_chunks = _make_tool_call_chunks("call_1", "shell", {"command": "rm -rf /tmp/test"})
    text_chunks = _make_text_chunks(["OK"])

    call_count = 0

    async def mock_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        src = tool_chunks if call_count == 1 else text_chunks
        for c in src:
            yield c

    llm.chat_stream = mock_stream

    registry = ToolRegistry()
    registry.register(ShellTool())
    checker = PermissionChecker()
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=checker)

    events = []
    async for event in agent.run_stream("delete stuff"):
        if isinstance(event, ConfirmRequest):
            event.future.set_result(False)
        events.append(event)

    result_events = [e for e in events if isinstance(e, ToolCallResult)]
    assert len(result_events) == 1
    assert "拒绝" in result_events[0].result


class ProgressTool(Tool):
    name = "progress_tool"
    description = "Tool with progress support"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    supports_progress = True

    def __init__(self):
        self._progress_callback = None

    async def execute(self, text: str) -> str:
        if self._progress_callback:
            self._progress_callback(BatchProgress(
                completed=1, total=2, task_index=0,
                task_prompt="sub-a", status="completed",
            ))
            self._progress_callback(BatchProgress(
                completed=2, total=2, task_index=1,
                task_prompt="sub-b", status="completed",
            ))
        return f"done: {text}"


@pytest.mark.asyncio
async def test_agent_progress_events():
    llm = LLMClient(api_key="test")

    tool_chunks = _make_tool_call_chunks("call_1", "progress_tool", {"text": "go"})
    text_chunks = _make_text_chunks(["OK"])

    call_count = 0

    async def mock_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        src = tool_chunks if call_count == 1 else text_chunks
        for c in src:
            yield c

    llm.chat_stream = mock_stream

    registry = ToolRegistry()
    registry.register(ProgressTool())
    checker = PermissionChecker()
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=checker)

    events = []
    async for event in agent.run_stream("do it"):
        events.append(event)

    progress_events = [e for e in events if isinstance(e, BatchProgress)]
    result_events = [e for e in events if isinstance(e, ToolCallResult)]
    assert len(progress_events) == 2
    assert progress_events[0].completed == 1
    assert progress_events[1].completed == 2
    assert len(result_events) == 1
    assert result_events[0].result == "done: go"


@pytest.mark.asyncio
async def test_loop_detection_same_tool_same_args():
    llm = LLMClient(api_key="test")

    tool_chunks = _make_tool_call_chunks("call_1", "echo", {"text": "same"})
    text_chunks = _make_text_chunks(["stopped"])

    call_count = 0

    async def mock_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= LOOP_DETECT_THRESHOLD:
            for c in _make_tool_call_chunks(f"call_{call_count}", "echo", {"text": "same"}):
                yield c
        else:
            for c in text_chunks:
                yield c

    llm.chat_stream = mock_stream

    registry = ToolRegistry()
    registry.register(EchoTool())
    checker = PermissionChecker()
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=checker)

    events = []
    async for event in agent.run_stream("repeat"):
        events.append(event)

    result_events = [e for e in events if isinstance(e, ToolCallResult)]
    loop_results = [e for e in result_events if "循环" in e.result]
    assert len(loop_results) == 1
    normal_results = [e for e in result_events if "echo:" in e.result]
    assert len(normal_results) == LOOP_DETECT_THRESHOLD - 1


@pytest.mark.asyncio
async def test_loop_detection_resets_on_different_args():
    llm = LLMClient(api_key="test")

    call_count = 0

    async def mock_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 4:
            for c in _make_tool_call_chunks(f"call_{call_count}", "echo", {"text": f"arg_{call_count}"}):
                yield c
        else:
            for c in _make_text_chunks(["done"]):
                yield c

    llm.chat_stream = mock_stream

    registry = ToolRegistry()
    registry.register(EchoTool())
    checker = PermissionChecker()
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=checker)

    events = []
    async for event in agent.run_stream("vary"):
        events.append(event)

    result_events = [e for e in events if isinstance(e, ToolCallResult)]
    loop_results = [e for e in result_events if "循环" in e.result]
    assert len(loop_results) == 0


@pytest.mark.asyncio
async def test_max_tool_calls_limit():
    llm = LLMClient(api_key="test")
    limit = 5

    call_count = 0

    async def mock_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= limit + 2:
            for c in _make_tool_call_chunks(f"call_{call_count}", "echo", {"text": f"t{call_count}"}):
                yield c
        else:
            for c in _make_text_chunks(["done"]):
                yield c

    llm.chat_stream = mock_stream

    registry = ToolRegistry()
    registry.register(EchoTool())
    checker = PermissionChecker()
    agent = AgentLoop(
        llm=llm, tool_registry=registry, permission_checker=checker,
        max_tool_calls=limit,
    )

    events = []
    async for event in agent.run_stream("many calls"):
        events.append(event)

    result_events = [e for e in events if isinstance(e, ToolCallResult)]
    limit_results = [e for e in result_events if "上限" in e.result]
    assert len(limit_results) >= 1
    normal_results = [e for e in result_events if "echo:" in e.result]
    assert len(normal_results) == limit


@pytest.mark.asyncio
async def test_tool_call_counters_reset_per_turn():
    llm = LLMClient(api_key="test")

    async def mock_stream(**kwargs):
        for c in _make_text_chunks(["hi"]):
            yield c

    llm.chat_stream = mock_stream

    registry = ToolRegistry()
    checker = PermissionChecker()
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=checker)

    agent._tool_call_count = 99
    agent._repeat_count = 5
    agent._last_tool_sig = "abc"

    async for _ in agent.run_stream("reset test"):
        pass

    assert agent._tool_call_count == 0
    assert agent._repeat_count == 0
    assert agent._last_tool_sig == ""


def test_tool_signature_deterministic():
    sig1 = AgentLoop._tool_signature("echo", {"text": "hello", "n": 1})
    sig2 = AgentLoop._tool_signature("echo", {"n": 1, "text": "hello"})
    assert sig1 == sig2

    sig3 = AgentLoop._tool_signature("echo", {"text": "world"})
    assert sig1 != sig3


def test_default_max_tool_calls():
    assert MAX_TOOL_CALLS_PER_TURN == 200


def test_default_loop_threshold():
    assert LOOP_DETECT_THRESHOLD == 3
