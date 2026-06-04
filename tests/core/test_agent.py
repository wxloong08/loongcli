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


class TestRebuildSystemPrompt:
    def test_rebuilds_when_builder_provided(self):
        registry = ToolRegistry()
        checker = PermissionChecker()
        llm = LLMClient(api_key="test")
        agent = AgentLoop(
            llm=llm, tool_registry=registry, permission_checker=checker,
            system_prompt="old prompt",
            system_prompt_builder=lambda: "new prompt",
        )
        assert agent.messages[0]["content"] == "old prompt"
        agent._rebuild_system_prompt()
        assert agent.messages[0]["content"] == "new prompt"

    def test_noop_without_builder(self):
        registry = ToolRegistry()
        checker = PermissionChecker()
        llm = LLMClient(api_key="test")
        agent = AgentLoop(
            llm=llm, tool_registry=registry, permission_checker=checker,
            system_prompt="original",
        )
        agent._rebuild_system_prompt()
        assert agent.messages[0]["content"] == "original"

    def test_noop_without_system_message(self):
        registry = ToolRegistry()
        checker = PermissionChecker()
        llm = LLMClient(api_key="test")
        agent = AgentLoop(
            llm=llm, tool_registry=registry, permission_checker=checker,
            system_prompt_builder=lambda: "rebuilt",
        )
        agent._rebuild_system_prompt()
        assert len(agent.messages) == 0


# ── agent-level retry tests ──


class FlakyTool(Tool):
    """Tool that fails once then succeeds."""
    name = "flaky"
    description = "Fails first call, succeeds on retry"
    parameters = {"type": "object", "properties": {}}

    def __init__(self):
        super().__init__()
        self.call_count = 0

    async def execute(self) -> str:
        self.call_count += 1
        if self.call_count == 1:
            from loongcli.tools.errors import ToolError
            raise ToolError("transient disk error", retryable=True)
        return f"succeeded on attempt {self.call_count}"


class PermanentFailTool(Tool):
    """Tool that always fails with a non-retryable error."""
    name = "permanent"
    description = "Always fails"
    parameters = {"type": "object", "properties": {}}

    async def execute(self) -> str:
        from loongcli.tools.errors import ToolError
        raise ToolError("permanent config error", retryable=False)


@pytest.mark.asyncio
async def test_agent_retries_on_retryable_tool_error():
    """Agent should retry once when ToolError(retryable=True) is raised."""
    from loongcli.core.events import AgentDone

    llm = LLMClient(api_key="test")
    tool = FlakyTool()
    tool_chunks = _make_tool_call_chunks("call_1", "flaky", {})
    text_chunks = _make_text_chunks(["done"])

    call_count = 0

    async def mock_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        src = tool_chunks if call_count == 1 else text_chunks
        for c in src:
            yield c

    llm.chat_stream = mock_stream

    registry = ToolRegistry()
    registry.register(tool)
    checker = PermissionChecker()

    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=checker)

    events = []
    async for event in agent.run_stream("use flaky"):
        events.append(event)

    results = [e for e in events if isinstance(e, ToolCallResult)]
    assert len(results) == 1
    assert "succeeded on attempt 2" in results[0].result
    assert tool.call_count == 2


@pytest.mark.asyncio
async def test_agent_does_not_retry_permanent_error():
    """Agent should NOT retry when ToolError(retryable=False)."""
    from loongcli.core.events import AgentDone

    llm = LLMClient(api_key="test")
    tool = PermanentFailTool()
    tool_chunks = _make_tool_call_chunks("call_1", "permanent", {})
    text_chunks = _make_text_chunks(["ok"])

    call_count = 0

    async def mock_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        src = tool_chunks if call_count == 1 else text_chunks
        for c in src:
            yield c

    llm.chat_stream = mock_stream

    registry = ToolRegistry()
    registry.register(tool)
    checker = PermissionChecker()

    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=checker)

    events = []
    async for event in agent.run_stream("use permanent"):
        events.append(event)

    results = [e for e in events if isinstance(e, ToolCallResult)]
    assert len(results) == 1
    assert "工具执行失败" in results[0].result
    assert "permanent config error" in results[0].result
