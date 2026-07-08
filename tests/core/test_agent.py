import asyncio
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


@pytest.mark.asyncio
async def test_invalid_tool_args_json_does_not_crash():
    """Malformed tool-argument JSON must become a tool error fed back to the
    model, never an exception that escapes run_stream and kills the TUI."""
    llm = LLMClient(api_key="test")

    bad_chunks = _make_tool_call_chunks("call_1", "echo", {"text": "x"})
    # Corrupt the streamed arguments into invalid JSON.
    bad_chunks[0].choices[0].delta.tool_calls[0].function.arguments = "{not valid"
    text_chunks = _make_text_chunks(["recovered"])

    call_count = 0

    async def mock_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        src = bad_chunks if call_count == 1 else text_chunks
        for c in src:
            yield c

    llm.chat_stream = mock_stream

    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=PermissionChecker())

    events = []
    async for event in agent.run_stream("go"):
        events.append(event)

    result_events = [e for e in events if isinstance(e, ToolCallResult)]
    assert len(result_events) == 1
    assert "不是合法 JSON" in result_events[0].result
    done = [e for e in events if isinstance(e, AgentDone)]
    assert done and done[0].content == "recovered"


def _img_user_msg():
    return {"role": "user", "content": [
        {"type": "text", "text": "看图"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,X"}},
    ]}


@pytest.mark.asyncio
async def test_image_rejected_without_vision():
    """vision=False 的模型收到图片 → 明确拒绝，且不发起任何 LLM 请求。"""
    llm = LLMClient(api_key="test", vision=False)
    created = {"v": False}

    async def _agen():
        for c in _make_text_chunks(["hi"]):
            yield c

    def mock_stream(**kwargs):
        created["v"] = True
        return _agen()

    llm.chat_stream = mock_stream

    agent = AgentLoop(llm=llm, tool_registry=ToolRegistry(), permission_checker=PermissionChecker())
    agent.messages.append(_img_user_msg())  # 模拟 Step 4 的图片入口

    events = []
    async for e in agent.run_stream("继续"):
        events.append(e)

    done = [e for e in events if isinstance(e, AgentDone)]
    assert done and "视觉" in done[0].content
    assert created["v"] is False  # chat_stream 未被调用


@pytest.mark.asyncio
async def test_image_proceeds_with_vision():
    """vision=True 的模型收到图片 → 正常进入发送路径。"""
    llm = LLMClient(api_key="test", vision=True)

    async def mock_stream(**kwargs):
        for c in _make_text_chunks(["看到了一个登录页"]):
            yield c

    llm.chat_stream = mock_stream

    agent = AgentLoop(llm=llm, tool_registry=ToolRegistry(), permission_checker=PermissionChecker())
    agent.messages.append(_img_user_msg())

    events = []
    async for e in agent.run_stream("这是什么"):
        events.append(e)

    done = [e for e in events if isinstance(e, AgentDone)]
    assert done and "视觉" not in done[0].content
    assert done[0].content == "看到了一个登录页"


@pytest.mark.asyncio
async def test_run_stream_builds_image_message(tmp_path):
    """--image 路径经 run_stream 构造成多模态 user 消息（text + loongimg:// 引用）。"""
    p = tmp_path / "m.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\ndata")
    llm = LLMClient(api_key="test", vision=True)

    async def mock_stream(**kwargs):
        for c in _make_text_chunks(["ok"]):
            yield c

    llm.chat_stream = mock_stream
    agent = AgentLoop(llm=llm, tool_registry=ToolRegistry(), permission_checker=PermissionChecker())

    async for _ in agent.run_stream("这是什么", images=[str(p)]):
        pass

    content = [m for m in agent.messages if m.get("role") == "user"][-1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "这是什么"}
    # 消息里只存引用，base64 由 chat_stream 发送时内联
    assert content[1]["image_url"]["url"].startswith("loongimg://")


@pytest.mark.asyncio
async def test_read_file_image_injected_as_user_message(tmp_path):
    """read_file 读图（vision 开）：tool 消息占位，图片以紧随其后的 user 消息注入。"""
    from loongcli.core.agent import IMAGE_READ_PLACEHOLDER
    from loongcli.tools.read_file import ReadFileTool

    p = tmp_path / "ui.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\ndata")

    llm = LLMClient(api_key="test", vision=True)
    tool_chunks = _make_tool_call_chunks("call_1", "read_file", {"path": str(p)})
    text_chunks = _make_text_chunks(["看到了"])
    call_count = 0

    async def mock_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        for c in (tool_chunks if call_count == 1 else text_chunks):
            yield c

    llm.chat_stream = mock_stream
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=PermissionChecker())

    events = []
    async for e in agent.run_stream("看看这张图"):
        events.append(e)

    # ToolCallResult 事件展示占位文本，而非原始 sentinel JSON
    results = [e for e in events if isinstance(e, ToolCallResult)]
    assert results[0].result == IMAGE_READ_PLACEHOLDER.format(n=1)

    # 消息序列：tool(占位) → user(图片引用块)，注入在 tool 结果之后
    tool_idx = next(i for i, m in enumerate(agent.messages) if m.get("role") == "tool")
    assert "见下一条消息" in agent.messages[tool_idx]["content"]
    inject = agent.messages[tool_idx + 1]
    assert inject["role"] == "user"
    assert str(p) in inject["content"][0]["text"]
    assert inject["content"][1]["image_url"]["url"].startswith("loongimg://")


@pytest.mark.asyncio
async def test_read_file_image_without_vision_explicit_text(tmp_path):
    """read_file 读图（vision 关）：明确无视觉文本，不注入图片。"""
    from loongcli.core.agent import NO_VISION_READ_MSG
    from loongcli.tools.read_file import ReadFileTool

    p = tmp_path / "ui.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\ndata")

    llm = LLMClient(api_key="test", vision=False)
    tool_chunks = _make_tool_call_chunks("call_1", "read_file", {"path": str(p)})
    text_chunks = _make_text_chunks(["好的"])
    call_count = 0

    async def mock_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        for c in (tool_chunks if call_count == 1 else text_chunks):
            yield c

    llm.chat_stream = mock_stream
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=PermissionChecker())

    async for _ in agent.run_stream("看看这张图"):
        pass

    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert NO_VISION_READ_MSG in tool_msgs[0]["content"]
    # 无任何图片注入
    assert not any(isinstance(m.get("content"), list) for m in agent.messages)


@pytest.mark.asyncio
async def test_run_stream_recycles_old_images(tmp_path):
    """多轮多图后，超出保留窗口的旧图被替换为占位文本，最近 N 张保留。"""
    from loongcli.core.messages import count_images, KEEP_RECENT_IMAGES, IMAGE_DROPPED_PLACEHOLDER

    llm = LLMClient(api_key="test", vision=True)

    async def mock_stream(**kwargs):
        for c in _make_text_chunks(["ok"]):
            yield c

    llm.chat_stream = mock_stream
    agent = AgentLoop(llm=llm, tool_registry=ToolRegistry(), permission_checker=PermissionChecker())

    n_turns = KEEP_RECENT_IMAGES + 2
    for i in range(n_turns):
        p = tmp_path / f"img{i}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + str(i).encode())  # 内容各异，哈希不同
        async for _ in agent.run_stream(f"看图{i}", images=[str(p)]):
            pass

    assert count_images(agent.messages) == KEEP_RECENT_IMAGES
    placeholders = [
        b for m in agent.messages if isinstance(m.get("content"), list)
        for b in m["content"]
        if b == {"type": "text", "text": IMAGE_DROPPED_PLACEHOLDER}
    ]
    assert len(placeholders) == n_turns - KEEP_RECENT_IMAGES
    # 最近一张仍是引用块
    last_user = [m for m in agent.messages if m.get("role") == "user"][-1]
    assert last_user["content"][1]["image_url"]["url"].startswith("loongimg://")


@pytest.mark.asyncio
async def test_run_stream_image_missing_file(tmp_path):
    """图片文件缺失 → 明确错误，不发起请求。"""
    llm = LLMClient(api_key="test", vision=True)
    created = {"v": False}

    def mock_stream(**kwargs):
        created["v"] = True
        async def _a():
            yield None
        return _a()

    llm.chat_stream = mock_stream
    agent = AgentLoop(llm=llm, tool_registry=ToolRegistry(), permission_checker=PermissionChecker())

    events = []
    async for e in agent.run_stream("看图", images=[str(tmp_path / "nope.png")]):
        events.append(e)

    done = [e for e in events if isinstance(e, AgentDone)]
    assert done and "图片加载失败" in done[0].content
    assert created["v"] is False


@pytest.mark.asyncio
async def test_run_stream_image_without_vision_rejected(tmp_path):
    """带 --image 但 role 未开视觉 → 立刻拒绝，连文件都不读。"""
    llm = LLMClient(api_key="test", vision=False)
    agent = AgentLoop(llm=llm, tool_registry=ToolRegistry(), permission_checker=PermissionChecker())
    # 传一个不存在的路径：若门控在前，不会因文件不存在报错，而是报视觉未开
    events = []
    async for e in agent.run_stream("看图", images=[str(tmp_path / "nope.png")]):
        events.append(e)
    done = [e for e in events if isinstance(e, AgentDone)]
    assert done and "视觉" in done[0].content


class FakeMCPTool(Tool):
    name = "searxng__web_search"
    description = "Search the web via MCP"
    parameters = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }
    is_mcp = True

    async def execute(self, q: str) -> str:
        return "results"


@pytest.mark.asyncio
async def test_mcp_tool_requires_confirmation():
    """External MCP tools must go through the confirmation layer, not auto-run."""
    llm = LLMClient(api_key="test")
    tool_chunks = _make_tool_call_chunks("call_1", "searxng__web_search", {"q": "x"})
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
    registry.register(FakeMCPTool())
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=PermissionChecker())

    confirms = []
    async for event in agent.run_stream("search it"):
        if isinstance(event, ConfirmRequest):
            confirms.append(event)
            event.future.set_result(True)

    assert len(confirms) == 1
    assert "MCP" in confirms[0].risk_reason


class FailingWriteTool(Tool):
    name = "write_file"
    description = "Writes a file (always raises, to simulate a mid-write crash)"
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path"],
    }

    async def execute(self, path: str, content: str = "") -> str:
        raise RuntimeError("disk full")


@pytest.mark.asyncio
async def test_checkpoint_restored_when_modify_tool_raises():
    """A modifying tool that raises must trigger a checkpoint restore (roll back
    the half-written file), not a discard."""
    from unittest.mock import MagicMock

    llm = LLMClient(api_key="test")
    tool_chunks = _make_tool_call_chunks("call_1", "write_file", {"path": "notes.txt", "content": "y"})
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
    registry.register(FailingWriteTool())
    ckpt = MagicMock()
    ckpt.save.return_value = "ckpt-1"
    agent = AgentLoop(
        llm=llm, tool_registry=registry,
        permission_checker=PermissionChecker(), checkpoint_manager=ckpt,
    )

    async for _ in agent.run_stream("write it"):
        pass

    ckpt.save.assert_called_once()
    ckpt.restore.assert_called_once_with("ckpt-1")
    ckpt.discard.assert_not_called()


@pytest.mark.asyncio
async def test_noninteractive_confirm_auto_denied():
    """interactive=False must turn a CONFIRM into a DENY without emitting a
    ConfirmRequest (nobody would answer it in a SubAgent) or running the tool."""
    llm = LLMClient(api_key="test")

    tool_chunks = _make_tool_call_chunks("call_1", "shell", {"command": "curl http://evil"})
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
    checker = PermissionChecker()  # DEFAULT mode → curl is CONFIRM
    agent = AgentLoop(
        llm=llm, tool_registry=registry,
        permission_checker=checker, interactive=False,
    )

    events = []
    async for event in agent.run_stream("fetch stuff"):
        events.append(event)

    assert not [e for e in events if isinstance(e, ConfirmRequest)]
    result_events = [e for e in events if isinstance(e, ToolCallResult)]
    assert len(result_events) == 1
    assert "自动拒绝" in result_events[0].result
    assert "ran:" not in result_events[0].result  # tool never executed


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
    agent._recent_tool_sigs.append("abc")
    agent._recent_tool_sigs.append("abc")

    async for _ in agent.run_stream("reset test"):
        pass

    assert agent._tool_call_count == 0
    assert len(agent._recent_tool_sigs) == 0


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


@pytest.mark.asyncio
async def test_plan_mode_rejects_out_of_list_tool():
    """派发层硬闸：规划模式下模型硬调非只读工具（真机：qwen 无视 schema 过滤发 shell）
    必须被拒绝且不执行，错误信息引导走 plan → exit_plan_mode 审批流。"""
    llm = LLMClient(api_key="test")

    executed = {"v": False}

    class SpyTool(EchoTool):
        async def execute(self, text: str) -> str:
            executed["v"] = True
            return f"echo: {text}"

    tool_chunks = _make_tool_call_chunks("call_1", "echo", {"text": "hi"})
    text_chunks = _make_text_chunks(["好的，我先调研。"])
    call_count = 0

    async def mock_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        for c in (tool_chunks if call_count == 1 else text_chunks):
            yield c

    llm.chat_stream = mock_stream
    registry = ToolRegistry()
    registry.register(SpyTool())
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=PermissionChecker())
    agent.enter_plan_mode()

    events = []
    async for e in agent.run_stream("统计文件数"):
        events.append(e)

    assert executed["v"] is False  # 工具没被执行
    results = [e for e in events if isinstance(e, ToolCallResult)]
    assert results and "规划模式" in results[0].result
    assert "exit_plan_mode" in results[0].result
    # 拒绝信息作为 tool 消息喂回了模型
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert any("规划模式" in m["content"] for m in tool_msgs)


@pytest.mark.asyncio
async def test_plan_approval_auto_edits_sets_flag():
    """审批菜单选「批准并自动接受编辑」→ 置 permission_checker.auto_accept_edits，
    安全语义收在 agent 层，TUI 只回传选择字符串。"""
    from loongcli.core.events import PlanApproval

    class FakeExitPlan(Tool):
        name = "exit_plan_mode"
        description = "submit plan"
        parameters = {"type": "object", "properties": {}}

        async def execute(self) -> str:
            return '{"__plan_approval__": true, "plan_id": "p1", "plan_summary": "步骤1"}'

    llm = LLMClient(api_key="test")
    tool_chunks = _make_tool_call_chunks("call_1", "exit_plan_mode", {})
    text_chunks = _make_text_chunks(["开始执行"])
    call_count = 0

    async def mock_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        for c in (tool_chunks if call_count == 1 else text_chunks):
            yield c

    llm.chat_stream = mock_stream
    registry = ToolRegistry()
    registry.register(FakeExitPlan())
    checker = PermissionChecker()
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=checker)
    agent.enter_plan_mode()

    results = []
    async for e in agent.run_stream("做个计划"):
        if isinstance(e, PlanApproval):
            e.future.set_result("approve_auto_edits")
        if isinstance(e, ToolCallResult):
            results.append(e.result)

    assert checker.auto_accept_edits is True
    assert agent.plan_mode is False  # 已退出规划模式
    assert any("自动接受" in r for r in results)


@pytest.mark.asyncio
async def test_exhausted_tools_withdrawn_and_force_stop():
    """撞单轮工具上限后：工具被收走（请求不带 tools）；模型仍硬发调用 → 两振强制终止，
    不陪它耗到 max_iterations（真机：单轮膨胀到 97 万 prompt tokens 的事故回归）。"""
    llm = LLMClient(api_key="test")
    limit = 3

    llm_rounds = 0
    tools_seen: list = []

    async def mock_stream(**kwargs):
        nonlocal llm_rounds
        llm_rounds += 1
        tools_seen.append(kwargs.get("tools"))
        # 永不悔改：每轮都发工具调用
        for c in _make_tool_call_chunks(f"call_{llm_rounds}", "echo", {"text": f"t{llm_rounds}"}):
            yield c

    llm.chat_stream = mock_stream
    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = AgentLoop(
        llm=llm, tool_registry=registry, permission_checker=PermissionChecker(),
        max_tool_calls=limit, max_iterations=50,
    )

    events = []
    async for e in agent.run_stream("统计一切"):
        events.append(e)

    done = [e for e in events if isinstance(e, AgentDone)]
    assert done and "强制结束" in done[0].content
    # 撞限后的迭代请求必须不带 tools（收走），且两振后终止——远小于 max_iterations
    assert tools_seen[-1] is None and tools_seen[-2] is None
    assert llm_rounds <= limit + 4
    # 每个 tool_call_id 都有响应（协议完整，不留悬空调用）
    tc_ids = {tc["id"] for m in agent.messages if m.get("role") == "assistant" for tc in (m.get("tool_calls") or [])}
    answered = {m.get("tool_call_id") for m in agent.messages if m.get("role") == "tool"}
    assert tc_ids <= answered


@pytest.mark.asyncio
async def test_plan_mode_allows_mcp_tool_via_confirm():
    """规划模式放行 MCP（用户确认制）：调用过确认闸后执行，而非被派发硬闸拒绝。
    用户拍板：自己配的 MCP 批准即用，别人的靠不批来拒——人是闸门。"""
    class FakeMcpSearch(Tool):
        name = "searx__web_search"
        description = "web search"
        is_mcp = True
        parameters = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}

        async def execute(self, q: str) -> str:
            return f"results for {q}"

    llm = LLMClient(api_key="test")
    tool_chunks = _make_tool_call_chunks("call_1", "searx__web_search", {"q": "行业调研"})
    text_chunks = _make_text_chunks(["调研完成，开始规划。"])
    call_count = 0

    async def mock_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        for c in (tool_chunks if call_count == 1 else text_chunks):
            yield c

    llm.chat_stream = mock_stream
    registry = ToolRegistry()
    registry.register(FakeMcpSearch())
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=PermissionChecker())
    agent.enter_plan_mode()

    confirms = []
    results = []
    async for e in agent.run_stream("帮我规划行业调研"):
        if isinstance(e, ConfirmRequest):
            confirms.append(e.tool_name)
            e.future.set_result(True)  # 用户批准
        if isinstance(e, ToolCallResult):
            results.append(e.result)

    assert confirms == ["searx__web_search"]           # 走的是确认闸，不是硬闸拒绝
    assert any("results for 行业调研" in r for r in results)  # 批准后真执行
    assert not any("规划模式下工具" in r for r in results)     # 没被派发闸拦


# ── 取消路径：supports_progress 工具跑在独立 exec_task 里，消费者被取消时必须收走 ──

@pytest.mark.asyncio
async def test_exec_tool_stream_reaps_exec_task_on_cancel():
    """回归：CancelledError 打在 queue.get() 上，exec_task 不随消费者消亡——
    不在 finally 收走的话，工具协程连同 shell 子进程一起变孤儿。"""

    class HangingProgressTool(Tool):
        name = "hangtool"
        description = "hangs forever"
        parameters = {"type": "object", "properties": {}}
        supports_progress = True

        def __init__(self):
            self._progress_callback = None
            self.got_cancelled = asyncio.Event()

        async def execute(self) -> str:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.got_cancelled.set()
                raise
            return "done"

    hang = HangingProgressTool()
    registry = ToolRegistry()
    registry.register(hang)
    agent = AgentLoop(
        llm=LLMClient(api_key="test"),
        tool_registry=registry,
        permission_checker=PermissionChecker(),
    )

    async def consume():
        async for _ in agent._exec_tool_stream("hangtool", {}):
            pass

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0.2)  # 进入 queue.get 轮询
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert hang.got_cancelled.is_set()  # exec_task 被收走，而非留在后台挂 60s


# ── 循环检测:周期 1(同参连打)与周期 2/3(交替模式)── 拍板 2026-07-07

def _loop_agent():
    return AgentLoop(
        llm=LLMClient(api_key="test"),
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(),
    )


def test_check_loop_period1_same_args():
    agent = _loop_agent()
    assert agent._check_loop("shell", {"command": "echo X"}) is None
    assert agent._check_loop("shell", {"command": "echo X"}) is None
    warn = agent._check_loop("shell", {"command": "echo X"})
    assert warn and "检测到循环" in warn and "3 次" in warn


def test_check_loop_period2_alternation():
    # A-B-A-B-A-B:旧实现每次签名都变、计数归零,抓不到;窗口检测在第 6 次拦下
    agent = _loop_agent()
    for i in range(5):
        call = ("read_file", {"path": "a.py"}) if i % 2 == 0 else ("read_file", {"path": "b.py"})
        assert agent._check_loop(*call) is None, f"第 {i+1} 次不应触发"
    warn = agent._check_loop("read_file", {"path": "b.py"})
    assert warn and "周期 2" in warn


def test_check_loop_period3_rotation():
    # A-B-C ×3:第 9 次拦下
    calls = [("shell", {"command": c}) for c in ("x", "y", "z")]
    agent = _loop_agent()
    for i in range(8):
        assert agent._check_loop(*calls[i % 3]) is None, f"第 {i+1} 次不应触发"
    warn = agent._check_loop(*calls[2])
    assert warn and "周期 3" in warn


def test_check_loop_varying_args_not_flagged():
    # 合法交替:编辑参数每轮都变,签名不同,整周期永不重复
    agent = _loop_agent()
    for i in range(12):
        if i % 2 == 0:
            r = agent._check_loop("edit_file", {"path": "a.py", "new": f"v{i}"})
        else:
            r = agent._check_loop("shell", {"command": "pytest"})
        assert r is None, f"第 {i+1} 次误报: {r}"


def test_check_loop_persists_after_warning():
    # 警告后模型仍硬打同一模式 → 持续拦截
    agent = _loop_agent()
    for _ in range(3):
        agent._check_loop("glob", {"pattern": "*"})
    assert agent._check_loop("glob", {"pattern": "*"}) is not None


# ── 回合中断:abort_turn 消息一致性修复(Ctrl+C 中断回合)── 拍板 2026-07-07

def test_abort_turn_repairs_dangling_tool_calls():
    # 中断可能停在「assistant 带 tool_calls、部分结果未回」的中间态,
    # 不补占位下一次请求会被 API 以孤儿 tool_call 拒绝
    agent = _loop_agent()
    agent.messages.append({"role": "user", "content": "做点事"})
    agent.messages.append({
        "role": "assistant", "content": "",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "shell", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "glob", "arguments": "{}"}},
        ],
    })
    agent.messages.append({"role": "tool", "tool_call_id": "c1", "content": "ok"})

    agent.abort_turn()

    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert {m["tool_call_id"] for m in tool_msgs} == {"c1", "c2"}
    c2 = next(m for m in tool_msgs if m["tool_call_id"] == "c2")
    assert "中断" in c2["content"]
    assert agent.messages[-1]["role"] == "user"
    assert "中断" in agent.messages[-1]["content"]


def test_abort_turn_already_complete_tool_results():
    # 结果齐全时不补占位,只追加中断标记
    agent = _loop_agent()
    agent.messages.append({"role": "user", "content": "hi"})
    agent.messages.append({
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "shell", "arguments": "{}"}}],
    })
    agent.messages.append({"role": "tool", "tool_call_id": "c1", "content": "ok"})
    n = len(agent.messages)
    agent.abort_turn()
    assert len(agent.messages) == n + 1
    assert agent.messages[-1]["role"] == "user"


def test_abort_turn_without_tool_calls_only_marks():
    agent = _loop_agent()
    agent.messages.append({"role": "user", "content": "hi"})
    agent.messages.append({"role": "assistant", "content": "写到一半..."})
    n = len(agent.messages)
    agent.abort_turn()
    assert len(agent.messages) == n + 1
    assert agent.messages[-1]["role"] == "user"
