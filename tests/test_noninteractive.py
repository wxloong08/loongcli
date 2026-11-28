import pytest
import json
import sys
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, AsyncMock
from argparse import Namespace

from loongcli.main import _build_prompt, _run_noninteractive, _brief, _async_main
from loongcli.core.agent import AgentLoop
from loongcli.core.llm import LLMClient
from loongcli.core.events import TextDelta, ToolCallStart, ToolCallResult, AgentDone, ConfirmRequest
from loongcli.tools.base import ToolRegistry
from loongcli.security.permissions import PermissionChecker


def _make_args(output_format="text", no_stream=False, verbose=False, prompt=None):
    return Namespace(
        prompt=prompt,
        output_format=output_format,
        no_stream=no_stream,
        verbose=verbose,
        image=None,
    )


def test_build_prompt_with_arg():
    args = Namespace(prompt="hello world")
    with patch("loongcli.main.sys") as mock_sys:
        mock_sys.stdin.isatty.return_value = True
        from loongcli.main import _build_prompt
        result = _build_prompt(args)
    assert result == "hello world"


def test_build_prompt_no_arg_tty():
    args = Namespace(prompt=None)
    with patch("loongcli.main.sys") as mock_sys:
        mock_sys.stdin.isatty.return_value = True
        result = _build_prompt(args)
    assert result is None


def test_build_prompt_piped_stdin():
    args = Namespace(prompt=None)
    with patch("loongcli.main.sys") as mock_sys:
        mock_sys.stdin.isatty.return_value = False
        mock_sys.stdin.read.return_value = "piped content\n"
        result = _build_prompt(args)
    assert result == "piped content"


def test_build_prompt_piped_stdin_with_arg():
    args = Namespace(prompt="analyze this")
    with patch("loongcli.main.sys") as mock_sys:
        mock_sys.stdin.isatty.return_value = False
        mock_sys.stdin.read.return_value = "file content here\n"
        result = _build_prompt(args)
    assert result == "file content here\n\nanalyze this"


def test_brief_args():
    assert _brief({"a": "short"}) == "a=short"
    assert "..." in _brief({"a": "x" * 50})


@pytest.mark.asyncio
async def test_async_main_uses_module_path_import_before_memory_migration():
    args = Namespace(
        prompt=None,
        continue_session=False,
        resume=False,
        dangerously_skip_permissions=False,
        output_format="text",
        no_stream=False,
        verbose=False,
        profile=None,
        image=None,
    )
    cfg = SimpleNamespace(
        api_key="test-key",
        providers={},
        skill_dirs=[],
        mcp_servers={},
        hooks={},
        model="test-model",
        compact_threshold=None,
    )
    router = MagicMock()
    router.client.side_effect = [
        SimpleNamespace(model="test-model"),
        SimpleNamespace(model="test-model"),
    ]

    with (
        patch("loongcli.main._parse_args", return_value=args),
        patch("loongcli.main._build_prompt", return_value=None),
        patch("loongcli.main.Config.load", return_value=cfg),
        patch("loongcli.main.ModelRouter.from_config", return_value=router),
        patch("loongcli.main.migrate_kv_to_markdown", side_effect=RuntimeError("migration reached")),
    ):
        with pytest.raises(RuntimeError, match="migration reached"):
            await _async_main()


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


@pytest.mark.asyncio
async def test_noninteractive_text_stream():
    llm = LLMClient(api_key="test")
    text_chunks = _make_text_chunks(["Hello", " World"])

    async def mock_stream(**kwargs):
        for c in text_chunks:
            yield c

    llm.chat_stream = mock_stream

    registry = ToolRegistry()
    checker = PermissionChecker()
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=checker)

    args = _make_args(output_format="text", no_stream=False)
    mcp = MagicMock()
    mcp.disconnect_all = AsyncMock()

    buf = StringIO()
    with patch("loongcli.main.sys") as mock_sys:
        mock_sys.stdout = buf
        mock_sys.stderr = sys.stderr
        await _run_noninteractive(agent, "hi", args, mcp)

    output = buf.getvalue()
    assert "Hello World" in output


@pytest.mark.asyncio
async def test_noninteractive_text_no_stream():
    llm = LLMClient(api_key="test")
    text_chunks = _make_text_chunks(["Hello", " World"])

    async def mock_stream(**kwargs):
        for c in text_chunks:
            yield c

    llm.chat_stream = mock_stream

    registry = ToolRegistry()
    checker = PermissionChecker()
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=checker)

    args = _make_args(output_format="text", no_stream=True)
    mcp = MagicMock()
    mcp.disconnect_all = AsyncMock()

    buf = StringIO()
    with patch("loongcli.main.sys") as mock_sys:
        mock_sys.stdout = buf
        mock_sys.stderr = sys.stderr
        await _run_noninteractive(agent, "hi", args, mcp)

    output = buf.getvalue()
    assert output == "Hello World\n"


@pytest.mark.asyncio
async def test_noninteractive_json_output():
    llm = LLMClient(api_key="test")
    text_chunks = _make_text_chunks(["Hello"])

    async def mock_stream(**kwargs):
        for c in text_chunks:
            yield c

    llm.chat_stream = mock_stream

    registry = ToolRegistry()
    checker = PermissionChecker()
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=checker)

    args = _make_args(output_format="json")
    mcp = MagicMock()
    mcp.disconnect_all = AsyncMock()

    buf = StringIO()
    with patch("loongcli.main.sys") as mock_sys:
        mock_sys.stdout = buf
        mock_sys.stderr = sys.stderr
        await _run_noninteractive(agent, "hi", args, mcp)

    data = json.loads(buf.getvalue())
    assert data["content"] == "Hello"


@pytest.mark.asyncio
async def test_noninteractive_confirm_auto_denied():
    """ConfirmRequest should be auto-denied in non-interactive mode."""
    llm = LLMClient(api_key="test")

    tc_delta = MagicMock()
    tc_delta.index = 0
    tc_delta.id = "call_1"
    tc_delta.type = "function"
    tc_delta.function = MagicMock()
    tc_delta.function.name = "shell"
    tc_delta.function.arguments = json.dumps({"command": "rm -rf /tmp/test"})

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

    text_chunks = _make_text_chunks(["OK"])

    call_count = 0

    async def mock_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        src = [chunk1, chunk2] if call_count == 1 else text_chunks
        for c in src:
            yield c

    llm.chat_stream = mock_stream

    from loongcli.tools.base import Tool

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

    registry = ToolRegistry()
    registry.register(ShellTool())
    checker = PermissionChecker()
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=checker)

    args = _make_args(verbose=True)
    mcp = MagicMock()
    mcp.disconnect_all = AsyncMock()

    buf = StringIO()
    err_buf = StringIO()
    with patch("loongcli.main.sys") as mock_sys:
        mock_sys.stdout = buf
        mock_sys.stderr = err_buf
        await _run_noninteractive(agent, "delete stuff", args, mcp)

    output = buf.getvalue()
    assert "OK" in output


@pytest.mark.asyncio
async def test_noninteractive_disconnects_mcp():
    llm = LLMClient(api_key="test")
    text_chunks = _make_text_chunks(["Done"])

    async def mock_stream(**kwargs):
        for c in text_chunks:
            yield c

    llm.chat_stream = mock_stream

    registry = ToolRegistry()
    checker = PermissionChecker()
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=checker)

    args = _make_args()
    mcp = MagicMock()
    mcp.disconnect_all = AsyncMock()

    buf = StringIO()
    with patch("loongcli.main.sys") as mock_sys:
        mock_sys.stdout = buf
        mock_sys.stderr = sys.stderr
        await _run_noninteractive(agent, "hi", args, mcp)

    mcp.disconnect_all.assert_called_once()


def _agent_with_stream(mock_stream):
    llm = LLMClient(api_key="test")
    llm.chat_stream = mock_stream
    return AgentLoop(llm=llm, tool_registry=ToolRegistry(), permission_checker=PermissionChecker())


async def _run_capture(agent, args):
    mcp = MagicMock()
    mcp.disconnect_all = AsyncMock()
    out, err = StringIO(), StringIO()
    with patch("loongcli.main.sys") as mock_sys:
        mock_sys.stdout = out
        mock_sys.stderr = err
        code = await _run_noninteractive(agent, "hi", args, mcp)
    return code, out.getvalue(), err.getvalue()


async def _failing_stream(**kwargs):
    raise RuntimeError("Error code: 402 - Insufficient Balance")
    yield  # 让它成为异步生成器


@pytest.mark.asyncio
async def test_noninteractive_success_exit_zero():
    async def ok(**kwargs):
        for c in _make_text_chunks(["Hello"]):
            yield c
    code, out, err = await _run_capture(_agent_with_stream(ok), _make_args())
    assert code == 0 and "Hello" in out and err == ""


@pytest.mark.asyncio
async def test_noninteractive_llm_failure_exit_nonzero_text():
    """模型调用失败（余额不足）→ 退出码 1，失败原因写 stderr，不混进 stdout 正文。"""
    code, out, err = await _run_capture(_agent_with_stream(_failing_stream), _make_args())
    assert code == 1
    assert "402" in err and "402" not in out


@pytest.mark.asyncio
async def test_noninteractive_llm_failure_json_error_field():
    code, out, err = await _run_capture(_agent_with_stream(_failing_stream), _make_args(output_format="json"))
    data = json.loads(out)
    assert code == 1
    assert "402" in data["error"]


@pytest.mark.asyncio
async def test_noninteractive_iteration_cap_note_shown_exit_zero():
    """达到迭代上限不算执行失败（退出 0），但上限提示要出现在输出里，此前被静默丢弃。"""
    agent = _agent_with_stream(_failing_stream)

    async def fake_run_stream(prompt, images=None):
        yield TextDelta(text="partial")
        yield AgentDone(content="⚠ 已达到迭代上限（30轮），自动停止。")

    agent.run_stream = fake_run_stream
    code, out, err = await _run_capture(agent, _make_args(output_format="json"))
    data = json.loads(out)
    assert code == 0 and "error" not in data
    assert data["content"].startswith("partial\n") and "迭代上限" in data["content"]
