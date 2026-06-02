import pytest
import platform
from loongcli.tools.shell import ShellTool
from loongcli.core.events import ShellOutput


@pytest.fixture
def tool():
    return ShellTool()


@pytest.fixture
def streaming_tool():
    t = ShellTool()
    t._collected = []
    t._progress_callback = lambda evt: t._collected.append(evt)
    return t


@pytest.mark.asyncio
async def test_echo_command(tool):
    result = await tool.execute(command="echo hello")
    assert "hello" in result


@pytest.mark.asyncio
async def test_command_timeout(tool):
    if platform.system() == "Windows":
        cmd = "ping -n 10 127.0.0.1"
    else:
        cmd = "sleep 10"
    result = await tool.execute(command=cmd, timeout=1)
    assert "超时" in result or "timeout" in result.lower()


@pytest.mark.asyncio
async def test_invalid_command(tool):
    result = await tool.execute(command="nonexistent_command_xyz_123")
    assert len(result) > 0


def test_tool_schema(tool):
    assert tool.name == "shell"
    assert "command" in tool.parameters["properties"]


def test_detects_platform(tool):
    if platform.system() == "Windows":
        assert "powershell" in tool._shell_cmd.lower() or "pwsh" in tool._shell_cmd.lower()
    else:
        assert "bash" in tool._shell_cmd or "sh" in tool._shell_cmd


def test_supports_progress(tool):
    assert tool.supports_progress is True


@pytest.mark.asyncio
async def test_streaming_echo(streaming_tool):
    if platform.system() == "Windows":
        cmd = "Write-Output 'line1'; Write-Output 'line2'; Write-Output 'line3'"
    else:
        cmd = "echo line1; echo line2; echo line3"
    result = await streaming_tool.execute(command=cmd)
    assert "line1" in result
    assert "line2" in result
    assert "line3" in result
    assert len(streaming_tool._collected) >= 3
    assert all(isinstance(e, ShellOutput) for e in streaming_tool._collected)
    stdout_events = [e for e in streaming_tool._collected if e.stream == "stdout"]
    assert len(stdout_events) >= 3


@pytest.mark.asyncio
async def test_streaming_stderr(streaming_tool):
    if platform.system() == "Windows":
        cmd = "Write-Error 'oops' 2>&1"
    else:
        cmd = "echo oops >&2"
    result = await streaming_tool.execute(command=cmd)
    assert "oops" in result
    assert len(streaming_tool._collected) >= 1


@pytest.mark.asyncio
async def test_streaming_no_callback(tool):
    result = await tool.execute(command="echo works")
    assert "works" in result


@pytest.mark.asyncio
async def test_streaming_timeout_partial(streaming_tool):
    if platform.system() == "Windows":
        cmd = "Write-Output 'before'; Start-Sleep -Seconds 10"
    else:
        cmd = "echo before; sleep 10"
    result = await streaming_tool.execute(command=cmd, timeout=2)
    assert "超时" in result
    before_events = [e for e in streaming_tool._collected if "before" in e.line]
    assert len(before_events) >= 1
