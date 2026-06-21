import pytest
import platform
import loongcli.tools.shell as shell_mod
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
    if tool.is_posix:
        cmd = "sleep 10"
    else:
        cmd = "ping -n 10 127.0.0.1"
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
        # git bash when Git is installed, else PowerShell fallback
        if tool.is_posix:
            assert tool._shell_cmd.lower().endswith("bash.exe")
        else:
            assert "powershell" in tool._shell_cmd.lower() or "pwsh" in tool._shell_cmd.lower()
    else:
        assert "bash" in tool._shell_cmd or "sh" in tool._shell_cmd


def test_supports_progress(tool):
    assert tool.supports_progress is True


@pytest.mark.asyncio
async def test_streaming_echo(streaming_tool):
    if streaming_tool.is_posix:
        cmd = "echo line1; echo line2; echo line3"
    else:
        cmd = "Write-Output 'line1'; Write-Output 'line2'; Write-Output 'line3'"
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
    if streaming_tool.is_posix:
        cmd = "echo oops >&2"
    else:
        cmd = "Write-Error 'oops' 2>&1"
    result = await streaming_tool.execute(command=cmd)
    assert "oops" in result
    assert len(streaming_tool._collected) >= 1


@pytest.mark.asyncio
async def test_streaming_no_callback(tool):
    result = await tool.execute(command="echo works")
    assert "works" in result


@pytest.mark.asyncio
async def test_streaming_timeout_partial(streaming_tool):
    if streaming_tool.is_posix:
        cmd = "echo before; sleep 10"
    else:
        cmd = "Write-Output 'before'; Start-Sleep -Seconds 10"
    result = await streaming_tool.execute(command=cmd, timeout=2)
    assert "超时" in result
    before_events = [e for e in streaming_tool._collected if "before" in e.line]
    assert len(before_events) >= 1


# ── exit code classification tests ──


@pytest.mark.asyncio
async def test_exit_code_1_shows_general_error(tool):
    result = await tool.execute(
        command="exit 1", timeout=5,
    )
    assert "exit code: 1" in result
    assert "一般错误" in result


@pytest.mark.asyncio
async def test_exit_code_2_shows_usage_error(tool):
    result = await tool.execute(command="exit 2", timeout=5)
    assert "exit code: 2" in result
    assert "用法错误" in result or "参数" in result


@pytest.mark.asyncio
async def test_exit_code_127_shows_not_found(tool):
    result = await tool.execute(command="exit 127", timeout=5)
    assert "exit code: 127" in result
    assert "不存在" in result or "不可执行" in result


@pytest.mark.asyncio
async def test_exit_code_0_no_prefix(tool):
    """Successful command should not show exit code prefix."""
    result = await tool.execute(command="echo success", timeout=10)
    assert "exit code" not in result
    assert "success" in result


@pytest.mark.asyncio
async def test_unknown_exit_code_no_hint(tool):
    """Exit code without a known hint shows just the code."""
    result = await tool.execute(command="exit 99", timeout=5)
    assert "exit code: 99" in result
    assert "—" not in result  # no hint appended


# ── shell resolution tests ──


def test_resolve_shell_prefers_git_bash(monkeypatch):
    """On Windows with Git installed, git bash wins over PowerShell."""
    monkeypatch.setattr(shell_mod.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        shell_mod, "_find_git_bash", lambda: r"C:\Program Files\Git\bin\bash.exe",
    )
    shell_mod.resolve_shell.cache_clear()
    try:
        spec = shell_mod.resolve_shell()
        assert spec.is_posix is True
        assert spec.label == "git bash"
        assert spec.args == ["-c"]
        assert spec.cmd.lower().endswith("bash.exe")
    finally:
        shell_mod.resolve_shell.cache_clear()


def test_resolve_shell_falls_back_to_powershell(monkeypatch):
    """On Windows without git bash, fall back to PowerShell (never breaks)."""
    monkeypatch.setattr(shell_mod.platform, "system", lambda: "Windows")
    monkeypatch.setattr(shell_mod, "_find_git_bash", lambda: None)
    shell_mod.resolve_shell.cache_clear()
    try:
        spec = shell_mod.resolve_shell()
        assert spec.is_posix is False
        assert spec.label == "PowerShell"
        assert "powershell" in spec.cmd.lower()
    finally:
        shell_mod.resolve_shell.cache_clear()


def test_find_git_bash_ignores_wsl_bash(monkeypatch):
    """_find_git_bash derives from `git`, never picks up System32 WSL bash."""
    # git resolves under an install root; bash.exe sits at <root>\bin\bash.exe
    monkeypatch.setattr(
        shell_mod.shutil, "which",
        lambda name: r"D:\program\Git\cmd\git.exe" if name == "git" else None,
    )
    seen = []

    def fake_isfile(p):
        seen.append(p)
        return p == r"D:\program\Git\bin\bash.exe"

    monkeypatch.setattr(shell_mod.os.path, "isfile", fake_isfile)
    result = shell_mod._find_git_bash()
    assert result == r"D:\program\Git\bin\bash.exe"
    # never probes the WSL launcher
    assert not any("System32" in p for p in seen)


@pytest.mark.asyncio
async def test_timeout_shows_terminated_notice(tool):
    if tool.is_posix:
        cmd = "sleep 10"
    else:
        cmd = "ping -n 10 127.0.0.1"
    result = await tool.execute(command=cmd, timeout=1)
    assert "终止" in result or "timeout" in result.lower()
