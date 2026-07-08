import asyncio
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


# ── .venv 环境接地：命令确定性解析到项目虚拟环境 ──

def test_venv_env_prepends_path(tmp_path, monkeypatch):
    import os
    import platform
    from loongcli.tools.shell import _venv_env

    bin_name = "Scripts" if platform.system() == "Windows" else "bin"
    (tmp_path / ".venv" / bin_name).mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    env = _venv_env()
    assert env is not None
    assert env["PATH"].startswith(str(tmp_path / ".venv" / bin_name) + os.pathsep)
    assert env["VIRTUAL_ENV"] == str(tmp_path / ".venv")


def test_venv_env_none_without_venv(tmp_path, monkeypatch):
    from loongcli.tools.shell import _venv_env

    monkeypatch.chdir(tmp_path)
    assert _venv_env() is None  # 无 .venv → 继承父环境，行为不变


# ── cd /d 归一：cmd 语法惯性（qwen 三次复发）确定性改写 ──

@pytest.mark.asyncio
async def test_cd_slash_d_normalized(tmp_path):
    """`cd /d <路径>` 在 bash 里必错（cd 只吃一个参数）→ 确定性改写为 `cd <路径>`。"""
    from loongcli.tools.shell import ShellTool

    tool = ShellTool()
    if not tool.is_posix:
        pytest.skip("非 POSIX shell 环境")
    posix_path = str(tmp_path).replace("\\", "/")
    result = await tool.execute(f'cd /d {posix_path} && echo NORMALIZED_OK')
    assert "NORMALIZED_OK" in result
    assert "too many arguments" not in result


@pytest.mark.asyncio
async def test_cd_single_arg_untouched():
    """正常的 `cd /d`（单参数，git bash 里=进 D: 根）与普通 cd 不受影响。"""
    from loongcli.tools.shell import ShellTool

    tool = ShellTool()
    if not tool.is_posix:
        pytest.skip("非 POSIX shell 环境")
    result = await tool.execute("cd . && echo PLAIN_OK")
    assert "PLAIN_OK" in result


# ── 取消路径：stop_task 取消协程后子进程必须随之死掉，不留孤儿 ──

@pytest.mark.asyncio
async def test_cancel_kills_subprocess(tool, monkeypatch):
    """回归：取消只在 await 点生效，此前没有 terminate→kill，子进程会孤儿续跑。"""
    captured = {}
    orig_exec = asyncio.create_subprocess_exec
    orig_shell = asyncio.create_subprocess_shell

    async def capture_exec(*args, **kwargs):
        p = await orig_exec(*args, **kwargs)
        captured["p"] = p
        return p

    async def capture_shell(*args, **kwargs):
        p = await orig_shell(*args, **kwargs)
        captured["p"] = p
        return p

    monkeypatch.setattr(shell_mod.asyncio, "create_subprocess_exec", capture_exec)
    monkeypatch.setattr(shell_mod.asyncio, "create_subprocess_shell", capture_shell)

    cmd = "sleep 30" if tool.is_posix else "Start-Sleep -Seconds 30"
    exec_task = asyncio.create_task(tool.execute(command=cmd, timeout=60))

    for _ in range(100):  # 等子进程真正起来
        if "p" in captured:
            break
        await asyncio.sleep(0.05)
    assert "p" in captured, "子进程未启动"

    exec_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await exec_task

    # CancelledError 重新抛出前必须走完 terminate→kill（returncode 已落定）
    assert captured["p"].returncode is not None


@pytest.mark.asyncio
async def test_cancel_kills_process_tree():
    """回归（真机 3.1 抓到）：MSYS2 bash 无法真 exec，只杀直接子进程 bash
    会让孙进程（sleep）孤儿续跑——取消后必须全树死透（taskkill /T）。"""
    tool = ShellTool()
    if platform.system() != "Windows" or not tool.is_posix:
        # POSIX 上 bash -c 单命令 exec 替换自身，直接子进程即全树；
        # 复合命令的树杀未做（诚实边界）。此测试只覆盖 Windows+git bash。
        pytest.skip("仅 Windows + git bash 环境")

    collected = []
    tool._progress_callback = lambda evt: collected.append(evt)

    # 后台起 sleep 并回报其 Windows PID（MSYS2 /proc/<pid>/winpid）
    cmd = "sleep 30 & echo CHILD_WINPID:$(cat /proc/$!/winpid); wait $!"
    exec_task = asyncio.create_task(tool.execute(command=cmd, timeout=60))

    winpid = None
    for _ in range(200):
        for evt in collected:
            if "CHILD_WINPID:" in evt.line:
                winpid = int(evt.line.split("CHILD_WINPID:")[1].strip())
                break
        if winpid:
            break
        await asyncio.sleep(0.05)
    assert winpid, "未拿到孙进程 Windows PID"

    exec_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await exec_task

    # 孙进程必须随树死掉；给 taskkill 一个短收尾窗
    import subprocess
    alive = True
    for _ in range(20):
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {winpid}"],
            capture_output=True, text=True,
        ).stdout
        alive = f" {winpid} " in out
        if not alive:
            break
        await asyncio.sleep(0.2)
    assert not alive, f"孙进程 {winpid} 仍存活——树杀失效"
