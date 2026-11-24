"""shell 子进程与 harness 控制台隔离的回归测试。

事故：2026-11-13 会话 b379c5b3aaa0，shell 子进程树附着在 TUI 的控制台上，期间控制台
输出模式被改，Rich Live 写入双宽字符命中 Windows Terminal 1.24 宿主死循环
（microsoft/terminal#20693），TUI 整体冻结、回合丢失。根因是「共享控制台」，
修法是 CREATE_NO_WINDOW + stdin=DEVNULL（见 loongcli/core/subproc.py）。
"""
import platform
import subprocess
import sys
import textwrap

import pytest

from loongcli.core.subproc import isolated_console_kwargs
from loongcli.tools.shell import ShellTool

CREATE_NO_WINDOW = 0x08000000

# 探针：打印附着在「我所在的控制台」上的全部进程 pid。
_CONSOLE_PIDS = (
    "import ctypes; a=(ctypes.c_ulong*128)(); "
    "n=ctypes.windll.kernel32.GetConsoleProcessList(a,128); "
    "print('CONSOLE_PIDS='+','.join(str(x) for x in a[:n]))"
)

# 驱动脚本：自己先拿到一个独占的隐藏控制台（扮演 TUI 的控制台），再通过 ShellTool
# 跑探针。pytest 进程自身可能根本没有控制台（CI/管道），那样测试会假阳性通过，
# 所以必须由这个确定「有控制台」的中间进程来当父进程。
_DRIVER = textwrap.dedent("""\
    import asyncio, os, sys
    from loongcli.tools.shell import ShellTool

    async def main():
        out = await ShellTool().execute(sys.argv[1], timeout=60)
        print("PARENT_PID=" + str(os.getpid()))
        print(out)

    asyncio.run(main())
""")


def _run_driver(tmp_path, command: str) -> str:
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER, encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(driver), command],
        capture_output=True, text=True, timeout=120,
        stdin=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW,
    )
    assert r.returncode == 0, f"driver 失败：{r.stderr}"
    return r.stdout


def _field(output: str, key: str) -> str:
    for line in output.splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip()
    raise AssertionError(f"输出缺少 {key}：{output}")


@pytest.mark.skipif(platform.system() != "Windows", reason="共享控制台是 Windows 专属语义")
class TestConsoleIsolationWindows:
    def test_child_not_attached_to_parent_console(self, tmp_path):
        """子进程看到的控制台进程列表里不得有父进程——即两者不在同一个控制台上。"""
        out = _run_driver(tmp_path, f'python -c "{_CONSOLE_PIDS}"')
        parent_pid = _field(out, "PARENT_PID")
        child_console_pids = _field(out, "CONSOLE_PIDS").split(",")
        assert parent_pid not in child_console_pids, (
            f"shell 子进程附着在父进程的控制台上（父 {parent_pid} ∈ {child_console_pids}）"
            "——共享控制台会让子进程树改写 TUI 的控制台模式/读走键盘输入"
        )

    def test_native_grandchild_also_isolated(self, tmp_path):
        """bash → 原生程序这条链（事故路径）上的孙进程同样不在父控制台上。"""
        out = _run_driver(tmp_path, f'bash -c "python -c \\"{_CONSOLE_PIDS}\\""')
        parent_pid = _field(out, "PARENT_PID")
        assert parent_pid not in _field(out, "CONSOLE_PIDS").split(",")

    def test_isolated_console_kwargs_on_windows(self):
        assert isolated_console_kwargs() == {"creationflags": CREATE_NO_WINDOW}


class TestStdinDetached:
    async def test_stdin_reads_eof_immediately(self):
        """读 stdin 的命令立即拿到 EOF，而不是挂到超时（或抢 TUI 的键盘）。"""
        result = await ShellTool().execute(
            "python -c \"import sys; print('STDIN=' + repr(sys.stdin.read()))\"",
            timeout=30,
        )
        assert "STDIN=''" in result, result
        assert "超时" not in result

    async def test_interactive_prompt_fails_fast(self):
        """要键盘输入的命令（这里是 bash read）立刻失败返回，不等到超时。"""
        result = await ShellTool().execute(
            "read -r line && echo GOT=$line || echo NO_INPUT", timeout=30,
        )
        assert "NO_INPUT" in result, result


class TestNonInteractiveGitEditor:
    """shell 子进程默认 GIT_EDITOR=true：git 拉起编辑器时立即返回，而不是等满超时。"""

    def test_child_env_defaults_git_editor(self, monkeypatch, tmp_path):
        from loongcli.tools.shell import _child_env
        monkeypatch.chdir(tmp_path)  # 无 .venv
        monkeypatch.delenv("GIT_EDITOR", raising=False)
        env = _child_env()
        assert env["GIT_EDITOR"] == "true"
        assert env["PATH"] == __import__("os").environ["PATH"]  # 其余环境原样继承

    def test_child_env_respects_user_setting(self, monkeypatch, tmp_path):
        from loongcli.tools.shell import _child_env
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GIT_EDITOR", "code --wait")
        assert _child_env()["GIT_EDITOR"] == "code --wait"

    def test_child_env_keeps_venv_prefix(self, monkeypatch, tmp_path):
        from loongcli.tools.shell import _child_env
        bin_dir = tmp_path / ".venv" / ("Scripts" if platform.system() == "Windows" else "bin")
        bin_dir.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GIT_EDITOR", raising=False)
        env = _child_env()
        assert env["PATH"].startswith(str(bin_dir))
        assert env["GIT_EDITOR"] == "true"

    async def test_bare_git_commit_returns_fast_instead_of_hanging(self, monkeypatch, tmp_path):
        """裸 `git commit`（无 -m）：修复前 vim 对着隐藏控制台等满超时；现在 git 因
        提交信息为空立即中止。超时设 60s，断言远早于它返回且不是超时路径。"""
        import time
        monkeypatch.delenv("GIT_EDITOR", raising=False)
        repo = tmp_path / "repo"
        repo.mkdir()
        setup = (
            f'cd "{repo.as_posix()}" && git init -q . && git config user.email t@t '
            "&& git config user.name t && echo x > f && git add f"
        )
        tool = ShellTool()
        assert "exit code" not in await tool.execute(setup, timeout=60)

        t0 = time.monotonic()
        result = await tool.execute(f'cd "{repo.as_posix()}" && git commit', timeout=60)
        elapsed = time.monotonic() - t0

        assert "超时" not in result, result
        assert elapsed < 30, f"裸 git commit 耗时 {elapsed:.1f}s，疑似仍在等编辑器"
        assert "empty commit message" in result.lower() or "exit code" in result, result


@pytest.mark.skipif(platform.system() == "Windows", reason="POSIX 无共享控制台模式")
def test_isolated_console_kwargs_empty_on_posix():
    assert isolated_console_kwargs() == {}
