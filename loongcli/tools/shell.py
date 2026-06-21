from __future__ import annotations
import asyncio
import functools
import os
import platform
import shutil
from pathlib import Path
from typing import NamedTuple
from loongcli.tools.base import Tool
from loongcli.core.events import ShellOutput

_GRACEFUL_WAIT = 3  # seconds to wait between terminate and kill


class ShellSpec(NamedTuple):
    cmd: str            # executable to invoke
    args: list[str]     # args preceding the command string
    label: str          # human-readable name for prompt display
    is_posix: bool      # True for bash-family shells (POSIX syntax)


def _find_git_bash() -> str | None:
    """Locate Git for Windows' bash.exe.

    Deliberately avoids ``shutil.which("bash")``: on machines with WSL that
    resolves to ``System32\\bash.exe`` (the WSL launcher), a completely
    different shell. We derive bash from the ``git`` install instead.
    """
    candidates: list[str] = []
    git = shutil.which("git")
    if git:
        # git is typically at <root>\cmd\git.exe or <root>\bin\git.exe
        root = Path(git).parent.parent
        candidates.append(str(root / "bin" / "bash.exe"))
    candidates.append(r"C:\Program Files\Git\bin\bash.exe")
    candidates.append(r"C:\Program Files (x86)\Git\bin\bash.exe")
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


@functools.lru_cache(maxsize=1)
def resolve_shell() -> ShellSpec:
    """Pick the shell for the current OS (stable for the process lifetime).

    Windows prefers git bash and falls back to PowerShell when it is absent,
    so the tool never breaks on a machine without Git installed.
    """
    if platform.system() == "Windows":
        bash = _find_git_bash()
        if bash:
            return ShellSpec(cmd=bash, args=["-c"], label="git bash", is_posix=True)
        return ShellSpec(
            cmd="powershell",
            args=["-NoProfile", "-NonInteractive", "-Command"],
            label="PowerShell",
            is_posix=False,
        )
    shell = os.environ.get("SHELL", "bash")
    return ShellSpec(cmd="bash", args=["-c"], label=os.path.basename(shell) or "bash", is_posix=True)


class ShellTool(Tool):
    name = "shell"
    description = "Execute a shell command and return its output. Uses git bash on Windows (falls back to PowerShell if Git is not installed), bash on Linux/Mac."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds. Default: 120",
            },
        },
        "required": ["command"],
    }
    supports_progress = True

    def __init__(self):
        spec = resolve_shell()
        self._shell_cmd = spec.cmd
        self._shell_args = spec.args
        self.shell_label = spec.label
        self.is_posix = spec.is_posix
        self._progress_callback = None

    async def _read_stream(self, stream, stream_name: str, lines: list[str]):
        async for raw in stream:
            line = raw.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")
            lines.append(line)
            if self._progress_callback:
                self._progress_callback(ShellOutput(line=line, stream=stream_name))

    async def execute(self, command: str, timeout: int = 120) -> str:
        try:
            if platform.system() == "Windows":
                full_cmd = [self._shell_cmd] + self._shell_args + [command]
                process = await asyncio.create_subprocess_exec(
                    *full_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    executable=self._shell_cmd,
                )

            stdout_lines: list[str] = []
            stderr_lines: list[str] = []

            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        self._read_stream(process.stdout, "stdout", stdout_lines),
                        self._read_stream(process.stderr, "stderr", stderr_lines),
                    ),
                    timeout=timeout,
                )
                await process.wait()
            except asyncio.TimeoutError:
                return await self._handle_timeout(
                    process, timeout, stdout_lines, stderr_lines,
                )

            output = _build_result(stdout_lines, stderr_lines, process.returncode)
            return output

        except Exception as e:
            return f"错误：{e}"

    async def _handle_timeout(
        self, process, timeout: int,
        stdout_lines: list[str], stderr_lines: list[str],
    ) -> str:
        """Graceful shutdown: terminate → wait → kill."""
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=_GRACEFUL_WAIT)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

        output_lines = stdout_lines + stderr_lines
        if output_lines:
            output = "\n".join(output_lines).strip()
            return f"⚠ 命令超时（{timeout}秒）\n{output}\n\n[... 进程已终止]"
        return f"⚠ 命令超时（{timeout}秒）\n[... 进程已终止]"


def _build_result(stdout_lines: list[str], stderr_lines: list[str], returncode: int) -> str:
    """Build result string with exit code hint."""
    parts = []
    if stdout_lines:
        parts.append("\n".join(stdout_lines))
    if stderr_lines:
        parts.append("\n".join(stderr_lines))

    output = "\n".join(parts).strip()

    if returncode != 0:
        hint = _exit_code_hint(returncode)
        if hint:
            output = f"[exit code: {returncode} — {hint}]\n{output}"
        else:
            output = f"[exit code: {returncode}]\n{output}"

    if len(output) > 10000:
        output = output[:10000] + "\n... (output truncated)"

    return output if output else "(no output)"


def _exit_code_hint(code: int) -> str:
    """Return a human-readable hint for common exit codes."""
    if code == 1:
        return "一般错误"
    if code == 2:
        return "命令用法错误（参数可能不正确）"
    if code in (126, 127):
        return "命令不存在或不可执行（请检查是否安装）"
    if platform.system() != "Windows":
        if code == 137:
            return "进程被 SIGKILL 终止（可能是 OOM）"
        if code == 139:
            return "段错误 SIGSEGV（内存访问越界）"
    return ""
