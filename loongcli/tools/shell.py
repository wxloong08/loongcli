from __future__ import annotations
import asyncio
import platform
from loongcli.tools.base import Tool
from loongcli.core.events import ShellOutput


class ShellTool(Tool):
    name = "shell"
    description = "Execute a shell command and return its output. Uses PowerShell on Windows, bash on Linux/Mac."
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
        system = platform.system()
        if system == "Windows":
            self._shell_cmd = "powershell"
            self._shell_args = ["-NoProfile", "-NonInteractive", "-Command"]
        else:
            self._shell_cmd = "bash"
            self._shell_args = ["-c"]
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
                process.kill()
                await process.wait()
                output = "\n".join(stdout_lines + stderr_lines).strip()
                if output:
                    return f"⚠ 命令超时（{timeout}秒）\n{output}"
                return f"⚠ 命令超时（{timeout}秒）"

            output_parts = []
            if stdout_lines:
                output_parts.append("\n".join(stdout_lines))
            if stderr_lines:
                output_parts.append("\n".join(stderr_lines))

            output = "\n".join(output_parts).strip()

            if process.returncode != 0:
                output = f"[exit code: {process.returncode}]\n{output}"

            if len(output) > 10000:
                output = output[:10000] + "\n... (output truncated)"

            return output if output else "(no output)"

        except asyncio.TimeoutError:
            return f"⚠ 命令超时（{timeout}秒）"
        except Exception as e:
            return f"错误：{e}"
