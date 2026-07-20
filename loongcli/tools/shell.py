from __future__ import annotations
import asyncio
import functools
import os
import platform
import re
import shutil
from pathlib import Path
from typing import NamedTuple
from loongcli.tools.base import Tool
from loongcli.core.events import ShellOutput

_GRACEFUL_WAIT = 3  # seconds to wait between terminate and kill
# 单条流的内存上限：超限后仍持续读取（让子进程别因管道满而阻塞），但不再累积进
# lines，防 `yes` 这类无限输出在超时窗口内把内存吃满。
_MAX_STREAM_CHARS = 5_000_000


class _WindowsJob:
    """Windows Job Object：把 shell 子进程及其全部后代圈进一个终止单元。

    MSYS2 bash 的 fork/exec 仿真会打断 Windows 父子链（真机实证：bash -c
    "sleep 120" 里 sleep 的父进程是一个已消亡的中间 stub，不在 bash 链上），
    taskkill /T、psutil 这类按父子链枚举的树杀全部结构性漏杀。Job 成员资格
    由内核在进程创建时继承，与父子链无关，是 Windows 上唯一可靠的整树终止。
    KILL_ON_JOB_CLOSE 兜底：句柄关闭（含 loongcli 崩溃）时内核自动收尸——
    命令正常结束后关句柄也会带走它遗留的后台进程，harness 不留隐形残留。

    ⚠ 不能加 JOB_OBJECT_LIMIT_BREAKAWAY_OK（看似是常驻 daemon 的 opt-in 出口）：
    MSYS2/Cygwin runtime 一旦发现所在 job 允许 breakaway，会给它创建的**所有**
    子进程加 CREATE_BREAKAWAY_FROM_JOB——git bash 的全部后代逃逸，整树终止
    彻底失效（2026-07-16 A/B 实测：0x2800 下孙进程 not-in-job 且关句柄后存活）。
    命令需要启动常驻 daemon 时，正解是经 job 外的中介创建进程（如 WMI
    Win32_Process.Create，新进程挂在 WmiPrvSE 下、不继承调用者的 job）——
    见 D:/skills/chrome-cli 的 _start_daemon。
    """

    def __init__(self, handle):
        self._handle = handle

    @classmethod
    def create_and_assign(cls, pid: int) -> "_WindowsJob | None":
        """创建 job 并圈入 pid；任一步失败返回 None（调用方退化为直接终止）。"""
        import ctypes
        from ctypes import wintypes

        class _BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _IoCounters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class _ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimits),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        k32 = ctypes.windll.kernel32
        handle = k32.CreateJobObjectW(None, None)
        if not handle:
            return None
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        ok = k32.SetInformationJobObject(
            handle, 9,  # JobObjectExtendedLimitInformation
            ctypes.byref(info), ctypes.sizeof(info),
        )
        if not ok:
            k32.CloseHandle(handle)
            return None
        PROCESS_SET_QUOTA, PROCESS_TERMINATE = 0x0100, 0x0001
        proc = k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not proc:
            k32.CloseHandle(handle)
            return None
        assigned = k32.AssignProcessToJobObject(handle, proc)
        k32.CloseHandle(proc)
        if not assigned:
            k32.CloseHandle(handle)
            return None
        return cls(handle)

    def terminate(self):
        import ctypes
        ctypes.windll.kernel32.TerminateJobObject(self._handle, 1)

    def close(self):
        import ctypes
        ctypes.windll.kernel32.CloseHandle(self._handle)
        self._handle = None


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
        # git.exe 可能位于 <root>\cmd\、<root>\bin\，或——当本进程本身运行在
        # git bash 里时——<root>\mingw64\bin\。固定 parent.parent 只覆盖前两种，
        # 第三种会推导失败静默退化到 PowerShell（真实复现）。沿祖先逐级找。
        for anc in list(Path(git).parents)[:4]:
            candidates.append(str(anc / "bin" / "bash.exe"))
    candidates.append(r"C:\Program Files\Git\bin\bash.exe")
    candidates.append(r"C:\Program Files (x86)\Git\bin\bash.exe")
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _venv_env() -> dict | None:
    """cwd 存在 .venv 时，把它的可执行目录前置到子进程 PATH。

    环境接地：不靠模型自觉记得用虚拟环境——真机三次事故都是模型裸调 `pytest`/
    `python` 解析到全局解释器（缺依赖 + 版本不符），先误诊"项目有语法错误"，
    最后一次甚至据此改了无关源码。PATH 前置后命令确定性解析到 .venv，
    与"激活虚拟环境"的语义一致。无 .venv 时返回 None（继承父环境，行为不变）。
    """
    venv = Path.cwd() / ".venv"
    bin_dir = venv / ("Scripts" if platform.system() == "Windows" else "bin")
    if not bin_dir.is_dir():
        return None
    env = os.environ.copy()
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(venv)
    return env


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
                "description": (
                    "The shell command to execute. POSIX/bash syntax ONLY, even on "
                    "Windows (runs in git bash): use forward slashes and plain `cd <dir>`; "
                    "never use cmd.exe/PowerShell syntax like `cd /d`, `dir`, or backslash paths."
                ),
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
        total = 0
        capped = False
        async for raw in stream:
            line = raw.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")
            if total < _MAX_STREAM_CHARS:
                lines.append(line)
                total += len(line) + 1
                if total >= _MAX_STREAM_CHARS and not capped:
                    lines.append("... (输出过多，后续内容已丢弃)")
                    capped = True
            # 超限后继续消费 stream（不 append）——避免管道写满阻塞子进程
            if self._progress_callback:
                self._progress_callback(ShellOutput(line=line, stream=stream_name))

    async def execute(self, command: str, timeout: int = 120) -> str:
        # cmd 语法归一：模型（qwen 三次复发）在 Windows 上惯写 `cd /d D:/x`，
        # bash 的 cd 只吃一个参数，`/d` 会导致 "too many arguments"。bash 里
        # `cd /d <路径>` 两参形式无合法含义，确定性改写零误伤；提示层已拦不住。
        if self.is_posix:
            command = re.sub(r"(?<![\w-])cd\s+/[dD]\s+(?=\S)", "cd ", command)
        process = None
        job = None
        try:
            env = _venv_env()
            if platform.system() == "Windows":
                full_cmd = [self._shell_cmd] + self._shell_args + [command]
                process = await asyncio.create_subprocess_exec(
                    *full_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
                # 立刻圈进 Job（后代自动继承成员资格）；失败退化为直接终止
                job = _WindowsJob.create_and_assign(process.pid)
            else:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    executable=self._shell_cmd,
                    env=env,
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
                    process, job, timeout, stdout_lines, stderr_lines,
                )

            output = _build_result(stdout_lines, stderr_lines, process.returncode)
            return output

        except asyncio.CancelledError:
            # stop_task 的取消只在 await 点生效，协程死了子进程不会跟着死——
            # 超时路径有 terminate→kill，取消路径此前没有，孤儿进程会一直跑
            if process is not None:
                await self._shutdown_process(process, job)
            raise
        except Exception as e:
            return f"错误：{e}"
        finally:
            # KILL_ON_JOB_CLOSE：正常结束时顺带收掉命令遗留的后台进程；
            # 超时/取消路径此时树已终止，关句柄只是释放内核对象
            if job is not None:
                job.close()

    async def _shutdown_process(self, process, job: "_WindowsJob | None" = None) -> None:
        """按整树收尾（超时与取消两条路共用）。

        Windows 走 Job Object 内核级全灭（MSYS2 会打断父子链，见 _WindowsJob）；
        POSIX 上 bash -c 单命令会 exec 替换自身，terminate 直接子进程即杀全树。
        """
        if process.returncode is not None:
            return
        if job is not None:
            job.terminate()
        else:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=_GRACEFUL_WAIT)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    async def _handle_timeout(
        self, process, job, timeout: int,
        stdout_lines: list[str], stderr_lines: list[str],
    ) -> str:
        await self._shutdown_process(process, job)

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
        # 留头+尾：测试/编译的关键信息（assertion、summary、报错）常在尾部，
        # 只留头会把最有用的部分丢掉。
        head = output[:7000]
        tail = output[-3000:]
        output = f"{head}\n... (中间 {len(output) - 10000} 字符已省略) ...\n{tail}"

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
