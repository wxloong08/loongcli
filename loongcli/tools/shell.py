from __future__ import annotations
import asyncio
import functools
import os
import platform
import re
import shutil
import signal
from pathlib import Path
from typing import NamedTuple
from loongcli.tools.base import Tool
from loongcli.core.events import ShellOutput
from loongcli.core.subproc import isolated_console_kwargs

_GRACEFUL_WAIT = 3  # seconds to wait between terminate and kill
# 单条流的内存上限：超限后仍持续读取（让子进程别因管道满而阻塞），但不再累积进
# lines，防 `yes` 这类无限输出在超时窗口内把内存吃满。
_MAX_STREAM_CHARS = 5_000_000
# 单条命令超时上限（与 Claude Code 的 10 分钟同）：Harbor 基线里模型给一条命令传过 920 秒，
# 而整题时限才 900 秒——一条命令吃光全部预算。更长的任务应后台运行并重定向输出。
_MAX_TIMEOUT = 600
# shell 本身退出后，再等多久把管道里剩余输出读完。超过仍不 EOF，说明命令里 `cmd &` 起的后台进程
# 继承了输出管道——不再等它（否则一直挂到超时），照常返回。
_DRAIN_AFTER_EXIT = 1.0
_POLL_INTERVAL = 0.05
_BACKGROUND_NOTE = (
    "[命令已结束；它启动的后台进程仍占用输出管道，之后的输出不再收集（后台进程照常运行）。"
    "需要它的日志请重定向，如 `cmd > /tmp/cmd.log 2>&1 &`]"
)
# 后台进程占着管道时，继续把输出读走丢弃（防管道写满让它阻塞）；持有引用防任务被回收
_detached_drains: set[asyncio.Task] = set()


async def _wait_exit(process, timeout: float) -> bool:
    """等 shell 进程本身退出（不等管道）。

    不能用 process.wait()：实测 Python 3.12 的 asyncio 要等全部管道关闭才返回，
    后台孙进程占着管道时会一直挂住。returncode 在进程退出时就有值。
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while process.returncode is None:
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(_POLL_INTERVAL)
    return True


async def _drain_detached(readers: asyncio.Future, process) -> None:
    try:
        await readers
    except BaseException:  # noqa: BLE001 — 包括退出时 asyncio.run 的取消
        pass
    finally:
        # 在事件循环关闭前关掉管道：否则 transport 在循环关闭后被回收，打出
        # 「Event loop is closed」traceback（Harbor 基线 query-optimize 实例）
        transport = getattr(process, "_transport", None)
        if transport is not None:
            transport.close()


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


def _child_env() -> dict:
    """shell 子进程的环境：.venv 前置（若有）+ 非交互默认值。

    GIT_EDITOR=true（仅当环境里没设时）：agent 的 shell 没有人坐在键盘前，git 一旦拉起
    编辑器（裸 `git commit`、`git rebase --continue`、非 ff 的 `git merge`）就只能等满
    超时才被整树杀掉——控制台隔离后 vim 抢不到 TUI 的键盘了，但它仍会对着自己的隐藏
    控制台干等（实测 25s 超时用满）。`true` 是「什么都不改、立即成功」的编辑器：
    裸 commit 因提交信息为空被 git 立即中止并报错，rebase/merge 接受默认信息继续。
    与 Claude Code 的做法一致。

    用 setdefault 而不是覆盖：用户在自己 shell 里显式设了 GIT_EDITOR 的，尊重；
    命令里内联的 `GIT_EDITOR=x git …` 本来就优先于环境，不受影响。GIT_SEQUENCE_EDITOR
    不单独设——它未设时回落到 GIT_EDITOR，一处管两处。
    """
    env = _venv_env() or os.environ.copy()
    env.setdefault("GIT_EDITOR", "true")
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
    description = (
        "Execute a shell command and return its output. Uses git bash on Windows (falls back to PowerShell "
        "if Git is not installed), bash on Linux/Mac. Long output is trimmed automatically to its beginning "
        "and end, and the full text is saved to a file you can read — so do not pipe commands into `head` or "
        "`tail` just to shorten output: a pipe replaces the command's exit code with that of `tail`."
    )
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
                "description": (
                    f"Timeout in seconds. Default: 120, maximum: {_MAX_TIMEOUT}. For longer jobs (servers, "
                    "big builds), run them in the background with output redirected "
                    "(`cmd > /tmp/cmd.log 2>&1 &`) and check the log in later calls."
                ),
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

    async def _read_stream(self, stream, stream_name: str, lines: list[str], state: dict | None = None):
        total = 0
        capped = False
        async for raw in stream:
            if state is not None and state.get("detached"):
                continue  # 工具已返回：只读走丢弃，不再累积、不再推进度
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
        clamp_note = ""
        if timeout > _MAX_TIMEOUT:
            clamp_note = (f"[timeout {timeout} 秒超过上限，已按 {_MAX_TIMEOUT} 秒执行；"
                          "更长的任务请后台运行并重定向输出]\n")
            timeout = _MAX_TIMEOUT
        process = None
        job = None
        readers = None
        try:
            env = _child_env()
            if platform.system() == "Windows":
                full_cmd = [self._shell_cmd] + self._shell_args + [command]
                # stdin=DEVNULL + 独立隐藏控制台：子进程树不得附着在 TUI 的控制台上
                # （共享控制台 = 共享输入缓冲与模式，见 core/subproc.py 的事故记录）。
                # 副带效果：要键盘输入的命令（编辑器、密码提示）读到 EOF 立即失败，
                # 而不是抢走 TUI 的按键挂到超时。
                process = await asyncio.create_subprocess_exec(
                    *full_cmd,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    **isolated_console_kwargs(),
                )
                # 立刻圈进 Job（后代自动继承成员资格）；失败退化为直接终止
                job = _WindowsJob.create_and_assign(process.pid)
            else:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    executable=self._shell_cmd,
                    env=env,
                    # 自成进程组：超时/取消时 killpg 整组。只杀 bash 收不掉 `a & b` 里的后台进程
                    start_new_session=True,
                )

            stdout_lines: list[str] = []
            stderr_lines: list[str] = []
            state = {"detached": False}
            readers = asyncio.ensure_future(asyncio.gather(
                self._read_stream(process.stdout, "stdout", stdout_lines, state),
                self._read_stream(process.stderr, "stderr", stderr_lines, state),
            ))
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout

            try:
                # 读输出的同时盯 shell 本身是否退出：只等管道 EOF 的话，后台进程占着管道就挂到超时
                while not readers.done():
                    if process.returncode is not None:
                        try:
                            await asyncio.wait_for(asyncio.shield(readers), timeout=_DRAIN_AFTER_EXIT)
                        except asyncio.TimeoutError:
                            state["detached"] = True
                        break
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    await asyncio.wait({readers}, timeout=min(_POLL_INTERVAL, remaining))
                if not state["detached"]:
                    readers.result()  # 读流本身出错时抛出，走下面的通用错误分支
                    # 管道都关了但进程还在（命令关掉了自己的 stdout/stderr）：等它退出，仍受总超时约束
                    if not await _wait_exit(process, max(0.0, deadline - loop.time())):
                        raise asyncio.TimeoutError
            except asyncio.TimeoutError:
                readers.cancel()
                return clamp_note + await self._handle_timeout(
                    process, job, timeout, stdout_lines, stderr_lines,
                )

            output = _build_result(stdout_lines, stderr_lines, process.returncode)
            if state["detached"]:
                task = asyncio.ensure_future(_drain_detached(readers, process))
                _detached_drains.add(task)
                task.add_done_callback(_detached_drains.discard)
                output = f"{output}\n{_BACKGROUND_NOTE}"
            return clamp_note + output

        except asyncio.CancelledError:
            # stop_task 的取消只在 await 点生效，协程死了子进程不会跟着死——
            # 超时路径有 terminate→kill，取消路径此前没有，孤儿进程会一直跑
            if readers is not None:
                readers.cancel()
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
        POSIX 上 shell 自成进程组（start_new_session），killpg 整组——只 terminate bash
        收不掉 `a & b` 这类复合命令里的后台进程。等退出用 _wait_exit，不用 process.wait()
        （后者要等管道关闭，有逃出进程组的孙进程占着管道就会永远挂住）。
        """
        if process.returncode is not None:
            return
        if job is not None:
            job.terminate()
        else:
            self._signal_group(process, signal.SIGTERM)
        if not await _wait_exit(process, _GRACEFUL_WAIT):
            self._signal_group(process, getattr(signal, "SIGKILL", signal.SIGTERM))
            await _wait_exit(process, _GRACEFUL_WAIT)

    @staticmethod
    def _signal_group(process, sig) -> None:
        if platform.system() == "Windows":  # 无 Job 时的退化路径
            if sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
            return
        try:
            os.killpg(process.pid, sig)  # start_new_session：pgid == shell 的 pid
        except (ProcessLookupError, PermissionError):
            try:
                process.send_signal(sig)
            except ProcessLookupError:
                pass

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

    # 不在此处自截：全文（≤ _MAX_STREAM_CHARS/流）交给入口 ToolResultManager——超过 60K 时
    # 它把完整结果落盘、上下文留头尾预览 + read_file 指针。入口要落「完整」必须拿到截断前的
    # 真全文，故落盘点必须在入口、shell 不能先截（此前这里先截 10000 再被入口砍成 2000 = 双重截断且尾丢）。
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
