"""shell 工具 Job Object 语义单测（Windows-only）。

验证两条对偶语义：默认后代在命令结束时整树回收（不留隐形残留）；
经 WMI（Win32_Process.Create）启动的进程脱离 job 常驻（合法 daemon 的正解）。

背景：jobhunter 的 opencli daemon 被 KILL_ON_JOB_CLOSE 每条命令收尸，每条
boss_api 都要重启 daemon 且打断 Chrome 扩展连接。⚠ 不能用 BREAKAWAY_OK 修
（MSYS2 会让 git bash 全部后代逃逸，整树终止失效，2026-07-16 A/B 实测）——
daemon 必须经 job 外中介（WMI）创建，见 shell.py _WindowsJob 文档。
"""
import platform
import subprocess
import textwrap
import time

import pytest

from loongcli.tools.shell import ShellTool

pytestmark = pytest.mark.skipif(
    platform.system() != "Windows", reason="Windows Job Object 专属语义"
)

CREATE_NO_WINDOW = 0x08000000


def _alive(pid: int) -> bool:
    import ctypes
    STILL_ACTIVE = 259
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return bool(ok) and code.value == STILL_ACTIVE


def _kill(pid: int) -> None:
    subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)


_DIRECT_SPAWNER = textwrap.dedent("""\
    import subprocess, sys
    p = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        creationflags=0x08000000,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    print("SLEEPER_PID=" + str(p.pid))
""")

_WMI_SPAWNER = textwrap.dedent("""\
    import subprocess, sys
    cmdline = f'"{sys.executable}" -c "import time; time.sleep(60)"'
    ps = ("Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
          f"-Arguments @{{CommandLine='{cmdline}'}} | "
          "Select-Object -ExpandProperty ProcessId")
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                       capture_output=True, text=True, timeout=30)
    print("SLEEPER_PID=" + r.stdout.strip())
""")


async def _spawn_via_shell(tmp_path, spawner_code: str) -> int:
    script = tmp_path / "spawner.py"
    script.write_text(spawner_code, encoding="utf-8")
    result = await ShellTool().execute(f'python "{script.as_posix()}"', timeout=60)
    pid = None
    for line in result.splitlines():
        if line.startswith("SLEEPER_PID=") and line.split("=", 1)[1].strip().isdigit():
            pid = int(line.split("=", 1)[1])
    assert pid, f"spawner 未输出 PID，命令结果：{result}"
    return pid


class TestJobSemantics:
    async def test_default_descendants_reaped(self, tmp_path):
        """默认后代：命令结束（job 句柄关闭）后被内核整树收尸。"""
        pid = await _spawn_via_shell(tmp_path, _DIRECT_SPAWNER)
        try:
            # KILL_ON_JOB_CLOSE 由内核异步收尸，给短暂窗口
            for _ in range(10):
                if not _alive(pid):
                    break
                time.sleep(0.3)
            assert not _alive(pid), "默认后代应在命令结束后被 job 回收"
        finally:
            _kill(pid)

    async def test_wmi_spawned_daemon_survives(self, tmp_path):
        """WMI 启动：进程挂在 WmiPrvSE 下、不在 job 里，命令结束后存活。"""
        pid = await _spawn_via_shell(tmp_path, _WMI_SPAWNER)
        try:
            time.sleep(1.0)
            assert _alive(pid), "WMI 启动的进程应在命令结束后存活（daemon 常驻语义）"
        finally:
            _kill(pid)
