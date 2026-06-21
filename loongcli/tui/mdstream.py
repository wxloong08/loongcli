"""流式 Markdown 渲染：边生成边显示完整格式化内容，零闪烁/堆叠。

技巧改编自 Aider 的 `aider/mdstream.py`（Apache-2.0）：每次更新把"目前为止的
Markdown"整体渲染成行，再切两半——

  - 已滚出底部 `live_window` 的行 = **稳定行** → `console.print` 到 Live 区上方，
    完整格式化落入终端 scrollback，**永不重绘**（所以不可能 stranding/堆叠）。
  - 最后 `live_window` 行 → 只在单个 Rich `Live` 里重绘。

Aider 的洞见原话：「Markdown 打到 console 对终端 scrollback 更友好；live 窗口和
scrollback 不对付。」`update()` 自适应节流（目标 ~20fps，渲染慢则退避），避免长
回答每个 token 都付 O(n²) 重渲染成本。

相对 Aider 修正了一个边界 bug：当总行数少于 `live_window` 时，原实现 `lines[num_lines:]`
用到负索引会丢掉开头几行；这里用 `max(num_lines, 0)` 守护。
"""
from __future__ import annotations

import io
import time

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.text import Text


class MarkdownStream:
    def __init__(self, console: Console, left_pad: int = 2, live_window: int = 6):
        self.console = console
        self.left_pad = left_pad
        self.live_window = live_window
        self.printed: list[str] = []        # 已作为稳定行输出的行（去重指针）
        self.min_delay = 1.0 / 20
        self.when = 0.0
        self._live: Live | None = None

    def _ensure_live(self) -> None:
        if self._live is None:
            self._live = Live(Text(""), console=self.console, refresh_per_second=20)
            self._live.start()

    def _render_to_lines(self, text: str) -> list[str]:
        width = max(20, self.console.width - self.left_pad)
        buf = io.StringIO()
        tmp = Console(
            file=buf, force_terminal=True, width=width,
            color_system=self.console.color_system or "standard",
        )
        tmp.print(Markdown(text))
        return buf.getvalue().splitlines(keepends=True)

    def _pad(self, text: Text):
        return Padding(text, (0, 0, 0, self.left_pad)) if self.left_pad else text

    def _emit_stable(self, lines: list[str]) -> None:
        block = "".join(lines).rstrip("\n")  # console.print 自带换行，避免行间多空行
        if not block:
            return
        self.console.print(self._pad(Text.from_ansi(block)))

    def update(self, text: str, final: bool = False) -> None:
        self._ensure_live()

        now = time.time()
        if not final and now - self.when < self.min_delay:
            return
        self.when = now

        start = time.time()
        lines = self._render_to_lines(text)
        render_time = time.time() - start
        # 渲染越慢，节流越松，避免长回答卡顿；区间 [1/20s, 2s]
        self.min_delay = min(max(render_time * 10, 1.0 / 20), 2)

        num_lines = len(lines)
        if not final:
            num_lines -= self.live_window

        # 有新稳定行就打印到 Live 区上方
        if final or num_lines > 0:
            num_printed = len(self.printed)
            if num_lines - num_printed > 0:
                self._emit_stable(lines[num_printed:num_lines])
                self.printed = lines[:num_lines]

        if final:
            self._teardown()
            return

        # Live 窗口始终刷新尾部剩余行（max 守护总行数 < live_window 的情形）
        rest = lines[max(num_lines, 0):]
        self._live.update(self._pad(Text.from_ansi("".join(rest))))

    def _teardown(self) -> None:
        if self._live is not None:
            try:
                self._live.update(Text(""))
                self._live.stop()
            except Exception:
                pass
            self._live = None

    def stop(self) -> None:
        """错误路径用的幂等拆除（不冲刷剩余内容）。"""
        self._teardown()


class StreamView:
    """统一管理一轮对话里那个唯一的 Rich Live。

    不变量：任一时刻至多一个 Live 活着。两种互斥模式——assistant 正文走
    `MarkdownStream`，thinking/工具执行/压缩等状态走单行 spinner。切换模式时先拆掉
    另一个，这正是消灭旧版 stop/restart churn 的关键。
    """

    def __init__(self, console: Console, left_pad: int = 2):
        self.console = console
        self.left_pad = left_pad
        self._md: MarkdownStream | None = None
        self._spinner: Live | None = None
        self._buf = ""  # 当前正文块累积文本（本块结束即冲刷进 scrollback）

    # ── assistant 正文 ──
    def append_text(self, delta: str) -> None:
        self._stop_spinner()
        self._buf += delta
        if self._md is None:
            self._md = MarkdownStream(self.console, left_pad=self.left_pad)
        self._md.update(self._buf)

    def flush_text(self) -> None:
        """把当前正文块完整冲刷进 scrollback，结束该块。"""
        if self._md is not None:
            try:
                self._md.update(self._buf, final=True)
            except Exception:
                self._md.stop()
            self._md = None
        self._buf = ""

    @property
    def has_text(self) -> bool:
        return bool(self._buf.strip())

    # ── 状态 spinner ──
    def status(self, renderable) -> None:
        """显示/更新单行状态。会先冲刷未结束的正文块（状态意味着正文已暂停/结束）。"""
        self.flush_text()
        if self._spinner is None:
            self._spinner = Live(renderable, console=self.console, refresh_per_second=8)
            self._spinner.start()
        else:
            self._spinner.update(renderable)

    def stop_status(self) -> None:
        self._stop_spinner()

    def _stop_spinner(self) -> None:
        if self._spinner is not None:
            try:
                self._spinner.update(Text(""))
                self._spinner.stop()
            except Exception:
                pass
            self._spinner = None

    # ── 收尾 ──
    def close(self) -> None:
        """finally 调用：冲刷正文 + 停 spinner，杜绝 Live 泄漏。"""
        self.flush_text()
        self._stop_spinner()
