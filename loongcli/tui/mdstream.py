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
from rich.markdown import Markdown, Heading
from rich.padding import Padding
from rich.text import Text

# emoji 变体选择符（VS15/16 及补充区）：Rich 宽度表记 0、且让前一个窄字符
# （如 ⚠ U+26A0）被终端按 emoji 呈现画成 2 格——Rich 记 1 格、终端画 2 格。
# 当折行后某行恰好被 Rich 填满时，真实宽度超出终端再折一次，Live 活动区
# 实际高度比记账多 1 行，transient 擦除/重绘全部错位 1 行：上一帧的行留成
# 残影、擦除落空成空洞（2026-07-17 截图"引用行三连+大片空洞"的真凶）。
# 只需剥这一类：它是唯一"Rich 低估宽度"的方向；ZWJ 合成序列 Rich 只会高估
# （提前折行，无害），故保留以维持正文观感。
_VS_STRIP = dict.fromkeys(list(range(0xFE00, 0xFE10)) + list(range(0xE0100, 0xE01F0)))


def strip_variation_selectors(text: str) -> str:
    """剥离变体选择符，让流式渲染的宽度记账与终端实际渲染一致。"""
    return text.translate(_VS_STRIP)


class _LeftHeading(Heading):
    """所有级别标题一律左对齐、不画盒子。

    Rich（14.x）的 Heading.__rich_console__ 硬编码 justify="center"，h1 还套
    Panel 盒子——agent 回复里出现居中带框标题，观感很怪。没有对齐配置钩子
    （LEVEL_ALIGN 之类的类属性不存在），只能覆写渲染方法本身。
    """

    def __rich_console__(self, console, options):
        text = self.text
        text.justify = "left"
        if self.tag in ("h1", "h2"):
            yield Text("")  # 大标题前留一空行，替代原 Panel 的视觉分隔
        yield text


class LeftMarkdown(Markdown):
    """标题一律左对齐的 Markdown（用于流式渲染、历史回放、计划展示）。"""
    elements = {**Markdown.elements, "heading_open": _LeftHeading}


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
            # auto_refresh=False：只在 update(refresh=True) 时重绘，杜绝 20fps 后台重绘
            # 与 console.print 交错造成的闪烁；重绘频率由 update() 的自适应节流控制。
            # transient=True：stop 时擦除 Live 区而非固化最后一帧——非 transient 的
            # stop 会把空帧固化成一个空行，每段正文冲刷都漏一行（块间距失控的根源）。
            self._live = Live(Text(""), console=self.console, auto_refresh=False, transient=True)
            self._live.start()

    def _render_to_lines(self, text: str) -> list[str]:
        width = max(20, self.console.width - self.left_pad)
        buf = io.StringIO()
        tmp = Console(
            file=buf, force_terminal=True, width=width,
            color_system=self.console.color_system or "standard",
        )
        tmp.print(LeftMarkdown(text))
        return buf.getvalue().splitlines(keepends=True)

    def _pad(self, text: Text):
        return Padding(text, (0, 0, 0, self.left_pad)) if self.left_pad else text

    def _emit_stable(self, lines: list[str]) -> None:
        block = "".join(lines).rstrip("\n")  # console.print 自带换行，避免行间多空行
        if not block:
            return
        self.console.print(self._pad(Text.from_ansi(block)))

    def update(self, text: str, final: bool = False) -> None:
        now = time.time()
        if not final and now - self.when < self.min_delay:
            return
        self.when = now
        text = strip_variation_selectors(text)

        start = time.time()
        lines = self._render_to_lines(text)
        render_time = time.time() - start
        # 渲染越慢，节流越松，避免长回答卡顿；区间 [1/20s, 2s]
        self.min_delay = min(max(render_time * 10, 1.0 / 20), 2)

        num_lines = len(lines)
        if not final:
            num_lines -= self.live_window

        # 有新稳定行就打印到 Live 区上方——关键：先拆掉 Live 再 print。
        # transient Live 拆除会擦净活动区，随后的 console.print 落进干净的 scrollback；
        # 此前是在 Live 存活期间 print，auto_refresh=False 下 Live 不重定位，print 与
        # 活动区交错 + 终端滚动会把旧帧留成残影（表格/多行块反复重复的真凶）。
        num_printed = len(self.printed)
        if (final or num_lines > 0) and num_lines - num_printed > 0:
            self._teardown()
            self._emit_stable(lines[num_printed:num_lines])
            self.printed = lines[:num_lines]

        if final:
            self._teardown()
            return

        # 尾部剩余行重绘进（可能刚重建的）Live 窗口
        self._ensure_live()
        rest = lines[max(num_lines, 0):]
        self._live.update(self._pad(Text.from_ansi("".join(rest))), refresh=True)

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


class _StatusGroup:
    """spinner 状态行 + 可选输入回显行的组合 renderable。

    provider 每帧被求值（spinner Live 以 refresh_per_second 重渲染同一
    renderable），返回 None 时第二行不渲染——Live 高度回到 1 行，与无回显
    时逐帧一致。provider 抛异常按 None 处理：回显是锦上添花，绝不能把
    渲染线程炸掉。"""

    def __init__(self, base, input_line):
        self._base = base
        self._input_line = input_line

    def __rich_console__(self, console, options):
        yield self._base
        try:
            line = self._input_line()
        except Exception:
            line = None
        if line is not None:
            yield line


class StreamView:
    """统一管理一轮对话里那个唯一的 Rich Live。

    不变量：任一时刻至多一个 Live 活着。两种互斥模式——assistant 正文走
    `MarkdownStream`，thinking/工具执行/压缩等状态走单行 spinner。切换模式时先拆掉
    另一个，这正是消灭旧版 stop/restart churn 的关键。

    input_line：可选的输入回显行 provider（() -> renderable | None），挂在
    spinner 状态行下方，用于"执行中打字"的实时回显。只作用于 spinner 模式；
    MarkdownStream 正文流不回显（其自适应节流最长 2s，回显会卡成幻灯片）。
    """

    def __init__(self, console: Console, left_pad: int = 2, input_line=None):
        self.console = console
        self.left_pad = left_pad
        self._md: MarkdownStream | None = None
        self._spinner: Live | None = None
        self._buf = ""  # 当前正文块累积文本（本块结束即冲刷进 scrollback）
        self._input_line = input_line

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
        if self._input_line is not None:
            renderable = _StatusGroup(renderable, self._input_line)
        if self._spinner is None:
            # transient：spinner 停止即擦除，不把空帧固化成空行（同 MarkdownStream）
            self._spinner = Live(renderable, console=self.console, refresh_per_second=8, transient=True)
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
