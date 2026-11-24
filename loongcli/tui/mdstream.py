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
import re
import time

from rich.console import Console
from rich.live import Live
from rich.markdown import CodeBlock, Heading, HorizontalRule, Markdown
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


class _ShortRule(HorizontalRule):
    """`---` 画成一小段线而不是满宽线：满宽线进了终端历史，窗口拉窄时会被终端折成一长一短两截。"""

    def __rich_console__(self, console, options):
        yield Text("─" * min(24, options.max_width), style=console.get_style("markdown.hr", default="dim"))


class _FitCodeBlock(CodeBlock):
    """代码块：不铺背景色，只留语法高亮（见 cards.code_block）。"""

    def __rich_console__(self, console, options):
        from loongcli.tui.cards import code_block
        yield code_block(str(self.text).rstrip(), self.lexer_name, self.theme)


class LeftMarkdown(Markdown):
    """标题左对齐、不画满宽元素的 Markdown（用于流式渲染、历史回放、计划展示）。"""
    elements = {**Markdown.elements, "heading_open": _LeftHeading, "hr": _ShortRule,
                "fence": _FitCodeBlock, "code_block": _FitCodeBlock}


_LIST_ITEM = re.compile(r"^ {0,3}([-*+]|\d{1,9}[.)])(\s|$)")
_TABLE_ROW = re.compile(r"^ {0,3}\|", re.M)


class MarkdownStream:
    def __init__(self, console: Console, left_pad: int = 2, live_window: int = 6, live_target=None):
        self.console = console
        self.left_pad = left_pad
        self.live_window = live_window
        self.printed: list[str] = []        # 当前未定稿块里已作为稳定行输出的行（去重指针）
        self._base = 0                      # text[:_base] 是已定稿、整块打印过的部分，不再参与渲染
        self.min_delay = 1.0 / 20
        self._render_cost = 0.0             # 单次渲染耗时的指数平滑（秒）
        self.when = 0.0
        self._live: Live | None = None
        # 常驻输入框的动态区出口（InputBar.set_live_renderable）。给了它就不建 Rich Live：
        # 尾部交给输入框上方的动态区，稳定行照常 print（经 patch_stdout 打到输入框上方）。
        self._live_target = live_target

    def _ensure_live(self) -> None:
        if self._live_target is not None:
            return
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

    def _render_block(self, chunk: str) -> list[str]:
        """渲染一段（从 _base 起）。不是开头的那段前面垫一个占位段落「x」再去掉第一行：Rich 块与块之间的空行、
        一二级标题自带的空行、列表开头的空行都看上下文，垫上之后这段前面的空行和整篇渲染时一模一样。"""
        if self._base == 0:
            return self._render_to_lines(chunk)
        return self._render_to_lines("x\n\n" + chunk)[1:]

    def _commit_point(self, text: str) -> int:
        """能定稿到哪：最后一个「空行之后、新块已经开头」的位置（在代码块围栏外）。返回 text 下标，没有就是 _base。

        不在两种块前定稿：列表项开头（松散列表分开渲染会多出空行）、缩进开头（列表续段 / 缩进代码，脱离上文含义就变了）。
        新块开头太短（不到 4 个字、行也没写完）先不判断：「1」可能长成「1. 」列表。"""
        best = self._base
        pos = self._base
        in_fence = False
        prev_blank = False
        for line in text[self._base:].splitlines(keepends=True):
            head = line.lstrip(" ")
            if head.startswith(("```", "~~~")):
                in_fence = not in_fence
            elif (not in_fence and prev_blank and pos > self._base and line[:1] not in (" ", "\t", "\n", "\r")
                  and (len(line.rstrip("\r\n")) >= 4 or line.endswith("\n")) and not _LIST_ITEM.match(line)):
                best = pos
            prev_blank = not line.strip()
            pos += len(line)
        return best

    def _pad(self, text: Text):
        return Padding(text, (0, 0, 0, self.left_pad)) if self.left_pad else text

    def _emit_stable(self, lines: list[str]) -> None:
        # 不能 rstrip("\n")：那会把这批末尾的空行全剥掉，块间空行留不留看合批边界落在哪（真机：有的标题前有空行、有的没有）。
        # 末尾那一个换行 Text.from_ansi 按 splitlines 拆行时已经吃掉，console.print 自带的换行不会叠出多余空行
        block = "".join(lines)
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
        # 写完的块定稿：整块打印、往后不再渲染。此前每次更新都从头渲染整篇，耗时随回复长度线性涨（4400 字 18ms），
        # 节流又按渲染耗时 ×10 放宽——长回复刷新拖到一秒三次、每次还卡住事件循环几十毫秒（真机：输出一顿一顿的）
        cut = len(text) if final else self._commit_point(text)
        if cut > self._base:
            done = self._render_block(text[self._base:cut])
            if len(done) > len(self.printed):
                if self._live_target is None:
                    self._teardown()
                self._emit_stable(done[len(self.printed):])
            self.printed = []
            self._base = cut
        lines = self._render_block(text[self._base:]) if self._base < len(text) else []
        render_time = time.time() - start
        # 渲染越慢，节流越松，区间 [1/20s, 0.5s]。按平滑后的耗时算：原先直接拿单次耗时 ×10，偶发一次 85ms 的渲染
        # （首个代码块加载词法分析器、GC）就停更 0.85 秒（实测后半程最大间隔 754ms）
        self._render_cost = 0.8 * self._render_cost + 0.2 * render_time
        self.min_delay = min(max(self._render_cost * 4, 1.0 / 20), 0.5)

        num_lines = len(lines)
        if not final:
            num_lines -= self.live_window
            # 在写的是表格：整张留在动态区、写完再定稿。表格每多一行列宽都可能变，表头和前几行要是先打出去，
            # 就和后面的行对不上（真机：表头「依据 / 结论」和下面的值错开）。太长（> 60 行）才退回边写边打，免得动态区比屏幕还高
            # 只是不再新打印：已经打出去的行（表格前面的段落）不能回到动态区——此前置 0，尾部从块开头算，
            # 打过的行在动态区又出现一遍，定稿时动态区多缩一截，输入框跳（e2e metric_jitter_markdown 偶发上跳）
            if _TABLE_ROW.search(text, self._base) and len(lines) <= 60:
                num_lines = min(num_lines, len(self.printed))

        # 有新稳定行就打印到 Live 区上方——关键：先拆掉 Live 再 print。
        # transient Live 拆除会擦净活动区，随后的 console.print 落进干净的 scrollback；
        # 此前是在 Live 存活期间 print，auto_refresh=False 下 Live 不重定位，print 与
        # 活动区交错 + 终端滚动会把旧帧留成残影（表格/多行块反复重复的真凶）。
        num_printed = len(self.printed)
        if (final or num_lines > 0) and num_lines - num_printed > 0:
            if self._live_target is None:
                self._teardown()   # 输入框模式不先清动态区：新尾部随稳定行同一帧换上（见 InputBar.set_live_renderable）
            self._emit_stable(lines[num_printed:num_lines])
            self.printed = lines[:num_lines]

        if final:
            self._teardown()
            return

        # 尾部剩余行重绘进（可能刚重建的）Live 窗口
        rest = lines[max(num_lines, 0):]
        tail = self._pad(Text.from_ansi("".join(rest)))
        if self._live_target is not None:
            self._live_target(tail)          # 和输入框之间的空行由输入框统一加（InputBar._message_text）
            return
        self._ensure_live()
        self._live.update(tail, refresh=True)

    def _teardown(self) -> None:
        if self._live_target is not None:
            self._live_target(None)
            return
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
    """统一管理一轮对话里的动态区：assistant 正文的尾部，或单行状态（thinking/工具执行/压缩等）。

    不变量：任一时刻至多一种动态内容。两种互斥模式——正文走 `MarkdownStream`，状态走 spinner；
    切换模式时先拆掉另一个，这正是消灭旧版 stop/restart churn 的关键。

    live_target：常驻输入框的动态区出口（InputBar.set_live_renderable）。给了它，动态内容画在输入框上方，
    不建任何 Rich Live（Rich Live 的光标重画与 patch_stdout 冲突，见 docs/tui-persistent-input.md）；
    不给则退回 Rich Live（非交互、测试）。
    """

    def __init__(self, console: Console, left_pad: int = 2, live_target=None):
        self.console = console
        self.left_pad = left_pad
        self._md: MarkdownStream | None = None
        self._spinner: Live | None = None
        self._spinner_shown = False  # live_target 模式下：状态是否正挂在动态区
        self.current_status = None   # 当前挂着的状态对象（调用方据此判断「还是不是同一个思考状态」）
        self._buf = ""  # 当前正文块累积文本（本块结束即冲刷进 scrollback）
        self._live_target = live_target

    # ── assistant 正文 ──
    def append_text(self, delta: str) -> None:
        self._stop_spinner()
        self._buf += delta
        if self._md is None:
            self._md = MarkdownStream(self.console, left_pad=self.left_pad, live_target=self._live_target)
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
        self.current_status = renderable
        if self._live_target is not None:
            # 状态行上下的空行由输入框加：上方要不要空行得看滚动区末尾是不是已经空着（只有输入框知道）
            self._live_target(renderable, status=True)
            self._spinner_shown = True
            return
        if self._spinner is None:
            # transient：spinner 停止即擦除，不把空帧固化成空行（同 MarkdownStream）
            self._spinner = Live(renderable, console=self.console, refresh_per_second=8, transient=True)
            self._spinner.start()
        else:
            self._spinner.update(renderable)

    def stop_status(self) -> None:
        self._stop_spinner()

    def _stop_spinner(self) -> None:
        self.current_status = None
        if self._spinner_shown:
            self._live_target(None)
            self._spinner_shown = False
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
