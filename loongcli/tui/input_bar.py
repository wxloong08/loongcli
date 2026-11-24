"""常驻输入框：整个会话只有一个输入框，模型运行期间也能编辑、提交（设计稿 docs/tui-persistent-input.md，步骤 14b）。

只管输入本身，不认识 agent——提交的内容放进 `submissions` 队列，由 TUI 决定是开新回合还是排队插话。

屏幕从下往上：
- 输入行：`❯ ` + 正在编辑的内容（prompt_toolkit 原生编辑；显示行上下导航沿用 input_nav）
- 动态区：正文还在变的尾部（set_live）+ 一行转圈状态（set_status），放进 prompt 的 message，随 refresh_interval 重画
- 滚动区：其余输出照常 print，由调用方用 patch_stdout(raw=True) 打到输入框上方

原型 v2（scratchpad/interject_probe_v2.py）在 Windows Terminal 验证过的三个设定沿用：状态行放动态区而非
bottom_toolbar（后者在 WT 下没显示出来）、reserve_space_for_menu=0（否则多行输入中间空出几行）、
ttimeoutlen 调短（否则 Esc 要等 0.5 秒才生效）。另：提交后清掉输入行（erase_when_done），回显由调用方
打印——原型里提交内容出现两次就是因为两边都留了。
"""
from __future__ import annotations

import asyncio
import io
import re
import sys
import threading
import time
from typing import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.renderer import Renderer
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from rich.console import Console

from loongcli.tui.input_nav import add_display_row_bindings

PROMPT = "❯ "
CONTINUATION = "  "
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
MENU_ROWS = 8          # 补全菜单预留行数（prompt_toolkit 默认值）
_UNSET = object()
PRINT_COALESCE = 0.03  # 打印合批窗口（秒）：窗口内的多次打印合成一次「擦除-写入-重画」


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _display_rows(text: str, width: int) -> int:
    """文字在 width 列宽的终端里占几行（去掉 ANSI 转义、按显示宽度折行）。"""
    cols = max(1, width)
    return sum(max(1, -(-get_cwidth(_ANSI_RE.sub("", line)) // cols)) for line in text.split("\n"))


def _rows_above_after_reflow(row_widths: list[int], cursor_x: int, new_cols: int) -> tuple[int, int]:
    """终端变窄后，旧布局里光标上方实际占几行、光标落在第几列。

    Windows Terminal 拉窄时会把超出新宽度的旧行重排成多行（一条 120 列横线放进 75 列 = 2 行）。prompt_toolkit 重画前
    只按旧布局行数上移光标，够不着顶部，旧横线就一截截残留（真机：拉窄一次多一摞横线）。row_widths 是光标所在行之上
    各行的显示宽度。"""
    cols = max(1, new_cols)
    up = sum(max(1, -(-w // cols)) for w in row_widths) + cursor_x // cols
    return up, cursor_x % cols


class _CompactRenderer(Renderer):
    """不把布局撑到终端底部的渲染器。

    prompt_toolkit 默认让布局占满光标以下的全部空间（高度取「光标下方行数」），底栏因此贴在终端最底边——
    刚启动时输入框在上面、状态行在几十行之外。这里让布局保持自身高度，底栏（下横线 + 状态行）紧贴输入框。
    底栏只在「高度已知」时显示，这里一律视为已知（不需要那个高度）。"""

    @property
    def height_is_known(self) -> bool:
        return True

    def request_absolute_cursor_position(self) -> None:
        pass

    def _correct_for_reflow(self) -> None:
        """终端变窄过：按重排后的行数校正光标位置，并强制整块重画（不再和旧画面做差分）。

        必须放在渲染器里而不是 resize 回调里：每 0.1 秒的定时重画会先发现尺寸变了、按旧行数上移重画并更新尺寸记录，
        等 resize 回调（约 0.5 秒轮询一次）被调用时尺寸已「没变」，修正永远轮不到（真机：改了回调仍然堆横线）。"""
        last, old = self._last_screen, self._last_size
        if last is None or old is None:
            return
        try:
            cols = self.output.get_size().columns
        except Exception:
            return
        if cols >= old.columns:
            return
        widths = []
        for y in range(self._cursor_pos.y):
            row = last.data_buffer.get(y, {})
            widths.append(max((x + max(1, get_cwidth(ch.char)) for x, ch in row.items() if ch.char not in ("", " ")),
                              default=0))
        up, x = _rows_above_after_reflow(widths, self._cursor_pos.x, cols)
        self._cursor_pos = self._cursor_pos._replace(x=x, y=up)
        self._last_screen = None

    def _shrink_if_needed(self, layout) -> None:
        """这一帧需要的高度比上一帧矮：丢掉上一帧，让父类从顶部擦到底再整块重画。

        prompt_toolkit 取高度 = max(上一帧高度, 这一帧需要的高度)（防抖），布局只长不缩：输入折成三行再删到一行，
        多出的空行一直留着、下横线停在撑大后的位置（真机）。丢掉上一帧后父类走「首帧」分支：光标回顶部、erase_down、
        整块画，和这一帧的其它输出在同一次 flush 里——不像 erase() 那样先单独 flush 一个空白帧。
        preferred_height 里格式化文字按 render_counter 缓存，父类随后同一帧再算一次不会重复渲染。"""
        last = self._last_screen
        if last is None:
            return
        try:
            size = self.output.get_size()
            want = layout.container.preferred_height(size.columns, size.rows).preferred
        except Exception:
            return
        if want < min(last.height, size.rows):
            self._last_screen = None

    def render(self, app, layout, is_done: bool = False) -> None:
        self._correct_for_reflow()
        if not is_done:
            self._shrink_if_needed(layout)
        super().render(app, layout, is_done)

    def erase(self, leave_alternate_screen: bool = True) -> None:
        self._correct_for_reflow()
        super().erase(leave_alternate_screen)


class InputBar:
    def __init__(
        self,
        *,
        history=None,
        completer=None,
        key_bindings: KeyBindings | None = None,
        on_interrupt: Callable[[], None] | None = None,
        status_provider: Callable[[], tuple[str, str]] | None = None,
        mode_provider: Callable[[], str] | None = None,
        input=None,
        output=None,
    ):
        """status_provider：输入框下方状态行的内容，返回 (左, 右)，每次重画都取一次（模型、上下文等实时变化）。
        mode_provider：当前模式名（如「规划」），非空时提示符变成彩色「规划 ❯ 」；每次重画都取，模式被别处改掉也跟得上。"""
        self.submissions: asyncio.Queue[str | None] = asyncio.Queue()
        self.on_interrupt = on_interrupt
        self.status_provider = status_provider
        self.mode_provider = mode_provider
        self._live = ""                 # 动态区：正文尾部（ANSI）
        self._status = ""               # 动态区：状态行文字；空 = 不显示
        self._renderable = None         # 动态区：Rich 可渲染对象（StreamView 的正文尾部 / 转圈状态），每次刷新重新渲染
        self._ask: tuple[str, asyncio.Future] | None = None    # 文字提问：(问题, future)
        self._choice: dict | None = None          # 选项对话框：title / options / selected / future
        self._ask_draft: Document | None = None   # 提问前输入框里的草稿，回答完放回去
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending: list[str] = []             # 待打印到输入框上方的文字（write 合批）
        self._pending_lock = threading.Lock()
        self._emit_scheduled = False
        self._staged = _UNSET                     # 等待随下一批打印一起生效的动态区内容：(renderable, 是否状态行)
        self._renderable_is_status = False        # 动态区挂的是状态行（思考中 / 工具执行中）还是正文尾部
        # 输入框「钉住」：提示区（动态区 + 对话框 + 上横线 + ❯ 行）的高度只在打印进滚动区的同一帧才缩，
        # 平时变矮就在顶部补空行。见 _message / _emit
        self._min_rows = 0
        self._min_width = 0
        self._ends_blank = True                   # 滚动区最后一行是不是空行（决定状态行上方要不要再空一行）

        kb = KeyBindings()

        @kb.add("c-j")                  # 换行；Windows Terminal 的 Shift+Enter 已映射为发送 \n（即 Ctrl+J）
        def _(event):
            event.current_buffer.insert_text("\n")

        # 不设 eager：Alt+V（贴图）在终端里是「Esc + v」，eager 的 Esc 一按就执行、v 永远等不到。
        # 单按 Esc 的等待由下面的 timeoutlen 压到 0.05 秒（默认 1 秒），手感不变
        @kb.add("escape")
        def _(event):
            self._interrupt()

        @kb.add("c-c")
        def _(event):
            self._interrupt()

        # 显示行上下导航先注册：同一个键有多个绑定时后注册的生效，选项对话框的上下键要盖过它
        add_display_row_bindings(kb, prompt=self._prompt_text, continuation=CONTINUATION)

        # 选项对话框（确认框、计划审批）：上下键移动、数字直选、回车确认、Esc 取消；其它键一律吞掉——
        # 正在打的插话不会漏进来当答案（14e 真机：插话草稿被当成确认答案、确认被拒、插话丢失）。
        # 最后一项可以是输入项（「提修改建议：」）：选中它时直接在这一行打字，回车连文字一起提交
        choosing = Condition(lambda: self._choice is not None)

        @kb.add("<any>", filter=choosing)
        def _(event):
            ch = event.data or ""
            if self._editing_choice():
                if ch.isprintable():
                    event.current_buffer.insert_text(ch)    # 输入项上：数字也是文字
                return
            n = len(self._choice["options"])
            if ch.isdigit() and 1 <= int(ch) <= n:
                if self._choice["input_last"] and int(ch) == n:
                    self._choice["selected"] = n - 1        # 数字选到输入项：跳过去等打字，不直接提交空文字
                else:
                    self._resolve_choice((int(ch) - 1, ""))
            elif self._choice["input_last"] and ch.isprintable() and ch.strip():
                self._choice["selected"] = n - 1            # 有输入项时直接打字 = 跳到输入项开始写
                event.current_buffer.insert_text(ch)

        @kb.add("up", filter=choosing, eager=True)
        def _(event):
            self._choice["selected"] = (self._choice["selected"] - 1) % len(self._choice["options"])

        @kb.add("down", filter=choosing, eager=True)
        def _(event):
            self._choice["selected"] = (self._choice["selected"] + 1) % len(self._choice["options"])

        @kb.add("enter", filter=choosing, eager=True)
        def _(event):
            text = event.current_buffer.text if self._editing_choice() else ""
            self._resolve_choice((self._choice["selected"], text))

        bindings = merge_key_bindings([kb, key_bindings]) if key_bindings else kb

        self.session = PromptSession(
            self._message,
            history=history,
            completer=completer,
            complete_while_typing=completer is not None,
            key_bindings=bindings,
            prompt_continuation=CONTINUATION,
            refresh_interval=0.1,       # 转圈动起来、动态区跟上
            reserve_space_for_menu=0,
            bottom_toolbar=self._toolbar,
            # 底栏默认反色整行高亮；改成普通底色，只是输入框下面的一条横线 + 一行暗色状态
            style=Style.from_dict({"bottom-toolbar": "noreverse", "bottom-toolbar.text": ""}),
            erase_when_done=True,
            input=input,
            output=output,
        )
        self.session.app.ttimeoutlen = 0.05
        # 多键绑定的等待：Esc 后 0.05 秒内没跟 v（Alt+V）就当单按 Esc。默认 1 秒——Esc 中断会慢一拍
        self.session.app.timeoutlen = 0.05
        self.session.app.renderer.__class__ = _CompactRenderer
        # 回车提交不结束应用（见 _accept）。默认缓冲区只在 PromptSession 构造时建一次，prompt_async 只重置内容，替换安全
        self.session.default_buffer.accept_handler = self._accept
        self._stopped = False
        # 补全菜单的预留空间每次重画前按「菜单实际有几项」给：常驻预留会让多行输入中间空出几行（原型 v2），
        # 一律不留则输入框贴底时菜单没地方画（14e 真机：/ 不出补全），按「以 / 开头」给则打完命令后空间还占着（真机）
        self.session.app.before_render += self._update_menu_space

    def _update_menu_space(self, _app=None) -> None:
        state = self.session.default_buffer.complete_state
        rows = min(MENU_ROWS, len(state.completions)) if state and state.completions else 0
        # 预留变少后布局高度变矮，由 _CompactRenderer._shrink_if_needed 整块重画缩回（此前这里单独 erase()，会多刷一个空白帧）
        self.session.reserve_space_for_menu = rows

    # ── 动态区 ──
    def set_live(self, ansi_text: str) -> None:
        self._live = ansi_text.rstrip("\n")
        self._invalidate()

    def set_status(self, text: str) -> None:
        self._status = text
        self._invalidate()

    def set_live_renderable(self, renderable, status: bool = False) -> None:
        """StreamView 的动态区出口（替代运行期的 Rich Live）。传 None 清空。status=True 表示挂的是状态行（上下留空行）。

        存对象而不是渲染好的文字：每次刷新（refresh_interval）重新渲染，Rich 的 Spinner 按时间取帧，自然就转起来了。"""
        with self._pending_lock:
            if self._pending:
                # 还有没写出的滚动区内容：动态区的变化先暂存，和那批内容同一帧生效。否则会先画出「尾部已变短、
                # 稳定行还没出现」的一帧，输入框往上跳一下（e2e：长 markdown 屏幕满后 8 秒上跳 9 次）
                self._staged = (renderable, status)
                return
            if renderable is None and self._renderable is not None and self._loop is not None \
                    and not self._loop.is_closed():
                # 清空动态区也晚一个合批窗口再生效：清空后几乎总是紧跟一次打印（工具执行完：先撤输出行、再打结果块），
                # 窗口内来了打印就同帧换上，不会先闪一下空白再出结果；没来，30ms 后照常清空（高度由 _min_rows 钉住）
                self._staged = (None, False)
                if not self._emit_scheduled:
                    self._emit_scheduled = True
                    self._loop.call_soon_threadsafe(self._loop.call_later, PRINT_COALESCE, self._emit)
                return
            # 直接生效的新内容作废还没生效的暂存清空：StreamView.status 先清再挂新状态，
            # 不作废的话 30ms 后那次清空会把刚挂上的新状态抹掉
            self._staged = _UNSET
        self._renderable, self._renderable_is_status = renderable, status
        self._invalidate()

    def _render_renderable(self) -> str:
        try:
            width = self.session.app.output.get_size().columns
        except Exception:
            width = 80
        buf = io.StringIO()
        # 少一列：Padding 会把行补满到整宽，正好顶满终端宽度时 prompt_toolkit 会再折一行
        Console(file=buf, force_terminal=True, color_system="truecolor", width=max(20, width - 1),
                legacy_windows=False).print(self._renderable)
        return buf.getvalue().rstrip("\n")

    def _width(self) -> int:
        try:
            return self.session.app.output.get_size().columns
        except Exception:
            return 80

    def _rule(self) -> str:
        return "\x1b[2m" + "─" * self._width() + "\x1b[0m"

    def _toolbar(self):
        """输入框下方：一条横线 + 状态行（左：按键提示；右：模型、上下文等）。"""
        left, right = "", ""
        if self.status_provider:
            try:
                left, right = self.status_provider()
            except Exception:
                pass                    # 状态行是锦上添花，取值出错不能把输入框炸掉
        if self._choice:
            # 对话框的按键提示放这里：对话框最后一行后面紧跟输入行，提示没法再写在对话框里
            if self._editing_choice():
                left = "直接打字 · 回车提交 · ↑↓ 换选项 · Esc 取消"
            else:
                left = "↑↓ 选择 · 回车确认 · 数字直选 · Esc 取消"
            if self._ask_draft is not None:
                left += "（正在打的内容已收起，选完放回）"
        width = self._width()
        left = "  " + left
        gap = width - get_cwidth(left) - get_cwidth(right) - 1
        if gap < 2:                     # 放不下：先砍右边
            right = ""
            gap = max(0, width - get_cwidth(left) - 1)
        line = left + " " * gap + right
        return ANSI(self._rule() + "\n\x1b[2m" + line + "\x1b[0m")

    def _mode(self) -> str:
        if not self.mode_provider:
            return ""
        try:
            return self.mode_provider() or ""
        except Exception:
            return ""                   # 同状态行：取值出错就当普通模式，不能把输入框炸掉

    def _prompt_text(self) -> str:
        mode = self._mode()
        return f"{mode} {PROMPT}" if mode else PROMPT

    def _invalidate(self) -> None:
        app = self.session.app
        if app.is_running:
            app.invalidate()

    def _pin(self, text: str) -> str:
        """提示区高度只增不减：比上一帧矮就在顶部补空行。真正变矮只发生在 _emit 里——打印进滚动区的行数
        同帧抵掉同样多的高度，输入框原地不动。

        为什么不逐个时刻去补：确认框关闭（5 行缩回 1 行）、思考转正文（3 行变 2 行）、工具结束……每个都是
        「先缩一下、30ms 后被打印顶回去」，按时刻打补丁已经第三次了，统一成这一条不变量（真机「输入框没有固定」）。"""
        width = self._width()
        if width != self._min_width:          # 改了窗口宽度：行数口径变了，重新起算
            self._min_width, self._min_rows = width, 0
        rows = _display_rows(text, width)
        if rows < self._min_rows:
            text = "\n" * (self._min_rows - rows) + text
            rows = self._min_rows
        self._min_rows = rows
        return text

    def _message(self):
        return ANSI(self._pin(self._message_text()))

    def _message_text(self) -> str:
        parts = [self._live] if self._live else []
        if self._renderable is not None:
            try:
                rendered = self._render_renderable()
            except Exception:
                rendered = ""           # 动态区是锦上添花，渲染出错不能把输入框炸掉
            if rendered:
                # 状态行（思考中… / 工具执行中）和上方正文隔一行；滚动区末尾已经是空行就不再加（否则回合开头两个空行）
                if self._renderable_is_status and not self._ends_blank:
                    parts.append("")
                parts.append(rendered)
        if parts:
            parts.append("")            # 动态区和输入框之间空一行（用户：贴在一起太密集）
        if self._choice:
            # 选项对话框占据输入框的位置（Claude Code 式）：两条横线之间是问题 + 选项，不显示输入行
            # 输入行只能接在提示内容的最后一行后面，所以输入项放最后一项、按键提示挪到下方状态行（_toolbar）
            c = self._choice
            parts.append(self._rule())
            parts.append(f"  \x1b[1m{c['title']}\x1b[0m")
            last = len(c["options"]) - 1
            for i, opt in enumerate(c["options"]):
                label = f"{i + 1}. {opt}" + ("：" if c["input_last"] and i == last else "")
                if i == c["selected"]:
                    parts.append(f"  \x1b[1;36m❯ {label}\x1b[0m")
                else:
                    parts.append(f"    {label}")
            return "\n".join(parts)
        if self._ask:
            parts.append(self._rule())
            parts.append(f"  \x1b[1m{self._ask[0]}\x1b[0m")
            return "\n".join(parts) + "\n\x1b[1;36m❯ \x1b[0m"
        if self._status:
            frame = _SPINNER[int(time.monotonic() * 10) % len(_SPINNER)]
            parts.append(f"\x1b[36m{frame}\x1b[0m \x1b[2m{self._status}\x1b[0m")
        parts.append(self._rule())     # 输入框上横线（下横线 + 状态行在 _toolbar）
        color = "1;35" if self._mode() else "1;32"   # 普通模式绿色，规划等模式品红，一眼能分出来
        return "\n".join(parts) + "\n" + f"\x1b[{color}m{self._prompt_text()}\x1b[0m"

    # ── 输入框上方打印 ──
    encoding = "utf-8"

    def write(self, text: str) -> None:
        """把文字打到输入框上方（滚动区）。线程安全；短时间内的多次写入合成一次。可直接当 Rich Console 的 file。

        为什么不用 patch_stdout：它每次打印分三次写终端（擦输入框 → 写内容 → 重画输入框），终端在中间刷新一帧，
        输入框就从屏幕上消失一下——这就是流式输出时的闪烁（e2e 实测 8 秒 15 次）。这里把三步攒在一个缓冲里一次写出。"""
        if not text:
            return
        with self._pending_lock:
            self._pending.append(text)
            if self._emit_scheduled:
                return
            self._emit_scheduled = True
        loop = self._loop
        if loop is None or loop.is_closed():
            self._emit()
            return
        loop.call_soon_threadsafe(loop.call_later, PRINT_COALESCE, self._emit)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return True

    def _emit(self) -> None:
        with self._pending_lock:
            text = "".join(self._pending)
            self._pending.clear()
            self._emit_scheduled = False
            staged, self._staged = self._staged, _UNSET
        if staged is not _UNSET:
            self._renderable, self._renderable_is_status = staged
        if not text:
            self._invalidate()
            return
        body = text[:-1] if text.endswith("\n") else text
        # 打进滚动区多少行，钉住的高度就放掉多少行——同一帧里上面多出来的正好补上下面缩掉的，输入框不动
        self._min_rows = max(0, self._min_rows - _display_rows(body, self._width()))
        self._ends_blank = not _ANSI_RE.sub("", body.rsplit("\n", 1)[-1]).strip()
        app = self.session.app
        if not app.is_running or app._running_in_terminal:
            # 输入框没在画（run_in_terminal 期间 / 退出后）：直接写，交给外层 patch_stdout 兜底
            sys.stdout.write(text)
            sys.stdout.flush()
            return
        out = app.output
        real_flush = out.flush
        out.flush = lambda: None             # 擦除、写入、重画期间不落终端，最后一次写出
        try:
            app.renderer.erase()
            out.enable_autowrap()
            out.write_raw(text)
            app.renderer.reset()
            app._redraw()
        finally:
            out.flush = real_flush
            out.flush()

    # ── 提问模式 ──
    def _flush_pending(self) -> None:
        """弹对话框 / 提问前先把合批里的打印写出去：对话框是立刻画的，它问的那件事（风险操作说明、计划）
        还在 30ms 合批里，会先看到对话框、过一会儿上面才出现说明。已排定的那次 _emit 到时发现没东西可写，无害。"""
        with self._pending_lock:
            pending = bool(self._pending) or self._staged is not _UNSET
        if pending and self.session.app.is_running:
            self._emit()

    def _stash_draft(self) -> None:
        buf = self.session.default_buffer
        self._ask_draft = buf.document if buf.text else None
        if buf.text:
            buf.set_document(Document(""), bypass_readonly=True)

    def _restore_draft(self) -> None:
        if self._ask_draft is not None and self.session.app.is_running:
            self.session.default_buffer.set_document(self._ask_draft, bypass_readonly=True)
            self._ask_draft = None

    async def ask(self, question: str) -> str | None:
        """输入框切成文字提问，回车提交即答案（不进 submissions）；Esc / Ctrl+C 返回 None（取消）。
        提问前输入框里正在打的草稿先收起来，答完原样放回——草稿不会被当成答案。"""
        fut = asyncio.get_running_loop().create_future()
        self._flush_pending()
        self._stash_draft()
        self._ask = (question, fut)
        self._invalidate()
        try:
            return await fut
        finally:
            self._ask = None
            self._restore_draft()
            self._invalidate()

    async def choose(self, title: str, options: list[str], selected: int = 0, *,
                     input_last: bool = False) -> tuple[int, str] | None:
        """选项对话框，返回 (选中项下标, 输入项里打的文字)；Esc / Ctrl+C 返回 None。草稿处理同 ask。

        input_last：最后一项是输入项——选中它就在这一行直接打字（不用先选再弹一行提问，用例 10 真机反馈）；
        其它项的文字恒为 ""。"""
        fut = asyncio.get_running_loop().create_future()
        self._flush_pending()
        self._stash_draft()
        self._choice = {"title": title, "options": list(options), "selected": selected, "future": fut,
                        "input_last": input_last}
        self._invalidate()
        try:
            return await fut
        finally:
            self._choice = None
            buf = self.session.default_buffer
            if buf.text:                # 输入项里打的字已随结果交出去，不能留在输入框里变成下一条消息
                buf.set_document(Document(""), bypass_readonly=True)
            self._restore_draft()
            self._invalidate()

    def _editing_choice(self) -> bool:
        """光标停在输入项上（此时按键是打字，不是选项操作）。"""
        c = self._choice
        return bool(c and c["input_last"] and c["selected"] == len(c["options"]) - 1)

    def _resolve_choice(self, result: tuple[int, str] | None) -> None:
        if self._choice and not self._choice["future"].done():
            self._choice["future"].set_result(result)

    @property
    def asking(self) -> bool:
        return self._ask is not None or self._choice is not None

    def _interrupt(self) -> None:
        if self._choice:
            self._resolve_choice(None)
            return
        if self._ask and not self._ask[1].done():
            self._ask[1].set_result(None)
            return
        if self.on_interrupt:
            self.on_interrupt()

    # ── 主循环 ──
    async def run(self) -> None:
        """一直接收输入直到 stop() 或 EOF（Ctrl+D）。EOF 时往队列放 None，告诉调用方用户要退出。"""
        self._loop = asyncio.get_running_loop()
        while not self._stopped:
            try:
                # 提交走 _accept、不结束应用，所以这里正常只在 stop() / Ctrl+D 时返回
                await self.session.prompt_async()
            except EOFError:
                await self.submissions.put(None)
                return
            except KeyboardInterrupt:
                continue                  # Ctrl+C 已由按键绑定处理（中断回合），这里不退出

    def _accept(self, buff) -> bool:
        """回车提交：收下文字、清空缓冲区，应用不退出。

        原先走 prompt_toolkit 默认的「回车 = 结束这次 prompt」：应用退出时把输入框整块擦掉（erase_when_done），
        回显打完再起一个新的——中间那段输入框不在屏幕上（e2e：每次提交消失约 9 个数据块，真机「输入框没有固定」）。
        现在输入框从头到尾是同一个应用，回显走 write 的「擦除-写入-重画」同帧路径。历史由 validate_and_handle 追加。"""
        text = buff.text
        if self._ask and not self._ask[1].done():
            self._ask[1].set_result(text)      # 提问模式的回答；草稿由 ask() 收尾时放回
        else:
            self.submissions.put_nowait(text)
        return False                           # False = 清空缓冲区

    def stop(self) -> None:
        self._stopped = True
        app = self.session.app
        if app.is_running:
            app.exit(exception=EOFError())
