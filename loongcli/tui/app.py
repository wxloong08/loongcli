from __future__ import annotations
import asyncio
import re
import time
import signal
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from rich.console import Console
from rich.cells import cell_len
from loongcli.tui.mdstream import LeftMarkdown as Markdown
from loongcli.tui.cards import logo, section
from loongcli.tui.clipboard_image import grab_clipboard_images
from loongcli.tui.dropped_paths import split_dropped_paths
from loongcli.tui.tool_display import arg_summary, result_lines
from rich.table import Table
from rich.text import Text
from rich.spinner import Spinner
from rich.padding import Padding
from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.patch_stdout import patch_stdout

from loongcli.core.agent import AgentLoop, PLAN_MODE_TOOLS
from loongcli.core.events import TextDelta, ThinkingDelta, ToolCallProgress, ToolCallStart, ToolCallResult, AgentDone, CompactStart, CompactNotice, TaskNotification, ConfirmRequest, BatchProgress, ShellOutput, PlanApproval, QueuedInputInjected
from loongcli.core.intent import StopIntent, detect_stop_intent
from loongcli.core.messages import message_text, extract_image_paths, is_image_file
from loongcli.memory.markdown_store import MarkdownMemoryStore
from loongcli.memory.conversation import ConversationStore
from loongcli.tui.commands import CommandContext, CommandRegistry, create_default_registry
from loongcli.tui.input_bar import InputBar
from loongcli.tui.mdstream import StreamView
from loongcli.core.sanitize import repair_surrogates


_BUILTIN_COMMANDS = [
    ("exit", "退出"),
]


_PLACEHOLDER = r"\[(?:Image #\d+|Pasted text #\d+ \+\d+ lines)\]"
_PLACEHOLDER_END = re.compile(_PLACEHOLDER + r"$")
_PLACEHOLDER_START = re.compile(_PLACEHOLDER)


class _ThinkingStatus:
    """「思考中… 12s」转圈状态：秒数在每次渲染时现算，挂上去之后不用再换内容，靠定时刷新自己走。"""

    def __init__(self, padding):
        self._start = time.monotonic()
        self._spinner = Spinner("dots", text="")
        self._padding = padding

    def _label(self) -> str:
        return "思考中…"

    def __rich_console__(self, console, options):
        secs = int(time.monotonic() - self._start)
        elapsed = f"{secs}s" if secs < 60 else f"{secs // 60}m{secs % 60:02d}s"
        self._spinner.update(text=Text(f"{self._label()} {elapsed}", style="dim"))
        yield Padding(self._spinner, self._padding)


class _ToolArgsStatus(_ThinkingStatus):
    """「正在生成 write_file 调用… 3.2K 字 12s」：模型流式写工具参数期间的状态（写大文件时要几十秒）。
    字数由 ToolCallProgress 直接改 chars，渲染时现取，不用每来一片参数就换一次动态区内容。"""

    def __init__(self, tool_name: str, padding):
        super().__init__(padding)
        self.tool_name = tool_name
        self.chars = 0

    def _label(self) -> str:
        n = self.chars
        size = f"{n} 字" if n < 1000 else f"{n / 1000:.1f}K 字"
        return f"正在生成 {self.tool_name} 调用… {size}"


_PLAN_STATUS_ZH = {"completed": "完成", "in_progress": "进行中", "skipped": "跳过", "pending": "待办",
                   "abandoned": "放弃", "active": "进行中", "failed": "失败"}


def _plan_progress_line(args: dict, result: str) -> Text:
    """plan 工具的一行展示「◆ 计划：步骤 1 已更新 → 完成（进度 1/4）」——结果首行本就是摘要，英文状态词翻成中文。"""
    if args.get("operation") == "create" and args.get("title"):
        summary = f"已创建「{args['title']}」（{len(args.get('steps') or [])} 步）"
    else:
        summary = (result.splitlines() or [""])[0]
        for en, zh in _PLAN_STATUS_ZH.items():
            summary = summary.replace(f"→ {en}", f"→ {zh}").replace(f"标记为 {en}", f"标记为{zh}")
    line = Text("◆ ", style="cyan")
    line.append("计划：", style="cyan")
    line.append(summary, style="dim")
    if summary.endswith(("：", ":")):
        # 被拒的情况（「不能标为完成：」「未标完成：」）首行只是引子，原因在后面几行，得一并显示
        for detail in [l for l in result.splitlines()[1:] if l.strip()][:4]:
            line.append(f"\n    {detail.strip()}", style="yellow")
    return line



def _user_echo(text: str) -> Table:
    """滚动区里用户输入的回显「❯ 原文」：两列表格，折行的续行对齐到 ❯ 后面（单个 Text 折行会回到第 0 列，真机）。
    原文按纯文字显示——用户输入里的 [ ] 不能被当成 Rich 标记。"""
    return _hang(Text("❯ ", style="bold green"), Text(text))


def _hang(prefix: Text, body: Text) -> Table:
    """悬挂缩进：前缀一列、正文一列，正文折行后续行对齐到前缀后面。"""
    grid = Table.grid()
    grid.add_column(width=cell_len(prefix.plain), no_wrap=True)
    grid.add_column(overflow="fold")      # 默认 ellipsis 会把恰好填满列宽的那行也截成「…」（Rich 的怪癖，实测）
    grid.add_row(prefix, body)
    return grid


def _fmt_tokens(n: int) -> str:
    """状态行里的 token 数：987 / 45K / 1.2M。"""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}".rstrip("0").rstrip(".") + "M"
    if n >= 1000:
        return f"{round(n / 1000)}K"
    return str(n)


def _sanitize_cells(s: str) -> str:
    """剥离会让 cell_len 与终端实际渲染宽度不一致的字符。

    _fit_cells 的"单行"承诺依赖 Rich 宽度表 == 终端渲染宽度，以下字符会打破它：
    - VS16(U+FE0F)：把 ⚠ 这类窄字符翻成双宽 emoji——Rich 记 1 格、终端画 2 格，
      行超宽换行 → 活动区高度抖动 → 残影（2026-07-17 截图思考行堆叠的真凶）；
    - \\t：终端展开到制表位（最多 8 格），Rich 记 1 格；
    - ZWJ 等格式字符：多个 emoji 合成单字形，宽度无法静态预测。
    只往安全方向偏：宁可高估宽度提前截断，绝不低估导致换行。"""
    out: list[str] = []
    for ch in s:
        o = ord(ch)
        if ch == "\t":
            out.append(" ")
        elif 0xFE00 <= o <= 0xFE0F or 0xE0100 <= o <= 0xE01EF:  # 变体选择符
            continue
        elif unicodedata.category(ch) in ("Cc", "Cf"):  # 控制/格式字符（含 ZWJ）
            continue
        else:
            out.append(ch)
    return "".join(out)


def _fit_cells(s: str, budget: int) -> str:
    """按显示宽度（cell）截断字符串到 budget 列内，超出补省略号。

    必须按 cell 而非字符数截断：中文/emoji 是双宽字符，用 len() 计数会让状态行
    实际宽度翻倍、超出终端换行——多行 transient Live 高度抖动会在 scrollback
    留下残影（思考行反复堆叠的真凶）。"""
    if budget <= 0:
        return ""
    s = _sanitize_cells(s)
    if cell_len(s) <= budget:
        return s
    budget -= 1  # 留 1 列给省略号
    out: list[str] = []
    used = 0
    for ch in s:
        w = cell_len(ch)
        if used + w > budget:
            break
        out.append(ch)
        used += w
    return "".join(out) + "…"





def _pack_status_lines(items: list[str], width: int) -> list[str]:
    """banner 状态条目打包成行：**只在条目边界断行**。

    整行 join 后交给 Rich 自动换行会在条目内部的空格处断开——真机截图实锤：
    「⚠ permissions: skip」被拆成行尾悬空的 ⚠ 和下一行的说明，警告变谜语。
    按 cell 宽度打包（未来条目可能含中文），单条目超宽也不拆、独占一行。"""
    lines: list[str] = []
    current = ""
    for item in items:
        candidate = f"{current} | {item}" if current else item
        if current and cell_len(candidate) > width:
            lines.append(current)
            current = item
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


class SlashCompleter(Completer):
    def __init__(self, command_registry, skill_registry=None):
        self._command_registry = command_registry
        self._skill_registry = skill_registry

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        prefix = text[1:].lower()
        seen = set()

        for name, desc in _BUILTIN_COMMANDS:
            if name.startswith(prefix):
                seen.add(name)
                yield Completion(
                    f"/{name}",
                    start_position=-len(text),
                    display_meta=desc,
                )

        for cmd in self._command_registry.all_commands():
            if cmd.name.startswith(prefix):
                seen.add(cmd.name)
                yield Completion(
                    f"/{cmd.name}",
                    start_position=-len(text),
                    display_meta=cmd.description,
                )

        if self._skill_registry:
            for skill in self._skill_registry.list_skills():
                if skill.name not in seen and skill.name.startswith(prefix):
                    desc = skill.description[:40] + "..." if len(skill.description) > 40 else skill.description
                    yield Completion(
                        f"/{skill.name}",
                        start_position=-len(text),
                        display_meta=f"[skill] {desc}",
                    )


@dataclass
class GoalState:
    """一次 /goal 的运行状态。暂停时保留，用户回复后据此续跑；目标结束（计划完成/放弃、/goal stop）时清掉。"""
    description: str
    plan_id: str
    status: str = "running"   # running / paused
    pause_reason: str = ""
    iterations: int = 0       # 累计执行迭代数（跨续跑累加；每次续跑另给一轮迭代预算）


class TUI:
    MAX_TOOL_RESULT_DISPLAY = 200
    PADDING = (0, 2)

    MAX_GOAL_ITERATIONS = 20
    GOAL_STALL_LIMIT = 3
    # /goal 规划与改计划阶段可用的工具：复用规划模式的只读清单（去掉只对 Shift+Tab 规划模式有意义的 exit_plan_mode）。
    # 此前只给 plan 一个工具，模型读不到目标里点名的文件，计划只能复述目标——dogfood exec-1/2 把后面阶段的内容排进了 P1。
    GOAL_PLANNING_TOOLS = frozenset(PLAN_MODE_TOOLS - {"exit_plan_mode"})
    # 规划期「先问再规划」最多几轮问答，之后必须出计划（防止无限追问）
    GOAL_QUESTION_ROUNDS = 3

    # 大块粘贴折叠阈值（仿 Claude Code）：超过则在输入区显示占位符，提交时还原完整原文
    PASTE_COLLAPSE_LINES = 4
    PASTE_COLLAPSE_CHARS = 300

    def __init__(
        self,
        memory: MarkdownMemoryStore | None = None,
        command_registry: CommandRegistry | None = None,
        config=None,
        skill_registry=None,
        plan_store=None,
    ):
        self.console = Console()
        self.memory = memory
        self.config = config
        self.command_registry = command_registry or create_default_registry()
        self.skill_registry = skill_registry
        self.plan_store = plan_store
        self._bar: InputBar | None = None   # 常驻输入框，start() 里建
        self._turn_interrupt = None          # 回合进行中：中断它的回调；空闲为 None
        self._busy: bool = False             # 正在跑回合或 goal 执行循环：此时的提交是插话
        # 常驻输入框是否在跑（start() 期间为 True）。不用 app.is_running：每次提交后输入框会短暂退出再重进
        # prompt_async，回合恰好在这个间隙开始，用它判断会误判成「没有输入框」而另建 Rich Live / 第二个输入框
        self._bar_active: bool = False
        self._goal_stop_requested: bool = False  # goal 运行中收到 /goal stop：循环在回合返回后结束目标
        self._goal_planning: bool = False        # /goal 的规划与审批阶段（此时 _goal_mode 还是 False）
        self._goal_mode: bool = False  # 执行循环正在跑（暂停时为 False）
        self._goal: GoalState | None = None  # 当前目标：执行中或暂停；None = 没有目标
        self._pastes: dict[str, str] = {}   # 占位符 -> 完整粘贴原文
        self._paste_counter: int = 0
        self._paste_images: dict[str, str] = {}   # [Image #N] 占位符 -> 真实图片路径
        self._image_counter: int = 0
        # 工具输出降噪：默认折叠模式（● 工具行 + ⎿ 统计/diff），/verbose 切回详细
        self.verbose: bool = False
        self._active_tool_args: dict = {}   # ToolCallStart 记参数，Result 时渲染用
        self._at_gap: bool = False          # 上一次输出是否已是空行（块间距统一为恰好 1 行）

    def _maybe_collapse_paste(self, data: str) -> str:
        """大块粘贴 → 折叠成占位符（存原文）；小块 → 原样返回。供 BracketedPaste 绑定调用。"""
        data = data.replace("\r\n", "\n").replace("\r", "\n")
        # 拖入文件（终端当一次粘贴送来，可能是一串路径）：图片折叠成 [Image #N] 占位符（显示友好、存真实路径），
        # 其它文件保留成路径交给模型自己读。只认一张图时，多张一起拖会变成一串裸路径（真机）——格式见 dropped_paths.py
        paths = split_dropped_paths(data)
        if paths and any(is_image_file(p) for p in paths):
            parts = [self._image_placeholder(p) if is_image_file(p) else (f'"{p}"' if " " in p else p)
                     for p in paths]
            out = parts[0]
            for prev, cur in zip(parts, parts[1:]):
                out += ("" if prev.startswith("[Image #") and cur.startswith("[Image #") else " ") + cur
            return out
        lines = data.splitlines()
        if len(lines) >= self.PASTE_COLLAPSE_LINES or len(data) >= self.PASTE_COLLAPSE_CHARS:
            self._paste_counter += 1
            placeholder = f"[Pasted text #{self._paste_counter} +{len(lines)} lines]"
            self._pastes[placeholder] = data
            return placeholder
        return data

    def _paste_clipboard(self, buf) -> None:
        """Alt+V：剪贴板里的图插成 [Image #N]（和拖入同一套占位符）；没图 / 缺 Pillow 时在输入框上方提示一行。"""
        paths, hint = grab_clipboard_images()
        if paths:
            buf.insert_text("".join(self._image_placeholder(p) for p in paths))
            return
        self.console.print(Padding(Text(hint, style="dim"), self.PADDING))

    def _image_placeholder(self, path: str) -> str:
        """图片路径 → 占位符；同一张图再拖一次复用原占位符，不重复附图。"""
        for ph, known in self._paste_images.items():
            if known == path:
                return ph
        self._image_counter += 1
        placeholder = f"[Image #{self._image_counter}]"
        self._paste_images[placeholder] = path
        return placeholder

    def _expand_pastes(self, text: str) -> str:
        """提交时把占位符还原成完整原文，发给 LLM 的是全文。"""
        if self._pastes:
            for placeholder, real in self._pastes.items():
                text = text.replace(placeholder, real)
            self._pastes.clear()
        return text

    def _resolve_images(self, text: str, vision: bool) -> tuple[str, list[str]]:
        """把输入里的图片解析成图片列表：拖入的 [Image #N] 占位符 + 打字的裸路径。

        vision=True：占位符/路径抽成图片、从文本移除。
        vision=False：占位符还原成路径文本、不抽图（当普通文本，别乱附）。
        """
        images: list[str] = []
        if self._paste_images:
            for ph, path in self._paste_images.items():
                if ph not in text:
                    continue
                if vision:
                    images.append(path)
                    text = text.replace(ph, "")
                else:
                    text = text.replace(ph, path)
            self._paste_images.clear()
        if vision:
            text, more = extract_image_paths(text)
            images += more
        return text.strip(), images

    def _count_images_in_input(self, text: str) -> int:
        """输入里有几张图（拖入的占位符 + 真实存在的图片路径）。只数不改——
        vision 关闭时用来决定要不要提示用户「图没发出去」。必须在 _resolve_images
        之前调用（后者会清空占位符表）。"""
        n = sum(1 for ph in self._paste_images if ph in text)
        _, typed = extract_image_paths(text)
        return n + len(typed)

    def _get_bar(self) -> InputBar:
        """整个会话唯一的常驻输入框（空闲与模型运行期间共用）。Ctrl+J/Shift+Enter 换行、
        显示行上下导航、Esc/Ctrl+C 中断由 InputBar 自带；这里补粘贴折叠与 Shift+Tab。"""
        if self._bar is None:
            kb = KeyBindings()

            @kb.add(Keys.BracketedPaste)
            def _(event):
                event.current_buffer.insert_text(self._maybe_collapse_paste(event.data))

            # 占位符（[Image #N]、[Pasted text #N +M lines]）当一个整体删：原先退格一个字一个字删，删到一半剩个「[Image」
            # 残片，既不是图也不是原文（真机）。光标紧挨着一个完整占位符时，退格 / Delete 一次删掉整个
            from prompt_toolkit.filters import Condition  # noqa: E402  只在建输入框时用

            before = Condition(lambda: bool(_PLACEHOLDER_END.search(
                get_app().current_buffer.document.text_before_cursor)))
            after = Condition(lambda: bool(_PLACEHOLDER_START.match(
                get_app().current_buffer.document.text_after_cursor)))

            @kb.add("backspace", filter=before)
            def _(event):
                buf = event.current_buffer
                buf.delete_before_cursor(len(_PLACEHOLDER_END.search(buf.document.text_before_cursor).group(0)))

            @kb.add("delete", filter=after)
            def _(event):
                buf = event.current_buffer
                buf.delete(len(_PLACEHOLDER_START.match(buf.document.text_after_cursor).group(0)))

            @kb.add("escape", "v")              # Alt+V：贴剪贴板里的截图（Ctrl+V 有图时被 Windows Terminal 吃掉）
            def _(event):
                self._paste_clipboard(event.current_buffer)

            @kb.add("s-tab")
            def _(event):
                # Shift+Tab 切换规划模式（Claude Code 式模式循环的最小版）。直接调、不走 run_in_terminal：
                # 那会先把输入框整块擦掉、打完再重画（输入框消失一下）；打印走 InputBar.write 合批，同帧写出
                self._toggle_plan_mode()

            # FileHistory：输入历史按项目落盘，新会话方向键直接可翻到过去的输入。
            # 此前 InMemoryHistory 只活在进程内，开新会话历史为空（真机反馈）。
            from loongcli.memory.conversation import input_history_path
            hist_path = input_history_path()
            hist_path.parent.mkdir(parents=True, exist_ok=True)
            self._bar = InputBar(
                history=FileHistory(str(hist_path)),
                completer=SlashCompleter(self.command_registry, self.skill_registry),
                key_bindings=kb,
                on_interrupt=self._on_interrupt_key,
                status_provider=self._status_line,
                # 规划模式除了 Shift+Tab，计划批准后也会被 agent 自动退出——所以每次重画现查，不在切换处推送
                mode_provider=lambda: "规划" if getattr(getattr(self, "_agent", None), "plan_mode", False) else "",
            )
        return self._bar

    def _status_line(self) -> tuple[str, str]:
        """输入框下方状态行：左边按键提示（空闲 / 运行中不同），右边模型、思考档位、上下文占用、规划模式。每次重画都取。"""
        busy = self._busy or self._goal_mode
        if self._bar is not None and self._bar.asking:
            left = ""                    # 对话框 / 提问自带操作提示，这里不再给（此时打字不会成为插话）
        elif busy:
            left = "打字即插话 · Esc 中断 · Shift+Enter 换行"
        elif getattr(getattr(self, "_agent", None), "plan_mode", False):
            # 切换时不再往滚动区打说明，含义放这里：一直保持到计划批准或再按 Shift+Tab
            left = "规划模式：先调研再出计划，批准后自动退出 · Shift+Tab 退出"
        else:
            left = "Shift+Enter 换行 · / 命令 · Alt+V 贴图 · Shift+Tab 规划模式"
        agent = getattr(self, "_agent", None)
        llm = getattr(agent, "llm", None)
        if llm is None:
            return left, ""
        bits = [str(getattr(llm, "model", ""))]
        bits.append(f"thinking {llm.reasoning_effort}" if getattr(llm, "thinking", False) else "thinking off")
        tokens = getattr(agent, "_last_prompt_tokens", 0)
        compactor = getattr(agent, "compactor", None)
        if tokens:
            # 分母用自动压缩的触发线而不是模型窗口：到这里就会压缩，对用户才有意义（且跟着 compact_threshold 配置走）
            line = f"上下文 {_fmt_tokens(tokens)}"
            if compactor is not None and getattr(compactor, "threshold", 0):
                line += f"（压缩线 {_fmt_tokens(compactor.threshold)}）"
            bits.append(line)
        if getattr(agent, "plan_mode", False):
            bits.append("规划模式")
        return left, " · ".join(b for b in bits if b)

    def _on_interrupt_key(self) -> None:
        """输入框里按了 Esc / Ctrl+C：回合进行中就中断它；空闲时什么也不做（退出用 /exit 或 Ctrl+D）。"""
        if self._turn_interrupt is not None:
            self._turn_interrupt()

    async def _ask(self, question: str) -> str | None:
        """文字提问、等用户回答；返回 None = 用户按 Esc 取消。

        常驻输入框在跑就切成提问模式；没有（非交互、测试）就临时开一个输入框。
        问答都留一行在滚动区——提问模式答完就收起，不留痕事后看不出当时答了什么（14e 真机）。"""
        if self._bar_active:
            answer = await self._bar.ask(question)
        else:
            session: PromptSession = PromptSession()
            with patch_stdout():
                answer = await session.prompt_async(f"  {question} ")
        shown = "（Esc 取消）" if answer is None else (answer.strip() or "（回车）")
        trace = Text(f"{question} → ", style="dim")
        trace.append(shown)
        self.console.print(Padding(trace, self.PADDING))
        return answer

    async def _choose(self, title: str, options: list[str], *,
                      input_last: bool = False) -> tuple[int, str] | None:
        """选项对话框（Claude Code 式，占据输入框位置）：返回 (选中项下标, 输入项文字)，None = Esc 取消。
        input_last：最后一项是输入项，选中后直接在那一行打字。滚动区留一行所选结果。

        常驻输入框不在（非交互、测试）就退回「输入编号」的一次性输入框，选到输入项再问一句文字。"""
        if self._bar_active:
            picked = await self._bar.choose(title, options, input_last=input_last)
        else:
            menu = "  ".join(f"{i + 1}. {o}" for i, o in enumerate(options))
            session: PromptSession = PromptSession()
            with patch_stdout():
                raw = (await session.prompt_async(f"  {title}  {menu}\n  输入编号: ")).strip()
                index = int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(options) else None
                text = ""
                if input_last and index == len(options) - 1:
                    text = await session.prompt_async(f"  {options[index]}：")
            picked = None if index is None else (index, text)
        trace = Text(f"{title} → ", style="dim")
        if picked is None:
            trace.append("（Esc 取消）")
        else:
            trace.append(options[picked[0]] + (f"：{picked[1].strip()}" if picked[1].strip() else ""))
        self.console.print(Padding(trace, self.PADDING))
        return picked

    def _toggle_plan_mode(self) -> None:
        """Shift+Tab 切换规划模式。工具过滤在 run_stream 开头按 flag 生效，无需另拆。"""
        agent = getattr(self, "_agent", None)
        if agent is None:
            return
        # 不往滚动区打提示：提示符变「规划 ❯」、状态行写明规划模式的含义，已经够了（真机：切三回打出 6 行；Claude Code 也不打）
        if agent.plan_mode:
            agent._plan_mode = False
            agent._rebuild_system_prompt()  # 退出必须重建，否则规划注入残留在系统提示里
        else:
            agent.enter_plan_mode()
        if self._bar is not None:
            self._bar._invalidate()

    _HISTORY_BACKFILL_MAX = 200

    def _backfill_input_history(self, store) -> None:
        """首次启用持久输入历史时，把项目历史会话的用户消息一次性回填进历史文件，
        让新会话立刻能用方向键翻旧输入。文件已存在则不动——正常输入由 FileHistory
        实时追加，重复回填会造成条目翻倍。"""
        from loongcli.memory.conversation import input_history_path
        path = input_history_path()
        if store is None or path.exists():
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            hist = FileHistory(str(path))
            entries: list[str] = []
            # list_sessions 新→旧，回填要旧→新（方向键↑先出最近的）
            for meta in reversed(store.list_sessions(limit=20)):
                sid = meta.get("session_id")
                if not sid:
                    continue
                for msg in store.full_history(sid):
                    if msg.get("role") != "user":
                        continue
                    content = message_text(msg)  # 多模态 list 取纯文本
                    if content and not self._is_internal_message(content):
                        entries.append(content)
            for entry in entries[-self._HISTORY_BACKFILL_MAX:]:
                hist.append_string(entry)
        except OSError:
            pass  # 回填失败不阻塞启动，历史从本次输入开始积累

    @staticmethod
    def _is_internal_message(content: str) -> bool:
        """Detect system-injected messages that shouldn't appear on resume."""
        if not content:
            return False
        return (
            content.startswith("[compact-boundary]")
            or content.startswith("[压缩后上下文恢复]")
            or content.startswith("[对话历史摘要]")
            or content.startswith("[snip]")
            or content.startswith("[工具")
            or content.startswith("## 验证")
            or content == "好的，我已了解之前的对话内容。请继续。"
            or content == "好的，我已了解恢复的上下文信息，继续工作。"
            or content == "好的，已了解。"
        )

    def render_history(self, messages: list[dict], max_recent: int = 5):
        conversations: list[tuple[str, str]] = []
        for msg in messages:
            role = msg.get("role", "")
            content = message_text(msg)  # 多模态 list 取纯文本，图片计 [图片]
            if role in ("system", "tool"):
                continue
            if self._is_internal_message(content):
                continue
            if role == "user":
                conversations.append(("user", content or ""))
            elif role == "assistant" and content:
                conversations.append(("assistant", content))

        if not conversations:
            return

        if len(conversations) > max_recent * 2:
            skipped = len(conversations) - max_recent * 2
            self.console.print(f"  [dim]... 还有 {skipped} 条历史消息 ...[/dim]\n")
            conversations = conversations[-(max_recent * 2):]

        # 回放与实时渲染同规矩：消息之间恰好一个空行，assistant 正文同款左缩进
        #（真机反馈：--continue 回放曾挤成一坨，正文段无间隔粘连）
        for role, content in conversations:
            self.console.print()
            if role == "user":
                self.console.print(_user_echo(content))
            elif role == "assistant":
                text = content if len(content) <= 500 else content[:500] + "..."
                try:
                    self.console.print(Padding(Markdown(text), self.PADDING))
                except Exception:
                    self.console.print(Padding(Text(text), self.PADDING))
        self.console.print()

    @staticmethod
    def _format_local_time(iso_str: str) -> str:
        if not iso_str:
            return "?"
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(iso_str)
            if dt.tzinfo is not None:
                dt = dt.astimezone()
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, OSError):
            return iso_str[:19].replace("T", " ") if len(iso_str) > 19 else iso_str

    async def pick_session(self, conversation: ConversationStore) -> str | None:
        from loongcli.tui.session_picker import SessionPicker
        picker = SessionPicker(self.console, conversation)
        return await picker.pick()

    async def start(self, agent: AgentLoop, resumed: bool = False, status_items: list[str] | None = None):
        # 欢迎页：渐变方块字 Logo + 状态 + 按键提示。不加边框：打出去就进了终端历史、程序没法重画，
        # 窗口拉窄时终端自己把超宽行折断，满宽边框会碎成一片（真机）。Logo 约 37 列，纯文字折行后仍可读
        self.console.print()
        self.console.print(Padding(logo(Text("cli v0.1.0", style="dim")), self.PADDING))
        lines = [""]
        if resumed:
            lines.append(f"[cyan]已恢复会话[/cyan] [bold]{agent.conversation_store.session_id}[/bold]")
        # 内宽 = 终端宽 − 左右缩进，再留 2 列余量
        inner_width = max(20, self.console.width - self.PADDING[1] * 2 - 2)
        for status_line in _pack_status_lines(status_items or [], inner_width):
            lines.append(f"[dim]{status_line}[/dim]")
        lines.append("[bold]/help[/bold] [dim]查看命令 ·[/dim] [bold]/exit[/bold] [dim]退出 ·[/dim] "
                     "[bold]Shift+Tab[/bold] [dim]规划模式[/dim]")
        self.console.print(Padding(Text.from_markup("\n".join(lines)), self.PADDING))

        if resumed:
            self.render_history(agent.messages)
        # 必须在 _get_bar 建 FileHistory 之前回填，否则首个 prompt 读到的还是空文件
        self._backfill_input_history(agent.conversation_store)

        ctx = CommandContext(
            console=self.console,
            agent=agent,
            memory=self.memory,
            registry=self.command_registry,
            tui=self,
            config=self.config,
        )
        self._agent = agent  # Shift+Tab 切换规划模式的键绑定要用
        bar = self._get_bar()
        # 空闲时的提交（新消息 / 命令）走这个队列；运行中的提交由 _route_submissions 转成插话
        self._idle_inputs: asyncio.Queue[str | None] = asyncio.Queue()

        # raw=True：Rich 的颜色转义码原样透传（原型 v1/v2 在 Windows Terminal 验证）
        # patch_stdout 只兜底零散的 print / 日志；Rich 的输出直接交给输入框整块写出（见 InputBar.write）
        with patch_stdout(raw=True):
            bar_task = asyncio.create_task(bar.run())
            router = asyncio.create_task(self._route_submissions(agent))
            self._bar_active = True
            console_file = self.console.file
            self.console.file = bar
            try:
                while True:
                    raw = await self._idle_inputs.get()
                    if raw is None:              # Ctrl+D
                        self.console.print(self._farewell(agent))
                        break
                    if not await self._process_input(agent, ctx, raw):
                        break
                    # goal 两次迭代之间（意图判断时）提交、没赶上注入的插话：当作下一条消息处理
                    for leftover in agent.take_queued_inputs():
                        self._idle_inputs.put_nowait(leftover)
            finally:
                bar._emit()                      # 收尾前把还在合批窗口里的输出写掉（告别语等）
                self.console.file = console_file
                self._bar_active = False
                router.cancel()
                bar.stop()
                try:
                    await asyncio.wait_for(bar_task, 2)
                except (Exception, asyncio.CancelledError):
                    pass

    async def _route_submissions(self, agent: AgentLoop) -> None:
        """输入框的每次提交在这里分流：运行中 → 排队成插话（下一迭代边界注入）；空闲 → 交给主循环。"""
        while True:
            text = await self._bar.submissions.get()
            if text is None:
                await self._idle_inputs.put(None)
                return
            if self._busy or self._goal_mode:
                self._queue_interjection(agent, text)
            else:
                await self._idle_inputs.put(text)

    def _queue_interjection(self, agent: AgentLoop, raw: str) -> None:
        text = repair_surrogates(self._expand_pastes(raw)).strip()
        if not text:
            return
        if text.lower().split() == ["/goal", "stop"] and (self._goal_mode or self._goal_planning):
            # 运行中唯一放行的命令：打断当前回合，goal 的规划 / 执行循环在回合返回后看到标志就结束目标
            self._goal_stop_requested = True
            self.console.print(Padding(Text("正在结束目标…", style="yellow"), self.PADDING))
            if self._turn_interrupt is not None:
                self._turn_interrupt()
            return
        if text.startswith("/"):
            # 运行中跑斜杠命令会和正在进行的回合抢状态（/goal、/compact 等），先不支持（/goal stop 除外，见上）
            self.console.print(Padding(Text(
                f"运行中不支持斜杠命令（{text.split()[0]}）——按 Esc 中断当前回合后再用；goal 运行中可用 /goal stop", style="yellow",
            ), self.PADDING))
            return
        agent.queue_user_input(text)
        line = Text("❯ ", style="green")
        line.append(repair_surrogates(raw).strip())
        line.append("  （已排队，模型下一步之前注入）", style="dim")
        self.console.print(Padding(line, (0, 0)))

    async def _process_input(self, agent: AgentLoop, ctx: CommandContext, raw: str) -> bool:
        """处理一条空闲时的输入。返回 False 表示要退出。"""
        # 还原折叠的大块粘贴：输入区显示的是占位符，发给 agent 的是完整原文。
        # 源头修复：Windows 控制台粘贴 emoji 会以 UTF-16 代理对传入，原样下传会污染
        # 历史/记忆并在存盘与 LLM 请求的 UTF-8 编码处崩溃。把代理对解码回真 emoji
        # （孤立代理丢弃），下游全拿到干净且忠实的输入。
        user_input = repair_surrogates(self._expand_pastes(raw)).strip()
        if not user_input:
            return True                         # 空提交：输入框已自清（erase_when_done），无需擦屏
        # 输入框提交后清掉自己那一行（避免回显两次），滚动区的 ❯ 原文由这里打印；显示的是折叠后的样子
        # 回合之间只空一行、不画满宽分隔线：满宽线进了终端历史，窗口拉窄时会被折成一长一短两截（真机）
        self.console.print()
        self.console.print(_user_echo(repair_surrogates(raw).strip()))
        if user_input == "/exit":
            self.console.print(self._farewell(agent))
            return False

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=3)
            cmd_name = parts[0][1:].lower()
            cmd = self.command_registry.get(cmd_name)
            if cmd:
                await cmd.run(parts[1:], ctx)
                return True
            if self.skill_registry:
                skill_content = self.skill_registry.load_content(cmd_name)
                if skill_content is not None:
                    meta = self.skill_registry.get(cmd_name)
                    skill_args = " ".join(parts[1:]) if len(parts) > 1 else ""
                    prompt = f"请按照以下技能指令执行：\n\n## 技能: {meta.name}\n{meta.description}\n\n{skill_content}"
                    if skill_args:
                        prompt += f"\n\n用户参数: {skill_args}"
                    await self._handle_agent_response(agent, prompt)
                    return True
            self.console.print(f"[yellow]未知命令: /{cmd_name}。输入 /help 查看可用命令。[/yellow]")
            return True

        # 目标暂停中：普通消息就是对目标的回复，续跑（斜杠命令在上面已分走，不触发续跑）
        if self._goal is not None and self._goal.status == "paused":
            await self.resume_goal(agent, user_input)
            return True

        await self._handle_agent_response(agent, user_input)
        return True

    def _farewell(self, agent: AgentLoop) -> str:
        """退出致辞。空会话（一轮都没跑过，_persist 从未触发、会话文件不存在）不能
        说"已保存"——那是撒谎，且会误导用户 --continue（恢复的会是上一个会话）。
        以会话文件是否落盘为准，不猜内存状态。"""
        store = agent.conversation_store
        if store and store.session_path.exists():
            return (
                f"[dim]再见！会话 [cyan]{store.session_id}[/cyan] 已保存，"
                f"[cyan]loongcli --continue[/cyan] 可继续[/dim]"
            )
        return "[dim]再见！（空会话，未保存）[/dim]"

    async def _handle_agent_response(
        self,
        agent: AgentLoop,
        user_input: str,
        allowed_tools: set[str] | None = None,
    ) -> bool:
        """回合外壳：把回合跑成可取消任务，Esc / Ctrl+C 中断回合而非崩出应用。返回是否被用户中断。

        中断键由常驻输入框的按键绑定接住（_on_interrupt_key → _turn_interrupt），回合期间输入框照常可用，
        提交的内容由 _route_submissions 排队成插话。取消传播复用 stop_task 同一条链：CancelledError 打进
        run_stream → _exec_tool_stream finally 收编 exec_task → shell 子进程随 Job 全灭。
        """
        # while 循环：轮正常结束后若插话队列有剩余（最后一次 drain 之后才提交的），
        # 逐条自动作为新轮发出——用户按了 Enter 就是提交意图，不该被吃掉。
        pending: str | None = user_input
        auto_send = False
        while pending is not None:
            if auto_send:
                # 自动发出的消息补一行 ❯ 回显，scrollback 上与手敲提交同构
                line = Text("❯ ", style="bold green")
                line.append(pending)
                self.console.print(line)

            turn = asyncio.create_task(
                self._run_turn_impl(agent, pending, allowed_tools)
            )
            loop = asyncio.get_running_loop()
            interrupted = False

            def _request_interrupt():
                nonlocal interrupted
                interrupted = True
                loop.call_soon_threadsafe(turn.cancel)

            self._turn_interrupt = _request_interrupt
            self._busy = True
            # SIGINT 兜底：常驻输入框在跑时 Ctrl+C 是按键（raw 模式不产生 SIGINT），这里只防输入框不在的场合
            prev_handler = signal.signal(signal.SIGINT, lambda *_: _request_interrupt())
            try:
                await turn
            except asyncio.CancelledError:
                if not interrupted:
                    raise  # 非用户中断的外部取消（如应用收尾），照常传播
                agent.abort_turn()
                self.console.print(Padding(Text.from_markup(
                    "[yellow]✗ 回合已中断——消息现场已修复，可继续输入[/yellow]"
                ), self.PADDING))
                self._at_gap = False
                # 中断路径：还没注入的插话放回输入框，不替用户发——用户刚按 Esc，此刻替他发消息违背中断意图
                self._stash_prefill(agent)
                return True
            finally:
                self._turn_interrupt = None
                self._busy = False
                signal.signal(signal.SIGINT, prev_handler)

            leftovers = agent.take_queued_inputs()
            if leftovers:
                pending = leftovers[0]
                for extra in leftovers[1:]:
                    agent.queue_user_input(extra)  # 重新排队，下一轮迭代边界注入
                auto_send = True
            else:
                pending = None
        return False

    def _stash_prefill(self, agent: AgentLoop) -> None:
        """中断时还没注入的插话：放回输入框（接在用户正在打的内容后面），由用户决定改不改、发不发。"""
        parts = agent.take_queued_inputs()
        if not parts or self._bar is None:
            return
        buf = self._bar.session.default_buffer
        buf.insert_text(("\n" if buf.text else "") + "\n".join(parts))

    async def _run_turn_impl(
        self,
        agent: AgentLoop,
        user_input: str,
        allowed_tools: set[str] | None = None,
    ):
        self.console.print()      # 原满宽分隔线 + 空行；分隔线去掉（同上），只留空行
        self._at_gap = True

        # 写入可见化：上一轮后台自动记忆的落库结果，在新一轮开头曝光
        # （人眼是第五道闸——毒记忆从"潜伏到发作"变成"落地即曝光"）
        notices = getattr(agent, "memory_notices", None)
        if notices:
            for n in notices:
                desc = f" — {n.get('description', '')}" if n.get("description") else ""
                self.console.print(Padding(Text(
                    f"已记忆: {n.get('name', '')}{desc}（不妥可 /forget {n.get('name', '')}）",
                    style="dim",
                ), self.PADDING))
            notices.clear()
            self._at_gap = False
        usage_before = {k: v for k, v in agent.token_usage.items()}
        cost_before = agent.cost_tracker.total_cost if agent.cost_tracker else 0
        # 常驻输入框在跑：正文尾部与转圈状态画在输入框上方的动态区，不建 Rich Live（与 patch_stdout 冲突）；
        # 输入框不在（非交互、测试）则退回 Rich Live。
        bar_live = self._bar_active
        view = StreamView(
            self.console, left_pad=self.PADDING[1],
            live_target=self._bar.set_live_renderable if bar_live else None,
        )
        view.status(_ThinkingStatus(self.PADDING))

        # 拖图/贴路径自动附图：[Image #N] 占位符（拖入）+ 裸路径（打字）一起解析。
        vision = bool(getattr(getattr(agent, "llm", None), "vision", False))
        # vision 关闭但输入里带了图：此前静默把路径当普通文本，模型拿到路径去 read_file，
        # 再被门控挡回，最后把「配置没开」转述成「模型不支持」（2026-11-09 真机）。
        # 配置问题在入口就该让用户看见，而不是绕模型一圈。
        if not vision:
            blocked = self._count_images_in_input(user_input)
            if blocked:
                self.console.print(Padding(Text(
                    f"⚠ 检测到 {blocked} 张图片，但当前角色的 vision 开关未开启，图片不会发给模型"
                    "（只当作路径文本）。若所用模型支持视觉：在 ~/.loongcli/config.json 把 "
                    "roles.main.vision 设为 true 后重启。",
                    style="yellow",
                ), self.PADDING))
        user_input, images = self._resolve_images(user_input, vision)
        images = images or None
        if images:
            self.console.print(Padding(
                Text.from_markup(f"[dim]已附加 {len(images)} 张图片[/dim]"),
                self.PADDING,
            ))

        try:
            async for event in agent.run_stream(user_input, allowed_tools=allowed_tools, images=images):
                if isinstance(event, ThinkingDelta):
                    # 思考内容不逐字上屏：只转圈 + 计时（用户 2026-11-21 提议）。此前每个思考片段都换一次动态区内容，
                    # 输入框跟着高频重画、看起来在闪；现在同一个状态对象挂着不动，只随定时刷新更新秒数
                    if not isinstance(view.current_status, _ThinkingStatus):
                        view.status(_ThinkingStatus(self.PADDING))

                elif isinstance(event, TextDelta):
                    if not view.has_text:
                        self._gap()  # 新正文块与上一个块之间恰好一个空行
                    view.append_text(event.text)
                    self._at_gap = False

                elif isinstance(event, ToolCallStart):
                    view.flush_text()
                    self._active_tool_args = event.arguments
                    if self.verbose:
                        self.console.print(Padding(
                            Text.from_markup(
                                f"[yellow]⚙ {event.tool_name}[/yellow]"
                                f" [dim]({self._brief_args(event.arguments)})[/dim]"
                            ), self.PADDING,
                        ))
                        self._at_gap = False
                        view.status(Padding(
                            Spinner("dots", text="[dim]执行中...[/dim]"), self.PADDING,
                        ))
                    else:
                        # 折叠模式：执行中只挂瞬态状态行，不落常驻痕迹
                        view.status(Padding(
                            self._tool_spinner(event.tool_name, event.arguments),
                            self.PADDING,
                        ))

                elif isinstance(event, ToolCallResult):
                    view.stop_status()
                    if self.verbose:
                        display = self._format_tool_result(event.tool_name, event.result)
                        self.console.print(Padding(
                            Text.from_markup(f"[green]✓ {event.tool_name}[/green] {display}"),
                            self.PADDING,
                        ))
                        self._at_gap = False
                    else:
                        self._print_tool_block(event.tool_name, event.result)
                    # 工具跑完到模型下一步出字之间也挂上转圈：模型不先输出思考时，这段原本一片空白（真机：以为卡死）
                    view.status(_ThinkingStatus(self.PADDING))

                elif isinstance(event, ToolCallProgress):
                    # 模型在流式写工具参数：同一个状态对象挂着不动，只更新字数（渲染时现取），不随每片参数重画动态区
                    st = view.current_status
                    if not (isinstance(st, _ToolArgsStatus) and st.tool_name == event.tool_name):
                        st = _ToolArgsStatus(event.tool_name, self.PADDING)
                        view.status(st)
                    st.chars = event.chars

                elif isinstance(event, CompactStart):
                    view.status(Padding(
                        Spinner("dots", text=f"[cyan]压缩上下文中... ({event.message_count} 条消息)[/cyan]"),
                        self.PADDING,
                    ))

                elif isinstance(event, CompactNotice):
                    view.stop_status()
                    self._gap()
                    self.console.print(Padding(
                        Text.from_markup(f"[cyan]上下文已压缩: {event.before} → {event.after} 条消息[/cyan]"),
                        self.PADDING,
                    ))
                    self._at_gap = False

                elif isinstance(event, TaskNotification):
                    view.stop_status()
                    self._gap()
                    self.console.print(Padding(
                        Text.from_markup(
                            f"[magenta]SubAgent {event.task_id} 完成[/magenta] "
                            f"{self._truncate(event.result, 200)}"
                        ), self.PADDING,
                    ))
                    self._at_gap = False

                elif isinstance(event, QueuedInputInjected):
                    # 执行中插话已注入：提交时已回显过「❯ 原文（已排队）」，这里只给一行简短确认。
                    # 用户文本是任意内容，纯 Text 组装防 [ ] 被当 markup 解析。
                    view.flush_text()
                    view.stop_status()
                    self._gap()
                    preview = event.text.replace("\n", " ")
                    line = Text("↳ 插话已注入：", style="dim")
                    line.append(preview[:60] + ("…" if len(preview) > 60 else ""), style="dim")
                    self.console.print(Padding(line, self.PADDING))
                    self._at_gap = False

                elif isinstance(event, ShellOutput):
                    style = "dim" if event.stream == "stdout" else "dim red"
                    # 命令输出是任意文本：纯 Text 组装防 [ ] 被当标记解析，
                    # 按 cell 截断防双宽字符撑到两行留残影（与思考行同一根因）。
                    budget = self.console.width - self.PADDING[1] * 2 - 2  # 2 = 左侧缩进
                    view.status(Padding(
                        Text("  " + _fit_cells(event.line, budget), style=style),
                        self.PADDING,
                    ))

                elif isinstance(event, BatchProgress):
                    status_style = "green" if event.status == "completed" else "red"
                    line = Text()
                    line.append(f"并行任务 {event.completed}/{event.total} ", style="cyan")
                    line.append(f"{event.status} ", style=status_style)
                    # 任务 prompt 多为中文/自由文本：按 cell 补满剩余预算并纯 Text
                    # 组装，保证状态行恒为一行且 [ ] 不被当标记解析。
                    budget = self.console.width - self.PADDING[1] * 2 - line.cell_len
                    line.append(_fit_cells(event.task_prompt, budget), style="dim")
                    view.status(Padding(line, self.PADDING))

                elif isinstance(event, ConfirmRequest):
                    view.flush_text()
                    view.stop_status()
                    self._gap()
                    self.console.print(Padding(
                        Text.from_markup(
                            f"[bold red]⚠ 风险操作:[/bold red] {event.tool_name} — {event.risk_reason}"
                        ), self.PADDING,
                    ))
                    self._at_gap = False
                    # 确认框是安全闸：参数展示不走 _brief_args 的 40 字符截断——
                    # 看不全的命令没法判断是否安全，等于让用户盲签。
                    self.console.print(Padding(
                        self._confirm_args_display(event.arguments), self.PADDING,
                    ))
                    # 选项对话框占据输入框（Claude Code 式）；Esc = 拒绝（不会中断整个回合）。
                    # 选「不执行」可以在那一行直接写一句说明，作为拒绝理由回给它（只拒不说，模型往往换个写法再试一遍）；
                    # 不写直接回车 = 只拒绝
                    feedback = ""
                    try:
                        picked = await self._choose(
                            "要执行这个操作吗？", ["执行", "不执行，告诉模型怎么改"], input_last=True)
                        approved = picked is not None and picked[0] == 0
                        if picked is not None and picked[0] == 1:
                            feedback = picked[1].strip()
                    except (EOFError, KeyboardInterrupt):
                        approved = False
                    event.future.set_result(True if approved else (feedback or False))
                    self._at_gap = False
                    # 不再另打「✓ 已确认 / ✗ 已拒绝」：_choose 留的「要执行这个操作吗？ → 执行」已经说了（真机：一件事说两遍）
                    if approved:
                        # 重挂执行动效——确认框弹出前 stop_status 掐掉了 spinner，
                        # 管道类命令（如 pytest | tail）到结束前零输出、也没有
                        # ShellOutput 兜底，不重挂会呈现"卡死"假象。
                        if self.verbose:
                            view.status(Padding(
                                Spinner("dots", text="[dim]执行中...[/dim]"), self.PADDING,
                            ))
                        else:
                            view.status(Padding(
                                self._tool_spinner(event.tool_name, event.arguments),
                                self.PADDING,
                            ))

                elif isinstance(event, PlanApproval):
                    view.flush_text()
                    view.stop_status()
                    self._gap()
                    # 标题 + 缩进内容，不画框（cards.py）：满宽框在窗口拉窄时会被终端折碎
                    self.console.print(section(Markdown(event.plan_summary), title="执行计划"))
                    try:
                        # 「提修改建议」是输入项，放最后一项（输入行只能接在对话框最后一行后面）
                        picked = await self._choose("批准这个计划吗？", [
                            "批准执行", "批准并自动接受编辑（敏感路径仍确认）", "取消", "提修改建议"], input_last=True)
                        choice = None if picked is None else picked[0]
                        answer = {0: "1", 1: "2", 2: "4", None: "4"}.get(choice, "")    # Esc = 取消
                        if choice == 3:
                            answer = picked[1].strip()
                            # 选了「提修改建议」又留空 → 视为取消，不把空串喂给模型
                        # 结果不再另打一行：「批准这个计划吗？ → 批准执行 / 提修改建议：……」已经说了（真机：说两遍）
                        if answer == "1":
                            event.future.set_result("approve")
                        elif answer == "2":
                            event.future.set_result("approve_auto_edits")
                        elif answer in ("4", ""):
                            event.future.set_result("cancel")
                            if answer == "":           # 只有这种问答留痕看不出来：选了提建议却没写
                                self.console.print("  [yellow]建议为空，计划已取消[/yellow]")
                        else:
                            event.future.set_result(answer)
                    except (EOFError, KeyboardInterrupt):
                        event.future.set_result("cancel")
                        self.console.print("  [yellow]✗ 计划已取消[/yellow]")

                elif isinstance(event, AgentDone):
                    # 正文内容已由 TextDelta 流式上屏（displayed=True）；错误/告警/
                    # 上限提示等从未上屏的终局消息在此补显——此前一律 pass，
                    # "LLM 调用失败"“空响应"等在 TUI 里无声无息（2026-07-17 事故）。
                    if event.content and not event.displayed:
                        view.flush_text()
                        view.stop_status()
                        self._gap()
                        # 纯 Text 组装：错误文本可能含异常原文的 [ ]，不能走 markup
                        self.console.print(Padding(
                            Text(event.content, style="yellow"), self.PADDING,
                        ))
                        self._at_gap = False

        finally:
            view.close()

        u = agent.token_usage
        delta = {k: u[k] - usage_before[k] for k in u}
        # 只在 /verbose 下打：状态行已有上下文占用，/usage 随时可查（Claude Code / Codex 都不按轮打，用户嫌输出太多）
        if self.verbose and delta["total_tokens"] > 0:
            self._gap()  # 统计行与正文之间同样恰好一个空行
            parts = [f"tokens: {delta['total_tokens']:,}"]
            prompt_detail = f"prompt {delta['prompt_tokens']:,}"
            if delta["prompt_cache_hit_tokens"] > 0:
                prompt_detail += f" (cache hit {delta['prompt_cache_hit_tokens']:,})"
            parts.append(prompt_detail)
            comp_detail = f"completion {delta['completion_tokens']:,}"
            if delta["reasoning_tokens"] > 0:
                comp_detail += f" (thinking {delta['reasoning_tokens']:,})"
            parts.append(comp_detail)
            if agent.cost_tracker:
                turn_cost = agent.cost_tracker.total_cost - cost_before
                if turn_cost > 0:
                    parts.append(f"cost: {agent.cost_tracker.format_cost(turn_cost)}")
            self.console.print(Padding(
                Text.from_markup(f"[dim]{' | '.join(parts)}[/dim]"),
                self.PADDING,
            ))

        self.console.print()

    def _gap(self) -> None:
        """块间距统一收口：需要时补恰好一个空行，已在空行处则不动（幂等）。"""
        if not self._at_gap:
            self.console.print()
            self._at_gap = True

    def _print_tool_block(self, tool_name: str, result: str) -> None:
        """折叠模式的工具结果块：`● tool(args)` + 缩进 `⎿` 统计/diff 行（错误为红 ✗）。"""
        args = self._active_tool_args or {}
        self._active_tool_args = {}
        ok, lines = result_lines(tool_name, args, result or "")
        if tool_name == "plan" and ok:
            self._gap()
            self.console.print(Padding(_plan_progress_line(args, result or ""), self.PADDING))
            self._at_gap = False
            return
        icon, style = ("●", "green") if ok else ("✗", "red")
        header = Text.assemble(
            (f"{icon} ", style), (tool_name, "bold"),
            ("(", "dim"), (arg_summary(tool_name, args), "dim"), (")", "dim"),
        )
        self._gap()
        self.console.print(Padding(header, self.PADDING))
        for i, line in enumerate(lines):
            # 前缀单独一列、只给它 dim（基础 dim 会叠加到 diff/错误文字上，把 #ffffff 压成灰——三轮发灰的真凶）；
            # 两列排版让长结果折行后对齐到 ⎿ 后面，不再回到最左边（真机）
            prefix = Text("  ⎿ " if i == 0 else "    ", style="dim")
            self.console.print(Padding(_hang(prefix, line), self.PADDING))
        self._at_gap = False

    # 确认框里非 command 参数的单值预览上限。command 永不截断（安全判断的全部依据）；
    # write_file 的 content 等长值给足预览即可——风险判断主要看路径，不需要整文件。
    CONFIRM_ARG_PREVIEW = 500

    def _confirm_args_display(self, args: dict) -> Text:
        """确认框参数展示：command 完整输出（自动换行），其余长值截 500 并标注总长。"""
        text = Text()
        for i, (k, v) in enumerate(args.items()):
            s = str(v)
            if k != "command" and len(s) > self.CONFIRM_ARG_PREVIEW:
                s = s[: self.CONFIRM_ARG_PREVIEW] + f"\n…（共 {len(str(v))} 字符，已截断预览）"
            if i:
                text.append("\n")
            text.append(f"{k}=", style="dim")
            text.append(s, style="bold" if k == "command" else "dim")
        return text

    def _tool_spinner(self, tool_name: str, arguments: dict) -> Spinner:
        """折叠模式的工具执行动效：整行按显示宽度压进单行。

        参数摘要是模型生成的自由文本——纯 Text 组装防 [ ] 被当标记解析，
        _fit_cells 防长参数/双宽字符撑到两行留残影（与思考行同一根因）。"""
        summary = arg_summary(tool_name, arguments)
        budget = self.console.width - self.PADDING[1] * 2 - 2  # 2 = spinner 字形 + 空格
        line = Text(_fit_cells(f"● {tool_name}({summary})", budget), style="dim")
        return Spinner("dots", text=line)

    def _brief_args(self, args: dict) -> str:
        parts = []
        for k, v in args.items():
            s = str(v)
            if len(s) > 40:
                s = s[:40] + "..."
            parts.append(f"{k}={s}")
        return ", ".join(parts)

    def _format_tool_result(self, tool_name: str, result: str) -> str:
        if not result:
            return ""
        if tool_name == "read_file":
            line_count = len(result.splitlines())
            return f"({line_count} 行)"
        if tool_name == "shell":
            lines = [l for l in result.strip().splitlines() if l.strip()]
            if lines:
                for line in reversed(lines):
                    stripped = line.strip().strip("=").strip("-").strip()
                    if any(kw in stripped for kw in ("passed", "failed", "error")):
                        if len(stripped) > 120:
                            stripped = stripped[:120] + "..."
                        return stripped
                last = lines[-1].strip()
                if last.startswith("[exit code:"):
                    last = lines[-2].strip() if len(lines) >= 2 else last
                if len(last) > 120:
                    last = last[:120] + "..."
                return last
        if len(result) > self.MAX_TOOL_RESULT_DISPLAY:
            return result[:self.MAX_TOOL_RESULT_DISPLAY] + "\n... (截断)"
        return result

    def _truncate(self, text: str, limit: int) -> str:
        if len(text) > limit:
            return text[:limit] + "..."
        return text

    def _goal_planning_prompt(self, description: str) -> str:
        return (
            f"目标: {description}\n\n"
            "现在只需要创建执行计划，不要开始执行。\n\n"
            "先调研再规划：目标里点名的文件（需求文档、任务清单、设计稿等）先用 read_file 读完；"
            "需要时用 shell 的只读命令核实环境和代码（git log/status、git grep、pip show、python --version 等），"
            "判据里写到的环境事实（版本、已装的包、git 状态）先查清再写。这一阶段 shell 只能跑只读查询："
            "不带重定向（>）和命令替换（$(...)），不跑脚本或 python -c；运行时行为要实测的，排进执行期的步骤里。\n"
            "- 文件里已经分了阶段或步骤的，计划只覆盖目标要求的那部分，步骤对齐文件里已有的分步；"
            "不要把后续阶段的内容提前排进来，也不要自造文件里没有的范围\n"
            "- 文件里写明的约束和验收要求，要落到具体步骤的判据里\n"
            "- 调研深度随任务：要做设计决定的（技术路线、架构、范围取舍），查到能为每个决定说出理由为止\n"
            "- 先问再规划：目标要产出设计（架构、方案、计划文档），而影响方案的关键决定在需求和文件里没定"
            "（技术路线、范围取舍、数据和凭证去向等）时，先不建计划——调研完列 3-6 个问题，每个写清选项、你的推荐和理由，"
            "然后停下等用户回答。需求里已经定了的不要再问；不是设计类的目标照常直接建计划\n\n"
            "然后用 plan(create) 创建计划。approach（方案）写四块：背景与范围（做什么、不做什么）；"
            "关键决定（每条写选了什么、放弃了什么、为什么）；风险与先验证什么；需要用户拍板的问题（单列，带你的推荐）。"
            "取舍和理由写在 approach，步骤只写怎么执行。步骤要求：\n"
            "- 每步是独立可验证的交付，不是操作指令；description 写清改什么、产出什么\n"
            "- 每步 1-5 条 criteria：执行期你自己能检查的事实（跑什么命令看到什么、文件里有什么）。"
            "标完成时每条都要交证据，写不出证据的判据就是没写好。判据查内容不查格式：章节数、行数、表格行数不能当判据"
            "（格式全对、内容全错也能过）；产出是文档或设计时，判据写文档里写明了哪些决定、覆盖了哪些风险\n"
            "- 需要用户本人操作才能验收的（实机、界面、外部账号）：目标要求拿到用户的验收结果时，把它单独作为最后一步，"
            "判据写「用户回复验收通过」，到那一步说清要用户做什么然后停下等回复（目标会暂停，用户回复后接着跑）；"
            "目标没要求等结果时，判据写「已告诉用户要验什么」。不要自己声称「实机通过」。"
            "用户不用动手的目标不要排「输出自查结论 / 验收清单」这类步骤——验证写进各步的判据，证据交给 plan\n"
            "- 3-8 步，包含验证\n\n"
            "只创建计划，不要执行。界面会把完整计划展示给用户审批，你不要再复述步骤；"
            "只用一两句话说需要用户拍板或特别注意的地方，没有就不说。"
        )

    def _goal_answer_prompt(self, description: str, answer: str, *, last_round: bool) -> str:
        """用户回答了规划期的提问（answer 为空 = 让模型按自己的推荐定）。"""
        if answer:
            head = f"用户对你问题的回答：{answer}\n\n没回答到的问题按你的推荐决定。"
        else:
            head = "用户让你按自己的推荐决定所有问题。"
        tail = ("不要再提问，现在就 plan(create)。" if last_round
                else "如果还有影响方案的关键决定没定，可以再问一轮；否则现在就 plan(create)。")
        return (f"目标: {description}\n\n{head}把这些决定（选了什么、放弃了什么、为什么）写进 approach 的关键决定。"
                f"{tail}只创建计划，不要执行。")

    def _goal_replan_prompt(self, description: str, feedback: str) -> str:
        return (
            f"目标: {description}\n\n"
            f"用户对计划的反馈：{feedback}\n\n"
            "请根据反馈用 plan(create) 重新创建计划；反馈涉及的文件或代码可以先用只读工具查看。"
            "反馈改变了决定的，approach 里同步改写（选了什么、放弃了什么、为什么），不要只改步骤。"
            "只创建计划，不要执行。"
        )

    def _goal_execution_prompt(self, description: str, plan_hint: str = "") -> str:
        parts = [
            f"目标: {description}",
            "用户已确认计划，现在开始执行。",
        ]
        if plan_hint:
            parts.append(plan_hint)
        parts.append(
            "逐步执行。每完成一步调用 plan(update_step, step_status=completed)，evidence 里逐条判据交证据"
            "（命令 + 关键输出、文件:行号）；证据要证明判据本身，不能拿相邻的事实充数。"
            "缺证据会被拒绝，按返回的清单补。step_output 记录关键结果。\n"
            "全部完成后调用 plan(complete)；做不了的步骤标 skipped 并在 step_output 写理由。\n"
            "步骤进度界面会显示（「◆ 计划：步骤 N 已更新」），不用播报「开始步骤 N」「登记第 N 步」；一步开始前最多一句话。\n"
            "遇到需要用户决策的问题直接提问，不要自行假设。"
        )
        return "\n".join(parts)

    def _goal_resume_prompt(self, description: str, user_input: str, plan_hint: str = "") -> str:
        """暂停后用户回复：原话放最前（它是这一轮的主角，也让采集脚本把它算作人工输入），目标与进度附后。"""
        parts = [user_input, "", "---", f"（目标模式续跑）目标: {description}"]
        if plan_hint:
            parts.append(plan_hint)
        parts.append("结合上面的回复继续执行计划。")
        return "\n".join(parts)

    def _goal_wrapup_prompt(self, description: str, plan_hint: str = "") -> str:
        parts = [f"目标: {description}"]
        if plan_hint:
            parts.append(plan_hint)
        parts.append(
            "本次自动执行的迭代预算已用完，目标还没完成。不要再调用工具，用几行话总结："
            "已完成什么（附证据位置）、卡在哪里、下一步具体做什么。这不是完成，不要声称目标已完成。"
        )
        return "\n".join(parts)

    def _goal_continue_prompt(self, description: str, plan_hint: str = "") -> str:
        parts = [f"目标: {description}"]
        if plan_hint:
            parts.append(plan_hint)
        parts.append("继续执行。")
        return "\n".join(parts)

    def _get_plan_hint(self, plan_id: str | None = None) -> tuple[str, bool]:
        """Returns (hint_text, is_completed)。

        执行期必须传 plan_id（这次 /goal 自己的计划）。不传时退回「最新创建的计划」——此前取的是
        「最近修改的计划」：exec-4 收尾时模型先把本计划标完成、再把上一轮遗留的空计划标放弃，后者成了
        最近修改的那份，目标完成被误判成「等待用户输入」；执行中途还会把旧计划的步骤当「下一步」喂给模型。"""
        if not self.plan_store:
            return ("", False)
        plan = self.plan_store.load(plan_id) if plan_id else self._get_latest_plan()
        if not plan:
            return ("", False)
        # 「完成」只认 plan complete（过了关闭闸）。步骤全部标完不算：p3-1 里模型把收尾步骤标完成的同时
        # 还在等用户回验收结果，旧逻辑按「步骤全完成」判目标完成，用户的回复成了脱离目标的普通对话。
        if plan.status == "completed":
            return ("", True)
        if plan.status != "active":
            return ("", False)
        done, total = plan.progress()
        if done >= total:
            return (
                f"当前进度: {done}/{total}\n所有步骤已标完成。对照目标原话与全部判据审计，"
                "确认无误后调用 plan complete 关闭计划；还在等用户的结果就说清楚要什么。",
                False,
            )
        next_step = None
        for step in plan.steps:
            if step.status in ("pending", "in_progress"):
                next_step = step
                break
        hint = f"当前进度: {done}/{total}"
        if next_step:
            hint += f"\n下一步（步骤 {next_step.number}）: {next_step.description}"
            missing = set(next_step.missing_evidence())
            for i, c in enumerate(next_step.criteria):
                hint += f"\n  判据 {i + 1}{'' if i in missing else '（已有证据）'}: {c}"
        return (hint, False)

    def _get_latest_plan(self, since: str | None = None):
        """最新创建的计划。按 created_at 而非文件修改时间：给别的计划标放弃、更新步骤都会刷新修改时间。

        给 since（规划开始时刻）则只认这之后创建、仍在执行态的——否则规划被打断时会拿到上一个目标已放弃的计划去审批
        （14e 真机：按 Esc 后执行了已放弃的计划）。"""
        if not self.plan_store:
            return None
        plans = self.plan_store.list_plans()
        if since is not None:
            plans = [p for p in plans if p.created_at >= since and p.status == "active"]
        return max(plans, key=lambda p: p.created_at) if plans else None

    def _display_plan(self, plan) -> None:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("idx", style="dim", width=4)
        table.add_column("step")
        for step in plan.steps:
            # 判据随步骤一起给用户审：批准的是「怎么算做完」，不只是「做什么」
            cell = Text(step.description)
            for c in step.criteria:
                cell.append(f"\n  · {c}", style="dim")
            table.add_row(f"{step.number}.", cell)          # 对外编号从 1 起（PlanStep.number）
        # 标题 + 缩进内容，不画框：计划内容再长、窗口拉多窄都不会被终端折碎（只按内容定宽挡不住长步骤）
        body = table
        approach = getattr(plan, "approach", "").strip()
        if approach:
            # 方案在步骤上面：先审「为什么这么做、放弃了什么」，再审「怎么做」（todo 第 23 项）
            from rich.console import Group
            body = Group(Markdown(approach), Text(""), Text("步骤", style="bold"), table)
        self.console.print(section(body, title=plan.title))
        self.console.print()             # 和下面的审批问答留痕隔一行

    async def _ask_plan_approval(self) -> str | None:
        """goal 计划审批。返回值沿用 run_goal 的约定："" = 执行，"n" = 取消，其它文字 = 修改意见；None = Esc 取消
        （Esc 不能当成「执行」：14e 真机按 Esc 执行了计划）。"""
        picked = await self._choose("开始执行这个计划吗？", ["执行", "取消", "提修改意见"], input_last=True)
        if picked is None:
            return None
        if picked[0] == 0:
            return ""
        if picked[0] == 2:
            return picked[1].strip() or "n"     # 选了「提修改意见」又留空 → 取消，不把空串当成「执行」
        return "n"

    def _last_assistant_text(self, agent: AgentLoop) -> str:
        for msg in reversed(agent.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"]
        return ""

    async def run_goal(self, agent: AgentLoop, description: str):
        if self._goal is not None:
            # 新目标顶替暂停中的旧目标：先正式结束旧的（未完成计划标放弃），免得两个目标的计划都挂着「活跃」
            self.stop_goal(agent)

        # 一行就够：目标原文上面「❯ /goal …」回显里已经有了，不再重复（用户嫌输出太多）
        # ── Phase 1: Planning (tool-restricted) ──
        self._goal_banner("规划中")

        self._goal_planning = True
        try:
            await self._goal_plan_and_approve(agent, description)
        finally:
            self._goal_planning = False
            self._goal_stop_requested = False

    def _cancel_goal_planning(self, since: str, reason: str) -> None:
        """规划 / 审批阶段被用户叫停：本次规划新建的计划一律标放弃（免得作为活跃计划进后续会话的系统提示）。"""
        if self.plan_store:
            for plan in self.plan_store.list_plans():
                if plan.created_at >= since and plan.status == "active":
                    plan.status = "abandoned"
                    self.plan_store.save(plan)
        self.console.print(Padding(Text.from_markup(f"[yellow]{reason}，目标未启动[/yellow]"), self.PADDING))

    async def _goal_plan_and_approve(self, agent: AgentLoop, description: str) -> None:
        since = datetime.now(timezone.utc).isoformat()

        async def planning_turn(prompt: str) -> bool:
            """跑一个规划回合；用户按 Esc 或发 /goal stop 则取消目标并返回 False。"""
            interrupted = await self._handle_agent_response(
                agent, prompt, allowed_tools=set(self.GOAL_PLANNING_TOOLS))
            if self._goal_stop_requested or interrupted:
                self._cancel_goal_planning(since, "已结束目标" if self._goal_stop_requested else "规划已中断")
                return False
            return True

        prompt = self._goal_planning_prompt(description)
        plan = None
        question_rounds = 0
        retries = 0
        while True:
            if not await planning_turn(prompt):
                return
            plan = self._get_latest_plan(since)
            if plan and plan.steps:
                break
            # 回合结束没建计划：模型说了话 = 在问关键决定（设计类目标先问再规划，todo 第 23.3 项），让用户答；
            # 没说话 = 偶发没建成，按原逻辑重试
            # 以问号为准：偶发没建成时模型多半也说了话（「我先读需求文档」），只看「有没有说话」会误弹问答框
            last_text = self._last_assistant_text(agent)
            asked = "？" in last_text or "?" in last_text
            if asked and question_rounds < self.GOAL_QUESTION_ROUNDS:
                question_rounds += 1
                picked = await self._choose(
                    "回答上面的问题？",
                    ["按你的推荐直接出计划", "取消", "回答问题（直接输入）"], input_last=True)
                if picked is None or picked[0] == 1:
                    self._cancel_goal_planning(since, "规划已取消")
                    return
                answer = picked[1].strip() if picked[0] == 2 else ""
                prompt = self._goal_answer_prompt(
                    description, answer, last_round=question_rounds >= self.GOAL_QUESTION_ROUNDS)
                continue
            retries += 1
            if retries >= 3:
                break
            self.console.print(Padding(
                Text.from_markup("[yellow]未能创建计划，重试...[/yellow]"),
                self.PADDING,
            ))
            prompt = self._goal_planning_prompt(description)
            if question_rounds >= self.GOAL_QUESTION_ROUNDS:
                prompt += "\n\n问答轮数已用完：不要再提问，没定的按你的推荐决定并写进 approach，现在就 plan(create)。"

        if not plan or not plan.steps:
            self.console.print(Padding(
                Text.from_markup("[red]未能创建执行计划，目标模式结束[/red]"),
                self.PADDING,
            ))
            return

        # ── Phase 2: Approval loop ──
        while True:
            self._display_plan(plan)
            response = await self._ask_plan_approval()

            if response is None or response.lower() in ("n", "no", "cancel", "/goal stop"):
                self.plan_store.load(plan.id)
                plan.status = "abandoned"
                self.plan_store.save(plan)
                self.console.print(Padding(
                    Text.from_markup("[yellow]计划已取消[/yellow]"),
                    self.PADDING,
                ))
                return

            if response in ("", "y", "yes"):
                break

            prompt = self._goal_replan_prompt(description, response)
            if not await planning_turn(prompt):
                return
            new_plan = self._get_latest_plan(since)
            if new_plan and new_plan.steps and new_plan.id != plan.id:
                plan.status = "abandoned"
                self.plan_store.save(plan)
                plan = new_plan
            elif new_plan and new_plan.steps:
                plan = new_plan

        # ── Phase 3: Execution ──
        self._goal_banner("执行中")
        self._goal = GoalState(description=description, plan_id=plan.id)
        agent.set_goal(description, plan.id)  # 目标原话 + 审计规则常驻系统提示，compact 不丢

        plan_hint, _ = self._get_plan_hint(plan.id)
        self._goal_planning = False          # 进入执行循环，/goal stop 改由 _goal_mode 放行
        await self._goal_execute(agent, self._goal_execution_prompt(description, plan_hint))

    def _goal_banner(self, phase: str) -> None:
        self._gap()                      # 阶段行和上面的内容（问答留痕、正文）空一行，别贴着
        line = Text("◆ ", style="cyan")
        line.append("目标模式", style="bold cyan")
        line.append(f" · {phase}", style="dim")
        self.console.print(Padding(line, self.PADDING))
        self._at_gap = False

    async def _goal_execute(self, agent: AgentLoop, prompt: str) -> None:
        """执行循环：跑到计划完成、暂停或用完本次迭代预算。run_goal 与暂停后续跑共用。

        停下来的出口只有两类：计划完成/放弃 → 结束目标；其余（等输入、受阻、无进展、自称完成但计划未完成、
        预算用完）→ 暂停，目标与计划保留，用户回复即续跑（Claude Code / Codex 的 /goal 都是这个语义）。"""
        goal = self._goal
        intent_llm = getattr(agent, '_intent_llm', None) or agent.llm
        plan_hint = ""
        self._goal_mode = True
        try:
            for i in range(1, self.MAX_GOAL_ITERATIONS + 1):
                goal.iterations += 1
                if i > 1:                    # 第 1 轮紧跟「执行中」那行，不再单独标
                    extra = f"（累计 {goal.iterations}）" if goal.iterations != i else ""
                    self._goal_banner(f"迭代 {i}/{self.MAX_GOAL_ITERATIONS}{extra}")
                interrupted = await self._handle_agent_response(agent, prompt)
                # 用户在运行中发了 /goal stop，或按了 Esc：都不能再自动续跑（此前 Esc 只停当前回合，循环照常往下走）
                if self._goal_stop_requested:
                    self.stop_goal(agent)
                    return
                if interrupted:
                    self._pause_goal("用户中断")
                    return

                plan_hint, plan_completed = self._get_plan_hint(goal.plan_id)
                if plan_completed:
                    self._end_goal(agent, "[bold green]✓ 目标完成[/bold green]（计划已完成）")
                    return
                plan = self.plan_store.load(goal.plan_id) if self.plan_store else None
                if plan is not None and plan.status == "abandoned":
                    self._end_goal(agent, "[yellow]计划已放弃，目标结束[/yellow]")
                    return

                # 无进展闸：一个工具都没调就不再自动续跑，也不必花一次模型调用判断意图（Codex / Claude Code 同做法）
                if agent.turn_tool_calls == 0:
                    self._pause_goal("本轮没有调用任何工具")
                    return

                intent = await detect_stop_intent(intent_llm, self._last_assistant_text(agent))
                if self._goal_stop_requested:      # 意图判断期间（两次迭代之间）发的 /goal stop
                    self.stop_goal(agent)
                    return
                if intent == StopIntent.NEEDS_INPUT:
                    self._pause_goal("等待用户输入")
                    return
                if intent == StopIntent.STUCK:
                    self._pause_goal("执行受阻")
                    return
                if intent == StopIntent.COMPLETED:
                    if plan is None or plan.current_step() is not None:
                        # 还有步骤没做完却自称完成 = 过早收尾的典型形态，交给用户裁决而不是记成完成
                        self._pause_goal("模型认为已完成，但计划还有步骤未完成")
                        return
                    # 步骤都标完了只差关闭：再续一轮让模型审计后 plan complete（关闭闸会再核一遍证据）

                self.console.print(Padding(
                    Text.from_markup(
                        f"[cyan]{plan_hint.splitlines()[0] if plan_hint else '继续执行...'}[/cyan]"
                    ),
                    self.PADDING,
                ))
                prompt = self._goal_continue_prompt(goal.description, plan_hint)

            # 预算用完：再给一轮不带工具的收尾，让模型交代进度与下一步；触顶 ≠ 完成
            self.console.print(Padding(
                Text.from_markup(
                    f"[bold yellow]⚠ 已达本次迭代上限（{self.MAX_GOAL_ITERATIONS}轮），目标未完成，收尾总结中[/bold yellow]"
                ),
                self.PADDING,
            ))
            await self._handle_agent_response(
                agent, self._goal_wrapup_prompt(goal.description, plan_hint), allowed_tools=set(),
            )
            self._pause_goal("迭代上限")
        finally:
            self._goal_mode = False
            self._goal_stop_requested = False

    def _pause_goal(self, reason: str) -> None:
        self._goal.status = "paused"
        self._goal.pause_reason = reason
        self.console.print(Padding(
            Text.from_markup(
                f"[cyan]⏸ 目标已暂停：{reason}[/cyan][dim]。直接回复即继续；/goal stop 结束目标[/dim]"
            ),
            self.PADDING,
        ))

    def _end_goal(self, agent: AgentLoop, markup: str) -> None:
        self._goal = None
        agent.clear_goal()
        self.console.print(Padding(Text.from_markup(markup), self.PADDING))

    async def resume_goal(self, agent: AgentLoop, user_input: str) -> None:
        """暂停的目标收到用户回复：用户原话在前、目标与进度在后，重入执行循环（重给一轮迭代预算）。"""
        goal = self._goal
        plan_hint, completed = self._get_plan_hint(goal.plan_id)
        plan = self.plan_store.load(goal.plan_id) if self.plan_store else None
        if completed or plan is None or plan.status != "active":
            # 计划已不在执行态：目标收尾，这条消息按普通对话处理
            self._end_goal(agent, "[dim]目标的计划已不在执行中，目标结束[/dim]")
            await self._handle_agent_response(agent, user_input)
            return
        goal.status = "running"
        goal.pause_reason = ""
        self.console.print(Padding(Text.from_markup("[dim]── 目标续跑 ──[/dim]"), self.PADDING))
        await self._goal_execute(agent, self._goal_resume_prompt(goal.description, user_input, plan_hint))

    def stop_goal(self, agent: AgentLoop) -> None:
        """/goal stop：结束目标，未完成的计划标放弃——否则它会作为「活跃计划」进入之后每个会话的系统提示。"""
        goal = self._goal
        if goal is None:
            self.console.print(Padding(Text.from_markup("[dim]当前没有目标[/dim]"), self.PADDING))
            return
        note = ""
        plan = self.plan_store.load(goal.plan_id) if self.plan_store else None
        if plan is not None and plan.status == "active":
            plan.status = "abandoned"
            self.plan_store.save(plan)
            note = f"（计划 {plan.id} 已标为放弃）"
        self._end_goal(agent, f"[yellow]目标已结束[/yellow]{note}")

    def describe_goal(self) -> str:
        """/goal 无参数时显示的当前目标状态；没有目标返回空串。"""
        goal = self._goal
        if goal is None:
            return ""
        state = f"暂停（{goal.pause_reason}）" if goal.status == "paused" else "执行中"
        hint, _ = self._get_plan_hint(goal.plan_id)
        return f"目标：{goal.description}\n状态：{state}，累计迭代 {goal.iterations}\n{hint}"
