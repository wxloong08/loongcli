from __future__ import annotations
import asyncio
import shutil
import signal
import sys
import unicodedata
from rich.console import Console
from rich.cells import cell_len
from loongcli.tui.mdstream import LeftMarkdown as Markdown
from loongcli.tui.tool_display import arg_summary, result_lines
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.spinner import Spinner
from rich.padding import Padding
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.patch_stdout import patch_stdout

from loongcli.core.agent import AgentLoop
from loongcli.core.events import TextDelta, ThinkingDelta, ToolCallStart, ToolCallResult, AgentDone, CompactStart, CompactNotice, TaskNotification, ConfirmRequest, BatchProgress, ShellOutput, PlanApproval, QueuedInputInjected
from loongcli.core.intent import StopIntent, detect_stop_intent
from loongcli.core.messages import message_text, extract_image_paths, is_image_file
from loongcli.memory.markdown_store import MarkdownMemoryStore
from loongcli.memory.conversation import ConversationStore
from loongcli.tui.commands import CommandContext, CommandRegistry, create_default_registry
from loongcli.tui.mdstream import StreamView
from loongcli.core.sanitize import repair_surrogates


_BUILTIN_COMMANDS = [
    ("exit", "退出"),
]


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


# 执行中插话队列上限：防轮中粘贴多行文本（无 bracketed paste，每个 \r 都是一次提交）
# 把队列灌爆——超限的提交静默丢弃
_INPUT_QUEUE_MAX = 20


def _fit_cells_tail(s: str, budget: int) -> str:
    """按显示宽度截断到 budget 列内，保留**尾部**、头部补省略号。

    输入回显要看的是正在打的那一端（尾部），与 _fit_cells 的方向相反；
    同样先剥宽度不可预测字符，单行承诺同一套防线。"""
    if budget <= 0:
        return ""
    s = _sanitize_cells(s)
    if cell_len(s) <= budget:
        return s
    budget -= 1  # 留 1 列给省略号
    out: list[str] = []
    used = 0
    for ch in reversed(s):
        w = cell_len(ch)
        if used + w > budget:
            break
        out.append(ch)
        used += w
    return "…" + "".join(reversed(out))


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


def _apply_key(ch: str, buffer: str) -> tuple[str, str | None, bool]:
    """键盘监听的纯函数内核：一个按键作用于行缓冲。

    返回 (新缓冲, 提交的消息或 None, 是否请求中断)。Esc 语义不变：永远中断，
    不做"先清缓冲再中断"的双击语义——零意外。功能键两字节序列（\\x00/\\xe0
    前缀）的第二字节由调用方吞掉（需要再读一次 kbhit，纯函数管不到）。
    非 BMP 字符（emoji）经 getwch 以代理对分两次到达：先原样累进缓冲，
    提交时 repair_surrogates 收口成真字符。"""
    if ch == "\x1b":
        return buffer, None, True
    if ch in ("\r", "\n"):
        submitted = repair_surrogates(buffer).strip()
        return "", (submitted or None), False
    if ch in ("\x08", "\x7f"):
        return buffer[:-1], None, False
    if ch >= " ":
        return buffer + ch, None, False
    return buffer, None, False


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


class TUI:
    MAX_TOOL_RESULT_DISPLAY = 200
    PADDING = (0, 2)

    MAX_GOAL_ITERATIONS = 20
    GOAL_STALL_LIMIT = 3

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
        self._session: PromptSession | None = None
        self._goal_mode: bool = False
        self._goal_description: str = ""
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
        # 拖入/粘贴单个图片文件 → 折叠成 [Image #N] 占位符（显示友好、存真实路径）。
        # 终端拖文件通常以 bracketed paste 送来一行路径，正好走这里。
        candidate = data.strip().strip('"').strip("'")
        if "\n" not in data.strip() and is_image_file(candidate):
            self._image_counter += 1
            placeholder = f"[Image #{self._image_counter}]"
            self._paste_images[placeholder] = candidate
            return placeholder
        lines = data.splitlines()
        if len(lines) >= self.PASTE_COLLAPSE_LINES or len(data) >= self.PASTE_COLLAPSE_CHARS:
            self._paste_counter += 1
            placeholder = f"[Pasted text #{self._paste_counter} +{len(lines)} lines]"
            self._pastes[placeholder] = data
            return placeholder
        return data

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

    def _get_session(self) -> PromptSession:
        if self._session is None:
            kb = KeyBindings()

            # 换行走 Ctrl+J（控制码 0x0A）：不经输入法（中文 IME 会把 `\` 键转成
            # 顿号，字符键方案全废）、也不被 Windows 终端抢去做全屏（Alt+Enter 的命）。
            # prompt_toolkit 在 Windows 上拿不到 Shift+Enter（读不到修饰键标志位，
            # Claude Code/Codex 用 crossterm/Ink 直读控制台事件才行）——Ctrl+J 是本栈
            # 下最接近 Shift+Enter 手感、且 IME/终端双躲的现成键。裸 Enter 保持默认。
            @kb.add("c-j")
            def _(event):
                event.current_buffer.insert_text("\n")

            @kb.add(Keys.BracketedPaste)
            def _(event):
                event.current_buffer.insert_text(self._maybe_collapse_paste(event.data))

            @kb.add("s-tab")
            def _(event):
                # Shift+Tab 切换规划模式（Claude Code 式模式循环的最小版）
                from prompt_toolkit.application import run_in_terminal
                run_in_terminal(self._toggle_plan_mode)

            # FileHistory：输入历史按项目落盘，新会话方向键直接可翻到过去的输入。
            # 此前 InMemoryHistory 只活在进程内，开新会话历史为空（真机反馈）。
            from loongcli.memory.conversation import input_history_path
            hist_path = input_history_path()
            hist_path.parent.mkdir(parents=True, exist_ok=True)
            self._session = PromptSession(
                "❯ ",
                history=FileHistory(str(hist_path)),
                key_bindings=kb,
                prompt_continuation="  ",
                completer=SlashCompleter(self.command_registry, self.skill_registry),
                complete_while_typing=True,
            )
        return self._session

    def _toggle_plan_mode(self) -> None:
        """Shift+Tab 切换规划模式。工具过滤在 run_stream 开头按 flag 生效，无需另拆。"""
        agent = getattr(self, "_agent", None)
        if agent is None:
            return
        # 无样式纯文本：run_in_terminal 挂起 prompt_toolkit 期间，Windows 下彩色
        # 转义码不被处理会裸露上屏（真机：?[36m…?[0m）。Rich 无样式时零转义码。
        if agent.plan_mode:
            agent._plan_mode = False
            agent._rebuild_system_prompt()  # 退出必须重建，否则规划注入残留在系统提示里
            self.console.print(Text("已退出规划模式"))
        else:
            agent.enter_plan_mode()
            self.console.print(Text("已进入规划模式（Shift+Tab 退出）— 下一条消息将先调研再规划"))

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
                self.console.print(f"[bold green]❯[/bold green] {content}")
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
        if resumed:
            title = (
                f"[bold cyan]loongcli[/bold cyan] v0.1.0 — 已恢复会话 "
                f"[bold]{agent.conversation_store.session_id}[/bold]"
            )
        else:
            title = "[bold cyan]loongcli[/bold cyan] v0.1.0 — Loong Agent CLI"

        lines = [title]
        # Panel 内宽 ≈ 终端宽 − 边框 2 − 默认 padding 2，再留 2 列余量
        inner_width = max(20, self.console.width - 6)
        for status_line in _pack_status_lines(status_items or [], inner_width):
            lines.append(f"[dim]{status_line}[/dim]")
        lines.append(
            "[bold]/help[/bold] 查看命令  [bold]/exit[/bold] 退出  "
            "[dim]Ctrl+J 换行  Esc 中断  Shift+Tab 规划模式[/dim]"
        )
        self.console.print(Panel("\n".join(lines), border_style="cyan"))

        if resumed:
            self.render_history(agent.messages)
        # 必须在 _get_session 建 FileHistory 之前回填，否则首个 prompt 读到的还是空文件
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
        session = self._get_session()

        need_divider = True
        while True:
            # 分隔线只在上一轮真的产出内容后打——空回车只重出提示符，
            # 否则连按回车会堆出一摞 ─── + ❯（真机 UI bug）
            if need_divider:
                self.console.print("─" * shutil.get_terminal_size().columns)
                need_divider = False
            try:
                with patch_stdout():
                    # default=中断/轮末未消化的插话预填——用户可编辑后再发，不替他发
                    user_input = await session.prompt_async(
                        default=getattr(self, "_input_prefill", ""),
                    )
                self._input_prefill = ""
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n" + self._farewell(agent))
                break

            raw_input = user_input  # 擦除空提交要用它的行数（Esc+Enter 换行会多行）
            # 还原折叠的大块粘贴：输入区显示的是占位符，发给 agent 的是完整原文。
            user_input = self._expand_pastes(user_input)
            # 源头修复：Windows 控制台粘贴 emoji 会以 UTF-16 代理对传入，原样下传会污染
            # 历史/记忆并在存盘与 LLM 请求的 UTF-8 编码处崩溃。把代理对解码回真 emoji
            # （孤立代理丢弃），下游全拿到干净且忠实的输入。
            user_input = repair_surrogates(user_input).strip()
            if not user_input:
                # 空/纯空白提交（连按回车、或 Esc+Enter 换行后空提交）：prompt_toolkit
                # 已把 ❯ 提交进滚动区，不擦就会堆一摞空 ❯。上移提示符块行数 + 清到屏末，
                # 下一轮 prompt_async 原位重画，视觉零堆叠。need_divider 未置 True → 不重画分隔线。
                self._erase_lines(raw_input.count("\n") + 1)
                continue
            need_divider = True  # 本轮有实际输入，处理后下一轮重新画分隔线
            if user_input == "/exit":
                self.console.print(self._farewell(agent))
                break

            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=3)
                cmd_name = parts[0][1:].lower()
                cmd = self.command_registry.get(cmd_name)
                if cmd:
                    await cmd.run(parts[1:], ctx)
                    continue
                if self.skill_registry:
                    skill_content = self.skill_registry.load_content(cmd_name)
                    if skill_content is not None:
                        meta = self.skill_registry.get(cmd_name)
                        skill_args = " ".join(parts[1:]) if len(parts) > 1 else ""
                        prompt = f"请按照以下技能指令执行：\n\n## 技能: {meta.name}\n{meta.description}\n\n{skill_content}"
                        if skill_args:
                            prompt += f"\n\n用户参数: {skill_args}"
                        await self._handle_agent_response(agent, prompt)
                        continue
                self.console.print(f"[yellow]未知命令: /{cmd_name}。输入 /help 查看可用命令。[/yellow]")
                continue

            await self._handle_agent_response(agent, user_input)

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

    def _erase_lines(self, n: int) -> None:
        """擦掉终端上最近 n 行（上移 n 行 + 清到屏末），光标回列首原位。

        用于空提交后回收堆叠的空提示符。走裸 stdout 而非 rich.console——
        rich 会转义 ANSI 控制序列。非 tty（管道）下静默跳过。"""
        if n < 1 or not sys.stdout.isatty():
            return
        sys.stdout.write(f"\033[{n}A\033[J")
        sys.stdout.flush()

    async def _handle_agent_response(
        self,
        agent: AgentLoop,
        user_input: str,
        allowed_tools: set[str] | None = None,
    ):
        """回合外壳：把回合跑成可取消任务，Ctrl+C 中断回合而非崩出应用。

        此前 KeyboardInterrupt 只在输入框和确认符两处被捕获，回合流式期间
        按 Ctrl+C 直接崩出整个应用（真机 3.1 附带确认的控制面缺口）。
        取消传播复用 stop_task 同一条链：CancelledError 打进 run_stream →
        _exec_tool_stream finally 收编 exec_task → shell 子进程随 Job 全灭。
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

            # Esc 为主触发键（无双义）；Ctrl+C 兜底（原行为=崩出应用，保留即降级保护）。
            # 只在回合期间接管，输入框阶段仍由 prompt_toolkit 处理（退出语义不变）。
            key_task = asyncio.create_task(self._key_listener(_request_interrupt, agent))
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
                # 中断路径：队列剩余 + 未提交缓冲一律预填输入框，不替用户发
                # ——用户刚按 Esc，此刻替他发消息违背中断意图
                self._stash_prefill(agent)
                return
            finally:
                key_task.cancel()
                signal.signal(signal.SIGINT, prev_handler)

            # 正常结束：未提交的半行缓冲预填输入框；队列剩余逐条自动发出
            buf = repair_surrogates(getattr(self, "_typed_buffer", "")).strip()
            self._typed_buffer = ""
            if buf:
                self._input_prefill = buf
            leftovers = agent.take_queued_inputs()
            if leftovers:
                pending = leftovers[0]
                for extra in leftovers[1:]:
                    agent.queue_user_input(extra)  # 重新排队，下一轮迭代边界注入
                auto_send = True
            else:
                pending = None

    def _typed_echo_line(self, agent: AgentLoop):
        """spinner Live 第二行的 provider：执行中已敲未提交的缓冲 + 排队数指示。

        空态（无缓冲无排队）返回 None——Live 高度回到 1 行，与无回显逐帧一致；
        确认框/审批框期间返回 None（缓冲已作废，别画尸体）。单行承诺三件套：
        repair_surrogates（半个代理对不进渲染）+ _fit_cells_tail（剥宽度不可预测
        字符 + 按 cell 截断，保尾部——正在打的那端）。在 Live 刷新线程被调用，
        只读状态、只造 renderable，不碰 console。"""
        if getattr(self, "_prompt_active", False):
            return None
        buf = getattr(self, "_typed_buffer", "")
        queued = len(getattr(agent, "_queued_user_inputs", []))
        if not buf and not queued:
            # 空态给常驻占位而非隐藏：可发现性是这个功能的生死线——没有提示的
            # 隐藏功能等于不存在（2026-07-19 真机测试反馈）。恒定 2 行高度也比
            # "空态 1 行 / 打字 2 行"的抖动更稳（高度抖动才是残影根因）。
            line = Text("❯ ", style="dim green")
            line.append("打字可插话 · Esc 中断", style="dim")
            return Padding(line, self.PADDING)
        suffix = f"  [+{queued} 排队]" if queued else ""
        budget = self.console.width - self.PADDING[1] * 2 - 2 - cell_len(suffix)  # 2 = "❯ "
        if budget <= 0:
            return None
        line = Text("❯ ", style="green")
        line.append(_fit_cells_tail(repair_surrogates(buf), budget))
        if suffix:
            line.append(suffix, style="dim")
        return Padding(line, self.PADDING)

    def _stash_prefill(self, agent: AgentLoop) -> None:
        """把插话队列剩余 + 未提交缓冲合并预填到下一次输入框（中断路径专用）。"""
        parts = agent.take_queued_inputs()
        buf = repair_surrogates(getattr(self, "_typed_buffer", "")).strip()
        self._typed_buffer = ""
        if buf:
            parts.append(buf)
        if parts:
            self._input_prefill = "\n".join(parts)

    async def _key_listener(self, cancel_cb, agent: AgentLoop):
        """回合期间监听键盘（Windows msvcrt 轮询；非 Windows 暂无，Ctrl+C 兜底）。

        Esc 中断语义不变；其余可见字符进行行缓冲，Enter 提交为执行中插话
        （agent.queue_user_input，下一迭代边界注入）。确认框/计划审批弹出时
        暂停读键并作废已敲缓冲（_prompt_active，避免抢 prompt_toolkit 的按键、
        防半行输入误提交）。每 tick 内层循环清空积压按键，粘贴突发不丢键。"""
        try:
            import msvcrt
        except ImportError:
            return
        self._typed_buffer = ""
        while True:
            if getattr(self, "_prompt_active", False):
                self._typed_buffer = ""
            else:
                while msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ("\x00", "\xe0"):
                        # 功能键两字节序列：吞掉第二字节，避免残留半个键
                        if msvcrt.kbhit():
                            msvcrt.getwch()
                        continue
                    self._typed_buffer, submitted, cancel = _apply_key(ch, self._typed_buffer)
                    if cancel:
                        cancel_cb()
                        return
                    if submitted and len(agent._queued_user_inputs) < _INPUT_QUEUE_MAX:
                        agent.queue_user_input(submitted)
            await asyncio.sleep(0.05)

    async def _run_turn_impl(
        self,
        agent: AgentLoop,
        user_input: str,
        allowed_tools: set[str] | None = None,
    ):
        self.console.print("─" * shutil.get_terminal_size().columns + "\n")
        self._at_gap = True  # 分隔线自带一个空行

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
        thinking_buffer = ""
        view = StreamView(
            self.console, left_pad=self.PADDING[1],
            # 实时打字回显：spinner 下方挂一行已敲缓冲。翻车回退点——去掉这个
            # 参数即逐字节恢复无回显行为，排队功能不受影响。
            input_line=lambda: self._typed_echo_line(agent),
        )
        view.status(Padding(Spinner("dots", text="[dim]思考中... (Esc 中断)[/dim]"), self.PADDING))

        # 拖图/贴路径自动附图：[Image #N] 占位符（拖入）+ 裸路径（打字）一起解析。
        vision = bool(getattr(getattr(agent, "llm", None), "vision", False))
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
                    thinking_buffer += event.text
                    last_line = thinking_buffer.split("\n")[-1]
                    # 按显示宽度截断到单行内（含左右 padding 与"思考中... "前缀），
                    # 保证 spinner 恒为一行，避免双宽中文撑到两行造成残影堆叠。
                    prefix = "思考中... "
                    budget = self.console.width - self.PADDING[1] * 2 - cell_len(prefix)
                    last_line = _fit_cells(last_line, budget)
                    # from_markup 会把正文里的 [ ] 当标记解析——思考文本含方括号会串样式，
                    # 故用 Text() 组装，仅前缀带样式。
                    line = Text.from_markup("[dim italic]思考中... [/dim italic]")
                    line.append(last_line, style="dim italic")
                    view.status(Padding(line, self.PADDING))

                elif isinstance(event, TextDelta):
                    thinking_buffer = ""
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
                    # 执行中插话已注入：停 spinner 后 scrollback 回显 ❯ 原文
                    # （与 CompactNotice 同款"停 Live 再 print"安全模式，零残影风险）。
                    # 用户文本是任意内容，纯 Text 组装防 [ ] 被当 markup 解析。
                    view.flush_text()
                    view.stop_status()
                    self._gap()
                    line = Text("❯ ", style="bold green")
                    line.append(event.text)
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
                    # 提前置位：从这里到 prompt_async 之间的每一次 print 都是
                    # key listener 抢键窗口（0.05s 轮询），置晚了会吞确认框首键
                    self._prompt_active = True
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
                    try:
                        confirm_session = PromptSession()
                        with patch_stdout():
                            answer = await confirm_session.prompt_async(
                                "  确认执行？(y/N) "
                            )
                        approved = answer.strip().lower() in ("y", "yes")
                    except (EOFError, KeyboardInterrupt):
                        approved = False
                    finally:
                        self._prompt_active = False
                    event.future.set_result(approved)
                    self._at_gap = False
                    if approved:
                        self.console.print("  [green]✓ 已确认[/green]")
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
                    else:
                        self.console.print("  [yellow]✗ 已拒绝[/yellow]")

                elif isinstance(event, PlanApproval):
                    # 提前置位，理由同 ConfirmRequest
                    self._prompt_active = True
                    view.flush_text()
                    view.stop_status()
                    self._gap()
                    self.console.print(Panel(
                        Markdown(event.plan_summary),
                        title="执行计划",
                        border_style="cyan",
                    ))
                    # 显式菜单（Claude Code 式）：编号或直接打字都行
                    self.console.print(Padding(Text.from_markup(
                        "[bold]1[/bold] 批准执行   [bold]2[/bold] 批准并自动接受编辑   "
                        "[bold]3[/bold] 提修改建议   [bold]4[/bold] 取消"
                    ), self.PADDING))
                    try:
                        plan_session = PromptSession()
                        with patch_stdout():
                            answer = await plan_session.prompt_async(
                                "  选择 (1/2/3/4)，或直接输入修改建议: "
                            )
                        answer = answer.strip()
                        if answer == "3":
                            with patch_stdout():
                                answer = (await plan_session.prompt_async("  修改建议: ")).strip()
                            # 选了 3 又留空 → 视为取消，不把空串喂给模型
                        if answer.lower() in ("1", "y", "yes"):
                            event.future.set_result("approve")
                            self.console.print("  [green]✓ 计划已批准，开始执行[/green]")
                        elif answer == "2":
                            event.future.set_result("approve_auto_edits")
                            self.console.print(
                                "  [green]✓ 计划已批准，开始执行[/green]"
                                "[dim]（文件编辑自动接受，敏感路径仍确认）[/dim]"
                            )
                        elif answer.lower() in ("4", "n", "no", ""):
                            event.future.set_result("cancel")
                            self.console.print("  [yellow]✗ 计划已取消[/yellow]")
                        else:
                            event.future.set_result(answer)
                            self.console.print(f"  [cyan]↻ 修改建议已提交[/cyan]")
                    except (EOFError, KeyboardInterrupt):
                        event.future.set_result("cancel")
                        self.console.print("  [yellow]✗ 计划已取消[/yellow]")
                    finally:
                        self._prompt_active = False

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
        if delta["total_tokens"] > 0:
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
        icon, style = ("●", "green") if ok else ("✗", "red")
        header = Text.assemble(
            (f"{icon} ", style), (tool_name, "bold"),
            ("(", "dim"), (arg_summary(tool_name, args), "dim"), (")", "dim"),
        )
        self._gap()
        self.console.print(Padding(header, self.PADDING))
        for i, line in enumerate(lines):
            # 前缀的 dim 只能做 span，不能做整行基础样式——基础 dim 会叠加到
            # append 进来的 diff/错误文字上，把 #ffffff 都压成灰（三轮发灰的真凶）
            row = Text()
            row.append("  ⎿ " if i == 0 else "    ", style="dim")
            row.append_text(line)
            self.console.print(Padding(row, self.PADDING))
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
            "现在只需要创建执行计划，不要开始执行。用 plan(create) 创建计划，步骤要求：\n"
            "- 每步是独立可验证的交付，不是操作指令\n"
            "- 描述中包含预期产出（如「分析 X 并输出对比表」）\n"
            "- 3-8 步，包含验证步骤\n\n"
            "只创建计划，等待用户确认后再执行。"
        )

    def _goal_replan_prompt(self, description: str, feedback: str) -> str:
        return (
            f"目标: {description}\n\n"
            f"用户对计划的反馈：{feedback}\n\n"
            "请根据反馈用 plan(create) 重新创建计划。只创建计划，不要执行。"
        )

    def _goal_execution_prompt(self, description: str, plan_hint: str = "") -> str:
        parts = [
            f"目标: {description}",
            "用户已确认计划，现在开始执行。",
        ]
        if plan_hint:
            parts.append(plan_hint)
        parts.append(
            "逐步执行，每完成一步调用 plan(update_step)，step_output 记录关键结果。\n"
            "全部完成后调用 plan(complete)。\n"
            "遇到需要用户决策的问题直接提问，不要自行假设。"
        )
        return "\n".join(parts)

    def _goal_continue_prompt(self, description: str, plan_hint: str = "") -> str:
        parts = [f"目标: {description}"]
        if plan_hint:
            parts.append(plan_hint)
        parts.append("继续执行。")
        return "\n".join(parts)

    def _get_plan_hint(self) -> tuple[str, bool]:
        """Returns (hint_text, is_completed)."""
        if not self.plan_store:
            return ("", False)
        plans = self.plan_store.list_plans()
        if not plans:
            return ("", False)
        plan = plans[0]
        if plan.status == "completed":
            return ("", True)
        if plan.status != "active":
            return ("", False)
        done, total = plan.progress()
        if done >= total:
            return ("", True)
        next_step = ""
        for step in plan.steps:
            if step.status in ("pending", "in_progress"):
                next_step = step.description
                break
        hint = f"当前进度: {done}/{total}"
        if next_step:
            hint += f"\n下一步: {next_step}"
        return (hint, False)

    def _get_latest_plan(self):
        if not self.plan_store:
            return None
        plans = self.plan_store.list_plans()
        return plans[0] if plans else None

    def _display_plan(self, plan) -> None:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("idx", style="dim", width=4)
        table.add_column("step")
        for step in plan.steps:
            table.add_row(f"{step.index}.", step.description)
        panel = Panel(
            table,
            title=f"[bold]{plan.title}[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )
        self.console.print(Padding(panel, self.PADDING))

    async def _ask_plan_approval(self) -> str:
        session: PromptSession = PromptSession()
        with patch_stdout():
            return (await session.prompt_async(
                "  [Enter] 执行  [n] 取消  或输入修改意见 > ",
            )).strip()

    def _last_assistant_text(self, agent: AgentLoop) -> str:
        for msg in reversed(agent.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"]
        return ""

    async def run_goal(self, agent: AgentLoop, description: str):
        self._goal_mode = True
        self._goal_description = description

        self.console.print(Padding(
            Text.from_markup(f"[bold cyan]目标模式[/bold cyan] {description}"),
            self.PADDING,
        ))

        # ── Phase 1: Planning (tool-restricted) ──
        self.console.print(Padding(
            Text.from_markup("[dim]── 规划阶段 ──[/dim]"),
            self.PADDING,
        ))

        prompt = self._goal_planning_prompt(description)
        plan = None
        for _attempt in range(3):
            await self._handle_agent_response(agent, prompt, allowed_tools={"plan"})
            plan = self._get_latest_plan()
            if plan and plan.steps:
                break
            self.console.print(Padding(
                Text.from_markup("[yellow]未能创建计划，重试...[/yellow]"),
                self.PADDING,
            ))
            prompt = self._goal_planning_prompt(description)

        if not plan or not plan.steps:
            self.console.print(Padding(
                Text.from_markup("[red]未能创建执行计划，目标模式结束[/red]"),
                self.PADDING,
            ))
            self._goal_mode = False
            self._goal_description = ""
            return

        # ── Phase 2: Approval loop ──
        while True:
            self._display_plan(plan)
            response = await self._ask_plan_approval()

            if response.lower() in ("n", "no", "cancel"):
                self.plan_store.load(plan.id)
                plan.status = "abandoned"
                self.plan_store.save(plan)
                self.console.print(Padding(
                    Text.from_markup("[yellow]计划已取消[/yellow]"),
                    self.PADDING,
                ))
                self._goal_mode = False
                self._goal_description = ""
                return

            if response in ("", "y", "yes"):
                break

            prompt = self._goal_replan_prompt(description, response)
            await self._handle_agent_response(agent, prompt, allowed_tools={"plan"})
            new_plan = self._get_latest_plan()
            if new_plan and new_plan.steps and new_plan.id != plan.id:
                plan.status = "abandoned"
                self.plan_store.save(plan)
                plan = new_plan
            elif new_plan and new_plan.steps:
                plan = new_plan

        # ── Phase 3: Execution ──
        self.console.print(Padding(
            Text.from_markup("[dim]── 执行阶段 ──[/dim]"),
            self.PADDING,
        ))

        plan_hint, _ = self._get_plan_hint()
        prompt = self._goal_execution_prompt(description, plan_hint)
        intent_llm = getattr(agent, '_intent_llm', None) or agent.llm

        for iteration in range(1, self.MAX_GOAL_ITERATIONS + 1):
            self.console.print(Padding(
                Text.from_markup(f"[dim]── 迭代 {iteration}/{self.MAX_GOAL_ITERATIONS} ──[/dim]"),
                self.PADDING,
            ))

            await self._handle_agent_response(agent, prompt)

            plan_hint, plan_completed = self._get_plan_hint()
            if plan_completed:
                self.console.print(Padding(
                    Text.from_markup("[bold green]✓ 目标完成[/bold green]（计划已完成）"),
                    self.PADDING,
                ))
                break

            last_text = self._last_assistant_text(agent)
            intent = await detect_stop_intent(intent_llm, last_text)

            if intent == StopIntent.NEEDS_INPUT:
                self.console.print(Padding(
                    Text.from_markup("[cyan]⏸ 等待用户输入...[/cyan]"),
                    self.PADDING,
                ))
                break

            if intent == StopIntent.STUCK:
                self.console.print(Padding(
                    Text.from_markup("[yellow]⚠ 执行受阻，目标模式停止[/yellow]"),
                    self.PADDING,
                ))
                break

            if intent == StopIntent.COMPLETED:
                self.console.print(Padding(
                    Text.from_markup("[bold green]✓ 目标完成[/bold green]"),
                    self.PADDING,
                ))
                break

            self.console.print(Padding(
                Text.from_markup(
                    f"[cyan]{plan_hint.splitlines()[0] if plan_hint else '继续执行...'}[/cyan]"
                ),
                self.PADDING,
            ))
            prompt = self._goal_continue_prompt(description, plan_hint)

        else:
            self.console.print(Padding(
                Text.from_markup(
                    f"[bold yellow]⚠ 已达目标模式迭代上限（{self.MAX_GOAL_ITERATIONS}轮）[/bold yellow]"
                ),
                self.PADDING,
            ))

        self._goal_mode = False
        self._goal_description = ""
