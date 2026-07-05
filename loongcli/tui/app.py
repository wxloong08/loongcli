from __future__ import annotations
import asyncio
import shutil
from rich.console import Console
from loongcli.tui.mdstream import LeftMarkdown as Markdown
from loongcli.tui.tool_display import arg_summary, result_lines
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.spinner import Spinner
from rich.padding import Padding
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.patch_stdout import patch_stdout

from loongcli.core.agent import AgentLoop
from loongcli.core.events import TextDelta, ThinkingDelta, ToolCallStart, ToolCallResult, AgentDone, CompactStart, CompactNotice, TaskNotification, ConfirmRequest, BatchProgress, ShellOutput, PlanApproval
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

            @kb.add("escape", "enter")
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

            self._session = PromptSession(
                "❯ ",
                history=InMemoryHistory(),
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

    def _seed_history(self, messages: list[dict]):
        session = self._get_session()
        for msg in messages:
            if msg.get("role") == "user":
                content = message_text(msg)  # 多模态 list 取纯文本，图片计 [图片]
                if content and not self._is_internal_message(content):
                    session.history.append_string(content)

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

    async def start(self, agent: AgentLoop, resumed: bool = False, status_line: str = ""):
        if resumed:
            title = (
                f"[bold cyan]loongcli[/bold cyan] v0.1.0 — 已恢复会话 "
                f"[bold]{agent.conversation_store.session_id}[/bold]"
            )
        else:
            title = "[bold cyan]loongcli[/bold cyan] v0.1.0 — Loong Agent CLI"

        lines = [title]
        if status_line:
            lines.append(f"[dim]{status_line}[/dim]")
        lines.append(
            "[bold]/help[/bold] 查看命令  [bold]/exit[/bold] 退出  "
            "[dim]Esc+Enter 换行  Shift+Tab 规划模式[/dim]"
        )
        self.console.print(Panel("\n".join(lines), border_style="cyan"))

        if resumed:
            self.render_history(agent.messages)
            self._seed_history(agent.messages)

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
                    user_input = await session.prompt_async()
            except (EOFError, KeyboardInterrupt):
                sid = agent.conversation_store.session_id
                self.console.print(f"\n[dim]再见！会话 [cyan]{sid}[/cyan] 已保存，[cyan]loongcli --continue[/cyan] 可继续[/dim]")
                break

            # 还原折叠的大块粘贴：输入区显示的是占位符，发给 agent 的是完整原文。
            user_input = self._expand_pastes(user_input)
            # 源头修复：Windows 控制台粘贴 emoji 会以 UTF-16 代理对传入，原样下传会污染
            # 历史/记忆并在存盘与 LLM 请求的 UTF-8 编码处崩溃。把代理对解码回真 emoji
            # （孤立代理丢弃），下游全拿到干净且忠实的输入。
            user_input = repair_surrogates(user_input).strip()
            if not user_input:
                continue
            need_divider = True  # 本轮有实际输入，处理后下一轮重新画分隔线
            if user_input == "/exit":
                sid = agent.conversation_store.session_id
                self.console.print(f"[dim]再见！会话 [cyan]{sid}[/cyan] 已保存，[cyan]loongcli --continue[/cyan] 可继续[/dim]")
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

    async def _handle_agent_response(
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
        view = StreamView(self.console, left_pad=self.PADDING[1])
        view.status(Padding(Spinner("dots", text="[dim]思考中...[/dim]"), self.PADDING))

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
                    if len(last_line) > 80:
                        last_line = last_line[:80] + "..."
                    view.status(Padding(
                        Text.from_markup(f"[dim italic]思考中... {last_line}[/dim italic]"),
                        self.PADDING,
                    ))

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
                        summary = arg_summary(event.tool_name, event.arguments)
                        view.status(Padding(
                            Spinner("dots", text=f"[dim]● {event.tool_name}({summary})[/dim]"),
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

                elif isinstance(event, ShellOutput):
                    style = "dim" if event.stream == "stdout" else "dim red"
                    display_line = event.line if len(event.line) <= 120 else event.line[:120] + "..."
                    view.status(Padding(
                        Text.from_markup(f"[{style}]  {display_line}[/{style}]"),
                        self.PADDING,
                    ))

                elif isinstance(event, BatchProgress):
                    prompt_preview = event.task_prompt[:50] + "..." if len(event.task_prompt) > 50 else event.task_prompt
                    status_style = "green" if event.status == "completed" else "red"
                    view.status(Padding(
                        Text.from_markup(
                            f"[cyan]并行任务 {event.completed}/{event.total}[/cyan] "
                            f"[{status_style}]{event.status}[/{status_style}] "
                            f"[dim]{prompt_preview}[/dim]"
                        ), self.PADDING,
                    ))

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
                    try:
                        confirm_session = PromptSession()
                        with patch_stdout():
                            answer = await confirm_session.prompt_async(
                                "  确认执行？(y/N) "
                            )
                        approved = answer.strip().lower() in ("y", "yes")
                    except (EOFError, KeyboardInterrupt):
                        approved = False
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
                            summary = arg_summary(event.tool_name, event.arguments)
                            view.status(Padding(
                                Spinner("dots", text=f"[dim]● {event.tool_name}({summary})[/dim]"),
                                self.PADDING,
                            ))
                    else:
                        self.console.print("  [yellow]✗ 已拒绝[/yellow]")

                elif isinstance(event, PlanApproval):
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

                elif isinstance(event, AgentDone):
                    pass

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
