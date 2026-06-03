from __future__ import annotations
import asyncio
import shutil
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.spinner import Spinner
from rich.padding import Padding
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout

from loongcli.core.agent import AgentLoop
from loongcli.core.events import TextDelta, ThinkingDelta, ToolCallStart, ToolCallResult, AgentDone, CompactStart, CompactNotice, TaskNotification, ConfirmRequest, BatchProgress, ShellOutput
from loongcli.core.intent import StopIntent, detect_stop_intent
from loongcli.memory.markdown_store import MarkdownMemoryStore
from loongcli.memory.conversation import ConversationStore
from loongcli.tui.commands import CommandContext, CommandRegistry, create_default_registry


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

    def _get_session(self) -> PromptSession:
        if self._session is None:
            kb = KeyBindings()

            @kb.add("escape", "enter")
            def _(event):
                event.current_buffer.insert_text("\n")

            self._session = PromptSession(
                "❯ ",
                history=InMemoryHistory(),
                key_bindings=kb,
                prompt_continuation="  ",
                completer=SlashCompleter(self.command_registry, self.skill_registry),
                complete_while_typing=True,
            )
        return self._session

    def _seed_history(self, messages: list[dict]):
        session = self._get_session()
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content:
                    session.history.append_string(content)

    def render_history(self, messages: list[dict], max_recent: int = 5):
        conversations: list[tuple[str, str]] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                continue
            if role == "user":
                conversations.append(("user", content or ""))
            elif role == "assistant" and content:
                conversations.append(("assistant", content))
            elif role == "tool":
                conversations.append(("tool", content or ""))

        if not conversations:
            return

        if len(conversations) > max_recent * 3:
            skipped = len(conversations) - max_recent * 3
            self.console.print(f"  [dim]... 还有 {skipped} 条历史消息 ...[/dim]\n")
            conversations = conversations[-(max_recent * 3):]

        for role, content in conversations:
            if role == "user":
                self.console.print(f"[bold green]❯[/bold green] {content}")
            elif role == "assistant":
                text = content if len(content) <= 500 else content[:500] + "..."
                try:
                    self.console.print(Markdown(text))
                except Exception:
                    self.console.print(text)
            elif role == "tool":
                preview = content if len(content) <= 120 else content[:120] + "..."
                self.console.print(f"  [dim]{preview}[/dim]")
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
        lines.append("[bold]/help[/bold] 查看命令  [bold]/exit[/bold] 退出  [dim]Esc+Enter 换行[/dim]")
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
        session = self._get_session()

        while True:
            self.console.print("─" * shutil.get_terminal_size().columns)
            try:
                with patch_stdout():
                    user_input = await session.prompt_async()
            except (EOFError, KeyboardInterrupt):
                break

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input == "/exit":
                self.console.print("[dim]再见！[/dim]")
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
        buffer = ""
        thinking_buffer = ""
        in_thinking = False
        live = Live(
            Padding(Spinner("dots", text="[dim]思考中...[/dim]"), self.PADDING),
            console=self.console, refresh_per_second=8,
        )
        live.start()

        try:
            async for event in agent.run_stream(user_input, allowed_tools=allowed_tools):
                if isinstance(event, ThinkingDelta):
                    thinking_buffer += event.text
                    in_thinking = True
                    lines = thinking_buffer.split("\n")
                    last_line = lines[-1] if lines else ""
                    if len(last_line) > 80:
                        last_line = last_line[:80] + "..."
                    live.update(Padding(
                        Text.from_markup(f"[dim italic]💭 思考中... {last_line}[/dim italic]"),
                        self.PADDING,
                    ))

                elif isinstance(event, TextDelta):
                    if in_thinking:
                        in_thinking = False
                        live.stop()
                        self.console.print(Padding(
                            Text.from_markup(
                                f"[dim]💭 思考完成 ({len(thinking_buffer)} chars)[/dim]"
                            ), self.PADDING,
                        ))
                        thinking_buffer = ""
                        live = Live(console=self.console, refresh_per_second=8)
                        live.start()
                    buffer += event.text
                    live.update(Padding(Text(buffer), self.PADDING))

                elif isinstance(event, ToolCallStart):
                    live.stop()
                    if in_thinking:
                        in_thinking = False
                        self.console.print(Padding(
                            Text.from_markup(
                                f"[dim]💭 思考完成 ({len(thinking_buffer)} chars)[/dim]"
                            ), self.PADDING,
                        ))
                        thinking_buffer = ""
                    if buffer.strip():
                        self.console.print(Padding(Markdown(buffer), self.PADDING))
                        buffer = ""
                    self.console.print(Padding(
                        Text.from_markup(
                            f"[yellow]⚙ {event.tool_name}[/yellow]"
                            f" [dim]({self._brief_args(event.arguments)})[/dim]"
                        ), self.PADDING,
                    ))
                    live = Live(
                        Padding(Spinner("dots", text="[dim]执行中...[/dim]"), self.PADDING),
                        console=self.console, refresh_per_second=8,
                    )
                    live.start()

                elif isinstance(event, ToolCallResult):
                    live.stop()
                    display = self._format_tool_result(event.tool_name, event.result)
                    self.console.print(Padding(
                        Text.from_markup(f"[green]✓ {event.tool_name}[/green] {display}"),
                        self.PADDING,
                    ))
                    buffer = ""
                    live = Live(console=self.console, refresh_per_second=8)
                    live.start()

                elif isinstance(event, CompactStart):
                    live.update(Padding(
                        Spinner("dots", text=f"[cyan]压缩上下文中... ({event.message_count} 条消息)[/cyan]"),
                        self.PADDING,
                    ))

                elif isinstance(event, CompactNotice):
                    live.stop()
                    self.console.print(Padding(
                        Text.from_markup(f"[cyan]⚡ 上下文已压缩: {event.before} → {event.after} 条消息[/cyan]"),
                        self.PADDING,
                    ))
                    live = Live(console=self.console, refresh_per_second=8)
                    live.start()

                elif isinstance(event, TaskNotification):
                    live.stop()
                    self.console.print(Padding(
                        Text.from_markup(
                            f"[magenta]📋 SubAgent {event.task_id} 完成[/magenta] "
                            f"{self._truncate(event.result, 200)}"
                        ), self.PADDING,
                    ))
                    live = Live(console=self.console, refresh_per_second=8)
                    live.start()

                elif isinstance(event, ShellOutput):
                    style = "dim" if event.stream == "stdout" else "dim red"
                    display_line = event.line if len(event.line) <= 120 else event.line[:120] + "..."
                    live.update(Padding(
                        Text.from_markup(f"[{style}]  {display_line}[/{style}]"),
                        self.PADDING,
                    ))

                elif isinstance(event, BatchProgress):
                    prompt_preview = event.task_prompt[:50] + "..." if len(event.task_prompt) > 50 else event.task_prompt
                    status_style = "green" if event.status == "completed" else "red"
                    live.update(Padding(
                        Text.from_markup(
                            f"[cyan]⏳ 并行任务 {event.completed}/{event.total}[/cyan] "
                            f"[{status_style}]{event.status}[/{status_style}] "
                            f"[dim]{prompt_preview}[/dim]"
                        ), self.PADDING,
                    ))

                elif isinstance(event, ConfirmRequest):
                    live.stop()
                    self.console.print(Padding(
                        Text.from_markup(
                            f"[bold red]⚠ 风险操作:[/bold red] {event.tool_name} — {event.risk_reason}"
                        ), self.PADDING,
                    ))
                    self.console.print(Padding(
                        Text.from_markup(f"[dim]{self._brief_args(event.arguments)}[/dim]"),
                        self.PADDING,
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
                    if approved:
                        self.console.print("  [green]✓ 已确认[/green]")
                    else:
                        self.console.print("  [yellow]✗ 已拒绝[/yellow]")
                    buffer = ""
                    live = Live(console=self.console, refresh_per_second=8)
                    live.start()

                elif isinstance(event, AgentDone):
                    pass

        finally:
            live.stop()
            if buffer.strip():
                self.console.print(Padding(Markdown(buffer), self.PADDING))
                buffer = ""

        u = agent.token_usage
        if u["total_tokens"] > 0:
            parts = [f"tokens: {u['total_tokens']:,}"]
            prompt_detail = f"prompt {u['prompt_tokens']:,}"
            if u["prompt_cache_hit_tokens"] > 0:
                prompt_detail += f" (cache hit {u['prompt_cache_hit_tokens']:,})"
            parts.append(prompt_detail)
            comp_detail = f"completion {u['completion_tokens']:,}"
            if u["reasoning_tokens"] > 0:
                comp_detail += f" (thinking {u['reasoning_tokens']:,})"
            parts.append(comp_detail)
            self.console.print(Padding(
                Text.from_markup(f"[dim]{' | '.join(parts)}[/dim]"),
                self.PADDING,
            ))

        self.console.print()

    def _brief_args(self, args: dict) -> str:
        parts = []
        for k, v in args.items():
            s = str(v)
            if len(s) > 40:
                s = s[:40] + "..."
            parts.append(f"{k}={s}")
        return ", ".join(parts)

    def _format_tool_result(self, tool_name: str, result: str) -> str:
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
            Text.from_markup(f"[bold cyan]🎯 目标模式[/bold cyan] {description}"),
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
