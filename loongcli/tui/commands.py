from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

if TYPE_CHECKING:
    from loongcli.core.agent import AgentLoop
    from loongcli.core.config import Config
    from loongcli.memory.markdown_store import MarkdownMemoryStore


class SlashCommand(ABC):
    name: str
    description: str
    usage: str = ""

    @abstractmethod
    async def run(self, args: list[str], ctx: CommandContext) -> None:
        ...


class CommandContext:
    def __init__(
        self,
        console: Console,
        agent: AgentLoop,
        memory: MarkdownMemoryStore | None = None,
        registry: CommandRegistry | None = None,
        tui: Any = None,
        config: Config | None = None,
    ):
        self.console = console
        self.agent = agent
        self.memory = memory
        self.registry = registry
        self.tui = tui
        self.config = config


class CommandRegistry:
    def __init__(self):
        self._commands: dict[str, SlashCommand] = {}

    def register(self, cmd: SlashCommand):
        self._commands[cmd.name] = cmd

    def get(self, name: str) -> SlashCommand | None:
        return self._commands.get(name)

    def all_commands(self) -> list[SlashCommand]:
        return list(self._commands.values())


class HelpCommand(SlashCommand):
    name = "help"
    description = "显示所有可用命令"

    async def run(self, args: list[str], ctx: CommandContext) -> None:
        lines = []
        for cmd in sorted(ctx.registry.all_commands(), key=lambda c: c.name):
            usage = f" {cmd.usage}" if cmd.usage else ""
            lines.append(f"  [bold cyan]/{cmd.name}[/bold cyan]{usage} — {cmd.description}")
        lines.append(f"  [bold cyan]/exit[/bold cyan] — 退出")
        ctx.console.print(Panel("\n".join(lines), title="命令列表", border_style="cyan"))


class ClearCommand(SlashCommand):
    name = "clear"
    description = "清空当前对话历史"

    async def run(self, args: list[str], ctx: CommandContext) -> None:
        agent = ctx.agent
        system_msgs = [m for m in agent.messages if m["role"] == "system"]
        agent.messages = system_msgs
        agent._last_prompt_tokens = 0
        ctx.console.print("[cyan]✓ 对话已清空[/cyan]")


class ModelCommand(SlashCommand):
    name = "model"
    description = "查看或切换模型（支持 profile 名称或直接模型名）"
    usage = "[profile|model_name]"

    async def run(self, args: list[str], ctx: CommandContext) -> None:
        if not args:
            ctx.console.print(f"当前模型: [bold cyan]{ctx.agent.llm.model}[/bold cyan]")
            if ctx.config and ctx.config.model_profiles:
                names = ", ".join(ctx.config.list_profiles())
                ctx.console.print(f"可用 profiles: [dim]{names}[/dim]")
            return

        name = args[0]
        if ctx.config:
            profile = ctx.config.get_profile(name)
            if profile:
                ctx.agent.llm.model = profile.model
                api_key = profile.effective_api_key(ctx.config.api_key)
                from openai import AsyncOpenAI
                ctx.agent.llm.client = AsyncOpenAI(api_key=api_key, base_url=profile.base_url)
                ctx.console.print(
                    f"[green]✓ 已切换到 profile[/green] [bold cyan]{name}[/bold cyan] "
                    f"(model={profile.model}, base_url={profile.base_url})"
                )
                return

        ctx.agent.llm.model = name
        ctx.console.print(f"[green]✓ 模型已切换为[/green] [bold cyan]{name}[/bold cyan]")


class CompactCommand(SlashCommand):
    name = "compact"
    description = "手动压缩对话历史"

    async def run(self, args: list[str], ctx: CommandContext) -> None:
        agent = ctx.agent
        if not agent.compactor:
            ctx.console.print("[yellow]Compact 未启用[/yellow]")
            return
        before = len(agent.messages)
        if before <= 12:
            ctx.console.print("[dim]对话太短，无需压缩[/dim]")
            return
        agent.messages = await agent.compactor.compact(agent.messages)
        after = len(agent.messages)
        ctx.console.print(f"[cyan]✓ 已压缩: {before} → {after} 条消息[/cyan]")


class RememberCommand(SlashCommand):
    name = "remember"
    description = "保存记忆"
    usage = "<name> <description> <content>"

    async def run(self, args: list[str], ctx: CommandContext) -> None:
        if not ctx.memory:
            ctx.console.print("[red]记忆功能未启用[/red]")
            return
        if len(args) < 3:
            ctx.console.print(f"[yellow]用法: /{self.name} {self.usage}[/yellow]")
            return
        name, description = args[0], args[1]
        content = " ".join(args[2:])
        saved = ctx.memory.save(name=name, description=description, type="project", content=content)
        ctx.console.print(f"[green]✓ 已保存[/green] {saved}: {description}")


class ForgetCommand(SlashCommand):
    name = "forget"
    description = "删除记忆"
    usage = "<name>"

    async def run(self, args: list[str], ctx: CommandContext) -> None:
        if not ctx.memory:
            ctx.console.print("[red]记忆功能未启用[/red]")
            return
        if len(args) < 1:
            ctx.console.print(f"[yellow]用法: /{self.name} {self.usage}[/yellow]")
            return
        name = args[0]
        if ctx.memory.delete(name):
            ctx.console.print(f"[green]✓ 已删除[/green] {name}")
        else:
            ctx.console.print("[yellow]未找到对应记忆[/yellow]")


class MemoriesCommand(SlashCommand):
    name = "memories"
    description = "查看所有记忆"

    async def run(self, args: list[str], ctx: CommandContext) -> None:
        if not ctx.memory:
            ctx.console.print("[red]记忆功能未启用[/red]")
            return
        entries = ctx.memory.list_all()
        if not entries:
            ctx.console.print("[dim]（暂无记忆）[/dim]")
            return
        lines = []
        for e in entries:
            lines.append(f"[{e['type']}] {e['name']} — {e['description']}")
        ctx.console.print(Panel("\n".join(lines), title="记忆", border_style="cyan"))


class InitCommand(SlashCommand):
    name = "init"
    description = "在当前目录生成 LOONG.md 项目配置模板"

    async def run(self, args: list[str], ctx: CommandContext) -> None:
        from pathlib import Path
        target = Path.cwd() / "LOONG.md"
        if target.exists():
            ctx.console.print(f"[yellow]LOONG.md 已存在: {target}[/yellow]")
            return
        target.write_text(LOONG_TEMPLATE, encoding="utf-8")
        ctx.console.print(f"[green]✓ 已创建 LOONG.md[/green] — {target}")
        ctx.console.print("[dim]编辑此文件来配置项目指引，重启 loongcli 后生效[/dim]")


EFFORT_LEVELS = ("low", "medium", "high", "max")


class ThinkCommand(SlashCommand):
    name = "think"
    description = "切换思考模式或设置强度"
    usage = "[on|off|low|medium|high|max]"

    async def run(self, args: list[str], ctx: CommandContext) -> None:
        llm = ctx.agent.llm
        if not args:
            status = "开启" if llm.thinking else "关闭"
            ctx.console.print(
                f"思考模式: [bold cyan]{status}[/bold cyan]  "
                f"强度: [bold cyan]{llm.reasoning_effort}[/bold cyan]"
            )
            ctx.console.print(f"[dim]可选强度: {', '.join(EFFORT_LEVELS)}[/dim]")
            return

        arg = args[0].lower()
        if arg == "on":
            llm.thinking = True
            ctx.console.print(f"[green]✓ 思考模式已开启[/green] (effort={llm.reasoning_effort})")
        elif arg == "off":
            llm.thinking = False
            ctx.console.print("[green]✓ 思考模式已关闭[/green]")
        elif arg in EFFORT_LEVELS:
            llm.thinking = True
            llm.reasoning_effort = arg
            ctx.console.print(f"[green]✓ 思考模式已开启[/green] effort=[bold cyan]{arg}[/bold cyan]")
        else:
            ctx.console.print(f"[yellow]用法: /think [on|off|{'/'.join(EFFORT_LEVELS)}][/yellow]")


class UsageCommand(SlashCommand):
    name = "usage"
    description = "查看当前会话 token 用量与费用"

    async def run(self, args: list[str], ctx: CommandContext) -> None:
        u = ctx.agent.token_usage
        lines = [
            f"  total:      [bold]{u['total_tokens']:,}[/bold]",
            f"  prompt:     [bold]{u['prompt_tokens']:,}[/bold]",
        ]
        if u["prompt_cache_hit_tokens"] > 0 or u["prompt_cache_miss_tokens"] > 0:
            lines.append(f"    cache hit:  [green]{u['prompt_cache_hit_tokens']:,}[/green]")
            lines.append(f"    cache miss: {u['prompt_cache_miss_tokens']:,}")
        lines.append(f"  completion: [bold]{u['completion_tokens']:,}[/bold]")
        if u["reasoning_tokens"] > 0:
            lines.append(f"    thinking:   [cyan]{u['reasoning_tokens']:,}[/cyan]")

        ct = ctx.agent.cost_tracker
        if ct and ct.total_cost > 0:
            lines.append("")
            lines.append(f"  [bold yellow]总费用: {ct.format_cost()}[/bold yellow]")
            for role_name, rc in ct.roles.items():
                if rc.calls > 0:
                    lines.append(
                        f"    {role_name}: {ct.format_cost(rc.cost_usd)} "
                        f"({rc.calls} calls, {rc.model})"
                    )

        ctx.console.print(Panel("\n".join(lines), title="Token 用量", border_style="cyan"))


MODEL_ALIASES = {
    "flash": "deepseek-v4-flash",
    "pro": "deepseek-v4-pro",
}


class FastCommand(SlashCommand):
    name = "fast"
    description = "切换到 deepseek-v4-flash（快速、低成本）"

    async def run(self, args: list[str], ctx: CommandContext) -> None:
        ctx.agent.llm.model = MODEL_ALIASES["flash"]
        ctx.console.print(
            f"[green]✓ 已切换到[/green] [bold cyan]{MODEL_ALIASES['flash']}[/bold cyan] "
            f"[dim](快速模式)[/dim]"
        )


class ProCommand(SlashCommand):
    name = "pro"
    description = "切换到 deepseek-v4-pro（强推理、高质量）"

    async def run(self, args: list[str], ctx: CommandContext) -> None:
        ctx.agent.llm.model = MODEL_ALIASES["pro"]
        ctx.console.print(
            f"[green]✓ 已切换到[/green] [bold cyan]{MODEL_ALIASES['pro']}[/bold cyan] "
            f"[dim](专业模式)[/dim]"
        )


LOONG_TEMPLATE = """\
# 项目指引

<!-- loongcli 会自动加载此文件到 Agent 的 system prompt 中 -->
<!-- 在此写入项目规则、代码规范、架构约定等 -->

## 项目概述
<!-- 简述项目用途、技术栈 -->

## 代码规范
<!-- 例如：使用 pytest 测试、函数命名用 snake_case -->

## 注意事项
<!-- 例如：不要修改 migrations/ 目录、API key 不要硬编码 -->
"""


PLAN_INSTRUCTION = (
    "请先制定一个详细的分步执行计划，不要直接执行任何工具。"
    "按步骤编号列出，每步说明要做什么、用什么工具、预期结果。"
    "等用户确认后再开始执行。"
)


class PlanCommand(SlashCommand):
    name = "plan"
    description = "先制定计划，确认后再执行"
    usage = "<任务描述>"

    async def run(self, args: list[str], ctx: CommandContext) -> None:
        if not args:
            ctx.console.print(f"[yellow]用法: /{self.name} {self.usage}[/yellow]")
            return

        task_desc = " ".join(args)
        plan_prompt = f"{PLAN_INSTRUCTION}\n\n任务：{task_desc}"

        ctx.console.print("[cyan]📋 正在制定计划...[/cyan]")

        from loongcli.core.events import TextDelta, AgentDone
        from rich.live import Live
        from rich.text import Text

        plan_text = ""
        live = Live(console=ctx.console, refresh_per_second=8)
        live.start()
        try:
            async for event in ctx.agent.run_stream(plan_prompt, disable_tools=True):
                if isinstance(event, TextDelta):
                    plan_text += event.text
                    try:
                        live.update(Markdown(plan_text))
                    except Exception:
                        live.update(Text(plan_text))
                elif isinstance(event, AgentDone):
                    break
        finally:
            live.stop()

        ctx.console.print(Panel(Markdown(plan_text), title="执行计划", border_style="cyan"))

        try:
            with patch_stdout():
                answer = await ctx.tui._session.prompt_async(
                    "确认执行此计划？(y/N) "
                )
            if answer.strip().lower() not in ("y", "yes"):
                ctx.console.print("[yellow]✗ 已取消[/yellow]")
                return
        except (EOFError, KeyboardInterrupt):
            ctx.console.print("[yellow]✗ 已取消[/yellow]")
            return

        ctx.console.print("[cyan]▶ 开始执行计划...[/cyan]")
        execute_prompt = f"用户已确认以下计划，请严格按步骤执行：\n\n{plan_text}"
        await ctx.tui._handle_agent_response(ctx.agent, execute_prompt)


class GoalCommand(SlashCommand):
    name = "goal"
    description = "设定目标，Agent 自动规划并执行直到完成"
    usage = "<目标描述>"

    async def run(self, args: list[str], ctx: CommandContext) -> None:
        if not args:
            ctx.console.print(f"[yellow]用法: /{self.name} {self.usage}[/yellow]")
            return

        description = " ".join(args)
        await ctx.tui.run_goal(ctx.agent, description)


def create_default_registry() -> CommandRegistry:
    registry = CommandRegistry()
    registry.register(HelpCommand())
    registry.register(ClearCommand())
    registry.register(ModelCommand())
    registry.register(CompactCommand())
    registry.register(RememberCommand())
    registry.register(ForgetCommand())
    registry.register(MemoriesCommand())
    registry.register(PlanCommand())
    registry.register(GoalCommand())
    registry.register(InitCommand())
    registry.register(UsageCommand())
    registry.register(ThinkCommand())
    registry.register(FastCommand())
    registry.register(ProCommand())
    return registry
