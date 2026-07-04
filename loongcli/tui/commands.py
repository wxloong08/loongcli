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
        if agent.conversation_store:
            agent.conversation_store.archive_segment(agent.messages, reason="clear")
        system_msgs = [m for m in agent.messages if m["role"] == "system"]
        agent.messages = system_msgs
        agent._last_prompt_tokens = 0
        ctx.console.print("[cyan]✓ 对话已清空[/cyan]")


def _vision_for_model(config, model: str) -> bool:
    """从 roles 配置里查该模型是否标了 vision（同模型继承）。查不到按 False——
    宁可拒图也别把图喂给可能是纯文本的模型（显式失败优于静默腐败）。"""
    if config and getattr(config, "role_bindings", None):
        for bind in config.role_bindings.values():
            if bind.model == model and bind.vision:
                return True
    return False


def _deepseek_provider(config):
    """找 deepseek 供应商配置：显式名字优先，其次 base_url 含 deepseek 的（含 _default）。"""
    providers = getattr(config, "providers", None) if config else None
    if not providers:
        return None
    if "deepseek" in providers:
        return providers["deepseek"]
    for prov in providers.values():
        if "deepseek" in (prov.base_url or ""):
            return prov
    return None


class ModelCommand(SlashCommand):
    name = "model"
    description = "查看或切换模型（provider:model 跨供应商 / profile / 直接模型名）"
    usage = "[provider:model|profile|model_name]"

    async def run(self, args: list[str], ctx: CommandContext) -> None:
        llm = ctx.agent.llm
        if not args:
            vision_label = " [magenta](vision)[/magenta]" if getattr(llm, "vision", False) else ""
            ctx.console.print(
                f"当前模型: [bold cyan]{llm.model}[/bold cyan]{vision_label} "
                f"[dim]{getattr(llm, 'base_url', '')}[/dim]"
            )
            if ctx.config and ctx.config.model_profiles:
                names = ", ".join(ctx.config.list_profiles())
                ctx.console.print(f"可用 profiles: [dim]{names}[/dim]")
            if ctx.config and ctx.config.providers:
                names = ", ".join(ctx.config.providers)
                ctx.console.print(f"可用 providers: [dim]{names}[/dim]（跨供应商用 /model <provider>:<model>）")
            return

        name = args[0]

        # provider:model 语法——跨供应商切换，端点/密钥/provider_type/vision 一起换
        if ":" in name:
            prov_name, _, model = name.partition(":")
            prov = (ctx.config.providers or {}).get(prov_name) if ctx.config else None
            if not prov or not model:
                ctx.console.print(f"[red]未配置 provider '{prov_name}'[/red]（见 config.json providers 段）")
                return
            llm.switch(model=model, api_key=prov.api_key, base_url=prov.base_url,
                       vision=_vision_for_model(ctx.config, model))
            ctx.console.print(
                f"[green]✓ 已切换到[/green] [bold cyan]{model}[/bold cyan] "
                f"[dim]@ {prov.base_url}[/dim]"
            )
            return

        if ctx.config:
            profile = ctx.config.get_profile(name)
            if profile:
                llm.switch(
                    model=profile.model,
                    api_key=profile.effective_api_key(ctx.config.api_key),
                    base_url=profile.base_url,
                    vision=_vision_for_model(ctx.config, profile.model),
                )
                ctx.console.print(
                    f"[green]✓ 已切换到 profile[/green] [bold cyan]{name}[/bold cyan] "
                    f"(model={profile.model}, base_url={profile.base_url})"
                )
                return

        # 裸模型名：同供应商换模型（端点不动）；vision 按 roles 同名绑定解析
        llm.switch(model=name, vision=_vision_for_model(ctx.config, name))
        ctx.console.print(
            f"[green]✓ 模型已切换为[/green] [bold cyan]{name}[/bold cyan] "
            f"[dim]（端点不变: {getattr(llm, 'base_url', '')}）[/dim]"
        )


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
        if agent.conversation_store:
            agent.conversation_store.archive_segment(agent.messages, reason="manual-compact")
        try:
            agent.messages = await agent.compactor.compact(agent.messages)
        except Exception as e:
            # 压缩失败（如空摘要守卫触发）保留原历史，不让手动 /compact 击穿 TUI
            ctx.console.print(f"[yellow]压缩失败，历史未变动: {e}[/yellow]")
            return
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
                        f"    {role_name}: {ct.format_cost(rc.cost, rc.currency)} "
                        f"({rc.calls} calls, {rc.model})"
                    )

        ctx.console.print(Panel("\n".join(lines), title="Token 用量", border_style="cyan"))


MODEL_ALIASES = {
    "flash": "deepseek-v4-flash",
    "pro": "deepseek-v4-pro",
}


async def _switch_deepseek(ctx: CommandContext, alias: str, label: str) -> None:
    """/fast /pro 共用：切到 DeepSeek 目标模型，端点/密钥随 provider 配置一起换。

    主力若是其他供应商（如 qwen），只换模型名会拿 deepseek 模型名打错端点（400）——
    必须找到 deepseek provider 配置整体切换；找不到且当前端点也不是 DeepSeek 则明确拒绝。
    """
    llm = ctx.agent.llm
    target = MODEL_ALIASES[alias]
    prov = _deepseek_provider(ctx.config)
    if prov:
        llm.switch(model=target, api_key=prov.api_key, base_url=prov.base_url, vision=False)
    elif "deepseek" in (getattr(llm, "base_url", "") or ""):
        llm.switch(model=target, vision=False)
    else:
        ctx.console.print(
            "[red]未配置 deepseek provider，无法切换[/red]"
            "（当前端点非 DeepSeek，需在 config.json providers 段配置 deepseek）"
        )
        return
    ctx.console.print(
        f"[green]✓ 已切换到[/green] [bold cyan]{target}[/bold cyan] [dim]({label})[/dim]"
    )


class FastCommand(SlashCommand):
    name = "fast"
    description = "切换到 deepseek-v4-flash（快速、低成本）"

    async def run(self, args: list[str], ctx: CommandContext) -> None:
        await _switch_deepseek(ctx, "flash", "快速模式")


class ProCommand(SlashCommand):
    name = "pro"
    description = "切换到 deepseek-v4-pro（强推理、高质量）"

    async def run(self, args: list[str], ctx: CommandContext) -> None:
        await _switch_deepseek(ctx, "pro", "专业模式")


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


class PlanCommand(SlashCommand):
    name = "plan"
    description = "进入规划模式：先调研、再规划、用户审批后执行"
    usage = "[任务描述]"

    async def run(self, args: list[str], ctx: CommandContext) -> None:
        if not args:
            if ctx.agent.plan_mode:
                ctx.console.print("[cyan]当前已在规划模式中[/cyan]")
            elif ctx.agent._active_plan_id and ctx.agent.plan_store:
                plan = ctx.agent.plan_store.load(ctx.agent._active_plan_id)
                if plan:
                    ctx.console.print(Panel(
                        Markdown(plan.format_summary()),
                        title="活跃计划", border_style="cyan",
                    ))
                else:
                    ctx.console.print("[dim]没有活跃计划[/dim]")
            else:
                ctx.console.print(f"[yellow]用法: /{self.name} {self.usage}[/yellow]")
            return

        ctx.agent.enter_plan_mode()
        ctx.console.print("[cyan]已进入规划模式 — Agent 将先调研再规划[/cyan]")

        task_desc = " ".join(args)
        await ctx.tui._handle_agent_response(ctx.agent, task_desc)


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


class WebCommand(SlashCommand):
    name = "web"
    description = "打开 Web 页面：会话浏览（只读）+ 记忆管理"

    _server = None  # 类级缓存，整个 TUI 进程共享一个 server

    async def run(self, args: list[str], ctx: CommandContext) -> None:
        import threading
        import webbrowser

        from loongcli.web.server import create_server, server_url

        cls = WebCommand
        if cls._server is None:
            try:
                cls._server = create_server()
            except OSError as exc:
                ctx.console.print(f"[red]Web 服务启动失败: {exc}[/red]")
                return
            thread = threading.Thread(target=cls._server.serve_forever, daemon=True)
            thread.start()

        url = server_url(cls._server)
        ctx.console.print(f"[cyan]Web 页面: [link={url}]{url}[/link]（随 TUI 退出关闭）[/cyan]")
        webbrowser.open(url)


class VerboseCommand(SlashCommand):
    name = "verbose"
    description = "切换工具输出模式：折叠（默认）↔ 详细"

    async def run(self, args: list[str], ctx: CommandContext) -> None:
        if ctx.tui is None:
            ctx.console.print("[yellow]当前上下文没有 TUI，无法切换[/yellow]")
            return
        ctx.tui.verbose = not getattr(ctx.tui, "verbose", False)
        if ctx.tui.verbose:
            ctx.console.print("[cyan]✓ 详细模式已开启[/cyan]（⚙ 常驻行 + 结果原文）")
        else:
            ctx.console.print("[cyan]✓ 折叠模式已恢复[/cyan]（● 工具行 + ⎿ 统计/diff）")


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
    registry.register(WebCommand())
    registry.register(VerboseCommand())
    return registry
