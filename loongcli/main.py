from __future__ import annotations
import argparse
import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console

from loongcli.core.config import Config
from loongcli.core.agent import AgentLoop
from loongcli.core.compact import Compactor, model_context_window, SUMMARY_TOKEN_RESERVE
from loongcli.core.prompts import get_system_prompt
from loongcli.core.git_context import collect_git_context
from loongcli.core.project_context import find_project_context_files
from loongcli.tools.base import ToolRegistry
from loongcli.tools.read_file import ReadFileTool
from loongcli.tools.write_file import WriteFileTool
from loongcli.tools.edit_file import EditFileTool
from loongcli.tools.shell import ShellTool
from loongcli.tools.glob_tool import GlobTool
from loongcli.tools.grep_tool import GrepTool
from loongcli.tools.recall import RecallTool
from loongcli.tools.memorize import MemorizeTool
from loongcli.tools.plan_tool import PlanTool
from loongcli.plan.store import PlanStore
from loongcli.tools.agent_tool import AgentTool
from loongcli.tools.batch_delegate import BatchDelegateTool
from loongcli.tools.send_message import SendMessageTool
from loongcli.tools.task_status import TaskStatusTool
from loongcli.tools.wait_tasks import WaitTasksTool
from loongcli.tools.stop_task import StopTaskTool
from loongcli.tools.enter_plan_mode import EnterPlanModeTool
from loongcli.tools.exit_plan_mode import ExitPlanModeTool
from loongcli.core.task import TaskManager
from loongcli.memory.markdown_store import MarkdownMemoryStore
from loongcli.memory.migrate import migrate_kv_to_markdown
from loongcli.memory.recall_engine import RecallEngine
from loongcli.memory.auto_extract import AutoExtractor
from loongcli.memory.conversation import ConversationStore
from loongcli.core.checkpoint import CheckpointManager
from loongcli.core.provider import ModelRouter
from loongcli.security.permissions import PermissionChecker, PermissionMode
from loongcli.mcp.manager import MCPManager
from loongcli.hooks.manager import HookManager, HookEvent
from loongcli.skills.registry import SkillRegistry
from loongcli.tools.skill import SkillTool
from loongcli.core.cost import CostTracker
from loongcli.lsp.server_manager import LSPServerManager
from loongcli.lsp.tools import register_lsp_tools
from loongcli.tui.app import TUI


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="loongcli",
        description="Loong Agent CLI — 通用 AI Agent 终端工具",
    )
    parser.add_argument(
        "prompt", nargs="?", default=None,
        help="直接提问（非交互模式），支持管道输入",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--continue", dest="continue_session", action="store_true",
        help="恢复最近一个会话",
    )
    group.add_argument(
        "--resume", action="store_true",
        help="浏览历史会话并选择一个继续",
    )
    parser.add_argument(
        "--dangerously-skip-permissions", action="store_true",
        help="跳过所有权限确认（灾难性操作仍会拦截）",
    )
    parser.add_argument(
        "--output-format", choices=["text", "json"], default="text",
        help="非交互模式输出格式（默认 text）",
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="非交互模式下不流式输出，等待完整回复",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="非交互模式下显示工具调用详情（输出到 stderr）",
    )
    return parser.parse_args()


def _build_prompt(args) -> str | None:
    parts = []
    if not sys.stdin.isatty():
        stdin_content = sys.stdin.read().strip()
        if stdin_content:
            parts.append(stdin_content)
    if args.prompt:
        parts.append(args.prompt)
    return "\n\n".join(parts) if parts else None


async def _run_noninteractive(agent: AgentLoop, prompt: str, args, mcp: MCPManager):
    from loongcli.core.events import TextDelta, ThinkingDelta, ToolCallStart, ToolCallResult, AgentDone, ConfirmRequest

    stderr = Console(stderr=True, no_color=True)
    output_json = args.output_format == "json"
    stream = not args.no_stream
    verbose = args.verbose

    buffer = ""
    thinking_buffer = ""
    tool_log: list[dict] = []

    async for event in agent.run_stream(prompt):
        if isinstance(event, ThinkingDelta):
            thinking_buffer += event.text

        elif isinstance(event, TextDelta):
            buffer += event.text
            if stream and not output_json:
                sys.stdout.write(event.text)
                sys.stdout.flush()

        elif isinstance(event, ToolCallStart):
            if verbose:
                stderr.print(f"⚙ {event.tool_name}({_brief(event.arguments)})")

        elif isinstance(event, ToolCallResult):
            if verbose:
                preview = event.result[:200] + "..." if len(event.result) > 200 else event.result
                stderr.print(f"✓ {event.tool_name}: {preview}")
            tool_log.append({"tool": event.tool_name, "result": event.result})

        elif isinstance(event, ConfirmRequest):
            if verbose:
                stderr.print(f"⚠ 自动拒绝: {event.tool_name} — {event.risk_reason}")
            event.future.set_result(False)

        elif isinstance(event, AgentDone):
            pass

    if output_json:
        result = {"content": buffer}
        if thinking_buffer:
            result["reasoning_content"] = thinking_buffer
        if verbose:
            result["tool_calls"] = tool_log
        result["usage"] = agent.token_usage
        if agent.cost_tracker:
            result["cost"] = agent.cost_tracker.summary_dict()
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2))
        sys.stdout.write("\n")
    elif not stream:
        sys.stdout.write(buffer)
        sys.stdout.write("\n")
    else:
        if not buffer.endswith("\n"):
            sys.stdout.write("\n")

    await mcp.disconnect_all()


def _brief(args: dict) -> str:
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 40:
            s = s[:40] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)


async def _async_main():
    args = _parse_args()
    console = Console()
    cfg = Config.load()

    if not cfg.api_key and not cfg.providers:
        console.print("[bold red]错误：[/bold red]未设置 API Key")
        console.print("配置文件: [cyan]~/.loongcli/config.json[/cyan]  字段: [cyan]api_key[/cyan]")
        console.print("  或环境变量: [cyan]DEEPSEEK_API_KEY[/cyan]")
        sys.exit(1)

    prompt = _build_prompt(args)
    noninteractive = prompt is not None

    router = ModelRouter.from_config(cfg)
    llm = router.client("main")
    sub_llm = router.client("sub")
    if sub_llm.model == llm.model:
        sub_llm = None
    memory_dir = Path.home() / ".loongcli" / "memory"
    migrate_kv_to_markdown(memory_dir)
    memory = MarkdownMemoryStore(base_dir=memory_dir)
    conversation = ConversationStore()

    resumed = False
    restored_messages: list[dict] | None = None
    structured_state: dict | None = None

    def _try_resume(sid: str) -> bool:
        nonlocal structured_state, restored_messages, resumed
        ss = conversation.resume_structured(sid)
        if ss:
            structured_state = ss
            resumed = True
            return True
        msgs = conversation.resume(sid)
        if msgs:
            restored_messages = msgs
            resumed = True
            return True
        return False

    if not noninteractive:
        if args.continue_session:
            sessions = conversation.list_sessions(limit=1)
            if sessions:
                sid = sessions[0]["session_id"]
                if _try_resume(sid):
                    mode = "structured" if structured_state else "compact"
                    console.print(
                        f"[dim]恢复会话 ({mode}): {sid} "
                        f"— {sessions[0].get('title', '(无标题)')}[/dim]"
                    )
            if not resumed:
                console.print("[yellow]没有可恢复的会话，启动新会话[/yellow]")

        elif args.resume:
            tui = TUI(memory=memory, config=cfg)
            session_id = await tui.pick_session(conversation)
            if session_id:
                _try_resume(session_id)
            if not resumed:
                console.print("[dim]启动新会话[/dim]")

    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    registry.register(ShellTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    registry.register(RecallTool(memory))
    registry.register(MemorizeTool(memory))
    plan_store = PlanStore()
    registry.register(PlanTool(plan_store))

    mcp = MCPManager(servers=cfg.mcp_servers)
    mcp_status = ""
    try:
        mcp_tools = await mcp.connect_all()
        if mcp_tools:
            mcp.register_tools(registry)
            mcp_status = f"MCP: {mcp.server_count} server(s), {mcp.tool_count} tool(s)"
    except Exception as e:
        if not noninteractive:
            console.print(f"[yellow]MCP 连接失败: {e}[/yellow]")
        else:
            print(f"MCP 连接失败: {e}", file=sys.stderr)

    skill_registry = SkillRegistry(project_dir=Path.cwd(), extra_dirs=cfg.skill_dirs)
    skill_tool = SkillTool(skill_registry)
    registry.register(skill_tool)

    perm_mode = PermissionMode.SKIP if args.dangerously_skip_permissions else PermissionMode.DEFAULT
    perm_checker = PermissionChecker(mode=perm_mode)
    task_manager = TaskManager()
    hook_manager = HookManager.from_config(cfg.hooks)

    registry.register(AgentTool(
        task_manager=task_manager,
        llm=llm,
        parent_registry=registry,
        security=perm_checker,
        sub_llm=sub_llm,
    ))
    registry.register(BatchDelegateTool(
        task_manager=task_manager,
        llm=llm,
        parent_registry=registry,
        security=perm_checker,
        sub_llm=sub_llm,
    ))
    registry.register(SendMessageTool(task_manager))
    registry.register(TaskStatusTool(task_manager))
    registry.register(WaitTasksTool(task_manager))
    registry.register(StopTaskTool(task_manager))
    enter_plan_tool = EnterPlanModeTool()
    exit_plan_tool = ExitPlanModeTool()
    exit_plan_tool.bind_plan_store(plan_store)
    registry.register(enter_plan_tool)
    registry.register(exit_plan_tool)

    lsp_manager = LSPServerManager(workspace_root=Path.cwd())
    register_lsp_tools(registry, lsp_manager)

    system_prompt = get_system_prompt(model=cfg.model, memory=memory, mcp=mcp, plan_store=plan_store)
    if cfg.compact_threshold:
        threshold = cfg.compact_threshold
    else:
        max_tokens = router.context_window_for("main")
        threshold = max(0, max_tokens - SUMMARY_TOKEN_RESERVE)

    compactor = Compactor(llm=llm, threshold=threshold, plan_store=plan_store, task_manager=task_manager)

    utility_llm = router.client("utility")
    recall_engine = RecallEngine(memory=memory, llm=utility_llm)
    auto_extractor = AutoExtractor(memory=memory, llm=utility_llm)

    checkpoint_mgr = CheckpointManager(cwd=Path.cwd())
    cost_tracker = CostTracker()

    agent = AgentLoop(
        llm=llm,
        tool_registry=registry,
        permission_checker=perm_checker,
        system_prompt=system_prompt,
        conversation_store=conversation,
        compactor=compactor,
        task_manager=task_manager,
        hook_manager=hook_manager,
        skill_registry=skill_registry,
        system_prompt_builder=lambda: get_system_prompt(
            model=cfg.model, memory=memory, mcp=mcp, plan_store=plan_store,
        ),
        recall_engine=recall_engine,
        auto_extractor=auto_extractor,
        checkpoint_manager=checkpoint_mgr,
    )
    agent.cost_tracker = cost_tracker
    agent.plan_store = plan_store
    agent.lsp_manager = lsp_manager
    enter_plan_tool.bind_agent(agent)
    exit_plan_tool.bind_agent(agent)

    if structured_state:
        # Structured resume: rebuild context from state instead of raw messages
        from loongcli.core.compact import SUMMARY_MARKER, SUMMARY_ACK
        from loongcli.core.attachments import (
            restore_files, plan_status, task_status,
            ATTACHMENT_MARKER, ATTACHMENT_ACK,
        )

        # Inject summary as conversation history
        if structured_state.get("summary"):
            agent.messages.append({
                "role": "user",
                "content": f"{SUMMARY_MARKER}\n{structured_state['summary']}",
            })
            agent.messages.append({"role": "assistant", "content": SUMMARY_ACK})

        # Rebuild file attachments + plan/task status from live stores
        sections: list[str] = []
        file_section = restore_files(structured_state.get("recent_files", []))
        if file_section:
            sections.append(file_section)
        ps = plan_status(plan_store)
        if ps:
            sections.append(ps)
        ts = task_status(task_manager)
        if ts:
            sections.append(ts)
        if sections:
            content = f"{ATTACHMENT_MARKER}\n\n" + "\n\n".join(sections)
            agent.messages.append({"role": "user", "content": content})
            agent.messages.append({"role": "assistant", "content": ATTACHMENT_ACK})
        resumed_plan_id = structured_state.get("plan_id")
        if resumed_plan_id and plan_store.load(resumed_plan_id):
            agent._active_plan_id = resumed_plan_id
    elif restored_messages:
        non_system = [m for m in restored_messages if m.get("role") != "system"]
        agent.messages = [agent.messages[0]] + non_system if agent.messages else non_system

    await hook_manager.run(HookEvent.SESSION_START, {
        "session_id": conversation.session_id,
        "model": cfg.model,
        "resumed": resumed,
        "noninteractive": noninteractive,
    })

    try:
        if noninteractive:
            await _run_noninteractive(agent, prompt, args, mcp)
        else:
            tui = TUI(memory=memory, config=cfg, skill_registry=skill_registry, plan_store=plan_store)
            status_parts = [f"model: {cfg.model}"]
            thinking_label = cfg.reasoning_effort if cfg.thinking else "off"
            status_parts.append(f"thinking: {thinking_label}")
            git_ctx = collect_git_context()
            if git_ctx.is_repo:
                git_label = f"git: {git_ctx.branch}"
                if git_ctx.dirty_files:
                    git_label += f" ({len(git_ctx.dirty_files)} changed)"
                status_parts.append(git_label)
            ctx_files = find_project_context_files()
            if ctx_files:
                status_parts.append(f"LOONG.md: {len(ctx_files)} loaded")
            skill_count = len(skill_registry.list_skills())
            if skill_count:
                status_parts.append(f"skills: {skill_count}")
            if mcp_status:
                status_parts.append(mcp_status)
            if perm_mode == PermissionMode.SKIP:
                status_parts.append("⚠ permissions: skip")
            await tui.start(agent, resumed=resumed, status_line=" | ".join(status_parts))
    finally:
        from loongcli.core.compact import _segment_turns, KEEP_RECENT_TURNS
        start = 1 if agent.messages and agent.messages[0].get("role") == "system" else 0
        turns = _segment_turns(agent.messages, start)
        if not noninteractive and len(turns) > KEEP_RECENT_TURNS:
            try:
                from rich.live import Live
                from rich.spinner import Spinner
                from loongcli.core.compact import SUMMARY_MARKER
                from loongcli.core.attachments import extract_recent_files
                from loongcli.core.task import TaskStatus

                spinner = Spinner("dots", text="[dim]压缩会话摘要中...[/dim]")
                with Live(spinner, console=console, refresh_per_second=8):
                    active_skill = agent._detect_active_skill()
                    compact_msgs = await compactor.compact(
                        agent.messages, active_skill=active_skill, mode="exit",
                    )
                    # Extract structured state for smart resume
                    recent_files = extract_recent_files(agent.messages)

                    summary = ""
                    marker = SUMMARY_MARKER
                    for m in compact_msgs:
                        if m.get("role") == "user" and marker in m.get("content", ""):
                            content = m["content"]
                            idx = content.find(marker)
                            if idx >= 0:
                                summary = content[idx + len(marker):].strip()
                            break

                    active_tasks = []
                    for t in task_manager._tasks.values():
                        if t.status == TaskStatus.RUNNING:
                            active_tasks.append({"id": t.id, "prompt": t.prompt[:200]})

                    plan_id = None
                    active_plans = plan_store.get_active_plans()
                    if active_plans:
                        plan_id = active_plans[0].id

                    conversation.save_compact(compact_msgs, structured_state={
                        "summary": summary,
                        "recent_files": recent_files,
                        "plan_id": plan_id,
                        "active_tasks": active_tasks,
                    })
                console.print("[dim]会话摘要已保存[/dim]")
            except Exception:
                pass
        await hook_manager.run(HookEvent.SESSION_END, {
            "session_id": conversation.session_id,
        })
        if not noninteractive:
            await mcp.disconnect_all()
        await lsp_manager.shutdown_all()


def main():
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
