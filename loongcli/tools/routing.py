from __future__ import annotations
from enum import Enum


class AgentRole(Enum):
    MAIN = "main"
    COORDINATOR = "coordinator"
    SUBAGENT = "subagent"
    BACKGROUND = "background"


SUBAGENT_BLACKLIST: frozenset[str] = frozenset({
    "delegate",
    "batch_delegate",
    "send_message",
    "task_status",
    "wait_tasks",
    "stop_task",
    "memorize",
    "plan",
    "enter_plan_mode",
    "exit_plan_mode",
})

BACKGROUND_WHITELIST: frozenset[str] = frozenset({
    "read_file",
    "write_file",
    "edit_file",
    "shell",
    "glob",
    "grep",
})

COORDINATOR_ALLOWED: frozenset[str] = frozenset({
    "delegate",
    "wait_tasks",
    "stop_task",
    "send_message",
    "task_status",
    "read_file",
    "glob",
    "grep",
    "lsp_goto_definition",
    "lsp_find_references",
    "lsp_symbol_search",
    "lsp_hover",
    "lsp_diagnostics",
})


def filter_tools(tools: dict[str, object], role: AgentRole) -> dict[str, object]:
    if role == AgentRole.MAIN:
        return tools
    if role == AgentRole.COORDINATOR:
        return {n: t for n, t in tools.items() if n in COORDINATOR_ALLOWED}
    if role == AgentRole.SUBAGENT:
        return {n: t for n, t in tools.items() if n not in SUBAGENT_BLACKLIST}
    if role == AgentRole.BACKGROUND:
        return {n: t for n, t in tools.items() if n in BACKGROUND_WHITELIST}
    return tools
