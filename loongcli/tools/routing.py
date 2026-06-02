from __future__ import annotations
from enum import Enum


class AgentRole(Enum):
    MAIN = "main"
    SUBAGENT = "subagent"
    BACKGROUND = "background"


SUBAGENT_BLACKLIST: frozenset[str] = frozenset({
    "delegate",
    "batch_delegate",
    "send_message",
    "task_status",
    "memorize",
    "plan",
})

BACKGROUND_WHITELIST: frozenset[str] = frozenset({
    "read_file",
    "write_file",
    "edit_file",
    "shell",
    "glob",
    "grep",
})


def filter_tools(tools: dict[str, object], role: AgentRole) -> dict[str, object]:
    if role == AgentRole.MAIN:
        return tools
    if role == AgentRole.SUBAGENT:
        return {n: t for n, t in tools.items() if n not in SUBAGENT_BLACKLIST}
    if role == AgentRole.BACKGROUND:
        return {n: t for n, t in tools.items() if n in BACKGROUND_WHITELIST}
    return tools
