from dataclasses import dataclass
from typing import Any


@dataclass
class ThinkingDelta:
    text: str


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCallStart:
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallResult:
    tool_name: str
    result: str


@dataclass
class AgentDone:
    content: str


@dataclass
class CompactStart:
    message_count: int


@dataclass
class CompactNotice:
    before: int
    after: int


@dataclass
class TaskNotification:
    task_id: str
    result: str


@dataclass
class BatchProgress:
    completed: int
    total: int
    task_index: int
    task_prompt: str
    status: str


@dataclass
class ShellOutput:
    line: str
    stream: str  # "stdout" or "stderr"


@dataclass
class ConfirmRequest:
    tool_name: str
    arguments: dict[str, Any]
    risk_reason: str
    future: Any  # asyncio.Future[bool]


@dataclass
class PlanApproval:
    plan_id: str
    plan_summary: str
    future: Any  # asyncio.Future[str] — "approve" / "cancel" / feedback text
