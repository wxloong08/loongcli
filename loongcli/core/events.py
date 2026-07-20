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
    # content 是否已在流式阶段（TextDelta）上过屏。错误/告警/上限提示等从未
    # 流式渲染的内容必须保持 False，TUI 见 False 才补显——否则"LLM 调用失败"
    # 这类终局消息静默消失（2026-07-17 空响应事故连带发现的洞）。默认 False
    # 是故障安全方向：新增错误路径忘了标，最坏是重复显示，而不是无声无息。
    displayed: bool = False


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
class QueuedInputInjected:
    """执行中排队的用户消息已注入 messages（迭代边界 drain）。TUI 收到后在
    scrollback 回显 ❯ 原文，让用户看到插话确实进了对话。"""
    text: str


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
