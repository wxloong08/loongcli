from __future__ import annotations
from datetime import datetime, timezone
import logging
import re

from loongcli.core.llm import LLMClient
from loongcli.core.stream_collector import StreamCollector
from loongcli.core.attachments import build_attachments

logger = logging.getLogger(__name__)

SUMMARY_TOKEN_RESERVE = 13000

# Model context windows (tokens). Used to compute compact threshold dynamically.
MODEL_MAX_TOKENS: dict[str, int] = {
    "deepseek-chat": 65536,
    "deepseek-reasoner": 65536,
    "deepseek-v4-flash": 1048576,
    "deepseek-v4-pro": 1048576,
}

DEFAULT_MODEL_MAX_TOKENS = 131072  # 128K safe fallback


def model_context_window(model: str) -> int:
    """Return the max context window (tokens) for a known model, or a safe default."""
    for prefix, size in MODEL_MAX_TOKENS.items():
        if model.startswith(prefix):
            return size
    return DEFAULT_MODEL_MAX_TOKENS


COMPACT_INSTRUCTION = """\
严禁调用任何工具。不要调用 read_file、shell、grep、glob 或任何其他工具。
工具调用会被拒绝，你只有这一次机会。你的全部回复必须是纯文本。

请总结以上对话历史。

先输出 <analysis> 块，按时间顺序梳理对话中的关键事件。然后输出 <summary> 块，包含以下 9 个固定章节：

1. **主要意图** — 用户的所有明确请求和目标
2. **关键技术概念** — 重要的技术栈、框架、架构决策
3. **文件与代码** — 查看、修改、创建的文件路径，附关键代码片段（路径:行号）和修改原因
4. **错误与修复** — 遇到的每个错误及其解决方式，避免重复踩坑
5. **问题解决** — 已解决的问题及方案
6. **所有用户消息** — 逐条列出所有非工具调用的用户消息原文（纠正、偏好、约束、需求变更一个不落）
7. **待办任务** — 明确请求但尚未完成的任务
8. **当前工作** — 用最细粒度描述摘要前正在进行的工作（精确到文件名、函数名、当前步骤）
9. **下一步** — 根据最近的请求建议下一步，引用对话原文作为依据

用中文输出。<summary> 部分要足够详细，确保仅凭摘要就能继续工作。

再次强调：不要调用任何工具，直接输出纯文本。"""

SUMMARY_MARKER = "[对话历史摘要]"
SUMMARY_ACK = "好的，我已了解之前的对话内容。请继续。"

BOUNDARY_TEMPLATE = (
    "[compact-boundary] mode={mode} | pre_tokens={pre_tokens} | "
    "pre_messages={pre_messages} | timestamp={timestamp}"
)

KEEP_RECENT_TURNS = 3
TOOL_RESULT_PLACEHOLDER = "[工具已执行，结果见摘要]"

# --- micro_compact: pre-LLM cleanup of old reclaimable tool results ---

RECLAIMABLE_TOOLS = frozenset({"read_file", "shell", "grep", "glob", "write_file", "edit_file"})
MICRO_COMPACT_KEEP_RECENT = 5
CLEARED_PLACEHOLDER = "[工具结果已清理，如需内容请重新调用]"


def _build_tool_call_index(messages: list[dict]) -> dict[str, str]:
    """Build mapping from tool_call_id -> tool_name by scanning assistant messages."""
    index: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                tc_id = tc.get("id", "")
                tc_name = tc.get("function", {}).get("name", "")
                if tc_id and tc_name:
                    index[tc_id] = tc_name
    return index


def micro_compact(messages: list[dict]) -> list[dict]:
    """Clear old reclaimable tool results, keeping the most recent MICRO_COMPACT_KEEP_RECENT.

    Reclaimable tools are those whose results can be re-obtained by calling the tool again
    (read_file, shell, grep, glob, write_file, edit_file).

    Non-reclaimable tools (delegate, recall, memorize, plan, task_status, send_message, skill, batch_delegate)
    are NEVER cleared.
    """
    if not messages:
        return messages

    tc_index = _build_tool_call_index(messages)

    # Find positions of all reclaimable tool results
    reclaimable_positions: list[int] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        tc_id = msg.get("tool_call_id", "")
        tool_name = tc_index.get(tc_id, "")
        if tool_name in RECLAIMABLE_TOOLS:
            reclaimable_positions.append(i)

    # If we have fewer than KEEP_RECENT, nothing to clear
    if len(reclaimable_positions) <= MICRO_COMPACT_KEEP_RECENT:
        return messages

    # Clear all but the last KEEP_RECENT
    to_clear = set(reclaimable_positions[:-MICRO_COMPACT_KEEP_RECENT])

    result = []
    for i, msg in enumerate(messages):
        if i in to_clear:
            result.append({**msg, "content": CLEARED_PLACEHOLDER})
        else:
            result.append(msg)
    return result


def _segment_turns(messages: list[dict], start: int) -> list[list[dict]]:
    turns: list[list[dict]] = []
    current: list[dict] = []
    for msg in messages[start:]:
        if msg["role"] == "user" and current:
            turns.append(current)
            current = []
        current.append(msg)
    if current:
        turns.append(current)
    return turns


def _replace_tool_results(messages: list[dict]) -> list[dict]:
    result = []
    for msg in messages:
        if msg["role"] == "tool":
            result.append({**msg, "content": TOOL_RESULT_PLACEHOLDER})
        else:
            result.append(msg)
    return result


def _fix_role_alternation(prefix: list[dict], kept: list[dict]) -> list[dict]:
    result = list(prefix)
    for msg in kept:
        if msg["role"] == "tool":
            result.append(msg)
            continue
        if result and msg["role"] == result[-1]["role"] == "assistant":
            prev = result[-1]
            merged_content = ((prev.get("content") or "") + "\n\n" + (msg.get("content") or "")).strip() or None
            result[-1] = {**prev, "content": merged_content}
            if msg.get("tool_calls"):
                result[-1]["tool_calls"] = msg["tool_calls"]
                result[-1]["content"] = result[-1].get("content") or None
        elif result and msg["role"] == result[-1]["role"] == "user":
            result[-1] = {**result[-1], "content": result[-1]["content"] + "\n\n" + msg["content"]}
        else:
            result.append(msg)
    return result


class Compactor:
    def __init__(self, llm: LLMClient, threshold: int = 800000, plan_store=None, task_manager=None):
        self.llm = llm
        self.threshold = threshold
        self.plan_store = plan_store
        self.task_manager = task_manager

    def should_compact(self, prompt_tokens: int, messages: list[dict]) -> bool:
        if prompt_tokens <= 0:
            return False
        if prompt_tokens <= self.threshold:
            return False
        start = 1 if messages and messages[0].get("role") == "system" else 0
        turns = _segment_turns(messages, start)
        return len(turns) > KEEP_RECENT_TURNS

    async def compact(
        self,
        messages: list[dict],
        active_skill: str | None = None,
        mode: str = "auto",
        pre_tokens: int = 0,
    ) -> list[dict]:
        system_msg = messages[0] if messages[0].get("role") == "system" else None
        start = 1 if system_msg else 0

        turns = _segment_turns(messages, start)
        if len(turns) <= KEEP_RECENT_TURNS:
            return messages

        keep_turns = turns[-KEEP_RECENT_TURNS:]
        summary = await self._summarize(messages, active_skill, mode=mode)
        logger.info("Compacted %d messages into summary (%d chars)", len(messages), len(summary))

        boundary = BOUNDARY_TEMPLATE.format(
            mode=mode,
            pre_tokens=pre_tokens,
            pre_messages=len(messages),
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        kept_messages = []
        for turn in keep_turns:
            kept_messages.extend(_replace_tool_results(turn))

        prefix: list[dict] = []
        if system_msg:
            prefix.append(system_msg)
        prefix.append({"role": "user", "content": f"{boundary}\n{SUMMARY_MARKER}\n{summary}"})
        prefix.append({"role": "assistant", "content": SUMMARY_ACK})

        attachments = build_attachments(messages, self.plan_store, self.task_manager)

        return _fix_role_alternation(prefix + attachments, kept_messages)

    async def _summarize(self, messages: list[dict], active_skill: str | None = None, mode: str = "auto") -> str:
        instruction = COMPACT_INSTRUCTION
        if active_skill:
            instruction += f"\n\n注意：当前正在执行技能 '{active_skill}'，在「活跃技能」部分详细记录进度。"
        if mode == "auto":
            instruction += "\n\n不要在摘要中提出新问题或建议用户回答任何内容。摘要应纯粹记录事实，不包含后续提问。"

        # Pre-process: clear old reclaimable tool results to reduce token load
        cleaned = micro_compact(list(messages))
        compact_messages = cleaned + [
            {"role": "user", "content": instruction},
        ]

        collector = StreamCollector()
        async for _ in collector.collect(
            self.llm.chat_stream(messages=compact_messages),
        ):
            pass

        return _extract_summary(collector.response.content)


def _extract_summary(raw: str) -> str:
    match = re.search(r"<summary>(.*?)</summary>", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    raw = re.sub(r"<analysis>.*?</analysis>", "", raw, flags=re.DOTALL)
    return raw.strip()
