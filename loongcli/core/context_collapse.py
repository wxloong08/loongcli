from __future__ import annotations
import logging
from loongcli.core.compact import (
    _build_tool_call_index, RECLAIMABLE_TOOLS, CLEARED_PLACEHOLDER,
)

logger = logging.getLogger(__name__)

COLLAPSE_THRESHOLD_RATIO = 0.9   # 90% window usage triggers Level 1
COLLAPSE_AGGRESSIVE_RATIO = 0.95  # 95% triggers Level 2

LONG_ASSISTANT_THRESHOLD = 500    # chars — Level 2 truncates assistant messages longer than this
LONG_ASSISTANT_KEEP = 200         # chars — keep first N chars when truncating

COLLAPSED_NOTICE = "[... 消息已压缩，保留前 {kept} 字符]"


def should_collapse(prompt_tokens: int, max_tokens: int) -> int:
    """Return collapse level: 0=none, 1=mild (clear old tool results), 2=aggressive (also truncate assistant)."""
    if max_tokens <= 0 or prompt_tokens <= 0:
        return 0
    ratio = prompt_tokens / max_tokens
    if ratio >= COLLAPSE_AGGRESSIVE_RATIO:
        return 2
    if ratio >= COLLAPSE_THRESHOLD_RATIO:
        return 1
    return 0


def collapse(messages: list[dict], level: int, keep_recent: int = 5) -> list[dict]:
    """Create a collapsed projection of messages. Does NOT modify the originals.

    Level 1: Clear old reclaimable tool results (same as micro_compact logic but in a projection).
    Level 2: Level 1 + truncate long assistant messages (older ones only, keep last `keep_recent`).
    """
    if level <= 0:
        return messages

    result = list(messages)  # shallow copy

    # Level 1: Clear old reclaimable tool results
    tc_index = _build_tool_call_index(messages)
    reclaimable_positions: list[int] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        tc_id = msg.get("tool_call_id", "")
        tool_name = tc_index.get(tc_id, "")
        if tool_name in RECLAIMABLE_TOOLS:
            reclaimable_positions.append(i)

    if len(reclaimable_positions) > keep_recent:
        to_clear = set(reclaimable_positions[:-keep_recent])
        for i in to_clear:
            result[i] = {**result[i], "content": CLEARED_PLACEHOLDER}

    if level < 2:
        return result

    # Level 2: Truncate long assistant messages (except recent ones)
    assistant_positions: list[int] = []
    for i, msg in enumerate(result):
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
            if len(msg["content"]) > LONG_ASSISTANT_THRESHOLD:
                assistant_positions.append(i)

    if len(assistant_positions) > keep_recent:
        to_truncate = assistant_positions[:-keep_recent]
        for i in to_truncate:
            content = result[i]["content"]
            truncated = content[:LONG_ASSISTANT_KEEP] + COLLAPSED_NOTICE.format(kept=LONG_ASSISTANT_KEEP)
            result[i] = {**result[i], "content": truncated}

    return result
