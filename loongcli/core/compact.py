from __future__ import annotations
import logging
import re

from loongcli.core.llm import LLMClient
from loongcli.core.stream_collector import StreamCollector

logger = logging.getLogger(__name__)

COMPACT_INSTRUCTION = """\
请总结以上对话历史。不要调用任何工具，直接输出纯文本。

先输出 <analysis> 块，按时间顺序梳理对话中的关键事件，然后输出 <summary> 块，包含以下结构化 section：

1. **主要意图** — 用户的所有明确请求和目标
2. **技术概念** — 重要的技术栈、框架、架构决策
3. **活跃技能** — 如果正在执行某个技能（skill），记录：技能名称、已读的 reference 文件、已完成和未完成的步骤、当前进度
4. **工具使用摘要** — 搜索查询和关键发现、读取的网页和要点、其他工具调用的关键结果（省略原始输出）
5. **文件与代码** — 如果涉及代码：查看、修改、创建的文件路径，附关键代码片段和修改原因
6. **错误与修复** — 遇到的每个错误及其解决方式
7. **用户原话** — 所有非工具调用的用户消息原文（保留纠正、偏好、约束）
8. **待办任务** — 明确请求但尚未完成的任务
9. **当前工作** — 摘要前正在进行的工作细节
10. **下一步** — 根据最近的请求建议下一步，引用对话原文作为依据

用中文输出。<summary> 部分要足够详细，确保仅凭摘要就能继续工作。"""

SUMMARY_MARKER = "[对话历史摘要]"
SUMMARY_ACK = "好的，我已了解之前的对话内容。请继续。"

KEEP_RECENT_TURNS = 3
TOOL_RESULT_PLACEHOLDER = "[工具已执行，结果见摘要]"


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
    def __init__(self, llm: LLMClient, threshold: int = 800000):
        self.llm = llm
        self.threshold = threshold

    def should_compact(self, prompt_tokens: int, messages: list[dict]) -> bool:
        if prompt_tokens <= 0:
            return False
        if prompt_tokens <= self.threshold:
            return False
        start = 1 if messages and messages[0].get("role") == "system" else 0
        turns = _segment_turns(messages, start)
        return len(turns) > KEEP_RECENT_TURNS

    async def compact(self, messages: list[dict], active_skill: str | None = None) -> list[dict]:
        system_msg = messages[0] if messages[0].get("role") == "system" else None
        start = 1 if system_msg else 0

        turns = _segment_turns(messages, start)
        if len(turns) <= KEEP_RECENT_TURNS:
            return messages

        keep_turns = turns[-KEEP_RECENT_TURNS:]
        summary = await self._summarize(messages, active_skill)
        logger.info("Compacted %d messages into summary (%d chars)", len(messages), len(summary))

        kept_messages = []
        for turn in keep_turns:
            kept_messages.extend(_replace_tool_results(turn))

        prefix: list[dict] = []
        if system_msg:
            prefix.append(system_msg)
        prefix.append({"role": "user", "content": f"{SUMMARY_MARKER}\n{summary}"})
        prefix.append({"role": "assistant", "content": SUMMARY_ACK})

        return _fix_role_alternation(prefix, kept_messages)

    async def _summarize(self, messages: list[dict], active_skill: str | None = None) -> str:
        instruction = COMPACT_INSTRUCTION
        if active_skill:
            instruction += f"\n\n注意：当前正在执行技能 '{active_skill}'，在「活跃技能」部分详细记录进度。"

        compact_messages = list(messages) + [
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
