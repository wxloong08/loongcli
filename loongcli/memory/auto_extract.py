from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loongcli.core.llm import LLMClient
    from loongcli.memory.markdown_store import MarkdownMemoryStore

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = """\
你是一个记忆提取器。分析以下对话，找出**值得跨会话保存**的信息。

用户和助手的对话：
{conversation}

## 什么值得提取

- **user**: 用户的角色、专长、偏好（"他是后端工程师"、"喜欢简洁回答"）
- **feedback**: 用户纠正或确认了助手的行为（"不要加注释"、"先设计方案再写代码"）
- **project**: 项目决策、约束、架构选择（"用 SQLite 因为简单"）
- **reference**: 外部资源引用（"bug 在 Linear INGEST 项目追踪"）

## 什么不要提取

- 临时任务状态（"正在写 test.py"）
- 当前对话的调试过程
- 能从代码文件直接获取的信息
- 琐碎的一次性问题

## 输出格式

只返回 JSON 数组，每个条目有 name、description、type、content 字段。如果没有值得提取的信息，返回空数组 []。

name 用 kebab-case（如 "user-prefers-python"），description 一行概括。
content 包含 why（为什么记住这个）和 how（什么场景下应用）。"""


class AutoExtractor:
    """Post-turn memory extraction using a lightweight LLM call.

    Fires asynchronously after each conversation turn completes.
    Non-blocking — failures are logged and silently ignored.
    """

    def __init__(self, memory: MarkdownMemoryStore, llm: LLMClient):
        self.memory = memory
        self.llm = llm

    def _build_prompt(self, messages: list[dict]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = (msg.get("content") or "")[:2000]
            if content.strip():
                lines.append(f"【{role}】{content}")
        return _EXTRACT_PROMPT.format(conversation="\n".join(lines))

    def _parse_memories(self, raw: str) -> list[dict]:
        """Parse JSON array from LLM response. Returns [] on any parse error."""
        try:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                return []
            data = json.loads(match.group(0))
            if not isinstance(data, list):
                return []
            return [
                m for m in data
                if isinstance(m, dict) and "name" in m and "content" in m
            ]
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.debug("AutoExtract: failed to parse LLM response: %s", raw[:200])
            return []

    async def extract(self, messages: list[dict]) -> int:
        """Extract and save memories from conversation. Returns count saved."""
        prompt = self._build_prompt(messages)
        try:
            response = await self.llm.chat(prompt)
        except Exception:
            logger.debug("AutoExtract: LLM call failed", exc_info=True)
            return 0

        extracted = self._parse_memories(response)
        count = 0
        for mem in extracted:
            try:
                self.memory.save(
                    name=mem["name"],
                    description=mem.get("description", ""),
                    type=mem.get("type", "project"),
                    content=mem["content"],
                )
                count += 1
            except Exception:
                logger.debug("AutoExtract: failed to save memory %s", mem.get("name"), exc_info=True)
        return count
