from __future__ import annotations

import logging
import re
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loongcli.core.llm import LLMClient
    from loongcli.memory.markdown_store import MarkdownMemoryStore

logger = logging.getLogger(__name__)

MAX_RECALL = 5
STALE_DAYS = 7


def _staleness_caveat(updated_at: str) -> str:
    try:
        updated = datetime.fromisoformat(updated_at)
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - updated
        if age >= timedelta(days=STALE_DAYS):
            return f"\n> ⚠ 此记忆已 {age.days} 天未更新，内容可能已过时。请验证后再使用。\n"
    except (ValueError, TypeError):
        pass
    return ""

_RECALL_PROMPT = """\
你是一个记忆检索系统。给定用户的消息和一组记忆条目的描述，判断哪些记忆与当前对话相关。

用户消息：
{user_input}

可用记忆：
{memory_list}

请只返回相关记忆的 name（逗号分隔），最多 5 个。如果没有相关记忆，返回空。
只返回 name 列表，不要解释。"""


class RecallEngine:
    """Semantic recall: uses a lightweight LLM sideQuery to pick the most
    relevant memories for a given user message."""

    def __init__(self, memory: MarkdownMemoryStore, llm: LLMClient):
        self.memory = memory
        self.llm = llm

    def _build_prompt(self, user_input: str) -> str:
        """Build the sideQuery prompt listing all memory descriptions."""
        entries = self.memory.list_all()
        lines = []
        for entry in entries:
            lines.append(f"- {entry['name']}: {entry['description']} [{entry['type']}]")
        memory_list = "\n".join(lines)
        return _RECALL_PROMPT.format(user_input=user_input, memory_list=memory_list)

    async def recall(self, user_input: str) -> list[dict]:
        """Return up to 5 relevant memories for the given user input.

        Returns [] if no memories exist, or on LLM failure (graceful fallback).
        """
        entries = self.memory.list_all()
        if not entries:
            return []

        prompt = self._build_prompt(user_input)

        try:
            response = await self.llm.chat(prompt)
        except Exception:
            logger.warning("RecallEngine: LLM call failed, returning empty recall", exc_info=True)
            return []

        names = self._parse_names(response)

        results: list[dict] = []
        for name in names:
            if len(results) >= MAX_RECALL:
                break
            mem = self.memory.load(name)
            if mem is not None:
                results.append(mem)

        return results

    def format_for_injection(self, memories: list[dict]) -> str:
        """Format recalled memories for injection into conversation."""
        if not memories:
            return ""

        parts = ["# 相关记忆（根据当前对话自动检索）\n"]
        for mem in memories:
            parts.append(f"## [{mem['type']}] {mem['name']}")
            parts.append(f"> {mem['description']}\n")
            caveat = _staleness_caveat(mem.get("updated_at", ""))
            if caveat:
                parts.append(caveat)
            parts.append(mem.get("content", ""))
            parts.append("")  # trailing blank line between entries

        return "\n".join(parts).rstrip("\n") + "\n"

    @staticmethod
    def _parse_names(response: str) -> list[str]:
        """Parse memory names from LLM response.

        Handles comma-separated and newline-separated formats.
        Strips whitespace and any surrounding quotes/backticks.
        """
        # Split by commas or newlines
        raw = re.split(r"[,\n]", response)
        names: list[str] = []
        for part in raw:
            cleaned = part.strip().strip("`\"'")
            if cleaned:
                names.append(cleaned)
        return names
