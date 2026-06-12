from __future__ import annotations

import json

from loongcli.tools.base import Tool

_SNIPPET_RADIUS = 200
_MAX_SNIPPET = 600
_ROLE_LABELS = {"user": "用户", "assistant": "助手", "tool": "工具结果", "system": "系统"}


def _message_text(msg: dict) -> str:
    """提取消息的可检索文本：content + 工具调用的参数。"""
    parts = []
    content = msg.get("content")
    if isinstance(content, str):
        parts.append(content)
    for tc in msg.get("tool_calls") or []:
        func = tc.get("function", {})
        parts.append(f"{func.get('name', '')} {func.get('arguments', '')}")
    return "\n".join(p for p in parts if p)


def _snippet(text: str, keyword: str) -> str:
    idx = text.lower().find(keyword)
    start = max(0, idx - _SNIPPET_RADIUS)
    end = min(len(text), idx + len(keyword) + _SNIPPET_RADIUS)
    snippet = text[start:end].strip()
    if len(snippet) > _MAX_SNIPPET:
        snippet = snippet[:_MAX_SNIPPET]
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


class SearchHistoryTool(Tool):
    name = "search_history"
    description = (
        "检索本会话的完整对话历史，包括已被 compact 压缩归档、当前上下文中看不到的部分。"
        "当你需要回忆早前的具体细节（报错原文、用户的原话、当时读到的文件内容、某个决策的依据）"
        "而当前上下文和摘要中找不到时使用。多个关键词用空格分隔（须同时命中）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索关键词，空格分隔多个词（AND 关系），不区分大小写",
            },
            "limit": {
                "type": "integer",
                "description": "最多返回的匹配消息数（默认 5，返回最近的匹配）",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(self, conversation_store):
        self.store = conversation_store

    async def execute(self, query: str, limit: int = 5) -> str:
        keywords = [k.lower() for k in query.split() if k.strip()]
        if not keywords:
            return "请提供至少一个检索关键词。"
        limit = max(1, min(int(limit), 20))

        history = self.store.full_history()
        if not history:
            return "当前会话还没有可检索的历史。"

        matches: list[tuple[int, dict, str]] = []
        seen_texts: set[str] = set()
        for i, msg in enumerate(history):
            text = _message_text(msg)
            if not text or len(text) < 2:
                continue
            lower = text.lower()
            if not all(k in lower for k in keywords):
                continue
            # 归档段与当前 messages 有重叠，按文本去重
            key = text[:300]
            if key in seen_texts:
                continue
            seen_texts.add(key)
            matches.append((i, msg, text))

        if not matches:
            return f"完整历史（{len(history)} 条消息）中没有匹配 '{query}' 的内容。"

        recent = matches[-limit:]
        lines = [f"在完整历史（{len(history)} 条消息）中找到 {len(matches)} 条匹配，显示最近 {len(recent)} 条：\n"]
        for i, msg, text in recent:
            role = _ROLE_LABELS.get(msg.get("role", ""), msg.get("role", "?"))
            lines.append(f"--- [#{i} {role}] ---")
            lines.append(_snippet(text, keywords[0]))
        return "\n".join(lines)
