from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from loongcli.core.messages import message_text

if TYPE_CHECKING:
    from loongcli.core.llm import LLMClient
    from loongcli.memory.markdown_store import MarkdownMemoryStore

logger = logging.getLogger(__name__)

# 闸 4：拦"搜了没找到 → 不存在"式否定断言。工具故障/搜错目录/扫描截断都会产生
# 假阴性，且这类记忆会被 recall 主动命中毒害后续会话（2026-07-04 真机复现：grep
# 扫描预算 bug 产出"cache_aware 不存在"毒记忆）。确定性拦截，不赌提取模型自觉。
_NEGATIVE_CLAIM_RE = re.compile(
    r"not[-_ ]?found|未找到|不存在|没有匹配|无匹配|no match", re.IGNORECASE,
)

_EXTRACT_PROMPT = """\
你是一个记忆提取器。分析以下对话，找出**值得跨会话保存**的信息。

用户和助手的对话：
{conversation}

## 已有记忆（不要重复写）

记忆库里已存在下列条目。**不要新建与之同义或重叠的记忆**；同一件事已经记过，就跳过。
{existing_memories}

## 提取的唯一标准

只有当一条信息**几周以后、在另一个会话里仍然有用**，才提取。问自己：这是「跨会话的稳定事实」，还是「只属于当前这轮对话的过程」？只有前者才存。

**大多数对话应该提取 0 条。** 一次超过 1~2 条，几乎一定是在记录过程噪音——重新审视。宁可漏，不可滥。

## 什么值得提取（四类）

- **user**: 用户的角色、专长、稳定偏好（"他是后端工程师"、"总是要求简洁回答"）
- **feedback**: 用户给出的、应改变你**今后**行为的稳定指导（"不要加注释"、"先设计方案再写代码"）——不是对某次具体产出的一次性意见
- **project**: 项目的决策、约束、架构选择（"用 SQLite 因为简单"）
- **reference**: 外部资源指针（"bug 在 Linear INGEST 项目追踪"）

## 什么绝不提取

核心原则：**只记 代码、git、当前对话 都推不出来、且过几周仍然成立的东西。**

- **"搜了没找到 → 不存在"式的否定断言**（"grep 未找到 X，项目里没有 X"）——工具故障、搜错目录、扫描截断都会产生假阴性；一次搜索只能证明那次没搜到，不能证明事实不存在。这类假事实会被 recall 命中并主动毒害后续会话
- **对某次产出的即时反应 / 自评 / 打断**（"这段不通顺"、"Q1 自评 2 分"、"用户刷到第 3 题时打断"）——这是会话过程，不是跨会话偏好
- 临时任务状态、当前对话的调试过程（"正在写 test.py"）
- 代码模式、架构、文件路径、项目结构（grep / 看代码就有，存了反而和真相不一致）
- git 历史和最近改动（git log 是权威，记忆只会落后）
- 调试方案、修复方法（fix 已在代码、commit 已记上下文）
- 已经写进 CLAUDE.md / LOONG.md 的内容
- 琐碎的一次性问题

## 输出格式

只返回 JSON 数组，每个条目有 name、description、type、content 字段。**没有值得提取的就返回空数组 []**（这是最常见的正确答案）。

- name：kebab-case（如 "user-prefers-python"）
- description：一行概括（决定能否被检索到）
- content：
  - **feedback / project 必须**含三段——1、 事实/规则本身；2、 `**Why:**` 为什么这么要求（往往是踩过的坑）；3、 `**How to apply:**` 什么情况下生效。**写不出真实 Why 的，说明只是一次性意见，不要提取。**
  - **project** 把相对日期转成绝对日期（"周四" → "2026-03-05"）。
  - user / reference 简洁陈述即可。"""


class AutoExtractor:
    """Post-turn memory extraction using a lightweight LLM call.

    Fires asynchronously after each conversation turn completes.
    Non-blocking — failures are logged and silently ignored.
    """

    def __init__(self, memory: MarkdownMemoryStore, llm: LLMClient, session_provider=None):
        self.memory = memory
        self.llm = llm
        # 溯源：返回当前会话 id 的可调用（晚绑定，resume 换 session 后仍准确）
        self._session_provider = session_provider

    def _build_prompt(self, messages: list[dict]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            # content 可能是多模态 list（含图片），经 message_text 取纯文本再截断
            content = message_text(msg)[:2000]
            if content.strip():
                lines.append(f"【{role}】{content}")
        return _EXTRACT_PROMPT.format(
            conversation="\n".join(lines),
            existing_memories=self._existing_memories(),
        )

    def _existing_memories(self) -> str:
        """已有记忆的 name — description 清单，用于写入前去重（闸 1）。"""
        try:
            items = self.memory.list_all()
        except Exception:
            logger.debug("AutoExtract: failed to list existing memories", exc_info=True)
            return "（暂无）"
        lines = [
            f"- {m.get('name', '')} — {m.get('description', '')}".rstrip(" —")
            for m in items
            if m.get("name") or m.get("description")
        ]
        return "\n".join(lines) if lines else "（暂无）"

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

    def _source(self) -> str:
        """溯源标记：auto-extract:<会话id>。"""
        sid = ""
        if self._session_provider:
            try:
                sid = str(self._session_provider() or "")
            except Exception:
                sid = ""
        return f"auto-extract:{sid}" if sid else "auto-extract"

    async def extract(self, messages: list[dict]) -> list[dict]:
        """Extract and save memories from conversation.

        返回已保存条目 [{name, description}, ...]——供 TUI 做写入可见化
        （落库即曝光，人眼是第五道闸），空列表即无写入。
        """
        prompt = self._build_prompt(messages)
        try:
            response = await self.llm.chat(prompt)
        except Exception:
            logger.debug("AutoExtract: LLM call failed", exc_info=True)
            return []

        extracted = self._parse_memories(response)
        saved: list[dict] = []
        for mem in extracted:
            mtype = mem.get("type", "project")
            # 闸 3: feedback/project 必须带真实 Why（一次性意见写不出耐久 Why，挡在门外）
            if mtype in ("feedback", "project") and "why" not in (mem.get("content") or "").lower():
                logger.debug("AutoExtract: dropped %s (%s) — missing Why", mem.get("name"), mtype)
                continue
            # 闸 4: project 类的否定存在断言一律拦下（见 _NEGATIVE_CLAIM_RE 注释）
            if mtype == "project" and _NEGATIVE_CLAIM_RE.search(
                f"{mem.get('name', '')} {mem.get('description', '')} {mem.get('content', '')}"
            ):
                logger.info("AutoExtract: dropped %s — 否定存在断言（闸 4）", mem.get("name"))
                continue
            try:
                final_name = self.memory.save(
                    name=mem["name"],
                    description=mem.get("description", ""),
                    type=mtype,
                    content=mem["content"],
                    source=self._source(),
                )
                # 用 save 返回的名字（查重可能合并改名），可见化提示才指得准
                saved.append({"name": final_name, "description": mem.get("description", "")})
            except Exception:
                logger.debug("AutoExtract: failed to save memory %s", mem.get("name"), exc_info=True)
        return saved
