from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from loongcli.plan.store import PlanStore
    from loongcli.core.task import TaskManager

POST_COMPACT_MAX_FILES = 5
POST_COMPACT_MAX_CHARS_PER_FILE = 5000
POST_COMPACT_CHAR_BUDGET = 50000

ATTACHMENT_MARKER = "[压缩后上下文恢复]"
ATTACHMENT_ACK = "好的，我已了解恢复的上下文信息，继续工作。"


def extract_recent_files(messages: list[dict]) -> list[str]:
    """Extract recently read file paths from messages, most recent first, deduplicated."""
    seen: set[str] = set()
    paths: list[str] = []
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            func = tc.get("function", {})
            if func.get("name") != "read_file":
                continue
            try:
                args = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                continue
            path = args.get("path", "")
            if path and path not in seen:
                seen.add(path)
                paths.append(path)
            if len(paths) >= POST_COMPACT_MAX_FILES:
                return paths
    return paths


def restore_files(paths: list[str]) -> str:
    if not paths:
        return ""
    sections: list[str] = []
    budget = POST_COMPACT_CHAR_BUDGET
    for p in paths:
        try:
            fp = Path(p)
            if not fp.exists() or not fp.is_file():
                continue
            content = fp.read_text(encoding="utf-8")
            if len(content) > POST_COMPACT_MAX_CHARS_PER_FILE:
                content = content[:POST_COMPACT_MAX_CHARS_PER_FILE] + "\n[... 文件已截断]"
            if budget - len(content) < 0:
                break
            budget -= len(content)
            sections.append(f"### {p}\n```\n{content}\n```")
        except Exception:
            logger.debug("Failed to restore file %s", p, exc_info=True)
            continue
    if not sections:
        return ""
    return "## 最近文件\n" + "\n\n".join(sections)


MAX_SKILL_REINJECT_CHARS = 8000


def skill_section(skill_registry, active_skill: str | None) -> str:
    """压缩后从磁盘重读活跃技能原文。

    技能指令是契约类内容：被摘要成梗概会导致后续执行悄悄偏离（静默腐败），
    且模型不知道自己忘了什么、不会主动找回——必须结构性保活，不能依赖检索。
    """
    if not skill_registry or not active_skill:
        return ""
    content = skill_registry.load_content(active_skill)
    if not content:
        return ""
    if len(content) > MAX_SKILL_REINJECT_CHARS:
        content = content[:MAX_SKILL_REINJECT_CHARS] + "\n[... 技能原文已截断]"
    # 头部必须用 "## 技能: <name>" 规范格式——_detect_active_skill 靠它在下一次
    # 压缩时识别活跃技能。换成别的写法会导致重注入只存活一代（已踩过）。
    return (
        f"## 技能: {active_skill}\n"
        f"（活跃技能原文已重新挂载，以此为准，摘要中的梗概仅供参考）\n"
        f"{content}"
    )


def plan_status(plan_store) -> str:
    if not plan_store:
        return ""
    text = plan_store.format_for_prompt(max_chars=2000)
    if not text:
        return ""
    return f"## 计划进度\n{text}"


def task_status(task_manager) -> str:
    if not task_manager:
        return ""
    from loongcli.core.task import TaskStatus
    running = [t for t in task_manager._tasks.values() if t.status == TaskStatus.RUNNING]
    completed_recent = [t for t in task_manager._tasks.values() if t.status == TaskStatus.COMPLETED][-3:]
    if not running and not completed_recent:
        return ""
    lines = ["## 子任务状态"]
    for t in running:
        lines.append(f"- [运行中] {t.id}: {t.prompt[:80]}")
    for t in completed_recent:
        result_text = t.result or ""
        result_preview = result_text[:120] + "..." if len(result_text) > 120 else result_text
        lines.append(f"- [已完成] {t.id}: {result_preview}")
    return "\n".join(lines)


def build_attachments(
    messages: list[dict],
    plan_store: PlanStore | None = None,
    task_manager: TaskManager | None = None,
    skill_registry=None,
    active_skill: str | None = None,
) -> list[dict]:
    sections: list[str] = []

    skill_text = skill_section(skill_registry, active_skill)
    if skill_text:
        sections.append(skill_text)

    file_paths = extract_recent_files(messages)
    file_section = restore_files(file_paths)
    if file_section:
        sections.append(file_section)

    plan_section = plan_status(plan_store)
    if plan_section:
        sections.append(plan_section)

    task_section = task_status(task_manager)
    if task_section:
        sections.append(task_section)

    if not sections:
        return []

    content = f"{ATTACHMENT_MARKER}\n\n" + "\n\n".join(sections)
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": ATTACHMENT_ACK},
    ]
