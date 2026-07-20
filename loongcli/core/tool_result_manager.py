from __future__ import annotations

MAX_SINGLE_RESULT = 8000
MAX_ERROR_RESULT = 16000        # Error messages get double the room
MAX_TOTAL_PER_TURN = 30000
PREVIEW_SIZE = 2000
ERROR_PREFIXES = ("错误：", "⚠")

TRUNCATION_NOTICE = "\n\n[... 已截断，共 {total} 字符。如需完整内容请重新调用工具]"

# 结果不可重放取回的工具，入口不做一刀切截断："重新调用工具"的恢复前提对它们不成立
# （重跑子代理非确定且烧钱；batch 不暴露 task_id；重调 task_status/skill 返回一模一样
# 的截断=死循环）。skill 尤其致命：技能是指令，截成 2000 预览等于按 12% 的工作流执行，
# 且现存 8 个技能超 8K（最大 45.9K）。这些工具各自自管预算（delegate 族
# 按 core.task 预算常量附 trace 指针；skill 按 SKILL_CONTENT_CAP 附文件路径）；仍计入
# turn 总额，保护后续普通工具结果的预算语义。
SELF_BUDGETED_TOOLS = frozenset({"batch_delegate", "task_status", "wait_tasks", "skill"})


class ToolResultManager:
    def __init__(self):
        self._turn_total = 0
        self.truncated_calls: list[dict] = []

    def process(self, tool_name: str, result: str) -> str:
        """Process a tool result, truncating if needed. Returns possibly-truncated result."""
        if tool_name in SELF_BUDGETED_TOOLS:
            self._turn_total += len(result)
            return result
        is_error = result.startswith(ERROR_PREFIXES)
        limit = MAX_ERROR_RESULT if is_error else MAX_SINGLE_RESULT
        if self._turn_total + len(result) > MAX_TOTAL_PER_TURN and len(result) > PREVIEW_SIZE:
            return self._truncate(tool_name, result)
        if len(result) > limit:
            return self._truncate(tool_name, result)
        self._turn_total += len(result)
        return result

    def _truncate(self, tool_name: str, result: str) -> str:
        preview = result[:PREVIEW_SIZE] + TRUNCATION_NOTICE.format(total=len(result))
        self.truncated_calls.append({"tool": tool_name, "original_size": len(result)})
        self._turn_total += len(preview)
        return preview

    def reset_turn(self):
        """Call at the start of each agent loop iteration."""
        self._turn_total = 0
        self.truncated_calls.clear()
