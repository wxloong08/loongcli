from __future__ import annotations

MAX_SINGLE_RESULT = 8000
MAX_ERROR_RESULT = 16000        # Error messages get double the room
MAX_TOTAL_PER_TURN = 30000
PREVIEW_SIZE = 2000
ERROR_PREFIXES = ("错误：", "⚠")

TRUNCATION_NOTICE = "\n\n[... 已截断，共 {total} 字符。如需完整内容请重新调用工具]"


class ToolResultManager:
    def __init__(self):
        self._turn_total = 0
        self.truncated_calls: list[dict] = []

    def process(self, tool_name: str, result: str) -> str:
        """Process a tool result, truncating if needed. Returns possibly-truncated result."""
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
