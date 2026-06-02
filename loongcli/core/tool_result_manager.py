from __future__ import annotations

MAX_SINGLE_RESULT = 8000        # Single tool result max chars
MAX_TOTAL_PER_TURN = 30000      # All tool results in one iteration max chars
PREVIEW_SIZE = 2000             # Chars kept when truncating

TRUNCATION_NOTICE = "\n\n[... 已截断，共 {total} 字符。如需完整内容请重新调用工具]"


class ToolResultManager:
    def __init__(self):
        self._turn_total = 0
        self.truncated_calls: list[dict] = []

    def process(self, tool_name: str, result: str) -> str:
        """Process a tool result, truncating if needed. Returns possibly-truncated result."""
        if self._turn_total + len(result) > MAX_TOTAL_PER_TURN and len(result) > PREVIEW_SIZE:
            return self._truncate(tool_name, result)
        if len(result) > MAX_SINGLE_RESULT:
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
