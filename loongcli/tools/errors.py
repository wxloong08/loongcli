from __future__ import annotations


class ToolError(Exception):
    """Raised by tools for errors the Agent layer can act on.

    Attributes:
        message: Human-readable error message (shown to LLM after retries exhausted).
        retryable: Whether the agent should retry the tool call once.
        retry_after: Seconds to wait before retry.
    """

    def __init__(self, message: str, retryable: bool = True, retry_after: float = 0.5):
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.retry_after = retry_after
