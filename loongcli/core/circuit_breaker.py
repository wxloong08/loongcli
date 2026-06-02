from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILURES = 3


class CompactCircuitBreaker:
    def __init__(self, max_failures: int = MAX_CONSECUTIVE_FAILURES):
        self._failures = 0
        self._max = max_failures

    @property
    def is_open(self) -> bool:
        return self._failures >= self._max

    def record_success(self):
        self._failures = 0

    def record_failure(self):
        self._failures += 1
        if self.is_open:
            logger.warning("Compact circuit breaker opened after %d consecutive failures", self._failures)
