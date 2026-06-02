from loongcli.core.circuit_breaker import CompactCircuitBreaker, MAX_CONSECUTIVE_FAILURES


class TestCompactCircuitBreaker:
    def test_starts_closed(self):
        cb = CompactCircuitBreaker()
        assert not cb.is_open

    def test_opens_after_max_failures(self):
        cb = CompactCircuitBreaker()
        for _ in range(MAX_CONSECUTIVE_FAILURES):
            cb.record_failure()
        assert cb.is_open

    def test_one_fewer_than_max_stays_closed(self):
        cb = CompactCircuitBreaker()
        for _ in range(MAX_CONSECUTIVE_FAILURES - 1):
            cb.record_failure()
        assert not cb.is_open

    def test_success_resets(self):
        cb = CompactCircuitBreaker()
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert not cb.is_open
        # Need full MAX_CONSECUTIVE_FAILURES again to open
        for _ in range(MAX_CONSECUTIVE_FAILURES - 1):
            cb.record_failure()
        assert not cb.is_open

    def test_custom_max(self):
        cb = CompactCircuitBreaker(max_failures=1)
        cb.record_failure()
        assert cb.is_open

    def test_success_after_open_resets(self):
        cb = CompactCircuitBreaker(max_failures=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open
        cb.record_success()
        assert not cb.is_open
