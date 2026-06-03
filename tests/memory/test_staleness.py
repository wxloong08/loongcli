from datetime import datetime, timezone, timedelta
from loongcli.memory.recall_engine import _staleness_caveat


def test_fresh_memory_no_caveat():
    now = datetime.now(timezone.utc).isoformat()
    assert _staleness_caveat(now) == ""


def test_stale_memory_has_caveat():
    old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    caveat = _staleness_caveat(old)
    assert "10" in caveat
    assert "过时" in caveat


def test_7_day_threshold():
    exactly_7 = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    assert _staleness_caveat(exactly_7) != ""

    within_7 = (datetime.now(timezone.utc) - timedelta(days=6)).isoformat()
    assert _staleness_caveat(within_7) == ""


def test_empty_string_no_caveat():
    assert _staleness_caveat("") == ""


def test_invalid_date_no_caveat():
    assert _staleness_caveat("not-a-date") == ""
