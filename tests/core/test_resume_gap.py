from datetime import datetime, timedelta, timezone

from loongcli.core.resume_gap import (
    RESUME_GAP_ACK,
    RESUME_GAP_MARKER,
    resume_gap_messages,
)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def test_short_gap_same_day_no_marker():
    now = datetime(2026, 8, 19, 15, 0, tzinfo=timezone.utc)
    last = now - timedelta(minutes=20)
    assert resume_gap_messages(_iso(last), now=now) == []


def test_long_gap_emits_marker_pair():
    now = datetime(2026, 8, 19, 15, 0, tzinfo=timezone.utc)
    last = now - timedelta(days=6)
    msgs = resume_gap_messages(_iso(last), now=now)

    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"].startswith(RESUME_GAP_MARKER)
    assert "6 天" in msgs[0]["content"]
    assert "以当时为准" in msgs[0]["content"]
    assert msgs[1] == {"role": "assistant", "content": RESUME_GAP_ACK}


def test_hours_gap_over_threshold():
    now = datetime(2026, 8, 19, 15, 0, tzinfo=timezone.utc)
    msgs = resume_gap_messages(_iso(now - timedelta(hours=5)), now=now)
    assert len(msgs) == 2
    assert "5 小时" in msgs[0]["content"]


def test_date_change_triggers_even_below_threshold():
    """跨日但间隔很短：日历日翻篇本身就会让「今天/昨天」失准，必须报。"""
    # 用本地时区构造，判据是本地日历日（与系统提示的「今天日期」同源）
    now = datetime.now().astimezone().replace(hour=0, minute=30, second=0, microsecond=0)
    last = now - timedelta(hours=1)  # 落到前一天 23:30
    assert last.date() != now.date()

    msgs = resume_gap_messages(_iso(last.astimezone(timezone.utc)), now=now)
    assert len(msgs) == 2
    assert "1 小时" in msgs[0]["content"]


def test_naive_timestamp_treated_as_utc():
    now = datetime(2026, 8, 19, 15, 0, tzinfo=timezone.utc)
    last = datetime(2026, 8, 13, 15, 0)  # 无时区
    msgs = resume_gap_messages(_iso(last), now=now)
    assert len(msgs) == 2
    assert "6 天" in msgs[0]["content"]


def test_missing_or_corrupt_timestamp_is_silent():
    """升级前的旧会话文件没有 updated_at，不该因此崩掉 resume。"""
    now = datetime(2026, 8, 19, 15, 0, tzinfo=timezone.utc)
    assert resume_gap_messages("", now=now) == []
    assert resume_gap_messages("not-a-date", now=now) == []
    assert resume_gap_messages(None, now=now) == []


def test_future_timestamp_is_silent():
    """时钟回拨会让 updated_at 落在未来，没有可信间隔可报。"""
    now = datetime(2026, 8, 19, 15, 0, tzinfo=timezone.utc)
    assert resume_gap_messages(_iso(now + timedelta(days=1)), now=now) == []


def test_marker_survives_raw_resume_system_filter():
    """raw resume 会滤掉 system 消息——标记必须活过这一刀，
    多次 resume 才能在历史上叠出完整时间脉络。"""
    now = datetime(2026, 8, 19, 15, 0, tzinfo=timezone.utc)
    msgs = resume_gap_messages(_iso(now - timedelta(days=2)), now=now)
    kept = [m for m in msgs if m.get("role") != "system"]
    assert kept == msgs


# ── 进程内跨日标记（会话不退出、开过午夜）──

def test_date_rollover_emits_marker_pair():
    from datetime import date, datetime
    from loongcli.core.resume_gap import (
        DATE_ROLLOVER_ACK, DATE_ROLLOVER_MARKER, date_rollover_messages,
    )
    msgs = date_rollover_messages(date(2026, 11, 10), datetime(2026, 11, 11, 1, 20))
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    text = msgs[0]["content"]
    assert text.startswith(DATE_ROLLOVER_MARKER)
    assert "2026-11-11 01:20" in text and "星期三" in text  # 新日期 + 星期
    assert "2026-11-10" in text                               # 点名过期的那个快照
    assert msgs[1]["content"] == DATE_ROLLOVER_ACK


def test_date_rollover_silent_when_not_needed():
    from datetime import date, datetime
    from loongcli.core.resume_gap import date_rollover_messages
    now = datetime(2026, 11, 11, 1, 20)
    assert date_rollover_messages(None, now) == []                 # 本进程第一回合
    assert date_rollover_messages(date(2026, 11, 11), now) == []   # 同一天
    assert date_rollover_messages(date(2026, 11, 12), now) == []   # 时钟回拨：闭嘴


def test_date_rollover_marker_survives_raw_resume_system_filter():
    """与 resume 标记同理：用 user/assistant 承载，raw resume 滤掉 system 后仍在。"""
    from datetime import date, datetime
    from loongcli.core.resume_gap import date_rollover_messages
    msgs = date_rollover_messages(date(2026, 11, 10), datetime(2026, 11, 11, 9, 0))
    assert [m for m in msgs if m["role"] != "system"] == msgs
