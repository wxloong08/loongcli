import pytest
from datetime import datetime, timezone, timedelta

from loongcli.tui.session_picker import (
    PickerState, render_picker,
    _relative_time, _format_size,
)


def _make_session(sid: str, title: str, minutes_ago: int = 0, turns: int = 3, size: int = 1024):
    dt = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return {
        "session_id": sid,
        "title": title,
        "updated_at": dt.isoformat(),
        "turn_count": turns,
        "file_size": size,
    }


class TestRelativeTime:
    def test_just_now(self):
        dt = datetime.now(timezone.utc).isoformat()
        assert _relative_time(dt) == "刚刚"

    def test_minutes(self):
        dt = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
        assert "15 分钟前" == _relative_time(dt)

    def test_hours(self):
        dt = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        assert "3 小时前" == _relative_time(dt)

    def test_days(self):
        dt = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        assert "5 天前" == _relative_time(dt)

    def test_months(self):
        dt = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        assert "2 个月前" == _relative_time(dt)

    def test_empty(self):
        assert _relative_time("") == "?"

    def test_invalid(self):
        assert _relative_time("not-a-date") == "?"


class TestFormatSize:
    def test_bytes(self):
        assert _format_size(512) == "512B"

    def test_kb(self):
        assert _format_size(1536) == "1.5KB"

    def test_mb(self):
        assert _format_size(1572864) == "1.5MB"


class TestPickerState:
    def _make_state(self, count=5) -> PickerState:
        sessions = [_make_session(f"s{i}", f"Session {i}", minutes_ago=i * 10) for i in range(count)]
        return PickerState(sessions=sessions, page_size=3)

    def test_move_down(self):
        state = self._make_state()
        assert state.cursor == 0
        state.move(1)
        assert state.cursor == 1

    def test_move_up_at_top(self):
        state = self._make_state()
        state.move(-1)
        assert state.cursor == 0

    def test_move_down_at_bottom(self):
        state = self._make_state(3)
        state.move(1)
        state.move(1)
        state.move(1)
        assert state.cursor == 2

    def test_scroll_offset_follows_cursor(self):
        state = self._make_state(10)
        for _ in range(5):
            state.move(1)
        assert state.cursor == 5
        assert state.scroll_offset == 3

    def test_scroll_offset_up(self):
        state = self._make_state(10)
        state.cursor = 5
        state.scroll_offset = 5
        state.move(-1)
        assert state.cursor == 4
        assert state.scroll_offset == 4

    def test_selected(self):
        state = self._make_state()
        state.move(2)
        sel = state.selected()
        assert sel["session_id"] == "s2"

    def test_selected_empty(self):
        state = PickerState()
        assert state.selected() is None

    def test_filter(self):
        state = self._make_state()
        state.search = "session 3"
        items = state.filtered
        assert len(items) == 1
        assert items[0]["title"] == "Session 3"

    def test_filter_case_insensitive(self):
        state = self._make_state()
        state.search = "SESSION"
        assert len(state.filtered) == 5

    def test_filter_no_match(self):
        state = self._make_state()
        state.search = "zzzzz"
        assert len(state.filtered) == 0

    def test_reset_cursor(self):
        state = self._make_state()
        state.cursor = 3
        state.scroll_offset = 2
        state.preview_id = "s1"
        state.reset_cursor()
        assert state.cursor == 0
        assert state.scroll_offset == 0
        assert state.preview_id is None

    def test_page_move(self):
        state = self._make_state(20)
        state.page_move(1)
        assert state.cursor == 3

    def test_page_move_clamps(self):
        state = self._make_state(5)
        state.page_move(1)
        assert state.cursor == 3
        state.page_move(1)
        assert state.cursor == 4


class TestRenderPicker:
    def test_render_basic(self):
        state = PickerState(
            sessions=[_make_session("abc123", "Test session", minutes_ago=5)],
        )
        panel = render_picker(state)
        assert panel.title is not None

    def test_render_empty_search(self):
        state = PickerState(
            sessions=[_make_session("a", "Hello")],
        )
        state.search = "zzz"
        panel = render_picker(state)
        text = str(panel.renderable)
        assert "没有匹配" in text

    def test_render_selected_marker(self):
        state = PickerState(
            sessions=[
                _make_session("a", "First"),
                _make_session("b", "Second"),
            ],
        )
        state.cursor = 1
        text = str(render_picker(state).renderable)
        assert ">" in text

    def test_render_more_indicator(self):
        sessions = [_make_session(f"s{i}", f"S{i}") for i in range(20)]
        state = PickerState(sessions=sessions, page_size=3)
        text = str(render_picker(state).renderable)
        assert "更多" in text

    def test_render_search_display(self):
        state = PickerState(
            sessions=[_make_session("a", "Hello world")],
        )
        state.search = "hello"
        text = str(render_picker(state).renderable)
        assert "hello" in text

    def test_render_count_with_filter(self):
        state = PickerState(
            sessions=[
                _make_session("a", "Alpha"),
                _make_session("b", "Beta"),
            ],
        )
        state.search = "alpha"
        panel = render_picker(state)
        assert "1/2" in panel.title

    def test_render_multiline_title_flattened(self):
        state = PickerState(
            sessions=[_make_session("a", "Line1\nLine2\nLine3")],
        )
        text = str(render_picker(state).renderable)
        assert "\nLine2" not in text
        assert "Line1 Line2 Line3" in text
