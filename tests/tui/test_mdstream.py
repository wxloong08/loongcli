import io
import re
from unittest.mock import MagicMock

from rich.console import Console

from loongcli.tui.mdstream import MarkdownStream, StreamView, LeftMarkdown


def test_left_markdown_headings_left_aligned():
    """LeftMarkdown 的 h1 应左对齐（默认 Rich 会居中，观感很怪）。"""
    buf = io.StringIO()
    Console(file=buf, width=40, color_system=None).print(LeftMarkdown("# 标题"))
    line = next(l for l in buf.getvalue().splitlines() if "标题" in l)
    assert line.index("标题") <= 1  # 左对齐，非居中（居中会有约 18 个前导空格）


def _make_md(render_sequences, live_window=3):
    """MarkdownStream with throttle off, a fake Live, and a scripted renderer.

    `render_sequences` is the list of line-lists `_render_to_lines` returns on
    successive calls. `_emit_stable` is recorded instead of printing.
    """
    md = MarkdownStream(MagicMock(), left_pad=2, live_window=live_window)
    md.min_delay = 0
    md._live = MagicMock()
    md._ensure_live = lambda: None
    emitted: list[list[str]] = []
    md._emit_stable = lambda lines: emitted.append(list(lines))
    tail_updates: list[list[str]] = []
    # record what the live tail is set to (the rest lines)
    orig_pad = md._pad
    md._pad = lambda text: text  # bypass Padding for easy inspection
    it = iter(render_sequences)
    md._render_to_lines = lambda text: next(it)
    return md, emitted


def _drive(md, n_nonfinal):
    """Call update n_nonfinal times (non-final) then once final, defeating the
    adaptive throttle by resetting `when` before each call so every call renders.
    """
    for _ in range(n_nonfinal):
        md.when = 0.0
        md.update("x")
    md.when = 0.0
    md.update("x", final=True)


class TestStableWindowSplit:
    def test_each_line_emitted_exactly_once_in_order(self):
        seqs = [
            ["L1\n", "L2\n"],                              # call1 (non-final)
            ["L1\n", "L2\n", "L3\n", "L4\n"],              # call2
            ["L1\n", "L2\n", "L3\n", "L4\n", "L5\n", "L6\n"],  # call3
            ["L1\n", "L2\n", "L3\n", "L4\n", "L5\n", "L6\n"],  # final
        ]
        md, emitted = _make_md(seqs, live_window=3)
        _drive(md, n_nonfinal=3)

        flat = [ln for block in emitted for ln in block]
        assert flat == ["L1\n", "L2\n", "L3\n", "L4\n", "L5\n", "L6\n"]
        assert len(flat) == len(set(flat))  # no duplicates

    def test_fewer_lines_than_window_no_loss(self):
        # total lines < live_window: nothing is "stable" until final, and final
        # must emit ALL lines (regression for the negative-index drop bug).
        seqs = [
            ["A\n", "B\n"],   # non-final: num_lines = 2-3 = -1 -> no stable
            ["A\n", "B\n"],   # final: emit everything
        ]
        md, emitted = _make_md(seqs, live_window=3)
        _drive(md, n_nonfinal=1)

        flat = [ln for block in emitted for ln in block]
        assert flat == ["A\n", "B\n"]

    def test_no_stable_emitted_before_window_fills(self):
        seqs = [["L1\n", "L2\n", "L3\n"]]  # exactly window size, non-final
        md, emitted = _make_md(seqs, live_window=3)
        md.update("x")  # num_lines = 3-3 = 0 -> nothing stable yet
        assert emitted == []

    def test_printed_pointer_monotonic(self):
        seqs = [
            ["a\n", "b\n", "c\n", "d\n"],          # emit a
            ["a\n", "b\n", "c\n", "d\n", "e\n"],   # emit b
            ["a\n", "b\n", "c\n", "d\n", "e\n"],   # final: emit c,d,e
        ]
        md, emitted = _make_md(seqs, live_window=3)
        _drive(md, n_nonfinal=2)
        flat = [ln for block in emitted for ln in block]
        assert flat == ["a\n", "b\n", "c\n", "d\n", "e\n"]


class TestThrottle:
    def test_throttle_skips_rapid_nonfinal_calls(self):
        md = MarkdownStream(MagicMock(), live_window=3)
        md._live = MagicMock()
        md._ensure_live = lambda: None
        md._pad = lambda t: t
        calls = []
        md._render_to_lines = lambda text: calls.append(text) or ["x\n"]
        md.min_delay = 999  # effectively block non-final updates
        md.when = __import__("time").time()
        md.update("a")  # should be throttled out
        assert calls == []
        md.update("a", final=True)  # final always renders
        assert calls == ["a"]


class TestRealRenderSmoke:
    def test_streams_markdown_without_crashing(self):
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=True, width=60)
        md = MarkdownStream(console, left_pad=2, live_window=3)
        md.min_delay = 0
        text = "# Title\n\nSome **bold** intro.\n\n- item one\n- item two\n\nEnd para.\n"
        for i in range(5, len(text), 7):
            md.update(text[:i])
        md.update(text, final=True)
        out = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", buf.getvalue())  # strip ANSI
        assert "Title" in out
        assert "item one" in out
        assert "End para" in out


def _real_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=True, width=60)


class TestStreamViewLifecycle:
    def test_only_one_live_active_text_then_status(self):
        from rich.text import Text
        view = StreamView(_real_console(), left_pad=2)
        # feed text -> creates md stream
        view.append_text("hello")
        assert view._md is not None and view._spinner is None
        # status -> flushes text (md gone) and starts spinner
        view.status(Text("working"))
        assert view._md is None and view._spinner is not None
        # back to text -> stops spinner, new md
        view.append_text("more")
        assert view._spinner is None and view._md is not None
        view.close()
        assert view._md is None and view._spinner is None

    def test_close_is_idempotent(self):
        view = StreamView(_real_console(), left_pad=2)
        view.close()
        view.close()  # no crash
