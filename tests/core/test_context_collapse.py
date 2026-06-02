import pytest
from loongcli.core.context_collapse import (
    should_collapse, collapse,
    COLLAPSE_THRESHOLD_RATIO, COLLAPSE_AGGRESSIVE_RATIO,
    LONG_ASSISTANT_THRESHOLD, LONG_ASSISTANT_KEEP,
    COLLAPSED_NOTICE,
)
from loongcli.core.compact import CLEARED_PLACEHOLDER


class TestShouldCollapse:
    def test_below_threshold(self):
        assert should_collapse(80000, 100000) == 0

    def test_at_level1(self):
        assert should_collapse(91000, 100000) == 1

    def test_at_level2(self):
        assert should_collapse(96000, 100000) == 2

    def test_zero_max(self):
        assert should_collapse(1000, 0) == 0

    def test_zero_prompt(self):
        assert should_collapse(0, 100000) == 0

    def test_exact_threshold(self):
        assert should_collapse(90000, 100000) == 1

    def test_exact_aggressive(self):
        assert should_collapse(95000, 100000) == 2


class TestCollapseLevel0:
    def test_returns_same(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        assert collapse(msgs, 0) is msgs


class TestCollapseLevel1:
    def test_clears_old_reclaimable_tool_results(self):
        msgs = []
        for i in range(8):
            msgs.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": f"tc{i}", "function": {"name": "read_file", "arguments": f'{{"path":"f{i}.py"}}'}}
            ]})
            msgs.append({"role": "tool", "tool_call_id": f"tc{i}", "content": f"content of f{i}"})

        result = collapse(msgs, 1, keep_recent=5)
        tool_msgs = [m for m in result if m["role"] == "tool"]
        cleared = [m for m in tool_msgs if m["content"] == CLEARED_PLACEHOLDER]
        preserved = [m for m in tool_msgs if "content of" in m["content"]]
        assert len(cleared) == 3  # 8 - 5
        assert len(preserved) == 5

    def test_preserves_non_reclaimable(self):
        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "function": {"name": "delegate", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "subtask result"},
        ]
        result = collapse(msgs, 1)
        assert result[1]["content"] == "subtask result"

    def test_does_not_modify_originals(self):
        msgs = []
        for i in range(8):
            msgs.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": f"tc{i}", "function": {"name": "read_file", "arguments": "{}"}}
            ]})
            msgs.append({"role": "tool", "tool_call_id": f"tc{i}", "content": f"original {i}"})

        collapse(msgs, 1, keep_recent=3)
        # Original messages should be unchanged
        for i in range(8):
            assert msgs[i * 2 + 1]["content"] == f"original {i}"


class TestCollapseLevel2:
    def test_truncates_long_assistant_messages(self):
        msgs = []
        for i in range(8):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": "x" * (LONG_ASSISTANT_THRESHOLD + 100)})

        result = collapse(msgs, 2, keep_recent=3)
        # Recent 3 long assistants preserved, older ones truncated
        truncated = [m for m in result if m["role"] == "assistant" and "已压缩" in m.get("content", "")]
        assert len(truncated) == 5  # 8 - 3

    def test_short_assistant_not_truncated(self):
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "short answer"},
        ]
        result = collapse(msgs, 2)
        assert result[1]["content"] == "short answer"

    def test_includes_level1_clearing(self):
        msgs = []
        # 8 read_file calls
        for i in range(8):
            msgs.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": f"tc{i}", "function": {"name": "read_file", "arguments": "{}"}}
            ]})
            msgs.append({"role": "tool", "tool_call_id": f"tc{i}", "content": f"content {i}"})

        result = collapse(msgs, 2, keep_recent=5)
        cleared = [m for m in result if m["role"] == "tool" and m["content"] == CLEARED_PLACEHOLDER]
        assert len(cleared) == 3  # Level 1 still applies
