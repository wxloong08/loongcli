import pytest
from unittest.mock import MagicMock
from loongcli.core.compact import (
    Compactor, KEEP_RECENT_TURNS, SUMMARY_MARKER, SUMMARY_ACK,
    COMPACT_INSTRUCTION, TOOL_RESULT_PLACEHOLDER, BOUNDARY_TEMPLATE,
    _extract_summary, _segment_turns, _replace_tool_results, _fix_role_alternation,
    micro_compact, RECLAIMABLE_TOOLS, MICRO_COMPACT_KEEP_RECENT, CLEARED_PLACEHOLDER,
)


def _make_messages(n: int, with_system: bool = True) -> list[dict]:
    msgs: list[dict] = []
    if with_system:
        msgs.append({"role": "system", "content": "you are helpful"})
    for i in range(n):
        msgs.append({"role": "user", "content": f"question {i}"})
        msgs.append({"role": "assistant", "content": f"answer {i}"})
    return msgs


def _make_turn_messages(with_system: bool = True) -> list[dict]:
    """Create messages with realistic turn structure including tool calls."""
    msgs: list[dict] = []
    if with_system:
        msgs.append({"role": "system", "content": "you are helpful"})
    # Turn 1: user + assistant with tool calls + tool results
    msgs.append({"role": "user", "content": "搜索 Python 教程"})
    msgs.append({"role": "assistant", "content": None, "tool_calls": [
        {"id": "tc1", "function": {"name": "search", "arguments": "{}"}}
    ]})
    msgs.append({"role": "tool", "tool_call_id": "tc1", "content": "x" * 2000})
    msgs.append({"role": "assistant", "content": "找到了以下教程..."})
    # Turn 2: user + assistant
    msgs.append({"role": "user", "content": "读取第一个链接"})
    msgs.append({"role": "assistant", "content": None, "tool_calls": [
        {"id": "tc2", "function": {"name": "url_read", "arguments": "{}"}}
    ]})
    msgs.append({"role": "tool", "tool_call_id": "tc2", "content": "y" * 3000})
    msgs.append({"role": "assistant", "content": "这篇文章讲的是..."})
    # Turn 3: user + assistant
    msgs.append({"role": "user", "content": "总结一下"})
    msgs.append({"role": "assistant", "content": "总结如下..."})
    # Turn 4: user + assistant
    msgs.append({"role": "user", "content": "谢谢"})
    msgs.append({"role": "assistant", "content": "不客气"})
    return msgs


def _mock_llm(summary_text: str = "<summary>这是摘要</summary>") -> MagicMock:
    llm = MagicMock()
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = summary_text
    chunk.choices[0].delta.tool_calls = None
    chunk.choices[0].finish_reason = "stop"
    chunk.usage = None

    async def mock_stream(**kwargs):
        yield chunk

    llm.chat_stream = mock_stream
    return llm


# --- _segment_turns ---

class TestSegmentTurns:
    def test_basic_turns(self):
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        turns = _segment_turns(msgs, 0)
        assert len(turns) == 2
        assert turns[0][0]["content"] == "q1"
        assert turns[1][0]["content"] == "q2"

    def test_turn_includes_tool_messages(self):
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "t1"}]},
            {"role": "tool", "tool_call_id": "t1", "content": "result"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        turns = _segment_turns(msgs, 0)
        assert len(turns) == 2
        assert len(turns[0]) == 4
        assert len(turns[1]) == 2

    def test_start_offset(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        turns = _segment_turns(msgs, 1)
        assert len(turns) == 1
        assert turns[0][0]["role"] == "user"

    def test_empty(self):
        assert _segment_turns([], 0) == []

    def test_single_turn(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        turns = _segment_turns(msgs, 0)
        assert len(turns) == 1


# --- _replace_tool_results ---

class TestReplaceToolResults:
    def test_tool_result_replaced(self):
        msgs = [{"role": "tool", "tool_call_id": "t1", "content": "x" * 5000}]
        result = _replace_tool_results(msgs)
        assert result[0]["content"] == TOOL_RESULT_PLACEHOLDER

    def test_short_tool_result_also_replaced(self):
        msgs = [{"role": "tool", "tool_call_id": "t1", "content": "short"}]
        result = _replace_tool_results(msgs)
        assert result[0]["content"] == TOOL_RESULT_PLACEHOLDER

    def test_non_tool_messages_unchanged(self):
        msgs = [
            {"role": "user", "content": "x" * 2000},
            {"role": "assistant", "content": "y" * 2000},
        ]
        result = _replace_tool_results(msgs)
        assert result[0]["content"] == msgs[0]["content"]
        assert result[1]["content"] == msgs[1]["content"]

    def test_preserves_other_fields(self):
        msgs = [{"role": "tool", "tool_call_id": "t1", "content": "x" * 1000}]
        result = _replace_tool_results(msgs)
        assert result[0]["tool_call_id"] == "t1"
        assert result[0]["role"] == "tool"


# --- _fix_role_alternation ---

class TestFixRoleAlternation:
    def test_no_conflict(self):
        prefix = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "summary"},
            {"role": "assistant", "content": "ack"},
        ]
        kept = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        result = _fix_role_alternation(prefix, kept)
        assert len(result) == 5

    def test_merges_consecutive_assistants(self):
        prefix = [
            {"role": "user", "content": "summary"},
            {"role": "assistant", "content": "ack"},
        ]
        kept = [
            {"role": "assistant", "content": "continued"},
            {"role": "user", "content": "next"},
        ]
        result = _fix_role_alternation(prefix, kept)
        assert result[1]["role"] == "assistant"
        assert "ack" in result[1]["content"]
        assert "continued" in result[1]["content"]
        assert result[2]["role"] == "user"

    def test_merges_consecutive_users(self):
        prefix = [
            {"role": "assistant", "content": "ack"},
        ]
        kept = [
            {"role": "user", "content": "part1"},
            {"role": "user", "content": "part2"},
        ]
        result = _fix_role_alternation(prefix, kept)
        assert result[1]["role"] == "user"
        assert "part1" in result[1]["content"]
        assert "part2" in result[1]["content"]

    def test_tool_messages_pass_through(self):
        prefix = [
            {"role": "user", "content": "summary"},
            {"role": "assistant", "content": "ack"},
        ]
        kept = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "t1"}]},
            {"role": "tool", "tool_call_id": "t1", "content": "result"},
            {"role": "assistant", "content": "done"},
        ]
        result = _fix_role_alternation(prefix, kept)
        merged_assistant = result[1]
        assert merged_assistant["role"] == "assistant"
        assert merged_assistant.get("tool_calls") == [{"id": "t1"}]
        assert result[2]["role"] == "tool"

    def test_preserves_tool_calls_on_merge(self):
        prefix = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "first"},
        ]
        kept = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc"}]},
        ]
        result = _fix_role_alternation(prefix, kept)
        assert len(result) == 2
        assert result[1]["tool_calls"] == [{"id": "tc"}]


# --- Compactor.should_compact ---

class TestShouldCompact:
    def test_under_threshold(self):
        c = Compactor(llm=MagicMock(), threshold=800000)
        msgs = _make_messages(3)
        assert c.should_compact(10000, msgs) is False

    def test_over_threshold_enough_turns(self):
        c = Compactor(llm=MagicMock(), threshold=800000)
        msgs = _make_messages(20)
        assert c.should_compact(900000, msgs) is True

    def test_over_threshold_too_few_turns(self):
        c = Compactor(llm=MagicMock(), threshold=100)
        msgs = _make_messages(1)
        assert c.should_compact(50000, msgs) is False

    def test_zero_tokens(self):
        c = Compactor(llm=MagicMock(), threshold=100)
        msgs = _make_messages(20)
        assert c.should_compact(0, msgs) is False


# --- Compactor.compact ---

class TestCompact:
    async def test_preserves_system_msg(self):
        c = Compactor(llm=_mock_llm(), threshold=100)
        msgs = _make_messages(15)
        result = await c.compact(msgs)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "you are helpful"

    async def test_has_summary_and_ack(self):
        c = Compactor(llm=_mock_llm(), threshold=100)
        msgs = _make_messages(15)
        result = await c.compact(msgs)
        assert SUMMARY_MARKER in result[1]["content"]
        assert result[2]["content"] == SUMMARY_ACK

    async def test_keeps_recent_turns(self):
        c = Compactor(llm=_mock_llm(), threshold=100)
        msgs = _make_messages(15)
        result = await c.compact(msgs)
        # Last KEEP_RECENT_TURNS turns should be preserved
        # Each turn = user + assistant = 2 messages
        last_user_content = result[-2]["content"]
        assert "question 14" in last_user_content

    async def test_short_messages_unchanged(self):
        c = Compactor(llm=MagicMock(), threshold=100)
        msgs = _make_messages(1)
        result = await c.compact(msgs)
        assert result == msgs

    async def test_replaces_tool_results_in_kept_turns(self):
        c = Compactor(llm=_mock_llm(), threshold=100)
        msgs = _make_turn_messages()
        result = await c.compact(msgs)
        for m in result:
            if m["role"] == "tool":
                assert m["content"] == TOOL_RESULT_PLACEHOLDER

    async def test_active_skill_in_instruction(self):
        calls: list[dict] = []

        async def capture_stream(**kwargs):
            calls.append(kwargs)
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "<summary>摘要</summary>"
            chunk.choices[0].delta.tool_calls = None
            chunk.choices[0].finish_reason = "stop"
            chunk.usage = None
            yield chunk

        llm = MagicMock()
        llm.chat_stream = capture_stream
        c = Compactor(llm=llm, threshold=100)

        msgs = _make_messages(15)
        await c.compact(msgs, active_skill="jobhunter")

        sent = calls[0]["messages"]
        assert "jobhunter" in sent[-1]["content"]

    async def test_no_consecutive_roles_in_result(self):
        c = Compactor(llm=_mock_llm(), threshold=100)
        msgs = _make_messages(15)
        result = await c.compact(msgs)
        for i in range(1, len(result)):
            if result[i]["role"] == "tool":
                continue
            if result[i - 1]["role"] == "tool":
                continue
            assert result[i]["role"] != result[i - 1]["role"], (
                f"Consecutive {result[i]['role']} at index {i - 1} and {i}"
            )

    async def test_sends_full_messages_for_cache(self):
        calls: list[dict] = []

        async def capture_stream(**kwargs):
            calls.append(kwargs)
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "<summary>摘要</summary>"
            chunk.choices[0].delta.tool_calls = None
            chunk.choices[0].finish_reason = "stop"
            chunk.usage = None
            yield chunk

        llm = MagicMock()
        llm.chat_stream = capture_stream
        c = Compactor(llm=llm, threshold=100)

        msgs = _make_messages(15)
        await c.compact(msgs)

        sent = calls[0]["messages"]
        assert sent[-1]["role"] == "user"
        assert sent[-1]["content"] == COMPACT_INSTRUCTION
        assert len(sent) == len(msgs) + 1


# --- _extract_summary ---

class TestExtractSummary:
    def test_with_tags(self):
        raw = "<analysis>some analysis</analysis>\n<summary>the summary</summary>"
        assert _extract_summary(raw) == "the summary"

    def test_without_tags(self):
        raw = "just a plain summary"
        assert _extract_summary(raw) == "just a plain summary"

    def test_strips_analysis(self):
        raw = "<analysis>thinking...</analysis>\nthe actual summary"
        assert _extract_summary(raw) == "the actual summary"


# --- Heavy tool result scenario ---

class TestHeavyToolResults:
    async def test_heavy_tool_results_replaced_with_placeholder(self):
        """Simulate the SearXNG scenario: user request + huge tool results."""
        c = Compactor(llm=_mock_llm(), threshold=100)
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(6):
            msgs.append({"role": "user", "content": f"搜索 query {i}"})
            msgs.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": f"tc{i}", "function": {"name": "search", "arguments": "{}"}}
            ]})
            msgs.append({"role": "tool", "tool_call_id": f"tc{i}", "content": "huge " * 2000})
            msgs.append({"role": "assistant", "content": f"结果分析 {i}"})

        result = await c.compact(msgs)
        for m in result:
            if m["role"] == "tool":
                assert m["content"] == TOOL_RESULT_PLACEHOLDER


# --- Compact Boundary ---

class TestCompactBoundary:
    async def test_boundary_marker_present(self):
        c = Compactor(llm=_mock_llm(), threshold=100)
        msgs = _make_messages(15)
        result = await c.compact(msgs, mode="auto", pre_tokens=150000)
        assert "compact-boundary" in result[1]["content"]
        assert "auto" in result[1]["content"]
        assert "150000" in result[1]["content"]

    async def test_boundary_before_summary(self):
        c = Compactor(llm=_mock_llm(), threshold=100)
        msgs = _make_messages(15)
        result = await c.compact(msgs, mode="manual", pre_tokens=0)
        summary_msg = result[1]
        assert "compact-boundary" in summary_msg["content"]
        assert SUMMARY_MARKER in summary_msg["content"]
        idx_boundary = summary_msg["content"].index("compact-boundary")
        idx_summary = summary_msg["content"].index(SUMMARY_MARKER)
        assert idx_boundary < idx_summary

    async def test_boundary_contains_message_count(self):
        c = Compactor(llm=_mock_llm(), threshold=100)
        msgs = _make_messages(15)
        result = await c.compact(msgs, mode="auto", pre_tokens=100000)
        assert f"pre_messages={len(msgs)}" in result[1]["content"]

    async def test_boundary_contains_timestamp(self):
        c = Compactor(llm=_mock_llm(), threshold=100)
        msgs = _make_messages(15)
        result = await c.compact(msgs, mode="auto")
        assert "timestamp=" in result[1]["content"]

    async def test_exit_mode(self):
        c = Compactor(llm=_mock_llm(), threshold=100)
        msgs = _make_messages(15)
        result = await c.compact(msgs, mode="exit")
        assert "mode=exit" in result[1]["content"]


# --- micro_compact ---

class TestMicroCompact:
    def test_preserves_recent_reclaimable(self):
        """Old reclaimable results are cleared, recent ones preserved."""
        msgs = []
        for i in range(8):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": f"tc{i}", "function": {"name": "read_file", "arguments": f'{{"path":"f{i}.py"}}'}}
            ]})
            msgs.append({"role": "tool", "tool_call_id": f"tc{i}", "content": f"content of f{i}"})
            msgs.append({"role": "assistant", "content": f"answer {i}"})
        result = micro_compact(msgs)
        tool_msgs = [m for m in result if m["role"] == "tool"]
        preserved = [m for m in tool_msgs if "content of" in m["content"]]
        cleared = [m for m in tool_msgs if m["content"] == CLEARED_PLACEHOLDER]
        assert len(preserved) == MICRO_COMPACT_KEEP_RECENT
        assert len(cleared) == 3  # 8 - 5 = 3

    def test_never_clears_non_reclaimable(self):
        """Non-reclaimable tool results (delegate, recall, etc.) are never touched."""
        msgs = []
        # 8 delegate calls — all should be preserved
        for i in range(8):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": f"tc{i}", "function": {"name": "delegate", "arguments": "{}"}}
            ]})
            msgs.append({"role": "tool", "tool_call_id": f"tc{i}", "content": f"subtask result {i}"})
            msgs.append({"role": "assistant", "content": f"done {i}"})
        result = micro_compact(msgs)
        tool_msgs = [m for m in result if m["role"] == "tool"]
        assert all("subtask result" in m["content"] for m in tool_msgs)

    def test_mixed_reclaimable_and_non_reclaimable(self):
        """Only reclaimable tools get cleared; non-reclaimable are untouched."""
        msgs = []
        # 6 read_file + 3 recall
        for i in range(6):
            name = "read_file"
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": f"tc{i}", "function": {"name": name, "arguments": "{}"}}
            ]})
            msgs.append({"role": "tool", "tool_call_id": f"tc{i}", "content": f"file {i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        for i in range(3):
            msgs.append({"role": "user", "content": f"recall {i}"})
            msgs.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": f"rc{i}", "function": {"name": "recall", "arguments": "{}"}}
            ]})
            msgs.append({"role": "tool", "tool_call_id": f"rc{i}", "content": f"memory {i}"})
            msgs.append({"role": "assistant", "content": f"noted {i}"})
        result = micro_compact(msgs)
        # 6 read_file - 5 kept = 1 cleared
        cleared = [m for m in result if m.get("role") == "tool" and m["content"] == CLEARED_PLACEHOLDER]
        assert len(cleared) == 1
        # All recall results preserved
        recall_msgs = [m for m in result if m.get("role") == "tool" and "memory" in m["content"]]
        assert len(recall_msgs) == 3

    def test_empty_messages(self):
        assert micro_compact([]) == []

    def test_no_tool_messages(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        assert micro_compact(msgs) == msgs

    def test_fewer_than_keep_recent_unchanged(self):
        """If fewer reclaimable results than KEEP_RECENT, nothing changes."""
        msgs = []
        for i in range(3):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": f"tc{i}", "function": {"name": "grep", "arguments": "{}"}}
            ]})
            msgs.append({"role": "tool", "tool_call_id": f"tc{i}", "content": f"match {i}"})
            msgs.append({"role": "assistant", "content": f"found {i}"})
        result = micro_compact(msgs)
        assert result == msgs
