import pytest
from loongcli.core.tool_result_manager import (
    ToolResultManager,
    MAX_SINGLE_RESULT,
    MAX_TOTAL_PER_TURN,
    PREVIEW_SIZE,
    SELF_BUDGETED_TOOLS,
    TRUNCATION_NOTICE,
)


def test_short_result_passes_through_unchanged():
    mgr = ToolResultManager()
    text = "hello world"
    result = mgr.process("echo", text)
    assert result == text
    assert mgr.truncated_calls == []


def test_large_result_gets_truncated():
    mgr = ToolResultManager()
    large = "x" * (MAX_SINGLE_RESULT + 1000)
    result = mgr.process("read_file", large)

    assert result.startswith("x" * PREVIEW_SIZE)
    assert "已截断" in result
    assert str(len(large)) in result
    assert len(result) < len(large)


def test_truncated_result_starts_with_preview_size_chars():
    mgr = ToolResultManager()
    large = "A" * 500 + "B" * (MAX_SINGLE_RESULT + 500)
    result = mgr.process("shell", large)

    expected_preview = large[:PREVIEW_SIZE]
    notice = TRUNCATION_NOTICE.format(total=len(large))
    assert result == expected_preview + notice


def test_cumulative_limit_triggers_truncation():
    mgr = ToolResultManager()
    # Directly set the turn total close to the budget limit so the next result
    # triggers the cumulative guard even though it's under MAX_SINGLE_RESULT.
    mgr._turn_total = MAX_TOTAL_PER_TURN - 100

    # This result is under the single-result limit but pushes past the turn budget,
    # and is longer than PREVIEW_SIZE so it qualifies for truncation.
    next_result = "z" * (PREVIEW_SIZE + 500)
    result = mgr.process("overflow_tool", next_result)

    assert "已截断" in result
    assert len(mgr.truncated_calls) == 1
    assert mgr.truncated_calls[0]["tool"] == "overflow_tool"


def test_reset_turn_clears_cumulative_counter():
    mgr = ToolResultManager()
    # Fill up the budget
    mgr._turn_total = MAX_TOTAL_PER_TURN - 100

    mgr.reset_turn()

    assert mgr._turn_total == 0
    assert mgr.truncated_calls == []

    # After reset, a large-but-under-single-limit result should pass through
    text = "a" * (PREVIEW_SIZE + 500)
    result = mgr.process("tool", text)
    assert result == text


def test_truncated_calls_tracks_tool_info():
    mgr = ToolResultManager()
    large1 = "x" * (MAX_SINGLE_RESULT + 100)
    large2 = "y" * (MAX_SINGLE_RESULT + 200)

    mgr.process("tool_a", large1)
    mgr.process("tool_b", large2)

    assert len(mgr.truncated_calls) == 2
    assert mgr.truncated_calls[0] == {"tool": "tool_a", "original_size": len(large1)}
    assert mgr.truncated_calls[1] == {"tool": "tool_b", "original_size": len(large2)}


def test_non_string_result_not_processed_by_manager():
    """ToolResultManager.process expects strings. The agent guards with isinstance."""
    mgr = ToolResultManager()
    # Just verify process works on strings; non-strings are guarded by the agent.
    result = mgr.process("tool", "short")
    assert result == "short"


def test_exactly_at_single_limit_not_truncated():
    mgr = ToolResultManager()
    text = "a" * MAX_SINGLE_RESULT
    result = mgr.process("tool", text)
    assert result == text
    assert mgr.truncated_calls == []


def test_one_over_single_limit_is_truncated():
    mgr = ToolResultManager()
    text = "a" * (MAX_SINGLE_RESULT + 1)
    result = mgr.process("tool", text)
    assert "已截断" in result
    assert len(mgr.truncated_calls) == 1


def test_error_result_uses_higher_limit():
    """Error messages get 16KB limit instead of 8KB."""
    mgr = ToolResultManager()
    error_text = "错误：" + "x" * (MAX_SINGLE_RESULT + 500)
    result = mgr.process("read_file", error_text)
    assert result == error_text  # within 16KB, not truncated


def test_error_result_still_truncated_when_too_large():
    """Error messages beyond 16KB still get truncated."""
    from loongcli.core.tool_result_manager import MAX_ERROR_RESULT
    mgr = ToolResultManager()
    error_text = "⚠ " + "y" * (MAX_ERROR_RESULT + 500)
    result = mgr.process("shell", error_text)
    assert "已截断" in result
    assert len(result) < len(error_text)


@pytest.mark.parametrize("tool_name", sorted(SELF_BUDGETED_TOOLS))
def test_self_budgeted_tools_exempt_from_truncation(tool_name):
    """delegate 类结果不可重放，入口不截断——预算由工具自身管理（真实事故：
    7 路子代理 37K 合成结果被截成 2K，合成层近盲）。"""
    mgr = ToolResultManager()
    big = "x" * (MAX_SINGLE_RESULT * 5)
    result = mgr.process(tool_name, big)
    assert result == big
    assert mgr.truncated_calls == []


def test_self_budgeted_result_still_counts_toward_turn_total():
    """豁免不等于隐身：仍计入 turn 总额，后续普通工具结果照常被总额闸压缩。"""
    mgr = ToolResultManager()
    big = "x" * (MAX_TOTAL_PER_TURN + 1000)
    mgr.process("batch_delegate", big)
    assert mgr._turn_total == len(big)

    normal = "y" * (PREVIEW_SIZE + 500)
    result = mgr.process("read_file", normal)
    assert "已截断" in result
