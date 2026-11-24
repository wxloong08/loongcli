"""工具结果保险丝：单个结果超过 HARD_CAP 才全文落盘 + 头尾预览，所有工具一个规则，没有豁免名单。"""
import pytest

from loongcli.core.tool_result_manager import (
    HARD_CAP,
    PREVIEW_HEAD,
    PREVIEW_TAIL,
    ToolResultManager,
)


def test_short_result_passes_through_unchanged():
    assert ToolResultManager().process("echo", "hello world") == "hello world"


def test_exactly_at_cap_not_truncated():
    text = "x" * HARD_CAP
    assert ToolResultManager().process("tool", text) == text


def test_one_over_cap_is_truncated():
    result = ToolResultManager().process("tool", "x" * (HARD_CAP + 1))
    assert "已截断" in result


def test_truncated_result_keeps_head_and_tail_with_explicit_marker():
    head, tail = "H" * PREVIEW_HEAD, "T" * PREVIEW_TAIL
    full = head + "M" * HARD_CAP + tail
    result = ToolResultManager().process("tool", full)
    assert result.startswith(head) and result.endswith(tail)
    assert str(len(full)) in result                                   # 原文总字数
    assert str(len(full) - PREVIEW_HEAD - PREVIEW_TAIL) in result     # 省略字数
    assert len(result) < 6000


@pytest.mark.parametrize("size", [8_871, 11_209, 13_620, 39_000])
def test_normal_sized_results_no_longer_cut(size):
    """回归：旧设计在 8000 处截断，历史 48 次截断全是 8K–40K 的正常结果（read_file、搜索、shell）。"""
    text = "y" * size
    assert ToolResultManager().process("searxng__web_search", text) == text


def test_no_per_response_cumulative_cap():
    """累计闸已删（152 个会话从未触发）：连续多个大结果各自只看单条上限。"""
    mgr = ToolResultManager()
    for _ in range(5):
        assert "已截断" not in mgr.process("read_file", "z" * 40_000)


@pytest.mark.parametrize("tool_name", ["shell", "read_file", "delegate", "skill", "searxng__url_read", "grep"])
def test_every_tool_spills_the_same_way(tool_name, tmp_path):
    """一个规则管所有工具：超限一律全文落盘 + 指针，没有按工具名分支，也没有「重新调用」。"""
    full = "A" * PREVIEW_HEAD + "M" * HARD_CAP + "Z" * PREVIEW_TAIL
    result = ToolResultManager(spill_dir=tmp_path).process(tool_name, full)
    spilled = list(tmp_path.glob("*.txt"))
    assert len(spilled) == 1
    assert spilled[0].read_text(encoding="utf-8") == full            # 落的是完整原文
    assert str(spilled[0]) in result and "read_file" in result
    assert "grep" in result                                           # 长日志找失败点，搜比分页读快
    assert "不要重跑" in result
    assert "重新调用" not in result


def test_small_result_not_spilled(tmp_path):
    ToolResultManager(spill_dir=tmp_path).process("shell", "ok\n" * 100)
    assert list(tmp_path.glob("*.txt")) == []


def test_no_spill_dir_falls_back_to_preview_and_says_so():
    result = ToolResultManager().process("shell", "x" * (HARD_CAP + 500))
    assert "已截断" in result and "未能落盘" in result
    assert "read_file 的 offset" not in result


def test_spill_write_failure_falls_back(tmp_path):
    """落盘失败（目录路径被文件占位，mkdir 抛错）→ 回落纯头尾预览，不崩。"""
    blocker = tmp_path / "blocked"
    blocker.write_text("i am a file, not a dir", encoding="utf-8")
    result = ToolResultManager(spill_dir=blocker).process("shell", "x" * (HARD_CAP + 500))
    assert "已截断" in result and "未能落盘" in result


def test_spill_filename_sanitizes_tool_name(tmp_path):
    """工具名进文件名前清洗：MCP 工具名来自外部服务器，不能让它带路径分隔符落盘。"""
    ToolResultManager(spill_dir=tmp_path).process("evil/../..\\name", "x" * (HARD_CAP + 500))
    spilled = list(tmp_path.rglob("*.txt"))
    assert len(spilled) == 1 and spilled[0].parent == tmp_path


def test_cap_sits_above_every_self_budget():
    """保险丝必须高于各工具自管上限，否则正常结果会被它截——那就又回到「中央替工具决定返回量」。"""
    from loongcli.core.task import BATCH_RESULT_BUDGET, SINGLE_TASK_RESULT_CAP
    from loongcli.tools.read_file import READ_CHAR_BUDGET
    from loongcli.tools.skill import SKILL_CONTENT_CAP
    for budget in (BATCH_RESULT_BUDGET, SINGLE_TASK_RESULT_CAP, READ_CHAR_BUDGET, SKILL_CONTENT_CAP):
        assert budget + 5_000 <= HARD_CAP   # 留出外层说明文字的余量
