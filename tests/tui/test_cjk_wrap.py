"""中日韩文字断行（loongcli/tui/cjk_wrap.py）与滚动区排版细节：块间空行、输入回显悬挂缩进。"""
from __future__ import annotations

import io

import loongcli.tui  # noqa: F401  导入即安装断行补丁
from rich.console import Console
from rich.text import Text

from loongcli.tui.app import _user_echo
from loongcli.tui.mdstream import LeftMarkdown, MarkdownStream


def _render(renderable, width: int) -> list[str]:
    c = Console(file=io.StringIO(), width=width, force_terminal=False)
    c.print(renderable)
    return [l.rstrip() for l in c.file.getvalue().splitlines()]


def test_cjk_after_space_fills_line_instead_of_jumping():
    """空格后面跟一长串中文：原来整串挪到下一行、上一行空半截（真机「都用 plan」后断行）；现在填满再断。"""
    lines = _render(LeftMarkdown("有一处需要说明：系统要求每完成一步都用 plan 记录进度，因此整个过程中唯一的工具调用就是这次计划记账。"), 60)
    from rich.cells import cell_len
    assert "plan 记录" in lines[0] and cell_len(lines[0]) >= 58, lines    # 60 列：至多空 2 格（一个汉字放不下）


def test_english_wrapping_unchanged():
    lines = _render(Text("The quick brown fox jumps over the lazy dog, again and again until it wraps."), 40)
    assert lines == ["The quick brown fox jumps over the lazy", "dog, again and again until it wraps."]


def test_cjk_punctuation_never_starts_or_ends_a_line_wrongly():
    """句读、右括号不放行首；左括号不放行尾。"""
    for width in range(20, 41):
        for line in _render(Text("（括号开头）在这里，句号也在。" * 6), width):
            assert not line[:1] in "，。、）」", (width, line)
            assert not line[-1:] in "（「", (width, line)


def test_blank_line_survives_batch_boundary():
    """稳定行分批写出：空行恰在一批末尾也要保留（原 rstrip 把它剥了，标题前空行时有时无）。"""
    buf = io.StringIO()
    md = MarkdownStream(Console(file=buf, width=40, force_terminal=False), left_pad=0)
    md._emit_stable(["标题一\n", "\n"])
    md._emit_stable(["标题二\n"])
    md._emit_stable(["甲\n"])
    assert buf.getvalue() == "标题一\n\n标题二\n甲\n"


def test_plan_tool_shows_one_progress_line():
    """plan 工具只显示一行进度，不 dump 原始参数；英文状态词翻成中文（用户嫌输出太多）。"""
    from loongcli.tui.app import _plan_progress_line
    upd = _plan_progress_line({"operation": "update_step", "plan_id": "x", "step": 1, "step_status": "completed"},
                              "步骤 1 已更新 → completed（进度 1/4）\n\n计划全文……")
    assert upd.plain == "◆ 计划：步骤 1 已更新 → 完成（进度 1/4）"     # 工具结果本身就从 1 起，界面原样显示
    new = _plan_progress_line({"operation": "create", "title": "速查表", "steps": [{}, {}, {}]}, "计划已创建: x\n…")
    assert new.plain == "◆ 计划：已创建「速查表」（3 步）"
    done = _plan_progress_line({"operation": "complete", "plan_id": "x"}, "计划 x 已标记为 completed (4/4 步骤完成)")
    assert done.plain == "◆ 计划：计划 x 已标记为完成 (4/4 步骤完成)"
    # 被拒：首行以冒号结尾只是引子，原因要跟着显示
    rej = _plan_progress_line({"operation": "update_step"}, "步骤 1 未标完成：以下判据还没有证据：\n- 判据 1：a.txt 里有 done")
    assert rej.plain == "◆ 计划：步骤 1 未标完成：以下判据还没有证据：\n    - 判据 1：a.txt 里有 done"


def test_tool_arg_summary_clips_by_display_width():
    """工具行的参数摘要按显示宽度截：中文两列，按字数截 60 个字会到 120 列，● 工具行折成两行。"""
    from rich.cells import cell_len
    from loongcli.tui.tool_display import arg_summary
    s = arg_summary("shell", {"command": "echo " + "很长的一行输出，" * 20})
    assert s.endswith("…") and cell_len(s) <= 61


def test_user_echo_wraps_with_hanging_indent_and_keeps_brackets():
    lines = _render(_user_echo("/goal 不要调用任何工具。用 Markdown 写一份约 100 行的 Python asyncio 速查表 [x]"), 50)
    assert lines[0].startswith("❯ /goal") and not lines[0].endswith("…")
    assert all(l.startswith("  ") for l in lines[1:]) and len(lines) > 1
    assert "[x]" in lines[-1]
