"""banner 状态条目的分行打包。

原本还测执行中打字回显（_fit_cells_tail/_StatusGroup/_typed_echo_line），2026-11-20 常驻输入框（InputBar）
接管运行期输入后这些都已删除，相应测试见 tests/tui/test_input_bar.py。
"""


class TestPackStatusLines:
    """banner 状态条目按条目边界打包——条目内断行会产生悬空 ⚠（2026-07-19 截图实锤）。"""

    def test_all_fit_one_line(self):
        from loongcli.tui.app import _pack_status_lines
        assert _pack_status_lines(["a: 1", "b: 2"], 40) == ["a: 1 | b: 2"]

    def test_breaks_at_item_boundary(self):
        from loongcli.tui.app import _pack_status_lines
        items = ["model: deepseek-v4-pro", "thinking: max", "⚠ permissions: skip"]
        lines = _pack_status_lines(items, 42)
        # 每个条目完整出现在某一行里，绝不跨行
        joined = "\n".join(lines)
        for item in items:
            assert item in joined
            assert not any(item.split()[0] == line.split(" | ")[-1] and len(item.split()) > 1
                           for line in lines[:-1] if not line.endswith(item))
        # ⚠ 条目与其说明在同一行
        warn_lines = [l for l in lines if "⚠" in l]
        assert len(warn_lines) == 1
        assert "permissions: skip" in warn_lines[0]

    def test_single_oversized_item_own_line_not_split(self):
        from loongcli.tui.app import _pack_status_lines
        lines = _pack_status_lines(["short", "x" * 100], 40)
        assert lines == ["short", "x" * 100]

    def test_cjk_width_counted(self):
        from loongcli.tui.app import _pack_status_lines
        # 4 个中文 = 8 cell；宽度 12 放不下两个条目（8+3+8=19）
        lines = _pack_status_lines(["中文条目", "中文条目"], 12)
        assert len(lines) == 2

    def test_empty(self):
        from loongcli.tui.app import _pack_status_lines
        assert _pack_status_lines([], 40) == []
