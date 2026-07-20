"""TUI 流式渲染回归：思考行单行截断（双宽感知）+ mdstream 稳定行去重/拆序。

背景（2026-07-16 截图"重复刷新重复显示"）：
- 思考行按字符数截断，中文双宽撑到两行 → transient Live 高度抖动留残影堆叠。
- 表格流式时活动区在 Live 存活期间 console.print 交错 → 残影重复。

背景（2026-07-17 截图残影复发）：
- 思考文本含 ⚠️（U+26A0+VS16）：Rich 宽度表记 1 格、Windows Terminal 按
  emoji 呈现画 2 格——cell 截断的"单行"承诺被打破，残影回归。
- 修法：_fit_cells 先剥离宽度不可预测字符（VS16/ZWJ/控制/格式字符、\\t）。
"""
from io import StringIO

from rich.console import Console
from rich.cells import cell_len

from loongcli.tui.app import _fit_cells
from loongcli.tui import mdstream


class TestFitCells:
    def test_no_truncation_when_fits(self):
        assert _fit_cells("hi", 10) == "hi"

    def test_ascii_truncated_by_width(self):
        out = _fit_cells("hello world", 5)
        assert cell_len(out) <= 5
        assert out.endswith("…")

    def test_chinese_double_width_respected(self):
        # 4 个中文 = 8 显示列，预算 4 列只能放 1 个中文 + 省略号
        out = _fit_cells("你好世界", 4)
        assert cell_len(out) <= 4
        assert out == "你…"

    def test_thinking_line_fits_one_terminal_row(self):
        # 端到端：80 个中文（=160 列）按 100 列终端预算截断后必须 ≤ 一行
        width = 100
        prefix = "思考中... "
        budget = width - 2 * 2 - cell_len(prefix)
        fitted = _fit_cells("难" * 80, budget)
        assert cell_len(prefix) + cell_len(fitted) <= width - 4

    def test_zero_budget(self):
        assert _fit_cells("x", 0) == ""


class TestSanitizeCells:
    """宽度不可预测字符必须先剥离——cell_len 与终端渲染不一致时单行承诺失效。"""

    def test_vs16_stripped_so_emoji_stays_narrow(self):
        # ⚠️ = U+26A0 + VS16：Rich 记 1 格、Windows Terminal 按 emoji 画 2 格。
        # 剥掉 VS16 后回落文本呈现（1 格），宽度记账恢复一致。
        out = _fit_cells("⚠️ 危险", 40)
        assert "️" not in out
        assert out == "⚠ 危险"

    def test_tab_replaced_with_space(self):
        # \t 由终端展开到制表位（最多 8 格），Rich 只记 1 格
        assert _fit_cells("a\tb", 40) == "a b"

    def test_zwj_stripped(self):
        # ZWJ 合成序列渲染宽度不可静态预测，拆开按组件计宽（只会高估不会低估）
        out = _fit_cells("👨‍👩‍👧", 40)
        assert "‍" not in out

    def test_emoji_line_still_respects_budget(self):
        out = _fit_cells("⚠️警告" * 20, 10)
        assert cell_len(out) <= 10


class TestStripVariationSelectors:
    """mdstream 源文本剥 VS16——Markdown 尾窗 Live 是宽度谎言的第二条爆破路径。

    2026-07-17 截图二复盘：以 ⚠️ 开头的长引用行折行后首行被 Rich 恰好填满，
    ⚠️ 实际 2 格（Rich 记 1）把该行挤出终端宽度再折一次，Live 高度记账差 1 行，
    每次刷新错位——尾窗里的旧行（"在看，方向对口的…"）留成三连残影 + 空洞。
    """

    def test_vs16_stripped_newline_and_zwj_kept(self):
        src = "⚠️ 警告\n👨‍👩‍👧 家庭"
        out = mdstream.strip_variation_selectors(src)
        assert "️" not in out
        assert "\n" in out      # 换行是 markdown 结构，绝不能剥
        assert "‍" in out  # ZWJ 只会让 Rich 高估宽度（提前折行），无害，保留观感

    def test_streaming_blockquote_with_emoji_never_emits_vs16(self):
        # 端到端：复刻事故素材——⚠️ 开头的长中文引用块流式喂入，
        # 断言 VS16 从不落到终端（含 Live 帧与稳定行）。
        buf = StringIO()
        console = Console(force_terminal=True, width=100, file=buf)
        ms = mdstream.MarkdownStream(console, left_pad=2)
        text = "## 回复话术\n\n> 在看，方向对口的可以聊。方便先说一下是哪家公司吗？\n\n---\n\n"
        tail = ("> ⚠️ BOSS直聘上的 `send` 命令由你手动执行，我没代发权限。"
                "三条话术都简洁回应了对方的问题，各带一个下一步推进问题。")
        for i in range(0, len(tail), 8):
            ms.min_delay = 0
            ms.when = 0
            ms.update(text + tail[: i + 8])
        ms.update(text + tail, final=True)
        assert "️" not in buf.getvalue()


class TestMarkdownStreamDedup:
    def _stream_table(self):
        """把逐行增长的 markdown 表喂进 MarkdownStream，收集稳定行 + 拆序。"""
        emitted: list[str] = []
        call_order: list[str] = []

        orig_emit = mdstream.MarkdownStream._emit_stable
        orig_teardown = mdstream.MarkdownStream._teardown

        def spy_emit(self, lines):
            call_order.append("emit")
            from rich.text import Text
            block = "".join(lines).rstrip("\n")
            for ln in Text.from_ansi(block).plain.splitlines():
                if ln.strip():
                    emitted.append(ln.rstrip())
            return orig_emit(self, lines)

        def spy_teardown(self):
            call_order.append("teardown")
            return orig_teardown(self)

        mdstream.MarkdownStream._emit_stable = spy_emit
        mdstream.MarkdownStream._teardown = spy_teardown
        try:
            console = Console(force_terminal=True, width=100, file=StringIO())
            ms = mdstream.MarkdownStream(console, left_pad=2)
            rows = [
                ("技术成长", "4.0", "AI Lab做安全Agent/RAG/算力网关，有真实场景"),
                ("薪资", "4.5", "50-55K深圳，上市公司福利顶级（免息房贷）"),
                ("稳定性", "4.0", "深信服上市公司，安全龙头"),
                ("工作强度", "1.5", "996常态化+56h/周"),
                ("团队氛围", "2.5", "狼性文化，HR面挂人不通知"),
                ("成长空间", "3.5", "安全AI方向前景好但内卷"),
            ]
            text = "## #9 深信服AI Lab\n\n关键发现：方向对口+高薪。\n\n| 维度 | 评分 | 说明 |\n|---|---|---|\n"
            for r in rows:
                text += f"| {r[0]} | {r[1]} | {r[2]} |\n"
                ms.min_delay = 0
                ms.when = 0
                ms.update(text)
            text += "\n背调完成。\n"
            ms.update(text, final=True)
        finally:
            mdstream.MarkdownStream._emit_stable = orig_emit
            mdstream.MarkdownStream._teardown = orig_teardown
        return emitted, call_order

    def test_no_duplicate_data_rows_in_scrollback(self):
        emitted, _ = self._stream_table()
        data = [e for e in emitted
                if any(k in e for k in ("技术成长", "稳定性", "工作强度", "团队氛围", "成长空间"))]
        assert len(data) == len(set(data)), f"稳定行出现重复：{data}"

    def test_teardown_precedes_each_stable_emit(self):
        _, order = self._stream_table()
        # 每个 emit 之前必有一次 teardown（拆 Live 后再 print，杜绝活动区残影）
        for i, ev in enumerate(order):
            if ev == "emit":
                assert i > 0 and order[i - 1] == "teardown", \
                    f"emit 前未先 teardown：{order[:i+1]}"

    def test_all_rows_present_after_final(self):
        emitted, _ = self._stream_table()
        joined = "\n".join(emitted)
        for name in ("技术成长", "稳定性", "工作强度", "团队氛围", "成长空间"):
            assert name in joined, f"缺失行 {name}"
