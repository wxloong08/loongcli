"""常驻输入框（loongcli/tui/input_bar.py）：用管道输入 + 假输出驱动真实的 PromptSession。"""
from __future__ import annotations

import asyncio

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from loongcli.tui.input_bar import InputBar

ENTER, ESC, UP, DOWN, CTRL_J, BACKSPACE = "\r", "\x1b", "\x1b[A", "\x1b[B", "\n", "\x7f"


class _Out(DummyOutput):
    def get_size(self) -> Size:
        return Size(rows=40, columns=80)


class _Rig:
    """起一个 InputBar 在后台跑，送键、取提交、收尾。"""

    def __init__(self, **kw):
        self.interrupts = 0
        self._pipe_ctx = create_pipe_input()
        self.inp = self._pipe_ctx.__enter__()
        self.bar = InputBar(input=self.inp, output=_Out(), on_interrupt=self._on_interrupt, **kw)
        self.task = None

    def _on_interrupt(self):
        self.interrupts += 1

    async def __aenter__(self):
        self.task = asyncio.create_task(self.bar.run())
        await asyncio.sleep(0.1)          # 等输入框起来
        return self

    async def __aexit__(self, *exc):
        self.bar.stop()
        await asyncio.wait_for(self.task, 2)
        self._pipe_ctx.__exit__(None, None, None)

    async def send(self, keys: str, settle: float = 0.15):
        self.inp.send_text(keys)
        await asyncio.sleep(settle)

    async def next_submission(self):
        return await asyncio.wait_for(self.bar.submissions.get(), 2)


@pytest.mark.asyncio
async def test_submit_goes_to_queue():
    async with _Rig() as r:
        await r.send("你好" + ENTER)
        assert await r.next_submission() == "你好"


@pytest.mark.asyncio
async def test_multiple_submissions_while_busy():
    """模型运行期间连续提交：每条都进队列，输入框始终可用（调用方据此排队插话）。"""
    async with _Rig() as r:
        await r.send("第一条" + ENTER)
        await r.send("第二条" + ENTER)
        assert [await r.next_submission(), await r.next_submission()] == ["第一条", "第二条"]


@pytest.mark.asyncio
async def test_ask_answer_does_not_enter_queue():
    async with _Rig() as r:
        ask = asyncio.create_task(r.bar.ask("确认执行？(y/N)"))
        await asyncio.sleep(0.1)
        assert r.bar.asking
        await r.send("y" + ENTER)
        assert await asyncio.wait_for(ask, 2) == "y"
        assert r.bar.submissions.empty()
        assert not r.bar.asking


@pytest.mark.asyncio
async def test_esc_calls_interrupt():
    async with _Rig() as r:
        await r.send("打到一半")
        await r.send(ESC, settle=0.3)
        assert r.interrupts == 1


@pytest.mark.asyncio
async def test_esc_during_ask_returns_none_not_interrupt():
    """提问模式下 Esc = 取消（None，区别于回车的空回答），不触发回合中断。"""
    async with _Rig() as r:
        ask = asyncio.create_task(r.bar.ask("确认执行？(y/N)"))
        await asyncio.sleep(0.1)
        await r.send(ESC, settle=0.3)
        assert await asyncio.wait_for(ask, 2) is None
        assert r.interrupts == 0


@pytest.mark.asyncio
async def test_choose_moves_with_arrows_and_enter():
    """选项对话框：上下键移动、回车确认；普通文字键吞掉、不入框、不作答。"""
    async with _Rig() as r:
        pick = asyncio.create_task(r.bar.choose("要执行这个操作吗？", ["执行", "不执行"]))
        await asyncio.sleep(0.1)
        await r.send("abc", settle=0.2)
        assert not pick.done() and r.bar.session.default_buffer.text == ""
        await r.send(DOWN, settle=0.2)
        await r.send(ENTER, settle=0.3)
        assert await asyncio.wait_for(pick, 2) == (1, "")
        assert r.bar.submissions.empty()


@pytest.mark.asyncio
async def test_choose_digit_selects_and_esc_cancels():
    async with _Rig() as r:
        pick = asyncio.create_task(r.bar.choose("批准这个计划吗？", ["批准", "改", "取消"]))
        await asyncio.sleep(0.1)
        await r.send("3", settle=0.3)
        assert await asyncio.wait_for(pick, 2) == (2, "")
        pick = asyncio.create_task(r.bar.choose("批准这个计划吗？", ["批准", "改", "取消"]))
        await asyncio.sleep(0.1)
        await r.send(ESC, settle=0.3)
        assert await asyncio.wait_for(pick, 2) is None
        assert r.interrupts == 0                      # 对话框里的 Esc 不中断回合


@pytest.mark.asyncio
async def test_draft_restored_after_choose():
    """弹框前正在打的草稿：对话框期间收起，选完原样放回，不被当成答案。"""
    async with _Rig() as r:
        await r.send("顺便加注释")
        pick = asyncio.create_task(r.bar.choose("要执行这个操作吗？", ["执行", "不执行"]))
        await asyncio.sleep(0.1)
        assert r.bar.session.default_buffer.text == ""
        await r.send("1", settle=0.3)
        assert await asyncio.wait_for(pick, 2) == (0, "")
        await asyncio.sleep(0.1)
        assert r.bar.session.default_buffer.text == "顺便加注释"


@pytest.mark.asyncio
async def test_choose_input_option_types_inline():
    """输入项（最后一项）：数字选到它只跳过去不提交；在这一行直接打字（数字也算文字），回车连文字一起交出；
    交出后文字不留在输入框里，弹框前的草稿原样放回。"""
    async with _Rig() as r:
        await r.send("草稿")
        pick = asyncio.create_task(r.bar.choose("批准这个计划吗？", ["批准", "取消", "提修改建议"], input_last=True))
        await asyncio.sleep(0.1)
        await r.send("3", settle=0.2)
        assert not pick.done() and r.bar._editing_choice()
        await r.send("用参数化写 3 个用例", settle=0.2)
        assert not pick.done()
        await r.send(ENTER, settle=0.3)
        assert await asyncio.wait_for(pick, 2) == (2, "用参数化写 3 个用例")
        await asyncio.sleep(0.1)
        assert r.bar.session.default_buffer.text == "草稿"
        assert r.bar.submissions.empty()


@pytest.mark.asyncio
async def test_choose_typing_jumps_to_input_option():
    """有输入项时，停在别的项上直接打字 = 跳到输入项开始写；移回别的项再回车，文字不跟着交出去。"""
    async with _Rig() as r:
        pick = asyncio.create_task(r.bar.choose("要执行这个操作吗？", ["执行", "不执行，告诉模型怎么改"], input_last=True))
        await asyncio.sleep(0.1)
        await r.send("别删", settle=0.2)
        assert r.bar._editing_choice() and r.bar.session.default_buffer.text == "别删"
        await r.send(UP, settle=0.2)
        await r.send(ENTER, settle=0.3)
        assert await asyncio.wait_for(pick, 2) == (0, "")
        await asyncio.sleep(0.1)
        assert r.bar.session.default_buffer.text == ""


@pytest.mark.asyncio
async def test_draft_restored_after_text_ask():
    """文字提问回车提交后输入框会重启清空，草稿也要放回。"""
    async with _Rig() as r:
        await r.send("草稿")
        ask = asyncio.create_task(r.bar.ask("修改建议:"))
        await asyncio.sleep(0.1)
        await r.send("用参数化" + ENTER, settle=0.3)
        assert await asyncio.wait_for(ask, 2) == "用参数化"
        await asyncio.sleep(0.2)
        assert r.bar.session.default_buffer.text == "草稿"
        assert r.bar.submissions.empty()


@pytest.mark.asyncio
async def test_menu_space_follows_completion_count():
    """预留行数 = 补全菜单实际项数（上限 MENU_ROWS）；没有补全项就不预留——打完完整命令、清掉输入都要缩回去。"""
    from prompt_toolkit.completion import WordCompleter
    words = ["/goal", "/compact", "/clear", "/help", "/exit"]
    async with _Rig(completer=WordCompleter(words, sentence=True)) as r:
        assert r.bar.session.reserve_space_for_menu == 0
        await r.send("/", settle=0.4)
        assert r.bar.session.reserve_space_for_menu == len(words)
        await r.send("go", settle=0.4)
        assert r.bar.session.reserve_space_for_menu == 1
        await r.send(BACKSPACE * 3 + "普通文字", settle=0.4)
        assert r.bar.session.reserve_space_for_menu == 0


@pytest.mark.asyncio
async def test_write_batches_into_single_flush():
    """输入框上方打印：擦除-写入-重画合成一次落终端（多次 flush 就是闪烁的来源）。"""
    async with _Rig() as r:
        out = r.bar.session.app.output
        r.bar._pending = ["第一行\n", "第二行\n"]   # 两次 write 合批后的待写内容
        flushes = []
        real = out.flush
        out.flush = lambda: (flushes.append(1), real())[1]
        try:
            r.bar._emit()                          # 只数这一次打印本身落终端几次（定时刷新的帧不算）
        finally:
            out.flush = real
        assert len(flushes) == 1
        assert r.bar._pending == []


@pytest.mark.asyncio
async def test_write_coalesces_calls():
    async with _Rig() as r:
        emits = []
        r.bar._emit_real, r.bar._emit = r.bar._emit, lambda: (emits.append(1), r.bar._emit_real())
        r.bar.write("a\n")
        r.bar.write("b\n")
        await asyncio.sleep(0.2)
        assert len(emits) == 1


@pytest.mark.asyncio
async def test_ctrl_j_newline_then_submit():
    async with _Rig() as r:
        await r.send("第一行" + CTRL_J + "第二行" + ENTER)
        assert await r.next_submission() == "第一行\n第二行"


@pytest.mark.asyncio
async def test_eof_puts_none():
    async with _Rig() as r:
        await r.send("\x04", settle=0.3)          # Ctrl+D 在空输入框上 = EOF
        assert await r.next_submission() is None


def test_message_shows_live_status_and_prompt():
    bar = InputBar(input=None, output=_Out())
    bar.set_live("正文尾部第一行\n正文尾部第二行")
    bar.set_status("模型思考中…")
    text = fragment_list_to_text(to_formatted_text(bar._message()))
    assert "正文尾部第二行" in text and "模型思考中…" in text
    assert text.rstrip().endswith("❯")
    assert text.index("正文尾部") < text.index("模型思考中") < text.index("❯")


def test_message_in_ask_mode_shows_question():
    bar = InputBar(input=None, output=_Out())
    bar._ask = ("修改建议:", None)
    text = fragment_list_to_text(to_formatted_text(bar._message()))
    assert "修改建议:" in text


def test_message_in_choose_mode_shows_options_not_prompt():
    bar = InputBar(input=None, output=_Out())
    bar._choice = {"title": "要执行这个操作吗？", "options": ["执行", "不执行"], "selected": 1, "future": None,
                   "input_last": False}
    lines = fragment_list_to_text(to_formatted_text(bar._message())).split("\n")
    assert lines[0] == "─" * 80 and "要执行这个操作吗？" in lines[1]
    assert lines[2] == "    1. 执行" and lines[3] == "  ❯ 2. 不执行"
    assert len(lines) == 4                            # 最后一行就是最后一项：输入行紧接其后
    assert "Esc 取消" in fragment_list_to_text(to_formatted_text(bar._toolbar()))   # 按键提示在下方状态行
    bar._choice["input_last"] = True
    lines = fragment_list_to_text(to_formatted_text(bar._message())).split("\n")
    assert lines[-1] == "  ❯ 2. 不执行："
    assert "直接打字" in fragment_list_to_text(to_formatted_text(bar._toolbar()))


def test_live_change_waits_for_pending_print():
    """有没写出的滚动区内容时，动态区变化暂存到那批内容写出的同一帧——否则输入框会先上跳一下再落回。"""
    bar = InputBar(input=None, output=_Out())
    bar.set_live_renderable("旧尾部")
    bar._pending = ["稳定行\n"]
    bar.set_live_renderable("新尾部")
    assert bar._renderable == "旧尾部"
    bar._emit()                       # 输入框没在跑：走直写兜底，同样要把暂存的动态区换上
    assert bar._renderable == "新尾部" and bar._pending == []


def test_toolbar_rule_and_status_aligned():
    """输入框下方：一条整宽横线 + 状态行（左提示、右模型等，右端对齐，放不下先砍右边）。"""
    bar = InputBar(input=None, output=_Out(), status_provider=lambda: ("Shift+Enter 换行", "deepseek-v4-flash · 上下文 45K"))
    rule, status = fragment_list_to_text(to_formatted_text(bar._toolbar())).split("\n")
    assert rule == "─" * 80
    assert status.startswith("  Shift+Enter 换行") and status.endswith("上下文 45K")
    narrow = InputBar(input=None, output=_Out(), status_provider=lambda: ("左" * 30, "右" * 30))
    _, status = fragment_list_to_text(to_formatted_text(narrow._toolbar())).split("\n")
    assert "右" not in status and status.startswith("  左")


def test_toolbar_survives_provider_error():
    def boom():
        raise RuntimeError("x")
    bar = InputBar(input=None, output=_Out(), status_provider=boom)
    assert fragment_list_to_text(to_formatted_text(bar._toolbar())).startswith("─")


def test_prompt_has_rule_above():
    bar = InputBar(input=None, output=_Out())
    lines = fragment_list_to_text(to_formatted_text(bar._message())).split("\n")
    assert lines[-2] == "─" * 80 and lines[-1].startswith("❯")


def test_rows_above_after_reflow():
    """拉窄窗口后终端把旧行重排成多行：上移的行数要按重排后算，否则够不着布局顶部、旧横线残留。"""
    from loongcli.tui.input_bar import _rows_above_after_reflow as f
    assert f([120, 0, 120], 0, 75) == (5, 0)      # 两条 120 列横线各占 2 行，空行占 1 行
    assert f([60, 60], 5, 75) == (2, 5)           # 都放得下：不变
    assert f([], 90, 75) == (1, 15)               # 光标所在行本身被折到第 2 行


def test_renderer_corrects_cursor_after_narrowing():
    """渲染器发现窗口变窄：光标按重排后的行数校正、强制整块重画；变宽或没变则不动。"""
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.layout.screen import Char, Screen
    from prompt_toolkit.styles import DummyStyle
    from loongcli.tui.input_bar import _CompactRenderer

    class _Sized(DummyOutput):
        cols = 120
        def get_size(self):
            return Size(rows=40, columns=self.cols)

    out = _Sized()
    r = _CompactRenderer(DummyStyle(), out)
    screen = Screen()
    for x in range(120):
        screen.data_buffer[0][x] = Char("─")
    r._last_screen, r._last_size, r._cursor_pos = screen, Size(rows=40, columns=120), Point(x=2, y=2)
    r._correct_for_reflow()                       # 宽度没变
    assert r._cursor_pos == Point(x=2, y=2) and r._last_screen is screen
    out.cols = 60
    r._correct_for_reflow()                       # 横线 120 列 → 2 行，空行 1 行，共上移 3 行
    assert r._cursor_pos == Point(x=2, y=3) and r._last_screen is None
