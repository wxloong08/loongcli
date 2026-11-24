"""常驻输入框接入 TUI（14c/14d）：提交分流、插话排队、StreamView 改走输入框动态区。"""
from __future__ import annotations

import asyncio
import io

import pytest
from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text
from rich.console import Console
from rich.spinner import Spinner
from rich.text import Text

from loongcli.tui.app import TUI
from loongcli.tui.input_bar import InputBar
from loongcli.tui.mdstream import StreamView

from tests.tui.test_input_bar import _Out


class _Agent:
    def __init__(self):
        self.queued: list[str] = []

    def queue_user_input(self, text):
        self.queued.append(text)


def _tui():
    tui = TUI()
    tui.console = Console(file=io.StringIO(), force_terminal=True, width=80)
    tui._bar = InputBar(input=None, output=_Out())
    tui._idle_inputs = asyncio.Queue()
    return tui


async def _route(tui, agent, *texts):
    task = asyncio.create_task(tui._route_submissions(agent))
    for t in texts:
        await tui._bar.submissions.put(t)
    await asyncio.sleep(0.05)
    task.cancel()


@pytest.mark.asyncio
async def test_idle_submission_goes_to_main_loop():
    tui, agent = _tui(), _Agent()
    await _route(tui, agent, "你好")
    assert tui._idle_inputs.get_nowait() == "你好"
    assert agent.queued == []


@pytest.mark.asyncio
async def test_busy_submission_becomes_interjection():
    tui, agent = _tui(), _Agent()
    tui._busy = True
    await _route(tui, agent, "顺便改一下 README")
    assert agent.queued == ["顺便改一下 README"]
    assert tui._idle_inputs.empty()
    assert "已排队" in tui.console.file.getvalue()


@pytest.mark.asyncio
async def test_goal_loop_between_turns_still_interjection():
    """goal 执行循环两次迭代之间 _busy 为 False，但提交仍应排队成插话，而不是另起一轮。"""
    tui, agent = _tui(), _Agent()
    tui._goal_mode = True
    await _route(tui, agent, "补充一句")
    assert agent.queued == ["补充一句"]


@pytest.mark.asyncio
async def test_slash_command_while_busy_rejected():
    tui, agent = _tui(), _Agent()
    tui._busy = True
    await _route(tui, agent, "/compact")
    assert agent.queued == [] and tui._idle_inputs.empty()
    assert "运行中不支持斜杠命令" in tui.console.file.getvalue()


@pytest.mark.asyncio
async def test_goal_stop_allowed_while_goal_running():
    tui, agent = _tui(), _Agent()
    tui._busy = tui._goal_mode = True
    hits = []
    tui._turn_interrupt = lambda: hits.append(1)
    await _route(tui, agent, "/goal stop")
    assert tui._goal_stop_requested and hits == [1]
    assert agent.queued == []


@pytest.mark.asyncio
async def test_eof_forwarded_to_main_loop():
    tui, agent = _tui(), _Agent()
    tui._busy = True                     # 运行中按 Ctrl+D 也要能退出
    await _route(tui, agent, None)
    assert tui._idle_inputs.get_nowait() is None


def test_interrupt_key_idle_is_noop_and_busy_interrupts():
    tui = _tui()
    tui._on_interrupt_key()              # 空闲：什么都不发生
    hits = []
    tui._turn_interrupt = lambda: hits.append(1)
    tui._on_interrupt_key()
    assert hits == [1]


def test_streamview_with_live_target_never_builds_rich_live():
    targets = []
    buf = io.StringIO()
    view = StreamView(Console(file=buf, force_terminal=True, width=60),
                      live_target=lambda r, status=False: targets.append(r))
    view.status(Spinner("dots", text="思考中"))
    assert view._spinner is None and targets[-1] is not None
    for ch in "第一行\n\n第二行\n\n第三行\n\n第四行\n\n第五行\n\n第六行\n\n第七行\n\n第八行\n":
        view.append_text(ch)
    assert view._md is not None and view._md._live is None
    view.close()
    assert targets[-1] is None                        # 收尾清空动态区
    out = buf.getvalue()
    assert "第一行" in out and "第八行" in out        # 正文完整落进滚动区
    assert out.count("第一行") == 1


def test_input_bar_renders_rich_renderable_above_prompt():
    bar = InputBar(input=None, output=_Out())
    bar.set_live_renderable(Text("正文尾部"))
    text = fragment_list_to_text(to_formatted_text(bar._message()))
    assert text.index("正文尾部") < text.index("❯")
    bar.set_live_renderable(None)
    text = fragment_list_to_text(to_formatted_text(bar._message()))
    assert "正文尾部" not in text


def test_status_line_reflects_agent_state():
    from loongcli.tui.app import _fmt_tokens
    assert [_fmt_tokens(n) for n in (987, 45321, 1_000_000, 1_234_567)] == ["987", "45K", "1M", "1.2M"]

    class _LLM:
        model, thinking, reasoning_effort = "deepseek-v4-pro", True, "max"

    class _Comp:
        threshold = 987_000

    class _A:
        llm, compactor, _last_prompt_tokens, plan_mode = _LLM(), _Comp(), 45_321, True

    tui = _tui()
    tui._agent = _A()
    left, right = tui._status_line()
    assert "Shift+Tab" in left
    assert right == "deepseek-v4-pro · thinking max · 上下文 45K（压缩线 987K） · 规划模式"
    tui._busy = True
    assert "打字即插话" in tui._status_line()[0]


def test_prompt_shows_mode_and_follows_provider():
    """有模式名时提示符是品红「规划 ❯」；模式被别处清掉（计划批准自动退出）下一次重画就变回绿色「❯」。
    app 侧接线（Shift+Tab 进出）由 e2e guard_input_frame 覆盖。"""
    mode = ["规划"]
    bar = InputBar(input=None, output=_Out(), mode_provider=lambda: mode[0])
    assert bar._message().value.endswith("\x1b[1;35m规划 ❯ \x1b[0m") and bar._prompt_text() == "规划 ❯ "
    mode[0] = ""
    assert bar._message().value.endswith("\x1b[1;32m❯ \x1b[0m") and bar._prompt_text() == "❯ "
    bar.mode_provider = lambda: 1 / 0          # 取值出错当普通模式，不炸输入框
    assert bar._prompt_text() == "❯ "


def test_thinking_status_shows_elapsed_not_content(monkeypatch):
    """思考状态只显示「思考中… 计时」，秒数在渲染时现算（同一个对象挂着不动也会走）。"""
    from loongcli.tui import app as app_mod
    now = [100.0]
    monkeypatch.setattr(app_mod.time, "monotonic", lambda: now[0])
    status = app_mod._ThinkingStatus((0, 2))

    def render():
        buf = io.StringIO()
        Console(file=buf, width=60, force_terminal=False).print(status)
        return buf.getvalue()

    assert "思考中… 0s" in render()
    now[0] += 12
    assert "思考中… 12s" in render()
    now[0] += 60
    assert "思考中… 1m12s" in render()
