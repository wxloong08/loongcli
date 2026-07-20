"""集成测试：驱动 _handle_agent_response 的完整事件流，验证流式 markdown 渲染
正确、无重复堆叠、工具调用穿插干净。覆盖 MarkdownStream/StreamView 在真实 TUI
路径里的端到端行为（单测在 test_mdstream.py 测算法，这里测集成）。"""
import asyncio
import io
import re

from rich.console import Console

from loongcli.tui.app import TUI
from loongcli.core.events import (
    TextDelta, ThinkingDelta, ToolCallStart, ToolCallResult, AgentDone,
)


class _FakeCost:
    total_cost = 0.0

    def format_cost(self, c):
        return f"¥{c:.4f}"


class _FakeAgent:
    def __init__(self, events):
        self._events = events
        self.token_usage = {
            "total_tokens": 0, "prompt_tokens": 0, "prompt_cache_hit_tokens": 0,
            "completion_tokens": 0, "reasoning_tokens": 0,
        }
        self.cost_tracker = _FakeCost()
        # 执行中插话队列接口（与 AgentLoop 对齐——轮末收尾会 take）
        self._queued_user_inputs: list[str] = []

    def queue_user_input(self, text: str) -> None:
        self._queued_user_inputs.append(text)

    def take_queued_inputs(self) -> list[str]:
        taken = self._queued_user_inputs
        self._queued_user_inputs = []
        return taken

    async def run_stream(self, user_input, allowed_tools=None, images=None):
        for e in self._events:
            yield e
            await asyncio.sleep(0)


def _strip_ansi(s: str) -> str:
    s = re.sub(r"\x1b\][^\x07\x1b]*(\x07|\x1b\\)", "", s)  # OSC
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", s)        # CSI


def _drive(events, width=80) -> str:
    buf = io.StringIO()
    tui = TUI()
    tui.console = Console(file=buf, force_terminal=True, width=width)
    asyncio.run(tui._handle_agent_response(_FakeAgent(events), "q"))
    return _strip_ansi(buf.getvalue())


_MD = (
    "# 报告标题\n\n这是**第一段**要点：\n\n- 要点一\n- 要点二\n- 要点三\n\n"
    "## 小节\n\n```python\ndef foo():\n    return 42\n```\n\n"
    "中间段落写几句凑多行，触发稳定行机制把内容滚出 live 窗口。第二句。第三句。\n"
)


def _text_events(md, chunk=4):
    return [TextDelta(text=md[i:i + chunk]) for i in range(0, len(md), chunk)]


def test_streams_full_markdown_no_duplication():
    events = [ThinkingDelta(text="想想")] + _text_events(_MD) + [AgentDone(content="")]
    out = _drive(events)
    # 完整内容都渲染出来
    for token in ["报告标题", "要点一", "要点二", "要点三", "def foo", "第三句"]:
        assert token in out, f"missing: {token}"
    # 关键回归：稳定行机制不得重复输出任何内容
    assert out.count("要点二") == 1
    assert out.count("def foo") == 1
    assert out.count("第三句") == 1


def test_tool_call_interleaves_cleanly():
    events = (
        _text_events(_MD)
        + [ToolCallStart(tool_name="read_file", arguments={"path": "x.py"})]
        + [ToolCallResult(tool_name="read_file", result="a\nb\nc\n")]
        + [TextDelta(text="\n\n收尾：完成。\n"), AgentDone(content="")]
    )
    out = _drive(events)
    # 正文在工具行之前完整落盘
    assert "要点三" in out
    # 折叠模式（默认）：● 工具行 + ⎿ 统计，无 ⚙ 常驻行
    assert "● read_file" in out and "x.py" in out
    assert "⎿ 3 行" in out
    assert "⚙" not in out
    # 工具后续写正常
    assert "收尾：完成" in out
    # 正文整体不重复
    assert out.count("要点二") == 1


def test_tool_call_verbose_mode_keeps_old_format():
    events = (
        [ToolCallStart(tool_name="read_file", arguments={"path": "x.py"})]
        + [ToolCallResult(tool_name="read_file", result="a\nb\nc\n")]
        + [AgentDone(content="")]
    )
    buf = io.StringIO()
    tui = TUI()
    tui.verbose = True
    tui.console = Console(file=buf, force_terminal=True, width=80)
    asyncio.run(tui._handle_agent_response(_FakeAgent(events), "q"))
    out = _strip_ansi(buf.getvalue())
    assert "⚙ read_file" in out
    assert "✓ read_file" in out and "(3 行)" in out


def test_tool_error_shows_cross():
    events = (
        [ToolCallStart(tool_name="edit_file", arguments={"path": "x.py", "old_string": "a", "new_string": "b"})]
        + [ToolCallResult(tool_name="edit_file", result="错误：未找到目标字符串")]
        + [AgentDone(content="")]
    )
    out = _drive(events)
    assert "✗ edit_file" in out
    assert "未找到目标字符串" in out


def test_text_only_turn_renders_and_no_live_leak():
    # 纯文本、无工具：MarkdownStream 必须在 finally 里冲刷干净
    events = _text_events("简单一段回答。\n\n第二段。\n") + [AgentDone(content="")]
    out = _drive(events)
    assert "简单一段回答" in out and "第二段" in out


def test_confirm_args_display_full_command():
    """确认框的 command 必须完整输出——看不全的命令没法判断安全，等于盲签。"""
    tui = TUI()
    long_cmd = "find /d -maxdepth 4 -type d -name 'loongcli' 2>/dev/null | head -5 && echo " + "x" * 100
    text = tui._confirm_args_display({"command": long_cmd})
    assert long_cmd in text.plain          # 一个字符不少
    assert "…" not in text.plain


def test_confirm_args_display_long_content_previewed():
    """非 command 长值（如 write_file 的 content）给 500 字符预览并标注总长。"""
    tui = TUI()
    text = tui._confirm_args_display({"path": "a.py", "content": "z" * 2000})
    assert "a.py" in text.plain
    assert "共 2000 字符" in text.plain
    assert "z" * 501 not in text.plain


def test_memory_notices_shown_and_drained():
    """写入可见化：上一轮后台记忆落库，在新一轮开头曝光一行并清空。"""
    events = [TextDelta(text="好的。"), AgentDone(content="好的。")]
    agent = _FakeAgent(events)
    agent.memory_notices = [{"name": "project-x-fact", "description": "某项目事实"}]
    buf = io.StringIO()
    tui = TUI()
    tui.console = Console(file=buf, force_terminal=True, width=100)
    asyncio.run(tui._handle_agent_response(agent, "q"))
    out = _strip_ansi(buf.getvalue())
    assert "已记忆: project-x-fact" in out
    assert "某项目事实" in out
    assert "/forget project-x-fact" in out
    assert agent.memory_notices == []  # 展示后清空，不重复提示


def test_diff_text_not_dimmed_true_white():
    """⎿ 前缀的 dim 不得污染 diff 文字（整行基础样式 dim 会把 #ffffff 压成灰——三轮发灰的真凶）。"""
    events = (
        [ToolCallStart(tool_name="edit_file", arguments={
            "path": "x.py", "old_string": "def hello():", "new_string": "def greet():",
        })]
        + [ToolCallResult(tool_name="edit_file", result="成功编辑 x.py")]
        + [AgentDone(content="")]
    )
    buf = io.StringIO()
    tui = TUI()
    tui.console = Console(file=buf, force_terminal=True, width=100, color_system="truecolor")
    asyncio.run(tui._handle_agent_response(_FakeAgent(events), "q"))
    raw = buf.getvalue()
    # truecolor 纯白前景码必须出现在 diff 行（38;2;255;255;255），且带背景色块（48;2;）
    assert "38;2;255;255;255" in raw
    assert "48;2;" in raw


def test_block_spacing_at_most_one_blank_line():
    """块间距统一：文本→工具块→文本 全程不得出现连续 2 个以上空行。"""
    events = (
        _text_events("第一段正文。\n")
        + [ToolCallStart(tool_name="read_file", arguments={"path": "x.py"})]
        + [ToolCallResult(tool_name="read_file", result="a\nb\nc\n")]
        + [TextDelta(text="第二段正文。\n"), AgentDone(content="")]
    )
    out = _drive(events)
    assert "第一段正文" in out and "⎿ 3 行" in out and "第二段正文" in out
    # 不允许 ≥2 个连续空行（3 个连续换行含首尾内容行 = 2 空行）
    assert "\n\n\n\n" not in out
