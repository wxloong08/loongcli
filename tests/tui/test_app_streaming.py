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

    async def run_stream(self, user_input, allowed_tools=None):
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
    assert "⚙ read_file" in out
    assert "✓ read_file" in out and "(3 行)" in out
    # 工具后续写正常
    assert "收尾：完成" in out
    # 正文整体不重复
    assert out.count("要点二") == 1


def test_text_only_turn_renders_and_no_live_leak():
    # 纯文本、无工具：MarkdownStream 必须在 finally 里冲刷干净
    events = _text_events("简单一段回答。\n\n第二段。\n") + [AgentDone(content="")]
    out = _drive(events)
    assert "简单一段回答" in out and "第二段" in out
