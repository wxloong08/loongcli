"""AUTO 档接线：本该确认的调用走审核器；DENY / 只读不调审核器；子代理同一路径；事件可对齐。"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from loongcli.core.agent import AgentLoop, AgentServices
from loongcli.core.llm import LLMClient
from loongcli.core.telemetry import EventLogger
from loongcli.security.auto_review import AutoReviewer
from loongcli.security.permissions import PermissionChecker, PermissionMode
from loongcli.tools.base import Tool, ToolRegistry

ALLOW = '{"verdict": "allow", "category": "普通开发", "reason": "运行项目脚本"}'
BLOCK = '{"verdict": "block", "category": "桌面与外设", "reason": "合成全局 Ctrl 按键"}'


class _RecordingShell(Tool):
    """名为 shell 的假工具：只记录收到的命令，不真跑。"""
    name = "shell"
    description = "fake shell"
    parameters = {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}

    def __init__(self):
        self.ran: list[str] = []

    async def execute(self, command: str) -> str:
        self.ran.append(command)
        return f"ran: {command}"


class _ReviewLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    async def chat(self, prompt: str) -> str:
        self.calls += 1
        return self.replies.pop(0)


def _chunk(content=None, tool_calls=None, finish=None):
    c = MagicMock()
    c.choices = [MagicMock()]
    c.choices[0].delta.content = content
    c.choices[0].delta.tool_calls = tool_calls
    c.choices[0].finish_reason = finish
    c.usage = None
    return c


def _tool_turn(call_id, command):
    d = MagicMock()
    d.index = 0
    d.id = call_id
    d.type = "function"
    d.function = MagicMock()
    d.function.name = "shell"
    d.function.arguments = json.dumps({"command": command})
    return [_chunk(tool_calls=[d]), _chunk(finish="tool_calls")]


def _make(tmp_path, command, review_llm, *, mode=PermissionMode.AUTO, interactive=True):
    llm = LLMClient(api_key="test")
    turns = [_tool_turn("c1", command), [_chunk(content="好"), _chunk(finish="stop")]]

    async def stream(**kwargs):
        for c in turns.pop(0):
            yield c

    llm.chat_stream = stream
    shell = _RecordingShell()
    reg = ToolRegistry()
    reg.register(shell)
    checker = PermissionChecker(mode)
    checker.auto_reviewer = AutoReviewer(review_llm)
    events = tmp_path / "e.jsonl"
    agent = AgentLoop(llm=llm, tool_registry=reg, permission_checker=checker, interactive=interactive,
                      services=AgentServices(telemetry=EventLogger(events)))
    return agent, shell, events


async def _run(agent):
    out = []
    async for ev in agent.run_stream("核实一下环境"):
        out.append(ev)
    return out


def _tool_result(agent):
    return next(m["content"] for m in agent.messages if m.get("role") == "tool")


def _events(path: Path, kind: str):
    return [e for e in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()) if e["type"] == kind]


@pytest.mark.asyncio
async def test_blocked_command_not_executed_and_reason_returned(tmp_path):
    rl = _ReviewLLM(BLOCK)
    agent, shell, events = _make(tmp_path, "python probe.py", rl)
    await _run(agent)
    assert shell.ran == []
    result = _tool_result(agent)
    assert "自动审核拦下" in result and "合成全局 Ctrl" in result and "停下来" in result
    ev = _events(events, "auto_review")
    assert len(ev) == 1 and ev[0]["verdict"] == "block" and ev[0]["tool_call_id"] == "c1"


@pytest.mark.asyncio
async def test_allowed_command_executes(tmp_path):
    rl = _ReviewLLM(ALLOW)
    agent, shell, events = _make(tmp_path, "python probe.py", rl)
    await _run(agent)
    assert shell.ran == ["python probe.py"]
    ev = _events(events, "auto_review")
    assert ev[0]["verdict"] == "allow"
    # 审核事件与工具执行事件按 tool_call_id 对得上
    assert _events(events, "tool_exec")[0]["tool_call_id"] == ev[0]["tool_call_id"] == "c1"


@pytest.mark.asyncio
@pytest.mark.parametrize("command,ran", [("rm -rf /", False), ("git status", True)])
async def test_deny_and_read_only_never_reviewed(tmp_path, command, ran):
    rl = _ReviewLLM()
    agent, shell, events = _make(tmp_path, command, rl)
    await _run(agent)
    assert rl.calls == 0
    assert bool(shell.ran) is ran
    assert _events(events, "auto_review") == []


@pytest.mark.asyncio
async def test_subagent_allowed_command_executes(tmp_path):
    # 非交互（子代理）以前 CONFIRM 一律自动拒绝；AUTO 下走审核器，放行的能执行
    rl = _ReviewLLM(ALLOW)
    agent, shell, _ = _make(tmp_path, "python probe.py", rl, interactive=False)
    await _run(agent)
    assert shell.ran == ["python probe.py"]


@pytest.mark.asyncio
async def test_default_mode_ignores_reviewer(tmp_path):
    # 默认档即使挂了审核器也不用：非交互下照旧自动拒绝，审核器零调用
    rl = _ReviewLLM(ALLOW)
    agent, shell, _ = _make(tmp_path, "python probe.py", rl, mode=PermissionMode.DEFAULT, interactive=False)
    await _run(agent)
    assert rl.calls == 0 and shell.ran == []
    assert "自动拒绝" in _tool_result(agent)


@pytest.mark.asyncio
async def test_review_failure_blocks(tmp_path):
    rl = _ReviewLLM("这不是 JSON")
    agent, shell, events = _make(tmp_path, "python probe.py", rl)
    await _run(agent)
    assert shell.ran == []
    assert "审核失败" in _tool_result(agent)
    assert _events(events, "auto_review")[0]["failed"] is True
