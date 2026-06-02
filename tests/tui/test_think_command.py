import pytest
from unittest.mock import MagicMock
from rich.console import Console
from io import StringIO

from loongcli.tui.commands import ThinkCommand, CommandContext, create_default_registry


def _make_ctx(thinking=False, reasoning_effort="max"):
    console = Console(file=StringIO())
    agent = MagicMock()
    agent.llm.thinking = thinking
    agent.llm.reasoning_effort = reasoning_effort
    return CommandContext(console=console, agent=agent)


@pytest.mark.asyncio
async def test_think_show_status():
    ctx = _make_ctx(thinking=False)
    cmd = ThinkCommand()
    await cmd.run([], ctx)
    output = ctx.console.file.getvalue()
    assert "关闭" in output


@pytest.mark.asyncio
async def test_think_on():
    ctx = _make_ctx(thinking=False)
    cmd = ThinkCommand()
    await cmd.run(["on"], ctx)
    assert ctx.agent.llm.thinking is True


@pytest.mark.asyncio
async def test_think_off():
    ctx = _make_ctx(thinking=True)
    cmd = ThinkCommand()
    await cmd.run(["off"], ctx)
    assert ctx.agent.llm.thinking is False


@pytest.mark.asyncio
async def test_think_max():
    ctx = _make_ctx(thinking=False)
    cmd = ThinkCommand()
    await cmd.run(["max"], ctx)
    assert ctx.agent.llm.thinking is True
    assert ctx.agent.llm.reasoning_effort == "max"


@pytest.mark.asyncio
async def test_think_high():
    ctx = _make_ctx(thinking=False)
    cmd = ThinkCommand()
    await cmd.run(["high"], ctx)
    assert ctx.agent.llm.thinking is True
    assert ctx.agent.llm.reasoning_effort == "high"


@pytest.mark.asyncio
async def test_think_low():
    ctx = _make_ctx()
    cmd = ThinkCommand()
    await cmd.run(["low"], ctx)
    assert ctx.agent.llm.thinking is True
    assert ctx.agent.llm.reasoning_effort == "low"


@pytest.mark.asyncio
async def test_think_medium():
    ctx = _make_ctx()
    cmd = ThinkCommand()
    await cmd.run(["medium"], ctx)
    assert ctx.agent.llm.thinking is True
    assert ctx.agent.llm.reasoning_effort == "medium"


def test_think_in_registry():
    reg = create_default_registry()
    assert reg.get("think") is not None
