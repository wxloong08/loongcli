import pytest
from unittest.mock import MagicMock, AsyncMock
from io import StringIO

from rich.console import Console

from loongcli.tui.commands import (
    CommandRegistry,
    CommandContext,
    SlashCommand,
    HelpCommand,
    ClearCommand,
    ModelCommand,
    CompactCommand,
    RememberCommand,
    ForgetCommand,
    MemoriesCommand,
    PlanCommand,
    create_default_registry,
)


def _make_ctx(memory=None, tui=None):
    console = Console(file=StringIO(), force_terminal=True)
    agent = MagicMock()
    agent.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    agent._last_prompt_tokens = 100
    agent.plan_mode = False
    agent._plan_mode = False
    agent._active_plan_id = None
    agent.plan_store = None
    agent.llm = MagicMock()
    agent.llm.model = "deepseek-v4-flash"
    agent.compactor = MagicMock()
    agent.compactor.compact = AsyncMock(return_value=[
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "summary"},
    ])
    registry = create_default_registry()
    return CommandContext(console=console, agent=agent, memory=memory, registry=registry, tui=tui)


def _output(ctx: CommandContext) -> str:
    ctx.console.file.seek(0)
    return ctx.console.file.read()


def test_registry_register_and_get():
    reg = CommandRegistry()
    cmd = HelpCommand()
    reg.register(cmd)
    assert reg.get("help") is cmd
    assert reg.get("nonexistent") is None


def test_create_default_registry():
    reg = create_default_registry()
    names = {c.name for c in reg.all_commands()}
    assert "help" in names
    assert "clear" in names
    assert "model" in names
    assert "compact" in names
    assert "remember" in names
    assert "forget" in names
    assert "memories" in names


@pytest.mark.asyncio
async def test_help_command():
    ctx = _make_ctx()
    cmd = HelpCommand()
    await cmd.run([], ctx)
    out = _output(ctx)
    assert "/help" in out
    assert "/exit" in out


@pytest.mark.asyncio
async def test_clear_command():
    ctx = _make_ctx()
    assert len(ctx.agent.messages) == 3
    cmd = ClearCommand()
    await cmd.run([], ctx)
    assert len(ctx.agent.messages) == 1
    assert ctx.agent.messages[0]["role"] == "system"
    assert ctx.agent._last_prompt_tokens == 0


@pytest.mark.asyncio
async def test_model_command_show():
    ctx = _make_ctx()
    cmd = ModelCommand()
    await cmd.run([], ctx)
    out = _output(ctx)
    assert "deepseek-v4-flash" in out


@pytest.mark.asyncio
async def test_model_command_switch():
    ctx = _make_ctx()
    cmd = ModelCommand()
    await cmd.run(["deepseek-v4-pro"], ctx)
    assert ctx.agent.llm.model == "deepseek-v4-pro"
    out = _output(ctx)
    assert "deepseek-v4-pro" in out


@pytest.mark.asyncio
async def test_compact_command_too_short():
    ctx = _make_ctx()
    cmd = CompactCommand()
    await cmd.run([], ctx)
    out = _output(ctx)
    assert "太短" in out


@pytest.mark.asyncio
async def test_compact_command_runs():
    ctx = _make_ctx()
    ctx.agent.messages = [{"role": "system", "content": "sys"}] + [
        {"role": "user", "content": f"msg{i}"} for i in range(20)
    ]
    cmd = CompactCommand()
    await cmd.run([], ctx)
    out = _output(ctx)
    assert "压缩" in out


@pytest.mark.asyncio
async def test_remember_command():
    memory = MagicMock()
    memory.save.return_value = "my-name"
    ctx = _make_ctx(memory=memory)
    cmd = RememberCommand()
    await cmd.run(["my-name", "description", "some content"], ctx)
    memory.save.assert_called_once_with(name="my-name", description="description", type="project", content="some content")
    out = _output(ctx)
    assert "已保存" in out


@pytest.mark.asyncio
async def test_remember_command_missing_args():
    ctx = _make_ctx(memory=MagicMock())
    cmd = RememberCommand()
    await cmd.run(["cat"], ctx)
    out = _output(ctx)
    assert "用法" in out


@pytest.mark.asyncio
async def test_forget_command():
    memory = MagicMock()
    memory.delete.return_value = True
    ctx = _make_ctx(memory=memory)
    cmd = ForgetCommand()
    await cmd.run(["my-name"], ctx)
    memory.delete.assert_called_once_with("my-name")
    out = _output(ctx)
    assert "已删除" in out


@pytest.mark.asyncio
async def test_memories_command():
    memory = MagicMock()
    memory.list_all.return_value = [
        {"name": "test-mem", "type": "user", "description": "a test memory"},
    ]
    ctx = _make_ctx(memory=memory)
    cmd = MemoriesCommand()
    await cmd.run([], ctx)
    out = _output(ctx)
    assert "test-mem" in out


@pytest.mark.asyncio
async def test_no_memory_shows_error():
    ctx = _make_ctx(memory=None)
    cmd = RememberCommand()
    await cmd.run(["a", "b", "c"], ctx)
    out = _output(ctx)
    assert "未启用" in out


def test_plan_command_in_registry():
    reg = create_default_registry()
    assert reg.get("plan") is not None


@pytest.mark.asyncio
async def test_plan_command_no_args():
    ctx = _make_ctx()
    cmd = PlanCommand()
    await cmd.run([], ctx)
    out = _output(ctx)
    assert "用法" in out
