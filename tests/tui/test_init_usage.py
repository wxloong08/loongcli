import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from rich.console import Console
from io import StringIO

from loongcli.tui.commands import InitCommand, UsageCommand, FastCommand, ProCommand, CommandContext, create_default_registry


def _make_ctx(tmp_path=None, token_usage=None):
    console = Console(file=StringIO())
    agent = MagicMock()
    agent.token_usage = token_usage or {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0,
        "reasoning_tokens": 0,
    }
    return CommandContext(console=console, agent=agent)


@pytest.mark.asyncio
async def test_init_creates_loongcli_md(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = _make_ctx()
    cmd = InitCommand()
    await cmd.run([], ctx)
    f = tmp_path / "DSCLI.md"
    assert f.exists()
    content = f.read_text(encoding="utf-8")
    assert "项目指引" in content
    assert "代码规范" in content


@pytest.mark.asyncio
async def test_init_refuses_if_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "DSCLI.md").write_text("existing", encoding="utf-8")
    ctx = _make_ctx()
    cmd = InitCommand()
    await cmd.run([], ctx)
    assert (tmp_path / "DSCLI.md").read_text(encoding="utf-8") == "existing"


@pytest.mark.asyncio
async def test_usage_shows_tokens():
    ctx = _make_ctx(token_usage={
        "prompt_tokens": 1500,
        "completion_tokens": 300,
        "total_tokens": 1800,
        "prompt_cache_hit_tokens": 1000,
        "prompt_cache_miss_tokens": 500,
        "reasoning_tokens": 100,
    })
    cmd = UsageCommand()
    await cmd.run([], ctx)
    output = ctx.console.file.getvalue()
    assert "1,500" in output
    assert "300" in output
    assert "1,800" in output


@pytest.mark.asyncio
async def test_usage_shows_cache_and_thinking():
    ctx = _make_ctx(token_usage={
        "prompt_tokens": 2000,
        "completion_tokens": 500,
        "total_tokens": 2500,
        "prompt_cache_hit_tokens": 1500,
        "prompt_cache_miss_tokens": 500,
        "reasoning_tokens": 200,
    })
    cmd = UsageCommand()
    await cmd.run([], ctx)
    output = ctx.console.file.getvalue()
    assert "1,500" in output
    assert "200" in output


@pytest.mark.asyncio
async def test_fast_command():
    ctx = _make_ctx()
    cmd = FastCommand()
    await cmd.run([], ctx)
    assert ctx.agent.llm.model == "deepseek-v4-flash"


@pytest.mark.asyncio
async def test_pro_command():
    ctx = _make_ctx()
    cmd = ProCommand()
    await cmd.run([], ctx)
    assert ctx.agent.llm.model == "deepseek-v4-pro"


def test_init_and_usage_in_default_registry():
    reg = create_default_registry()
    assert reg.get("init") is not None
    assert reg.get("usage") is not None
    assert reg.get("fast") is not None
    assert reg.get("pro") is not None
