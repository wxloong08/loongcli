import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from rich.console import Console
from io import StringIO

from loongcli.tui.commands import InitCommand, UsageCommand, FastCommand, ProCommand, CommandContext, create_default_registry


def _make_ctx(tmp_path=None, token_usage=None):
    from loongcli.core.llm import LLMClient

    console = Console(file=StringIO())
    agent = MagicMock()
    agent.llm = LLMClient(api_key="k", model="deepseek-v4-pro")  # 默认 deepseek 端点
    agent.token_usage = token_usage or {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0,
        "reasoning_tokens": 0,
    }
    agent.cost_tracker = None
    return CommandContext(console=console, agent=agent)


@pytest.mark.asyncio
async def test_init_creates_loongcli_md(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = _make_ctx()
    cmd = InitCommand()
    await cmd.run([], ctx)
    f = tmp_path / "LOONG.md"
    assert f.exists()
    content = f.read_text(encoding="utf-8")
    assert "项目指引" in content
    assert "代码规范" in content


@pytest.mark.asyncio
async def test_init_refuses_if_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "LOONG.md").write_text("existing", encoding="utf-8")
    ctx = _make_ctx()
    cmd = InitCommand()
    await cmd.run([], ctx)
    assert (tmp_path / "LOONG.md").read_text(encoding="utf-8") == "existing"


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
    # 无 config（providers 未知）但当前端点是 deepseek → 原地换模型
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


@pytest.mark.asyncio
async def test_fast_switches_endpoint_from_qwen_main():
    """主力 qwen 时 /fast 必须整体切到 deepseek provider（端点+密钥），不能只换模型名打错端点。"""
    from loongcli.core.config import Config
    from loongcli.core.llm import LLMClient
    from loongcli.core.provider import ProviderConfig

    ctx = _make_ctx()
    ctx.agent.llm = LLMClient(
        api_key="qk", model="qwen3.7-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", vision=True,
    )
    cfg = Config()
    cfg.providers = {"deepseek": ProviderConfig(name="deepseek", api_key="dk", base_url="https://api.deepseek.com")}
    ctx.config = cfg

    await FastCommand().run([], ctx)
    llm = ctx.agent.llm
    assert llm.model == "deepseek-v4-flash"
    assert "deepseek" in llm.base_url
    assert llm.vision is False          # vision 不残留
    assert llm._provider_type == "deepseek"  # provider_type 已重推断（cache_aware 属性已删）


@pytest.mark.asyncio
async def test_fast_refuses_without_deepseek_provider():
    from loongcli.core.config import Config
    from loongcli.core.llm import LLMClient

    ctx = _make_ctx()
    ctx.agent.llm = LLMClient(
        api_key="qk", model="qwen3.7-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    ctx.config = Config()  # 无 providers

    await FastCommand().run([], ctx)
    assert ctx.agent.llm.model == "qwen3.7-plus"  # 未切换
    assert "无法切换" in ctx.console.file.getvalue()


def test_init_and_usage_in_default_registry():
    reg = create_default_registry()
    assert reg.get("init") is not None
    assert reg.get("usage") is not None
    assert reg.get("fast") is not None
    assert reg.get("pro") is not None
