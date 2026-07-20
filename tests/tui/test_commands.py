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
    from loongcli.core.llm import LLMClient
    agent.llm = LLMClient(api_key="k", model="deepseek-v4-flash")
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
async def test_model_command_provider_syntax():
    """/model provider:model 跨供应商切换：端点/密钥/vision 一起换。"""
    from loongcli.core.config import Config
    from loongcli.core.provider import ProviderConfig, RoleBinding

    ctx = _make_ctx()
    cfg = Config()
    cfg.providers = {"qwen": ProviderConfig(
        name="qwen", api_key="qk", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )}
    cfg.role_bindings = {"main": RoleBinding(provider="qwen", model="qwen3.7-plus", vision=True)}
    ctx.config = cfg

    await ModelCommand().run(["qwen:qwen3.7-plus"], ctx)
    llm = ctx.agent.llm
    assert llm.model == "qwen3.7-plus"
    assert "dashscope" in llm.base_url
    assert llm.vision is True  # 从 roles 同名绑定继承


@pytest.mark.asyncio
async def test_model_command_provider_syntax_unknown_provider():
    from loongcli.core.config import Config

    ctx = _make_ctx()
    ctx.config = Config()
    await ModelCommand().run(["nope:some-model"], ctx)
    assert ctx.agent.llm.model == "deepseek-v4-flash"  # 未切换
    assert "未配置 provider" in _output(ctx)


@pytest.mark.asyncio
async def test_model_command_bare_switch_resets_stale_vision():
    """裸模型名切换：roles 里查不到 vision 的一律关掉，防止 vision 残留把图喂给纯文本模型。"""
    ctx = _make_ctx()
    ctx.agent.llm.vision = True  # 模拟此前是视觉模型
    await ModelCommand().run(["deepseek-v4-flash"], ctx)
    assert ctx.agent.llm.vision is False


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


@pytest.mark.asyncio
async def test_verbose_command_toggles():
    from loongcli.tui.commands import VerboseCommand

    class _FakeTui:
        verbose = False

    tui = _FakeTui()
    ctx = _make_ctx(tui=tui)
    cmd = VerboseCommand()

    await cmd.run([], ctx)
    assert tui.verbose is True
    assert "详细模式已开启" in _output(ctx)

    await cmd.run([], ctx)
    assert tui.verbose is False
    assert "折叠模式已恢复" in _output(ctx)


@pytest.mark.asyncio
async def test_verbose_command_no_tui():
    ctx = _make_ctx(tui=None)
    from loongcli.tui.commands import VerboseCommand
    await VerboseCommand().run([], ctx)
    assert "无法切换" in _output(ctx)


# ── /model --save 持久化 + /config 查看 ──

def test_persist_main_role_writes_config(tmp_path):
    from loongcli.tui.commands import _persist_main_role

    path = tmp_path / "config.json"
    path.write_text('{"api_key": "sk-old", "roles": {"sub": {"provider": "deepseek", "model": "x"}}}',
                    encoding="utf-8")
    _persist_main_role("deepseek-v4-pro", "deepseek", False, config_path=path)
    data = __import__("json").loads(path.read_text(encoding="utf-8"))
    # vision=False 不写入（loader 缺省即 False，等价）——避免覆盖用户手工设的 vision:true
    assert data["roles"]["main"] == {"provider": "deepseek", "model": "deepseek-v4-pro"}
    assert data["roles"]["sub"]["model"] == "x"      # 其他角色不动
    assert data["api_key"] == "sk-old"               # 其他字段不动


def test_persist_main_role_writes_vision_true_and_keeps_prior(tmp_path):
    """vision=True 显式写入；且已有的 vision:true 不被裸模型切换（vision=False）清掉。"""
    from loongcli.tui.commands import _persist_main_role

    path = tmp_path / "config.json"
    # 用户手工设了 main.vision:true
    path.write_text('{"roles": {"main": {"provider": "qwen", "model": "v", "vision": true}}}',
                    encoding="utf-8")
    # 裸模型名切换（vision 解析不到 → False）不应清掉已有 vision
    _persist_main_role("v2", None, False, config_path=path)
    data = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert data["roles"]["main"]["vision"] is True
    # 显式 vision=True 正常写入
    _persist_main_role("v3", None, True, config_path=path)
    data = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert data["roles"]["main"]["vision"] is True


@pytest.mark.asyncio
async def test_model_save_calls_persist(monkeypatch, tmp_path):
    """/model provider:model --save → 切换成功后把 roles.main 写回 config。"""
    from loongcli.core.config import Config
    from loongcli.core.provider import ProviderConfig
    import loongcli.tui.commands as cmds

    calls = {}
    monkeypatch.setattr(cmds, "_persist_main_role",
                        lambda model, provider, vision, config_path=None:
                        calls.update(model=model, provider=provider, vision=vision) or "p")

    ctx = _make_ctx()
    cfg = Config()
    cfg.providers = {"deepseek": ProviderConfig(name="deepseek", api_key="dk",
                                                base_url="https://api.deepseek.com")}
    ctx.config = cfg
    await cmds.ModelCommand().run(["deepseek:deepseek-v4-pro", "--save"], ctx)
    assert ctx.agent.llm.model == "deepseek-v4-pro"
    assert calls == {"model": "deepseek-v4-pro", "provider": "deepseek", "vision": False}


@pytest.mark.asyncio
async def test_config_command_masks_keys():
    from loongcli.core.config import Config
    from loongcli.core.provider import ProviderConfig
    from loongcli.tui.commands import ConfigCommand

    ctx = _make_ctx()
    cfg = Config()
    cfg.providers = {"deepseek": ProviderConfig(
        name="deepseek", api_key="sk-3271234567890abcdefa595",
        base_url="https://api.deepseek.com",
    )}
    ctx.config = cfg
    await ConfigCommand().run([], ctx)
    out = _output(ctx)
    assert "sk-3271234567890abcdefa595" not in out   # 完整密钥绝不出现
    assert "sk-32" in out and "a595" in out          # 打码形态
    assert "deepseek-v4-flash" in out                # 当前生效模型
