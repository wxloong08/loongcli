"""配置向导测试：providers+roles 规范结构、DS-first 回车路径、双供应商视觉支线。"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from loongcli.core.config import Config
from loongcli.main import _onboarding, _parse_args


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    config_dir = home / ".loongcli"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(json.dumps({
        "api_key": "",
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
    }), encoding="utf-8")
    return home


def _config_data(home) -> dict:
    return json.loads((home / ".loongcli" / "config.json").read_text(encoding="utf-8"))


class TestOnboarding:
    def test_enter_all_the_way_gives_deepseek(self, fake_home):
        """DS-first：回车到底（选默认、默认模型、不配第二供应商）即可用。"""
        console = Console(quiet=True)
        # 顺序：供应商选择、API Key、模型、第二供应商
        with patch("builtins.input", side_effect=["", "sk-test123", "", ""]):
            result = _onboarding(console, Config.load())

        data = _config_data(fake_home)
        assert data["providers"]["deepseek"]["api_key"] == "sk-test123"
        # 默认主力 deepseek-v4-pro（用户改定），utility 仍走便宜档 flash
        assert data["roles"]["main"] == {
            "provider": "deepseek", "model": "deepseek-v4-pro",
            "thinking": True, "reasoning_effort": "max",
        }
        assert data["roles"]["utility"]["model"] == "deepseek-v4-flash"
        # 遗留顶层字段同步写（无 roles 时的回退路径）
        assert data["api_key"] == "sk-test123"
        # 返回的 Config 已加载新结构
        assert "deepseek" in result.providers
        assert result.role_bindings["main"].model == "deepseek-v4-pro"

    def test_qwen_provider(self, fake_home):
        console = Console(quiet=True)
        with patch("builtins.input", side_effect=["2", "sk-q", "", ""]):
            _onboarding(console, Config.load())
        data = _config_data(fake_home)
        assert "dashscope" in data["providers"]["qwen"]["base_url"]
        assert data["roles"]["main"]["model"] == "qwen3.7-plus"

    def test_custom_provider_asks_name_url_model(self, fake_home):
        console = Console(quiet=True)
        with patch("builtins.input", side_effect=[
            "3", "glm", "https://open.bigmodel.cn/api/paas/v4", "sk-glm", "glm-5", "",
        ]):
            _onboarding(console, Config.load())
        data = _config_data(fake_home)
        assert data["providers"]["glm"]["base_url"] == "https://open.bigmodel.cn/api/paas/v4"
        assert data["roles"]["main"] == {
            "provider": "glm", "model": "glm-5", "thinking": True, "reasoning_effort": "max",
        }
        # 非 deepseek 供应商：utility 与主力同模型
        assert data["roles"]["utility"]["model"] == "glm-5"

    def test_second_provider_vision_role(self, fake_home):
        """双供应商支线：第二供应商记成独立 vision 角色，不动主力。"""
        console = Console(quiet=True)
        with patch("builtins.input", side_effect=[
            "", "sk-d", "", "y",      # deepseek 主力
            "", "sk-q", "", "",       # 第二供应商：qwen 默认、key、默认模型、vision 默认 Y
        ]):
            _onboarding(console, Config.load())
        data = _config_data(fake_home)
        assert set(data["providers"]) == {"deepseek", "qwen"}
        assert data["roles"]["main"]["provider"] == "deepseek"
        assert data["roles"]["vision"] == {
            "provider": "qwen", "model": "qwen3.7-plus", "vision": True,
        }

    def test_empty_key_returns_original(self, fake_home):
        console = Console(quiet=True)
        with patch("builtins.input", side_effect=["", ""]):
            result = _onboarding(console, Config.load())
        assert result.api_key == ""
        assert _config_data(fake_home)["api_key"] == ""

    def test_keyboard_interrupt_returns_original(self, fake_home):
        console = Console(quiet=True)
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            result = _onboarding(console, Config.load())
        assert result.api_key == ""

    def test_eof_returns_original(self, fake_home):
        console = Console(quiet=True)
        with patch("builtins.input", side_effect=EOFError):
            result = _onboarding(console, Config.load())
        assert result.api_key == ""

    def test_invalid_choice_defaults_to_deepseek(self, fake_home):
        console = Console(quiet=True)
        with patch("builtins.input", side_effect=["9", "sk-x", "", ""]):
            _onboarding(console, Config.load())
        data = _config_data(fake_home)
        assert "deepseek" in data["providers"]


def test_setup_flag_parsed(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["loongcli", "--setup"])
    args = _parse_args()
    assert args.setup is True
