import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from rich.console import Console

from loongcli.core.config import Config
from loongcli.main import _onboarding


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    config_dir = home / ".loongcli"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps({
        "api_key": "",
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
    }), encoding="utf-8")
    return home


class TestOnboarding:
    def test_saves_api_key_and_provider(self, fake_home):
        console = Console(quiet=True)
        cfg = Config.load()
        with patch("builtins.input", side_effect=["sk-test123", "1"]):
            result = _onboarding(console, cfg)

        assert result.api_key == "sk-test123"
        assert result.base_url == "https://api.deepseek.com"
        assert result.model == "deepseek-v4-flash"

        data = json.loads(
            (fake_home / ".loongcli" / "config.json").read_text(encoding="utf-8")
        )
        assert data["api_key"] == "sk-test123"

    def test_openai_provider(self, fake_home):
        console = Console(quiet=True)
        cfg = Config.load()
        with patch("builtins.input", side_effect=["sk-oai", "2"]):
            result = _onboarding(console, cfg)

        assert result.api_key == "sk-oai"
        assert result.base_url == "https://api.openai.com/v1"
        assert result.model == "gpt-4o"

    def test_custom_provider_asks_url_and_model(self, fake_home):
        console = Console(quiet=True)
        cfg = Config.load()
        with patch("builtins.input", side_effect=[
            "sk-custom",
            "3",
            "https://my-proxy.com/v1",
            "claude-sonnet-4-20250514",
        ]):
            result = _onboarding(console, cfg)

        assert result.base_url == "https://my-proxy.com/v1"
        assert result.model == "claude-sonnet-4-20250514"

    def test_empty_key_returns_original(self, fake_home):
        console = Console(quiet=True)
        cfg = Config.load()
        with patch("builtins.input", side_effect=[""]):
            result = _onboarding(console, cfg)

        assert result.api_key == ""

    def test_keyboard_interrupt_returns_original(self, fake_home):
        console = Console(quiet=True)
        cfg = Config.load()
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            result = _onboarding(console, cfg)

        assert result.api_key == ""

    def test_eof_returns_original(self, fake_home):
        console = Console(quiet=True)
        cfg = Config.load()
        with patch("builtins.input", side_effect=EOFError):
            result = _onboarding(console, cfg)

        assert result.api_key == ""

    def test_ollama_provider(self, fake_home):
        console = Console(quiet=True)
        cfg = Config.load()
        with patch("builtins.input", side_effect=["sk-ollama", "4", "llama3"]):
            result = _onboarding(console, cfg)

        assert result.base_url == "http://localhost:11434/v1"
        assert result.model == "llama3"

    def test_invalid_choice_defaults_to_deepseek(self, fake_home):
        console = Console(quiet=True)
        cfg = Config.load()
        with patch("builtins.input", side_effect=["sk-test", "9"]):
            result = _onboarding(console, cfg)

        assert result.base_url == "https://api.deepseek.com"
        assert result.model == "deepseek-v4-flash"
