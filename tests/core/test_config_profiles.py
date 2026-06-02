import json
import pytest
from pathlib import Path
from loongcli.core.config import Config, ModelProfile


def test_load_no_profiles(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"api_key": "test"}), encoding="utf-8")
    cfg = Config.load(cfg_path)
    assert cfg.model_profiles == {}
    assert cfg.list_profiles() == []


def test_load_with_profiles(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "api_key": "default-key",
        "models": {
            "gpt4o": {
                "model": "gpt-4o",
                "api_key": "openai-key",
                "base_url": "https://api.openai.com/v1",
            },
            "claude": {
                "model": "claude-sonnet-4-20250514",
                "api_key": "anthropic-key",
                "base_url": "https://api.anthropic.com/v1",
            },
        },
    }), encoding="utf-8")
    cfg = Config.load(cfg_path)
    assert len(cfg.model_profiles) == 2
    assert "gpt4o" in cfg.list_profiles()
    assert "claude" in cfg.list_profiles()

    gpt = cfg.get_profile("gpt4o")
    assert gpt.model == "gpt-4o"
    assert gpt.api_key == "openai-key"
    assert gpt.base_url == "https://api.openai.com/v1"


def test_profile_effective_api_key():
    p = ModelProfile(model="test", api_key="own-key")
    assert p.effective_api_key("fallback") == "own-key"

    p2 = ModelProfile(model="test", api_key="")
    assert p2.effective_api_key("fallback") == "fallback"


def test_get_nonexistent_profile(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"api_key": "test"}), encoding="utf-8")
    cfg = Config.load(cfg_path)
    assert cfg.get_profile("nonexistent") is None


def test_profile_model_defaults_to_name(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "api_key": "test",
        "models": {
            "deepseek-v4-pro": {
                "base_url": "https://api.deepseek.com",
            },
        },
    }), encoding="utf-8")
    cfg = Config.load(cfg_path)
    p = cfg.get_profile("deepseek-v4-pro")
    assert p.model == "deepseek-v4-pro"
