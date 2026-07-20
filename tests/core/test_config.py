import json
import os
import pytest
from pathlib import Path

from loongcli.core.config import Config, parse_token_size


def test_defaults():
    cfg = Config.load(path=Path("/nonexistent/config.json"))
    assert cfg.api_key == ""
    assert cfg.model == "deepseek-v4-flash"
    assert cfg.base_url == "https://api.deepseek.com"
    assert cfg.mcp_servers == {}


def test_load_from_file(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "api_key": "sk-file",
        "model": "deepseek-v4-pro",
        "base_url": "https://custom.api.com",
        "mcpServers": {"demo": {"command": "python", "args": ["s.py"]}},
    }), encoding="utf-8")

    cfg = Config.load(path=p)
    assert cfg.api_key == "sk-file"
    assert cfg.model == "deepseek-v4-pro"
    assert cfg.base_url == "https://custom.api.com"
    assert "demo" in cfg.mcp_servers


def test_role_vision_parsed(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "api_key": "sk-x",
        "roles": {
            "main": {"provider": "_default", "model": "qwen3.7-plus", "vision": True},
            "utility": {"provider": "_default", "model": "deepseek-v4-flash"},
        },
    }), encoding="utf-8")

    cfg = Config.load(path=p)
    assert cfg.role_bindings["main"].vision is True
    # 未配置 vision 的 role 默认 False
    assert cfg.role_bindings["utility"].vision is False


def test_env_overrides_file(tmp_path, monkeypatch):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "api_key": "sk-file",
        "model": "file-model",
        "base_url": "https://file.api.com",
    }), encoding="utf-8")

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
    monkeypatch.setenv("DSCLI_MODEL", "env-model")
    monkeypatch.setenv("DSCLI_BASE_URL", "https://env.api.com")

    cfg = Config.load(path=p)
    assert cfg.api_key == "sk-env"
    assert cfg.model == "env-model"
    assert cfg.base_url == "https://env.api.com"


def test_invalid_json(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{broken", encoding="utf-8")
    cfg = Config.load(path=p)
    assert cfg.model == "deepseek-v4-flash"


def test_malformed_token_size_does_not_crash(tmp_path):
    """畸形 token 字段（如 context_window: "abc"）按 0 处理，不击穿启动。"""
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "api_key": "sk-test",
        "model_max_tokens": "not-a-number",
        "roles": {"main": {"model": "x", "context_window": "abc"}},
    }), encoding="utf-8")
    cfg = Config.load(path=p)  # 不抛
    assert cfg.model_max_tokens == 0
    assert cfg.role_bindings["main"].context_window == 0


def test_sub_model_default():
    cfg = Config.load(path=Path("/nonexistent/config.json"))
    assert cfg.sub_model == ""


def test_sub_model_from_file(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "api_key": "sk-test",
        "model": "deepseek-v4-pro",
        "sub_model": "deepseek-v4-flash",
    }), encoding="utf-8")
    cfg = Config.load(path=p)
    assert cfg.sub_model == "deepseek-v4-flash"


def test_sub_model_from_env(tmp_path, monkeypatch):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"api_key": "sk-test"}), encoding="utf-8")
    monkeypatch.setenv("DSCLI_SUB_MODEL", "deepseek-v4-flash")
    cfg = Config.load(path=p)
    assert cfg.sub_model == "deepseek-v4-flash"


class TestAutoCreateConfig:
    def test_creates_dir_and_file(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "home"
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        config_path = fake_home / ".loongcli" / "config.json"
        assert not config_path.exists()

        cfg = Config.load()

        assert config_path.exists()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert "api_key" in data
        assert data["model"] == "deepseek-v4-flash"

    def test_does_not_overwrite_existing(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "home"
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        config_dir = fake_home / ".loongcli"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "config.json"
        config_path.write_text(json.dumps({"api_key": "sk-existing"}), encoding="utf-8")

        cfg = Config.load()

        assert cfg.api_key == "sk-existing"
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert data["api_key"] == "sk-existing"

    def test_custom_path_skips_creation(self):
        cfg = Config.load(path=Path("/nonexistent/config.json"))
        assert cfg.api_key == ""

    def test_default_template_is_valid_json(self):
        from loongcli.core.config import _DEFAULT_CONFIG
        data = json.loads(_DEFAULT_CONFIG)
        assert isinstance(data, dict)
        assert "api_key" in data
        assert "mcpServers" in data


class TestParseTokenSize:
    def test_plain_int(self):
        assert parse_token_size(131072) == 131072
        assert parse_token_size(0) == 0

    def test_k_suffix_lower(self):
        assert parse_token_size("128k") == 128000
        assert parse_token_size("1k") == 1000

    def test_k_suffix_upper(self):
        assert parse_token_size("128K") == 128000

    def test_m_suffix(self):
        assert parse_token_size("1M") == 1_000_000
        assert parse_token_size("1m") == 1_000_000
        assert parse_token_size("1.5M") == 1_500_000

    def test_b_suffix(self):
        assert parse_token_size("2B") == 2_000_000_000
        assert parse_token_size("0.5B") == 500_000_000

    def test_no_suffix(self):
        assert parse_token_size("65536") == 65536

    def test_whitespace(self):
        assert parse_token_size(" 1M ") == 1_000_000

    def test_empty(self):
        assert parse_token_size("") == 0
        assert parse_token_size(0) == 0

    def test_zero_value(self):
        assert parse_token_size("0") == 0
        assert parse_token_size("0k") == 0

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="无法解析"):
            parse_token_size("abc")
        with pytest.raises(ValueError, match="无法解析"):
            parse_token_size("1X")


class TestConfigTokenSizeParsing:
    def test_model_max_tokens_from_string(self, tmp_path):
        p = tmp_path / "config.json"
        p.write_text(json.dumps({
            "api_key": "sk-test",
            "model_max_tokens": "1M",
        }), encoding="utf-8")
        cfg = Config.load(path=p)
        assert cfg.model_max_tokens == 1_000_000

    def test_compact_threshold_from_string(self, tmp_path):
        p = tmp_path / "config.json"
        p.write_text(json.dumps({
            "api_key": "sk-test",
            "compact_threshold": "128k",
        }), encoding="utf-8")
        cfg = Config.load(path=p)
        assert cfg.compact_threshold == 128000

    def test_model_max_tokens_zero_means_auto(self, tmp_path):
        p = tmp_path / "config.json"
        p.write_text(json.dumps({"api_key": "sk-test"}), encoding="utf-8")
        cfg = Config.load(path=p)
        assert cfg.model_max_tokens == 0
