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
        "model": "deepseek-reasoner",
        "base_url": "https://custom.api.com",
        "mcpServers": {"demo": {"command": "python", "args": ["s.py"]}},
    }), encoding="utf-8")

    cfg = Config.load(path=p)
    assert cfg.api_key == "sk-file"
    assert cfg.model == "deepseek-reasoner"
    assert cfg.base_url == "https://custom.api.com"
    assert "demo" in cfg.mcp_servers


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


def test_sub_model_default():
    cfg = Config.load(path=Path("/nonexistent/config.json"))
    assert cfg.sub_model == ""


def test_sub_model_from_file(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "api_key": "sk-test",
        "model": "deepseek-reasoner",
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
