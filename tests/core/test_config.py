import json
import os
import pytest
from pathlib import Path

from loongcli.core.config import Config


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
