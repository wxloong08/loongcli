from __future__ import annotations
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _safe_token_size(value: Any, field_name: str = "") -> int:
    """load 里用的容错版 parse_token_size：畸形值（如 context_window: "abc"）记一条
    警告并按 0（自动）处理，而不是让一个配置打字击穿整个启动。"""
    try:
        return parse_token_size(value)
    except ValueError as e:
        logger.warning("config %s 解析失败，按自动处理：%s", field_name or "token 字段", e)
        return 0


def parse_token_size(value: Any) -> int:
    """Parse human-readable token sizes like '128k', '1M', '1.5M', or plain ints."""
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        return 0
    value = value.strip().upper()
    if not value:
        return 0
    match = re.match(r'^([\d.]+)\s*(K|M|B)?$', value)
    if not match:
        raise ValueError(f"无法解析 token 大小: {value!r}，支持格式如 128k, 1M, 2B")
    num = float(match.group(1))
    unit = match.group(2)
    multiplier = {"K": 1000, "M": 1_000_000, "B": 1_000_000_000}.get(unit, 1)
    return int(num * multiplier)


@dataclass
class ModelProfile:
    model: str
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    max_tokens: int = 0  # 0 = auto-detect from model name

    def effective_api_key(self, fallback: str) -> str:
        return self.api_key or fallback


@dataclass
class Config:
    api_key: str = ""
    model: str = "deepseek-v4-flash"
    sub_model: str = ""
    base_url: str = "https://api.deepseek.com"
    model_max_tokens: int = 0  # 0 = auto-detect from model name
    compact_threshold: int = 0  # 0 = auto (model context - 13K summary reserve)
    thinking: bool = True
    reasoning_effort: str = "max"
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    hooks: dict[str, Any] = field(default_factory=dict)
    skill_dirs: list[str] = field(default_factory=list)
    model_profiles: dict[str, ModelProfile] = field(default_factory=dict)
    providers: dict[str, Any] = field(default_factory=dict)
    role_bindings: dict[str, Any] = field(default_factory=dict)

    def get_profile(self, name: str) -> ModelProfile | None:
        return self.model_profiles.get(name)

    def list_profiles(self) -> list[str]:
        return list(self.model_profiles.keys())

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        default = path is None
        path = path or Path.home() / ".loongcli" / "config.json"
        if default:
            cls._ensure_config_dir(path)
        file_data: dict[str, Any] = {}
        if path.exists():
            try:
                file_data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        profiles: dict[str, ModelProfile] = {}
        for name, pdata in file_data.get("models", {}).items():
            profiles[name] = ModelProfile(
                model=pdata.get("model", name),
                api_key=pdata.get("api_key", ""),
                base_url=pdata.get("base_url", "https://api.deepseek.com"),
                max_tokens=_safe_token_size(pdata.get("max_tokens", 0), f"models.{name}.max_tokens"),
            )

        from loongcli.core.provider import ProviderConfig, RoleBinding
        providers: dict[str, ProviderConfig] = {}
        for name, pdata in file_data.get("providers", {}).items():
            providers[name] = ProviderConfig(
                name=name,
                api_key=pdata.get("api_key", ""),
                base_url=pdata.get("base_url", "https://api.deepseek.com"),
            )

        role_bindings: dict[str, RoleBinding] = {}
        for name, rdata in file_data.get("roles", {}).items():
            role_bindings[name] = RoleBinding(
                provider=rdata.get("provider", "_default"),
                model=rdata.get("model", "deepseek-v4-flash"),
                thinking=rdata.get("thinking", False),
                reasoning_effort=rdata.get("reasoning_effort", "max"),
                context_window=_safe_token_size(rdata.get("context_window", 0), f"roles.{name}.context_window"),
                vision=rdata.get("vision", False),
            )

        return cls(
            api_key=os.environ.get("DEEPSEEK_API_KEY") or file_data.get("api_key", ""),
            model=os.environ.get("DSCLI_MODEL") or file_data.get("model", "deepseek-v4-flash"),
            sub_model=os.environ.get("DSCLI_SUB_MODEL") or file_data.get("sub_model", ""),
            base_url=os.environ.get("DSCLI_BASE_URL") or file_data.get("base_url", "https://api.deepseek.com"),
            model_max_tokens=_safe_token_size(file_data.get("model_max_tokens", 0), "model_max_tokens"),
            compact_threshold=_safe_token_size(file_data.get("compact_threshold", 0), "compact_threshold"),
            thinking=file_data.get("thinking", True),
            reasoning_effort=file_data.get("reasoning_effort", "max"),
            skill_dirs=file_data.get("skill_dirs", []),
            mcp_servers=file_data.get("mcpServers", {}),
            hooks=file_data.get("hooks", {}),
            model_profiles=profiles,
            providers=providers,
            role_bindings=role_bindings,
        )

    @staticmethod
    def _ensure_config_dir(config_path: Path) -> None:
        config_dir = config_path.parent
        config_dir.mkdir(parents=True, exist_ok=True)
        if not config_path.exists():
            config_path.write_text(_DEFAULT_CONFIG, encoding="utf-8")


_DEFAULT_CONFIG = json.dumps({
    "api_key": "",
    "model": "deepseek-v4-flash",
    "base_url": "https://api.deepseek.com",
    "thinking": True,
    "reasoning_effort": "max",
    "mcpServers": {},
    "hooks": {},
    "providers": {},
    "roles": {},
}, indent=2, ensure_ascii=False) + "\n"
