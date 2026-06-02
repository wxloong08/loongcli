from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelProfile:
    model: str
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"

    def effective_api_key(self, fallback: str) -> str:
        return self.api_key or fallback


@dataclass
class Config:
    api_key: str = ""
    model: str = "deepseek-v4-flash"
    sub_model: str = ""
    base_url: str = "https://api.deepseek.com"
    compact_threshold: int = 800000
    thinking: bool = True
    reasoning_effort: str = "max"
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    hooks: dict[str, Any] = field(default_factory=dict)
    skill_dirs: list[str] = field(default_factory=list)
    model_profiles: dict[str, ModelProfile] = field(default_factory=dict)

    def get_profile(self, name: str) -> ModelProfile | None:
        return self.model_profiles.get(name)

    def list_profiles(self) -> list[str]:
        return list(self.model_profiles.keys())

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        path = path or Path.home() / ".loongcli" / "config.json"
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
            )

        return cls(
            api_key=os.environ.get("DEEPSEEK_API_KEY") or file_data.get("api_key", ""),
            model=os.environ.get("DSCLI_MODEL") or file_data.get("model", "deepseek-v4-flash"),
            sub_model=os.environ.get("DSCLI_SUB_MODEL") or file_data.get("sub_model", ""),
            base_url=os.environ.get("DSCLI_BASE_URL") or file_data.get("base_url", "https://api.deepseek.com"),
            compact_threshold=int(file_data.get("compact_threshold", 800000)),
            thinking=file_data.get("thinking", True),
            reasoning_effort=file_data.get("reasoning_effort", "max"),
            skill_dirs=file_data.get("skill_dirs", []),
            mcp_servers=file_data.get("mcpServers", {}),
            hooks=file_data.get("hooks", {}),
            model_profiles=profiles,
        )
