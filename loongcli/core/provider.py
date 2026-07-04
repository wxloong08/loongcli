from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loongcli.core.llm import LLMClient

logger = logging.getLogger(__name__)

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "deepseek-v4-flash": 1_048_576,
    "deepseek-v4-pro": 1_048_576,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_048_576,
    "gpt-4.1-mini": 1_048_576,
    "gpt-4.1-nano": 1_048_576,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-haiku-4-5": 200_000,
    # qwen3.7 旗舰系（plus/max 均 1M，百炼官方；前缀匹配覆盖 -latest 等变体）
    "qwen3.7-plus": 1_048_576,
    "qwen3.7-max": 1_048_576,
}

DEFAULT_CONTEXT_WINDOW = 131_072


@dataclass
class ProviderConfig:
    """Authentication and endpoint for one API provider."""
    name: str
    api_key: str
    base_url: str

    @property
    def provider_type(self) -> str:
        """Infer provider type from base_url for parameter adaptation."""
        url = self.base_url.lower()
        if "deepseek" in url:
            return "deepseek"
        if "dashscope" in url or "aliyuncs" in url:
            return "qwen"
        if "openai" in url or "api.openai.com" in url:
            return "openai"
        if "anthropic" in url:
            return "anthropic"
        if "localhost" in url or "127.0.0.1" in url:
            return "local"
        return "openai"


@dataclass
class RoleBinding:
    """Binds a role to a provider + model + parameters."""
    provider: str
    model: str
    thinking: bool = False
    reasoning_effort: str = "max"
    context_window: int = 0
    # 该 role 的模型是否具备视觉能力（能吃图片）。只认显式配置，不按模型名猜——
    # 原生多模态旗舰（如 qwen3.7-plus）名字里没 -vl-，名字表必然误判。
    vision: bool = False


DEFAULT_ROLES: dict[str, RoleBinding] = {
    "main": RoleBinding(provider="_default", model="deepseek-v4-flash", thinking=True, reasoning_effort="max"),
    "sub": RoleBinding(provider="_default", model="deepseek-v4-flash", thinking=False),
    "utility": RoleBinding(provider="_default", model="deepseek-v4-flash", thinking=False),
}


def context_window(model: str, override: int = 0) -> int:
    """Return context window for a model. Override > 0 takes precedence."""
    if override > 0:
        return override
    for prefix, size in MODEL_CONTEXT_WINDOWS.items():
        if model.startswith(prefix):
            return size
    return DEFAULT_CONTEXT_WINDOW


class ModelRouter:
    """Route LLMClient instances by role (main / sub / utility).

    Lazily creates clients on first access and caches them.
    """

    def __init__(
        self,
        providers: dict[str, ProviderConfig],
        roles: dict[str, RoleBinding],
    ):
        self._providers = providers
        self._roles = roles
        self._clients: dict[str, LLMClient] = {}

    def client(self, role: str = "main") -> "LLMClient":
        """Get or create an LLMClient for the given role."""
        if role in self._clients:
            return self._clients[role]

        binding = self._roles.get(role)
        if not binding:
            binding = self._roles.get("main") or DEFAULT_ROLES["main"]

        provider = self._providers.get(binding.provider)
        if not provider:
            provider = self._providers.get("_default")
        if not provider:
            raise ValueError(f"Provider '{binding.provider}' not found for role '{role}'")

        from loongcli.core.llm import LLMClient
        client = LLMClient(
            api_key=provider.api_key,
            model=binding.model,
            base_url=provider.base_url,
            thinking=binding.thinking,
            reasoning_effort=binding.reasoning_effort,
            provider_type=provider.provider_type,
            vision=binding.vision,
        )
        self._clients[role] = client
        return client

    def context_window_for(self, role: str = "main") -> int:
        binding = self._roles.get(role) or DEFAULT_ROLES.get("main")
        if not binding:
            return DEFAULT_CONTEXT_WINDOW
        return context_window(binding.model, binding.context_window)

    @classmethod
    def from_config(cls, cfg) -> ModelRouter:
        """Build a ModelRouter from a Config object, with backward compatibility."""
        from loongcli.core.config import Config
        assert isinstance(cfg, Config)

        providers: dict[str, ProviderConfig] = {}
        roles: dict[str, RoleBinding] = {}

        if cfg.providers:
            for name, prov in cfg.providers.items():
                providers[name] = prov
        if cfg.role_bindings:
            for name, bind in cfg.role_bindings.items():
                roles[name] = bind

        if "_default" not in providers and cfg.api_key:
            providers["_default"] = ProviderConfig(
                name="_default",
                api_key=cfg.api_key,
                base_url=cfg.base_url,
            )

        for role_name, default_binding in DEFAULT_ROLES.items():
            if role_name not in roles:
                if role_name == "main":
                    roles["main"] = RoleBinding(
                        provider="_default",
                        model=cfg.model,
                        thinking=cfg.thinking,
                        reasoning_effort=cfg.reasoning_effort,
                        context_window=cfg.model_max_tokens,
                    )
                elif role_name == "sub" and cfg.sub_model:
                    roles["sub"] = RoleBinding(
                        provider="_default",
                        model=cfg.sub_model,
                        thinking=False,
                    )
                else:
                    roles[role_name] = RoleBinding(
                        provider=default_binding.provider,
                        model=default_binding.model,
                        thinking=default_binding.thinking,
                    )

        return cls(providers=providers, roles=roles)
