import pytest
from loongcli.core.provider import (
    ProviderConfig, RoleBinding, ModelRouter, context_window,
    DEFAULT_CONTEXT_WINDOW, DEFAULT_ROLES,
)


class TestProviderConfig:
    def test_deepseek_detection(self):
        p = ProviderConfig(name="ds", api_key="k", base_url="https://api.deepseek.com")
        assert p.provider_type == "deepseek"

    def test_openai_detection(self):
        p = ProviderConfig(name="oai", api_key="k", base_url="https://api.openai.com/v1")
        assert p.provider_type == "openai"

    def test_anthropic_detection(self):
        p = ProviderConfig(name="ant", api_key="k", base_url="https://api.anthropic.com")
        assert p.provider_type == "anthropic"

    def test_local_detection(self):
        p = ProviderConfig(name="local", api_key="k", base_url="http://localhost:11434/v1")
        assert p.provider_type == "local"

    def test_qwen_detection(self):
        p = ProviderConfig(name="qwen", api_key="k",
                           base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        assert p.provider_type == "qwen"

    def test_unknown_defaults_to_openai(self):
        p = ProviderConfig(name="x", api_key="k", base_url="https://custom.api.com")
        assert p.provider_type == "openai"


class TestContextWindow:
    def test_known_model(self):
        assert context_window("deepseek-v4-flash") == 1_048_576
        assert context_window("gpt-4o") == 128_000

    def test_unknown_model(self):
        assert context_window("unknown-model") == DEFAULT_CONTEXT_WINDOW

    def test_override(self):
        assert context_window("deepseek-v4-flash", override=200_000) == 200_000

    def test_prefix_match(self):
        assert context_window("deepseek-v4-flash-0601") == 1_048_576


class TestModelRouter:
    def _make_router(self, **role_overrides):
        providers = {
            "_default": ProviderConfig(name="_default", api_key="sk-test", base_url="https://api.deepseek.com"),
        }
        roles = dict(DEFAULT_ROLES)
        roles.update(role_overrides)
        return ModelRouter(providers=providers, roles=roles)

    def test_client_caching(self):
        router = self._make_router()
        c1 = router.client("main")
        c2 = router.client("main")
        assert c1 is c2

    def test_different_roles_different_clients(self):
        router = self._make_router(
            sub=RoleBinding(provider="_default", model="deepseek-v4-pro"),
        )
        main = router.client("main")
        sub = router.client("sub")
        assert main is not sub
        assert sub.model == "deepseek-v4-pro"

    def test_unknown_role_falls_back_to_main(self):
        router = self._make_router()
        c = router.client("nonexistent")
        assert c.model == router.client("main").model

    def test_missing_provider_raises(self):
        roles = {"main": RoleBinding(provider="missing", model="m")}
        router = ModelRouter(providers={}, roles=roles)
        with pytest.raises(ValueError, match="missing"):
            router.client("main")

    def test_context_window_for(self):
        router = self._make_router()
        assert router.context_window_for("main") == 1_048_576

    def test_context_window_for_with_override(self):
        router = self._make_router(
            main=RoleBinding(provider="_default", model="deepseek-v4-flash", context_window=200_000),
        )
        assert router.context_window_for("main") == 200_000

    def test_provider_type_propagated(self):
        router = self._make_router()
        client = router.client("main")
        assert client._provider_type == "deepseek"

    def test_vision_defaults_false(self):
        assert RoleBinding(provider="_default", model="m").vision is False
        router = self._make_router()
        assert router.client("main").vision is False

    def test_vision_propagated_to_client(self):
        router = self._make_router(
            main=RoleBinding(provider="_default", model="qwen3.7-plus", vision=True),
        )
        assert router.client("main").vision is True


class TestMultiProvider:
    def test_two_providers(self):
        providers = {
            "ds": ProviderConfig(name="ds", api_key="sk-ds", base_url="https://api.deepseek.com"),
            "oai": ProviderConfig(name="oai", api_key="sk-oai", base_url="https://api.openai.com/v1"),
        }
        roles = {
            "main": RoleBinding(provider="oai", model="gpt-4o", thinking=False),
            "utility": RoleBinding(provider="ds", model="deepseek-v4-flash"),
        }
        router = ModelRouter(providers=providers, roles=roles)

        main = router.client("main")
        utility = router.client("utility")
        assert main.model == "gpt-4o"
        assert main._provider_type == "openai"
        assert utility.model == "deepseek-v4-flash"
        assert utility._provider_type == "deepseek"


class TestFromConfig:
    def test_backward_compat_no_providers(self):
        from loongcli.core.config import Config
        cfg = Config(
            api_key="sk-test",
            model="deepseek-v4-pro",
            sub_model="deepseek-v4-flash",
            thinking=True,
            reasoning_effort="high",
        )
        router = ModelRouter.from_config(cfg)
        main = router.client("main")
        sub = router.client("sub")
        utility = router.client("utility")

        assert main.model == "deepseek-v4-pro"
        assert main.thinking is True
        assert main.reasoning_effort == "high"
        assert sub.model == "deepseek-v4-flash"
        assert utility.model == "deepseek-v4-flash"

    def test_with_providers_and_roles(self):
        from loongcli.core.config import Config
        from loongcli.core.provider import ProviderConfig, RoleBinding

        cfg = Config(
            api_key="sk-fallback",
            providers={
                "ds": ProviderConfig(name="ds", api_key="sk-ds", base_url="https://api.deepseek.com"),
            },
            role_bindings={
                "main": RoleBinding(provider="ds", model="deepseek-v4-pro", thinking=True),
                "utility": RoleBinding(provider="ds", model="deepseek-v4-flash"),
            },
        )
        router = ModelRouter.from_config(cfg)
        assert router.client("main").model == "deepseek-v4-pro"
        assert router.client("utility").model == "deepseek-v4-flash"
