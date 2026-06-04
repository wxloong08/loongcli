from loongcli.core.llm import LLMClient


class TestBuildThinkingParams:
    def test_deepseek_thinking_enabled(self):
        client = LLMClient(api_key="k", thinking=True, provider_type="deepseek")
        kwargs = {}
        client._build_thinking_params(kwargs)
        assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
        assert kwargs["reasoning_effort"] == "max"

    def test_deepseek_thinking_disabled(self):
        client = LLMClient(api_key="k", thinking=False, provider_type="deepseek")
        kwargs = {}
        client._build_thinking_params(kwargs)
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
        assert "reasoning_effort" not in kwargs

    def test_openai_thinking_enabled(self):
        client = LLMClient(api_key="k", thinking=True, reasoning_effort="high", provider_type="openai")
        kwargs = {}
        client._build_thinking_params(kwargs)
        assert kwargs["reasoning_effort"] == "high"
        assert "extra_body" not in kwargs

    def test_openai_thinking_disabled(self):
        client = LLMClient(api_key="k", thinking=False, provider_type="openai")
        kwargs = {}
        client._build_thinking_params(kwargs)
        assert "extra_body" not in kwargs
        assert "reasoning_effort" not in kwargs

    def test_anthropic_thinking_enabled(self):
        client = LLMClient(api_key="k", thinking=True, reasoning_effort="high", provider_type="anthropic")
        kwargs = {}
        client._build_thinking_params(kwargs)
        assert kwargs["extra_body"]["thinking"]["type"] == "enabled"
        assert kwargs["extra_body"]["thinking"]["budget_tokens"] == 16384

    def test_local_no_thinking_params(self):
        client = LLMClient(api_key="k", thinking=True, provider_type="local")
        kwargs = {}
        client._build_thinking_params(kwargs)
        assert "extra_body" not in kwargs
        assert "reasoning_effort" not in kwargs

    def test_infer_provider_from_url(self):
        client = LLMClient(api_key="k", base_url="https://api.deepseek.com")
        assert client._provider_type == "deepseek"

        client2 = LLMClient(api_key="k", base_url="https://api.openai.com/v1")
        assert client2._provider_type == "openai"

        client3 = LLMClient(api_key="k", base_url="http://localhost:11434/v1")
        assert client3._provider_type == "local"
