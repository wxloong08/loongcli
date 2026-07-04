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

    # ── Qwen(dashscope 兼容)：思考走 enable_thinking，绝不发 reasoning_effort ──

    def test_qwen_inferred_from_dashscope_url(self):
        client = LLMClient(api_key="k", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        assert client._provider_type == "qwen"

    def test_qwen_thinking_streaming_enables_thinking(self):
        client = LLMClient(api_key="k", thinking=True, provider_type="qwen")
        kwargs = {}
        client._build_thinking_params(kwargs, stream=True)
        assert kwargs["extra_body"] == {"enable_thinking": True}
        # 关键回归：Qwen 绝不能收到 reasoning_effort（否则 400：max 非法值）
        assert "reasoning_effort" not in kwargs

    def test_qwen_thinking_nonstream_disables_thinking(self):
        # 非流式（chat）即便 thinking=True 也关——Qwen 思考模式要求流式
        client = LLMClient(api_key="k", thinking=True, provider_type="qwen")
        kwargs = {}
        client._build_thinking_params(kwargs, stream=False)
        assert kwargs["extra_body"] == {"enable_thinking": False}
        assert "reasoning_effort" not in kwargs

    def test_qwen_thinking_off(self):
        client = LLMClient(api_key="k", thinking=False, provider_type="qwen")
        kwargs = {}
        client._build_thinking_params(kwargs, stream=True)
        assert kwargs["extra_body"] == {"enable_thinking": False}
        assert "reasoning_effort" not in kwargs
