from __future__ import annotations
import asyncio
import logging
import random
from typing import AsyncIterator

from openai import AsyncOpenAI

from loongcli.core.messages import inline_image_refs

logger = logging.getLogger(__name__)

# 单次请求的图片配额。图片 token 成本高且不可部分截断，设上限防失控累积/巨图。
MAX_IMAGES_PER_REQUEST = 20
MAX_IMAGE_DATA_URL_CHARS = 15 * 1024 * 1024  # ~15MB base64 data URL（约 11MB 原图）


def count_and_validate_images(messages: list[dict]) -> int:
    """扫描 messages 里的 image_url 块，校验单图体积与总数量。返回图片总数。

    纯文本消息（content 非 list）直接跳过，开销可忽略。超限抛 ValueError，
    由调用方转成明确错误——图片是原子的，宁可提前拒绝也别让 API 端糊涂报错。
    """
    n = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image_url":
                continue
            n += 1
            url = (block.get("image_url") or {}).get("url", "")
            if isinstance(url, str) and len(url) > MAX_IMAGE_DATA_URL_CHARS:
                raise ValueError(
                    f"单张图片过大（{len(url) // 1024 // 1024}MB base64，"
                    f"上限 {MAX_IMAGE_DATA_URL_CHARS // 1024 // 1024}MB）"
                )
    if n > MAX_IMAGES_PER_REQUEST:
        raise ValueError(f"单次请求图片过多（{n} 张，上限 {MAX_IMAGES_PER_REQUEST}）")
    return n


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        thinking: bool = True,
        reasoning_effort: str = "max",
        provider_type: str = "",
        vision: bool = False,
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.base_url = base_url
        self._api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self._provider_type = provider_type or _infer_provider(base_url)
        # 该模型是否具备视觉能力（能吃图片）。由 role 配置显式传入，门控图片输入。
        self.vision = vision

    def switch(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        vision: bool | None = None,
    ) -> None:
        """一致性切换模型/供应商：模型名、端点、密钥、provider_type、vision 一起更新。

        防两类半切换事故：换模型名不换 base_url（跨供应商 400）；vision 残留
        （把图喂给纯文本模型，或反向误拒）。base_url/api_key 未变时复用现有连接。
        vision 传 None 表示调用方明确"保持不变"（仅限同模型场景，谨慎使用）。
        """
        new_key = api_key or self._api_key
        new_url = base_url or self.base_url
        if new_url != self.base_url or new_key != self._api_key:
            self.client = AsyncOpenAI(api_key=new_key, base_url=new_url)
            self.base_url = new_url
            self._api_key = new_key
        self.model = model
        self._provider_type = _infer_provider(self.base_url)
        if vision is not None:
            self.vision = vision

    @property
    def cache_aware(self) -> bool:
        """该 provider 是否有强自动前缀缓存，值得「保前缀稳定 > 减 token」。

        开启后压缩走缓存友好路径（跳过 collapse、摘要优先喂完整历史）；其他 provider
        默认 False，走完整压缩金字塔（减 token）。已开：
        - DeepSeek：自动硬盘缓存，命中远便宜于未命中（早期实测 ~22x 省）。
        - Qwen(dashscope)：隐式缓存自动开启，2026-07-03 真机实测（tests/e2e_cache.py）：
          稳态命中 72%、命中按输入价 20% 计（稳态输入成本 ≈ 原价 42%）、位置敏感——
          collapse/snip 改写历史即全部击穿，故同样适用保前缀策略。
        Kimi/GLM 也有隐式缓存但未接入未实测，接入后跑 e2e_cache 再开。

        判断同时看 base_url 推断的 provider_type 和 model 名——后者是为了兜住「走代理 base_url
        访问」的场景：那时 base_url 推断的 provider_type 会误判，但 model 名通常仍带家族前缀，
        靠它避免静默退化回完整金字塔。"""
        pt = self._provider_type
        model = self.model.lower()
        return (
            pt in ("deepseek", "qwen")
            or model.startswith("deepseek")
            or model.startswith("qwen")
        )

    def _build_thinking_params(self, kwargs: dict, stream: bool = False) -> None:
        """Inject provider-specific thinking/reasoning parameters."""
        pt = self._provider_type
        # Qwen(dashscope 兼容)：思考模式用 extra_body.enable_thinking 控制，不认
        # reasoning_effort（它只接受 none/minimal/.../xhigh，发 max 会 400）。且思考
        # 模式要求流式，非流式一律关，避免"thinking mode requires stream"报错。
        if pt == "qwen":
            kwargs.setdefault("extra_body", {})["enable_thinking"] = bool(self.thinking and stream)
            return

        if not self.thinking:
            if pt == "deepseek":
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            return

        if pt == "deepseek":
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            kwargs["reasoning_effort"] = self.reasoning_effort
        elif pt == "openai":
            kwargs["reasoning_effort"] = self.reasoning_effort
        elif pt == "anthropic":
            budget = {"low": 2048, "medium": 8192, "high": 16384, "max": 32768}
            tokens = budget.get(self.reasoning_effort, 16384)
            kwargs["extra_body"] = {"thinking": {"type": "enabled", "budget_tokens": tokens}}

    async def chat(self, prompt: str) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        self._build_thinking_params(kwargs)
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                logger.warning("LLM chat failed (attempt %d/%d): %s", attempt, self.max_retries, e)
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    await asyncio.sleep(delay)
        raise last_error

    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator:
        # 图片引用（loongimg://）在此单点内联成 base64 data URL——所有调用方
        # （agent 主循环、compactor 摘要、子代理）都经这里，消息历史里只留引用。
        # 随后校验图片配额，放在重试循环外——参数错误重试无意义。
        messages = inline_image_refs(messages)
        count_and_validate_images(messages)
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
        self._build_thinking_params(kwargs, stream=True)

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                stream = await self.client.chat.completions.create(**kwargs)
                async for chunk in stream:
                    yield chunk
                return
            except Exception as e:
                last_error = e
                logger.warning("LLM request failed (attempt %d/%d): %s", attempt, self.max_retries, e)
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    await asyncio.sleep(delay)

        raise last_error


def _infer_provider(base_url: str) -> str:
    url = base_url.lower()
    if "deepseek" in url:
        return "deepseek"
    if "dashscope" in url or "aliyuncs" in url:
        return "qwen"
    if "openai" in url:
        return "openai"
    if "anthropic" in url:
        return "anthropic"
    if "localhost" in url or "127.0.0.1" in url:
        return "local"
    return "openai"
