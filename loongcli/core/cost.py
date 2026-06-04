from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelPricing:
    """Price per million tokens (USD)."""
    input_per_m: float
    cached_per_m: float
    output_per_m: float


PRICING: dict[str, ModelPricing] = {
    # DeepSeek
    "deepseek-v4-flash": ModelPricing(0.14, 0.014, 0.28),
    "deepseek-v4-pro": ModelPricing(2.19, 0.55, 8.76),
    # OpenAI
    "gpt-4o": ModelPricing(2.50, 1.25, 10.00),
    "gpt-4o-mini": ModelPricing(0.15, 0.075, 0.60),
    "gpt-4.1": ModelPricing(2.00, 0.50, 8.00),
    "gpt-4.1-mini": ModelPricing(0.40, 0.10, 1.60),
    "gpt-4.1-nano": ModelPricing(0.10, 0.025, 0.40),
    "o3": ModelPricing(10.00, 2.50, 40.00),
    "o3-mini": ModelPricing(1.10, 0.275, 4.40),
    "o4-mini": ModelPricing(1.10, 0.275, 4.40),
    # Anthropic
    "claude-opus-4-6": ModelPricing(15.00, 3.75, 75.00),
    "claude-opus-4-7": ModelPricing(15.00, 3.75, 75.00),
    "claude-sonnet-4-6": ModelPricing(3.00, 0.75, 15.00),
    "claude-haiku-4-5": ModelPricing(0.80, 0.08, 4.00),
}


def get_pricing(model: str, overrides: dict[str, ModelPricing] | None = None) -> ModelPricing | None:
    if overrides and model in overrides:
        return overrides[model]
    if model in PRICING:
        return PRICING[model]
    for prefix in PRICING:
        if model.startswith(prefix):
            return PRICING[prefix]
    return None


def _calc_cost(pricing: ModelPricing, usage: dict) -> float:
    cache_hit = usage.get("prompt_cache_hit_tokens", 0)
    cache_miss = usage.get("prompt_cache_miss_tokens", 0)
    prompt_total = usage.get("prompt_tokens", 0)

    if cache_hit > 0 or cache_miss > 0:
        input_tokens = cache_miss
        cached_tokens = cache_hit
    else:
        input_tokens = prompt_total
        cached_tokens = 0

    output_tokens = usage.get("completion_tokens", 0)

    return (
        input_tokens * pricing.input_per_m
        + cached_tokens * pricing.cached_per_m
        + output_tokens * pricing.output_per_m
    ) / 1_000_000


@dataclass
class RoleCost:
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0


class CostTracker:
    def __init__(self, pricing_overrides: dict[str, ModelPricing] | None = None):
        self._overrides = pricing_overrides
        self._roles: dict[str, RoleCost] = {}
        self.total_cost: float = 0.0

    def record(self, role: str, model: str, usage: dict) -> float:
        """Record token usage for a role. Returns the incremental cost."""
        rc = self._roles.get(role)
        if not rc:
            rc = RoleCost(model=model)
            self._roles[role] = rc

        rc.prompt_tokens += usage.get("prompt_tokens", 0)
        rc.completion_tokens += usage.get("completion_tokens", 0)
        rc.cache_hit_tokens += usage.get("prompt_cache_hit_tokens", 0)
        rc.cache_miss_tokens += usage.get("prompt_cache_miss_tokens", 0)
        rc.reasoning_tokens += usage.get("reasoning_tokens", 0)
        rc.calls += 1

        pricing = get_pricing(model, self._overrides)
        if pricing:
            delta = _calc_cost(pricing, usage)
            rc.cost_usd += delta
            self.total_cost += delta
            return delta
        return 0.0

    @property
    def roles(self) -> dict[str, RoleCost]:
        return dict(self._roles)

    def format_cost(self, usd: float | None = None) -> str:
        v = usd if usd is not None else self.total_cost
        if v < 0.01:
            return f"${v:.4f}"
        if v < 1.0:
            return f"${v:.3f}"
        return f"${v:.2f}"

    def summary_dict(self) -> dict:
        roles = {}
        for name, rc in self._roles.items():
            roles[name] = {
                "model": rc.model,
                "prompt_tokens": rc.prompt_tokens,
                "completion_tokens": rc.completion_tokens,
                "cache_hit_tokens": rc.cache_hit_tokens,
                "cache_miss_tokens": rc.cache_miss_tokens,
                "reasoning_tokens": rc.reasoning_tokens,
                "cost_usd": round(rc.cost_usd, 6),
                "calls": rc.calls,
            }
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "roles": roles,
        }
