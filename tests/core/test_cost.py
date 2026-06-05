from __future__ import annotations

import pytest
from loongcli.core.cost import (
    CostTracker, ModelPricing, PRICING,
    get_pricing, _calc_cost,
)


# ── get_pricing ─────────────────────────────────────────────────────


class TestGetPricing:

    def test_exact_match(self):
        p = get_pricing("deepseek-v4-flash")
        assert p is not None
        assert p.input_per_m == 1.0

    def test_prefix_match(self):
        p = get_pricing("deepseek-v4-flash-0601")
        assert p is not None
        assert p.input_per_m == 1.0

    def test_unknown_model(self):
        assert get_pricing("totally-unknown-model") is None

    def test_override_takes_precedence(self):
        custom = {"my-model": ModelPricing(1.0, 0.5, 2.0)}
        p = get_pricing("my-model", overrides=custom)
        assert p.input_per_m == 1.0

    def test_override_shadows_builtin(self):
        custom = {"deepseek-v4-flash": ModelPricing(99.0, 99.0, 99.0)}
        p = get_pricing("deepseek-v4-flash", overrides=custom)
        assert p.input_per_m == 99.0


# ── _calc_cost ──────────────────────────────────────────────────────


class TestCalcCost:

    def test_basic_no_cache(self):
        pricing = ModelPricing(input_per_m=2.0, cached_per_m=0.5, output_per_m=8.0)
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}
        cost = _calc_cost(pricing, usage)
        assert cost == pytest.approx(2.0 + 4.0)

    def test_with_cache_breakdown(self):
        pricing = ModelPricing(input_per_m=2.0, cached_per_m=0.5, output_per_m=8.0)
        usage = {
            "prompt_tokens": 1_000_000,
            "prompt_cache_hit_tokens": 800_000,
            "prompt_cache_miss_tokens": 200_000,
            "completion_tokens": 100_000,
        }
        cost = _calc_cost(pricing, usage)
        expected = (200_000 * 2.0 + 800_000 * 0.5 + 100_000 * 8.0) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_zero_tokens(self):
        pricing = ModelPricing(1.0, 0.5, 2.0)
        cost = _calc_cost(pricing, {})
        assert cost == 0.0

    def test_deepseek_flash_realistic(self):
        pricing = PRICING["deepseek-v4-flash"]
        usage = {
            "prompt_tokens": 50_000,
            "prompt_cache_hit_tokens": 40_000,
            "prompt_cache_miss_tokens": 10_000,
            "completion_tokens": 2_000,
        }
        cost = _calc_cost(pricing, usage)
        expected = (10_000 * 1.0 + 40_000 * 0.02 + 2_000 * 2.0) / 1_000_000
        assert cost == pytest.approx(expected)


# ── CostTracker ─────────────────────────────────────────────────────


class TestCostTracker:

    def test_single_record(self):
        ct = CostTracker()
        usage = {"prompt_tokens": 1000, "completion_tokens": 500}
        delta = ct.record("main", "deepseek-v4-flash", usage)
        assert delta > 0
        assert ct.total_cost == pytest.approx(delta)

    def test_multiple_records_accumulate(self):
        ct = CostTracker()
        usage = {"prompt_tokens": 1000, "completion_tokens": 500}
        d1 = ct.record("main", "deepseek-v4-flash", usage)
        d2 = ct.record("main", "deepseek-v4-flash", usage)
        assert ct.total_cost == pytest.approx(d1 + d2)
        assert ct.roles["main"].calls == 2

    def test_multi_role(self):
        ct = CostTracker()
        ct.record("main", "deepseek-v4-flash", {"prompt_tokens": 1000, "completion_tokens": 500})
        ct.record("utility", "deepseek-v4-flash", {"prompt_tokens": 200, "completion_tokens": 50})
        assert "main" in ct.roles
        assert "utility" in ct.roles
        assert ct.roles["main"].calls == 1
        assert ct.roles["utility"].calls == 1
        assert ct.total_cost > 0

    def test_unknown_model_zero_cost(self):
        ct = CostTracker()
        delta = ct.record("main", "unknown-local-model", {"prompt_tokens": 9999, "completion_tokens": 9999})
        assert delta == 0.0
        assert ct.total_cost == 0.0
        assert ct.roles["main"].prompt_tokens == 9999

    def test_token_accumulation(self):
        ct = CostTracker()
        ct.record("main", "deepseek-v4-flash", {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 20,
            "reasoning_tokens": 10,
        })
        rc = ct.roles["main"]
        assert rc.prompt_tokens == 100
        assert rc.completion_tokens == 50
        assert rc.cache_hit_tokens == 80
        assert rc.cache_miss_tokens == 20
        assert rc.reasoning_tokens == 10

    def test_pricing_override(self):
        custom = {"my-llm": ModelPricing(10.0, 5.0, 20.0)}
        ct = CostTracker(pricing_overrides=custom)
        delta = ct.record("main", "my-llm", {"prompt_tokens": 1_000_000, "completion_tokens": 0})
        assert delta == pytest.approx(10.0)


# ── format_cost ─────────────────────────────────────────────────────


class TestFormatCost:

    def test_tiny_cost(self):
        ct = CostTracker()
        assert ct.format_cost(0.00023) == "$0.0002"

    def test_small_cost(self):
        ct = CostTracker()
        assert ct.format_cost(0.0456) == "$0.046"

    def test_large_cost(self):
        ct = CostTracker()
        assert ct.format_cost(3.1415) == "$3.14"

    def test_total_cost_default_deepseek(self):
        ct = CostTracker()
        ct.record("main", "deepseek-v4-flash", {"prompt_tokens": 1000, "completion_tokens": 500})
        formatted = ct.format_cost()
        assert formatted.startswith("¥")

    def test_total_cost_default_openai(self):
        ct = CostTracker()
        ct.record("main", "gpt-4o", {"prompt_tokens": 1000, "completion_tokens": 500})
        formatted = ct.format_cost()
        assert formatted.startswith("$")

    def test_currency_override(self):
        ct = CostTracker()
        assert ct.format_cost(0.05, "¥") == "¥0.050"
        assert ct.format_cost(0.05, "$") == "$0.050"


# ── summary_dict ────────────────────────────────────────────────────


class TestSummaryDict:

    def test_structure(self):
        ct = CostTracker()
        ct.record("main", "deepseek-v4-flash", {"prompt_tokens": 1000, "completion_tokens": 500})
        d = ct.summary_dict()
        assert "total_cost" in d
        assert "currency" in d
        assert "roles" in d
        assert "main" in d["roles"]
        role = d["roles"]["main"]
        assert role["model"] == "deepseek-v4-flash"
        assert role["prompt_tokens"] == 1000
        assert role["completion_tokens"] == 500
        assert role["calls"] == 1
        assert role["cost"] > 0
        assert role["currency"] == "¥"

    def test_empty_tracker(self):
        ct = CostTracker()
        d = ct.summary_dict()
        assert d["total_cost"] == 0
        assert d["roles"] == {}


# ── PRICING table sanity ────────────────────────────────────────────


class TestPricingTable:

    def test_all_prices_positive(self):
        for model, p in PRICING.items():
            assert p.input_per_m >= 0, f"{model} input price negative"
            assert p.cached_per_m >= 0, f"{model} cached price negative"
            assert p.output_per_m >= 0, f"{model} output price negative"

    def test_cached_cheaper_than_input(self):
        for model, p in PRICING.items():
            assert p.cached_per_m <= p.input_per_m, f"{model} cached > input"

    def test_key_models_present(self):
        assert "deepseek-v4-flash" in PRICING
        assert "deepseek-v4-pro" in PRICING
        assert "gpt-4o" in PRICING
        assert "claude-opus-4-6" in PRICING

    def test_deepseek_currency_is_rmb(self):
        assert PRICING["deepseek-v4-flash"].currency == "¥"
        assert PRICING["deepseek-v4-pro"].currency == "¥"

    def test_openai_anthropic_currency_is_usd(self):
        assert PRICING["gpt-4o"].currency == "$"
        assert PRICING["claude-opus-4-6"].currency == "$"
