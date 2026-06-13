"""缓存探针：量化 recall 注入位置对 DeepSeek 前缀缓存命中率的真实影响。

A/B 实验，不碰产品代码——在脚本里直接构造两种 message 布局：
- 布局 A（现状）：[system, recall(每轮变), ...对话历史..., user]   recall 在 position 1
- 布局 B（优化）：[system, ...对话历史..., recall(每轮变), user]   recall 在末尾

唯一变量是 recall 注入位置。system 前缀、对话历史、每轮的 recall 内容、user 内容
两种布局完全一致。连续发 N 轮（对话历史累积），读 API 返回的真实
prompt_cache_hit_tokens，对比两种布局的命中率随轮次的走势 + 折算成本差。

用法：python bench/cache_probe.py [--turns 10]
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

from loongcli.core.prompts import get_system_prompt
from loongcli.memory.recall_engine import RecallEngine

# DeepSeek 价格（¥/M tokens），用于把命中率差折算成真实成本
PRICE_HIT = 0.025
PRICE_MISS = 3.0


def make_memories(n=15):
    now = datetime.now(timezone.utc).isoformat()
    types = ["user", "feedback", "project", "reference"]
    mems = []
    for i in range(n):
        mems.append({
            "type": types[i % 4],
            "name": f"demo-memory-{i:02d}",
            "description": f"第 {i} 条演示记忆的一句话描述，用于缓存探针测试",
            "updated_at": now,
            "content": f"这是第 {i} 条记忆的正文内容。" + "用于模拟真实记忆体积的填充文本。" * 18,
        })
    return mems


def user_msg(i: int) -> str:
    return f"第 {i} 轮的用户问题：请分析问题 {i} 并给出建议。" + "补充一些背景上下文。" * 5


def assistant_reply(i: int) -> str:
    return f"这是第 {i} 轮的助手回复。" + "针对问题给出详细的分析、推理和结论。" * 30


def layout_A(system, recall, history, user):
    # recall 在 position 1（现状）
    return [
        {"role": "system", "content": system},
        {"role": "system", "content": recall},
        *history,
        {"role": "user", "content": user},
    ]


def layout_B(system, recall, history, user):
    # recall 在末尾（紧贴当前 user 之前）
    return [
        {"role": "system", "content": system},
        *history,
        {"role": "system", "content": recall},
        {"role": "user", "content": user},
    ]


async def run_layout(client, model, system, mems, label, build, turns):
    fmt = RecallEngine(None, None).format_for_injection
    print(f"\n=== 布局 {label} ===", flush=True)
    history: list[dict] = []
    rows = []
    for i in range(1, turns + 1):
        # 每轮选不同的 5 条记忆，模拟召回集合逐轮变化（位置影响最大的情形）
        picks = [mems[(i * 3 + k) % len(mems)] for k in range(5)]
        recall = fmt(picks)
        u = user_msg(i)
        messages = build(system, recall, history, u)
        resp = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=16,
        )
        usage = resp.usage
        pt = usage.prompt_tokens or 0
        hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        rate = hit / pt * 100 if pt else 0
        rows.append({"turn": i, "prompt": pt, "hit": hit, "miss": pt - hit, "rate": rate})
        print(f"  轮{i:2d}: prompt={pt:6d}  hit={hit:6d}  命中率={rate:4.0f}%", flush=True)
        history.append({"role": "user", "content": u})
        history.append({"role": "assistant", "content": assistant_reply(i)})
    steady = rows[1:]  # 第 1 轮无历史缓存，从第 2 轮起看稳态
    avg = sum(r["rate"] for r in steady) / len(steady) if steady else 0
    print(f"  → {label} 第 2 轮起平均命中率: {avg:.0f}%", flush=True)
    return rows


def cost_of(rows):
    """这批轮次的总 input 成本（¥），按命中/未命中分别计价。"""
    hit = sum(r["hit"] for r in rows)
    miss = sum(r["miss"] for r in rows)
    return (hit * PRICE_HIT + miss * PRICE_MISS) / 1_000_000


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=10)
    args = parser.parse_args()

    cfg = json.loads((Path.home() / ".loongcli" / "config.json").read_text(encoding="utf-8"))
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    model = cfg["model"]

    # 固定前缀：真实 system prompt（不含记忆索引，排除无关变量），只取一次
    system = get_system_prompt(model=model)
    mems = make_memories(15)

    print(f"缓存探针 | model={model} | {args.turns} 轮 | system={len(system)} 字符 | 记忆池 15 条")

    rows_a = await run_layout(client, model, system, mems, "A (recall在前/现状)", layout_A, args.turns)
    rows_b = await run_layout(client, model, system, mems, "B (recall在末尾)", layout_B, args.turns)

    cost_a = cost_of(rows_a)
    cost_b = cost_of(rows_b)
    print("\n" + "=" * 56)
    print(f"布局 A 总 input 成本: ¥{cost_a:.4f}")
    print(f"布局 B 总 input 成本: ¥{cost_b:.4f}")
    if cost_a > 0:
        print(f"末尾注入相对现状省: {(1 - cost_b / cost_a) * 100:.0f}%  (绝对 ¥{cost_a - cost_b:.4f} / {args.turns} 轮)")
    print(f"折算每轮节省: ¥{(cost_a - cost_b) / args.turns:.5f}")


if __name__ == "__main__":
    asyncio.run(main())
