"""【存档探针，不可再运行】验证 snip/micro_compact 对「摘要调用」缓存的影响。

结论（2026-07 真机）：B（完整历史吃缓存）比 A（snip+micro"省 token"）便宜 ~22 倍、
命中率 98% vs 12%——这个结论直接导致 2026-07-19 压缩金字塔（collapse/micro_compact）
被整体删除。本探针的被测对象已不存在，脚本保留作证据链，见 interview-qa Q43。

原实验设计：
1. 预热——发一次完整历史（模拟 compact 触发前、主对话上一轮已发过并被缓存）
2. A（旧现状）：snip + micro_compact 处理后的历史 + COMPACT_INSTRUCTION，读真实命中
3. B（不删）：完整历史 + COMPACT_INSTRUCTION，读真实命中
"""
from __future__ import annotations

raise SystemExit(
    "已存档：micro_compact 已于 2026-07-19 随压缩金字塔删除（本探针的结论即删除依据），"
    "不可再运行。结论与叙事见 tasks/interview-qa.md Q43。"
)

import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

from loongcli.core.prompts import get_system_prompt
from loongcli.core.compact import snip, micro_compact, COMPACT_INSTRUCTION

PRICE_HIT = 0.025   # ¥/M tokens
PRICE_MISS = 3.0


def build_history(system: str) -> list[dict]:
    """构造一段够大的对话历史：25 轮，穿插 10 个大工具结果。
    25 轮 > SNIP_AGE_THRESHOLD(20) → snip 会删最老几轮；
    10 个工具结果 > MICRO_COMPACT_KEEP_RECENT(5) → micro_compact 会清理最老 5 个。"""
    msgs: list[dict] = [{"role": "system", "content": system}]
    tc = 0
    for i in range(1, 26):
        msgs.append({"role": "user", "content": f"第 {i} 轮：请处理任务 {i}。" + "补充背景说明。" * 20})
        if i % 2 == 0 and tc < 10:
            tc += 1
            tid = f"call_{tc:02d}"
            msgs.append({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": tid, "type": "function",
                                "function": {"name": "read_file", "arguments": json.dumps({"path": f"src/module_{i}.py"})}}],
            })
            msgs.append({"role": "tool", "tool_call_id": tid,
                         "content": f"# module_{i}.py 文件内容\n" + f"def func_{i}(): pass  # 行内容填充\n" * 120})
            msgs.append({"role": "assistant", "content": f"第 {i} 轮：已读取并分析 module_{i}。" + "分析结论填充。" * 30})
        else:
            msgs.append({"role": "assistant", "content": f"第 {i} 轮：任务 {i} 已完成。" + "结论填充内容。" * 30})
    return msgs


async def call(client, model, msgs) -> tuple[int, int, int]:
    resp = await client.chat.completions.create(model=model, messages=msgs, max_tokens=16)
    u = resp.usage
    pt = u.prompt_tokens or 0
    hit = getattr(u, "prompt_cache_hit_tokens", 0) or 0
    miss = getattr(u, "prompt_cache_miss_tokens", 0) or (pt - hit)
    return pt, hit, miss


def cost(hit: int, miss: int) -> float:
    return (hit * PRICE_HIT + miss * PRICE_MISS) / 1_000_000


async def main():
    cfg = json.loads((Path.home() / ".loongcli" / "config.json").read_text(encoding="utf-8"))
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    model = cfg["model"]
    system = get_system_prompt(model=model)

    messages = build_history(system)

    # 预热：发完整历史，建立前缀缓存（模拟 compact 触发前主对话已发过）
    print("预热（发完整历史建立缓存）...", flush=True)
    await call(client, model, messages)
    await asyncio.sleep(3)

    # A：现状——snip + micro_compact 后摘要
    snipped, dropped = snip(list(messages))
    cleaned = micro_compact(snipped)
    A = cleaned + [{"role": "user", "content": COMPACT_INSTRUCTION}]
    a_pt, a_hit, a_miss = await call(client, model, A)

    # B：不删——完整历史摘要
    B = list(messages) + [{"role": "user", "content": COMPACT_INSTRUCTION}]
    b_pt, b_hit, b_miss = await call(client, model, B)

    print(f"\n（snip 删了 {dropped} 条远古消息）")
    print("=" * 60)
    print(f"{'':22} prompt    hit    miss   命中率   成本(¥)")
    print(f"A 现状(snip+micro) {a_pt:7d}{a_hit:7d}{a_miss:7d}   {a_hit/a_pt*100:4.0f}%   {cost(a_hit,a_miss):.4f}")
    print(f"B 不删(完整历史)   {b_pt:7d}{b_hit:7d}{b_miss:7d}   {b_hit/b_pt*100:4.0f}%   {cost(b_hit,b_miss):.4f}")
    print("=" * 60)
    ca, cb = cost(a_hit, a_miss), cost(b_hit, b_miss)
    if cb < ca:
        print(f"结论：B(不删) 更便宜 —— A 是 B 的 {ca/cb:.1f} 倍成本。snip/micro_compact 在缓存场景下是净亏。")
    else:
        print(f"结论：A(现状) 更便宜（{cb/ca:.1f} 倍）。snip/micro_compact 的减 token 收益 > 缓存损失。")
    print(f"（A 省了 {(1-a_pt/b_pt)*100:.0f}% 的 token 量，但命中率 {a_hit/a_pt*100:.0f}% vs {b_hit/b_pt*100:.0f}%）")


if __name__ == "__main__":
    asyncio.run(main())
