"""Qwen(dashscope) 隐式缓存真机测量 — cache_aware 决策依据。

手动运行：python tests/e2e_cache.py [model]
key 来源：DASHSCOPE_API_KEY / QWEN_API_KEY 环境变量，或 ~/.loongcli/config.json 的 providers.qwen。

测三件事：
1. 重复前缀第二次请求，implicit cache 是否命中（usage.prompt_tokens_details.cached_tokens）；
2. 命中率（cached / prompt）；
3. 改写历史中段（模拟 recycle/snip 的动作）后，缓存是否从改写点起失效。

判定：命中率 ≥ 60% 且稳定 → 值得把 qwen 标为 cache_aware（compact 走缓存友好路径）；
命中为 0（社区曾报部分模型隐式缓存 bug）→ 不标。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _get_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if key:
        return key
    cfg = Path.home() / ".loongcli" / "config.json"
    if cfg.exists():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        key = (data.get("providers", {}).get("qwen", {}) or {}).get("api_key", "")
        if key:
            return key
    sys.exit("未找到 Qwen key（环境变量或 ~/.loongcli/config.json providers.qwen）")


def _long_system() -> str:
    # ~4K token 的长系统提示（内容互不相同，避免任何去重伪影；positional 前缀缓存只看前缀）
    paras = []
    for i in range(60):
        paras.append(
            f"第{i}节：项目模块 M{i} 负责数据管道的第 {i} 阶段，输入格式为 schema-v{i % 7}，"
            f"输出投递到队列 q-{i * 13 % 97}。该模块的重试上限是 {i % 5 + 1} 次，超时 {i * 3 + 10} 秒，"
            f"负责人是工程师 E{i * 7 % 41}，上游依赖 M{max(0, i - 2)} 与 M{max(0, i - 3)}，"
            f"监控指标前缀 metrics.m{i}，告警阈值 p99 大于 {i * 20 + 100} 毫秒。"
        )
    return "你是项目知识库助手。以下是项目模块清单：\n" + "\n".join(paras)


def _call(client: OpenAI, model: str, messages: list[dict]) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=30,
        extra_body={"enable_thinking": False},
    )
    u = resp.usage
    details = getattr(u, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    return {
        "prompt": u.prompt_tokens,
        "cached": cached,
        "ratio": cached / u.prompt_tokens if u.prompt_tokens else 0.0,
        "answer": (resp.choices[0].message.content or "")[:40],
    }


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.7-plus"
    client = OpenAI(api_key=_get_key(), base_url=BASE_URL)
    system = _long_system()

    print(f"模型: {model}  |  系统提示约 {len(system)} 字符\n")

    # 第 1 次：冷启动（预期 cached=0 或极低）
    msgs1 = [{"role": "system", "content": system},
             {"role": "user", "content": "M5 的重试上限是多少？只答数字。"}]
    r1 = _call(client, model, msgs1)
    print(f"[1] 冷启动          prompt={r1['prompt']:>6} cached={r1['cached']:>6} ({r1['ratio']:.0%})")

    time.sleep(2)

    # 第 2 次：同前缀追加一轮（预期命中≈第 1 次的 prompt）
    msgs2 = msgs1 + [{"role": "assistant", "content": r1["answer"] or "1"},
                     {"role": "user", "content": "M8 的超时是多少秒？只答数字。"}]
    r2 = _call(client, model, msgs2)
    print(f"[2] 同前缀追加一轮  prompt={r2['prompt']:>6} cached={r2['cached']:>6} ({r2['ratio']:.0%})")

    time.sleep(2)

    # 第 3 次：再追加一轮（多轮会话的稳态命中率）
    msgs3 = msgs2 + [{"role": "assistant", "content": r2["answer"] or "34"},
                     {"role": "user", "content": "M12 的负责人是谁？只答编号。"}]
    r3 = _call(client, model, msgs3)
    print(f"[3] 再追加一轮      prompt={r3['prompt']:>6} cached={r3['cached']:>6} ({r3['ratio']:.0%})")

    time.sleep(2)

    # 第 4 次：改写历史中段（把 system 开头一节改掉，模拟 recycle/snip 改写旧消息）
    mutated = system.replace("第0节", "第零节", 1)
    msgs4 = [{"role": "system", "content": mutated}] + msgs3[1:]
    r4 = _call(client, model, msgs4)
    print(f"[4] 改写前缀开头    prompt={r4['prompt']:>6} cached={r4['cached']:>6} ({r4['ratio']:.0%})")

    print("\n判定：")
    steady = max(r2["ratio"], r3["ratio"])
    if steady >= 0.6:
        print(f"  稳态命中率 {steady:.0%} —— 隐式缓存有效，建议 qwen 标为 cache_aware。")
    elif r2["cached"] == 0 and r3["cached"] == 0:
        print("  命中始终为 0 —— 隐式缓存未生效（社区报过此 bug），不建议标 cache_aware。")
    else:
        print(f"  稳态命中率 {steady:.0%}（偏低）—— 边际收益有限，建议维持现状再观察。")
    if r4["cached"] < r3["cached"]:
        print(f"  改写前缀后命中从 {r3['cached']} 降到 {r4['cached']} —— 证实位置敏感：回收/snip 改写历史会击穿其后缓存。")


if __name__ == "__main__":
    main()
