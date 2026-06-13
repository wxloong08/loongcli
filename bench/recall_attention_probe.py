"""召回注意力探针：验证 recall 注入位置（position 1 vs 末尾）是否影响模型对召回记忆的利用。

背景：把 recall 注入从 position 1 挪到 user 之前（缓存友好）后，唯一未验证的风险是
「模型挪了位置后还重不重视那些召回的记忆」。这个探针做 A/B 量化它。

设计（为了有区分度）：
- 关键事实是人造、不可从常识/训练知识猜到的（灰度观察窗口 6 小时、回滚阈值 0.3%）
- 埋在 5 条召回记忆的中间，其余 4 条是无关干扰
- 加几轮与问题无关的对话历史，把「recall 离当前问题的距离」在两种布局间真正拉开
- 同一问题、同一记忆、同一历史，唯一变量是 recall 插入位置；各跑 N 次看答对率

判读：两种都高且接近 → 改位置不退化（要的结论）；末尾明显低 → 退化，需重新考虑。

用法：python bench/recall_attention_probe.py [--trials 6]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

from loongcli.core.prompts import get_system_prompt
from loongcli.memory.recall_engine import RecallEngine

NOW = datetime.now(timezone.utc).isoformat()

KEY_MEM = {
    "type": "project", "name": "project-canary-policy",
    "description": "灰度发布的观察窗口与自动回滚阈值", "updated_at": NOW,
    "content": "本项目灰度发布的观察窗口固定为 6 小时；窗口内如果错误率超过 0.3%，系统自动回滚到上一版本。",
}
DISTRACTORS = [
    {"type": "user", "name": "user-editor", "description": "用户的编辑器偏好", "updated_at": NOW,
     "content": "用户用 Neovim，缩进偏好 4 空格，不喜欢自动格式化打乱 import 顺序。"},
    {"type": "feedback", "name": "feedback-commit-style", "description": "提交信息风格", "updated_at": NOW,
     "content": "提交信息用祈使句，首行不超过 50 字符，正文解释 why 而非 what。"},
    {"type": "reference", "name": "reference-staging-url", "description": "预发环境地址", "updated_at": NOW,
     "content": "预发环境在 staging.internal.example，需要 VPN 才能访问。"},
    {"type": "project", "name": "project-test-cmd", "description": "测试命令", "updated_at": NOW,
     "content": "本项目测试用 pytest -q，CI 里额外跑 ruff 和 mypy。"},
]
# 关键事实埋在第 3 位（中间）
RECALL_MEMS = DISTRACTORS[:2] + [KEY_MEM] + DISTRACTORS[2:]

HISTORY = [
    {"role": "user", "content": "帮我看看 utils.py 里那个 parse_config 函数，参数太多了想重构。"},
    {"role": "assistant", "content": "建议把那 6 个参数收敛成一个 dataclass 配置对象传入，"
     "既减少参数列表长度，也便于以后扩展字段。我可以先读一下 parse_config 的当前实现。"},
    {"role": "user", "content": "好，顺便把它的单元测试也更新一下。"},
    {"role": "assistant", "content": "明白。重构成 dataclass 后，测试里构造入参的地方也要同步改成"
     "传配置对象，我会一并更新断言。"},
    {"role": "user", "content": "另外那个日志格式我想统一成 JSON，方便后面接 ELK。"},
    {"role": "assistant", "content": "可以用 structlog 或标准库 logging 配 JSON formatter。"
     "考虑到你想接 ELK，建议字段里带上 trace_id 和 service 名，便于检索。"},
]

QUESTION = "对了，提醒我一下：咱们项目灰度发布的观察窗口是多长时间？错误率到多少会自动回滚？"


def layout_front(system, recall):
    return [{"role": "system", "content": system}, {"role": "system", "content": recall},
            *HISTORY, {"role": "user", "content": QUESTION}]


def layout_end(system, recall):
    return [{"role": "system", "content": system}, *HISTORY,
            {"role": "system", "content": recall}, {"role": "user", "content": QUESTION}]


def is_correct(ans: str) -> bool:
    has_window = bool(re.search(r"6\s*(小时|个小时|h\b|hour)", ans, re.I))
    has_threshold = "0.3" in ans
    return has_window and has_threshold


async def run_layout(client, model, system, recall, label, build, trials):
    print(f"\n=== {label} ===", flush=True)
    correct = 0
    for t in range(1, trials + 1):
        resp = await client.chat.completions.create(
            model=model, messages=build(system, recall), max_tokens=256,
        )
        ans = resp.choices[0].message.content or ""
        ok = is_correct(ans)
        correct += ok
        print(f"  试{t}: {'✓' if ok else '✗'}  {ans.strip()[:70]}", flush=True)
    rate = correct / trials * 100
    print(f"  → {label} 答对率: {correct}/{trials} ({rate:.0f}%)", flush=True)
    return correct, trials


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=6)
    args = parser.parse_args()

    cfg = json.loads((Path.home() / ".loongcli" / "config.json").read_text(encoding="utf-8"))
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    model = cfg["model"]
    system = get_system_prompt(model=model)
    recall = RecallEngine(None, None).format_for_injection(RECALL_MEMS)

    print(f"召回注意力探针 | model={model} | 每布局 {args.trials} 次 | 关键事实埋在 5 条召回的第 3 位")
    cf, nf = await run_layout(client, model, system, recall, "布局 A (recall 在 position 1)", layout_front, args.trials)
    ce, ne = await run_layout(client, model, system, recall, "布局 B (recall 在末尾/新方案)", layout_end, args.trials)

    print("\n" + "=" * 56)
    print(f"position 1 答对率: {cf}/{nf} ({cf / nf * 100:.0f}%)")
    print(f"末尾注入  答对率: {ce}/{ne} ({ce / ne * 100:.0f}%)")
    if ce >= cf:
        print("结论: ✓ 末尾注入召回利用 ≥ position 1，改动在注意力维度不退化（甚至更好）")
    elif cf - ce <= 1:
        print("结论: ~ 两者接近（差 ≤1 次），无明显退化，可视为通过")
    else:
        print("结论: ✗ 末尾注入明显更差，注意力退化，需重新考虑改动")


if __name__ == "__main__":
    asyncio.run(main())
