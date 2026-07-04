"""A/B: does the system-prompt accuracy clause cut number-fabrication?

Reproduces the jobhunter failure in isolation (no Boss/chrome): hand the model
the resume line that says "1010 单元测试" and ask it to draft an HR reply that
mentions the agent-runtime project's test count. Measure how often it writes a
WRONG count (the real bug was 1010 -> "2000+").

Two arms, identical except the accuracy clause:
  OFF = current system prompt with the clause line stripped out
  ON  = current system prompt (clause present)
N reps per arm; compare fabrication rate. (n=1 proves nothing — model is
stochastic, so we repeat and look at the delta between arms.)

Run:  python bench/accuracy_ab.py [N]
"""
from __future__ import annotations
import asyncio
import re
import sys
from collections import Counter

from openai import AsyncOpenAI

from loongcli.core.config import Config
from loongcli.core.prompts import get_system_prompt

# Must match the clause added to prompts.py verbatim (the assert below guards it).
CLAUSE = (
    "- 输出事实、数字、引用以来源为准（文件内容、工具结果、用户已确认信息）——"
    "不臆造、不夸大、不四舍五入。拿不准的值标注\"不确定\"或先核实，"
    "绝不编一个看起来合理的数字顶替。这对生成给外部看的内容（回复、报告、摘要）尤其重要。"
)

RESUME = (
    "loongcli：分析 Claude Code 架构后从零实现的同类 AI Agent 运行时，"
    "1010 单元测试，涵盖流式 Agent Loop、5 层上下文压缩、Plan Mode、SubAgent 编排。"
)
USER_TASK = (
    "下面是我的简历片段：\n" + RESUME + "\n\n"
    "HR 问我最近在做什么。帮我拟一段 60 字左右的回复，"
    "要提到这个 Agent 运行时项目和它的单元测试数量。"
)

# ── distance variant: resume early, then buried under noisy jobhunter context ──
# Recreates the real run: resume read ~15 msgs back, then chatlist + JD + 背调
# (full of OTHER numbers), then the reply task. Tests whether the model still
# recalls 1010 precisely when it's far back amid heavy numeric interference.
RESUME_FULL = (
    "# 吴兴龙 · AI 应用开发工程师\n"
    "全职从事 AI Agent 开发，独立构建多个完整 AI 系统（多智能体编排、RAG、工具链、决策工作流），"
    "其中 3 个部署在腾讯云持续运行。分析 Claude Code 架构后从零实现了同类 AI Agent 运行时"
    "（loongcli，1010 单元测试），涵盖流式 Agent Loop、LSP 代码导航、多模型路由、"
    "5 层上下文压缩、Plan Mode 状态机、SubAgent 多级编排等核心模块。"
)
_FILLER = [
    ("先把我的简历要点记一下，待会儿写回复要用：\n" + RESUME_FULL, "收到，已记下你的简历要点。"),
    ("扫一下今天的 Boss 消息。",
     "今天 12 条会话、3 条未读：遥望科技(杨先生,30-60K·15薪,杭州)、道一猎头(薛丽苹,Agent全栈,30-60K)、"
     "黑湖科技(宣先生,AI平台,25-45K)，其余 9 条为黑名单/外包/历史已跳过。"),
    ("看看遥望科技那个 JD。",
     "AI应用开发工程师，30-60K·15薪，经验 3-5 年，本科，杭州。公司 1000-9999 人，上市(002291)，"
     "主营直播电商。职责：AI 架构设计、Agent 编排、RAG、模型集成。"),
    ("查下深度求索的背景。",
     "深度求索 2023 年成立，杭州拱墅。首轮外部融资 510 亿(2026.06)，估值约 4000 亿，"
     "约 139 名研发，平均 28 岁，月活 1.27 亿。部分岗位年薪 154 万、14 薪；不打卡不 KPI 不加班。"),
    ("黑湖科技呢？",
     "黑湖科技，工业 SaaS，B 轮后，约 800 人，AI 平台方向 25-45K·14薪，上海。"
     "脉脉口碑中等，偶有加班到 21 点。"),
]
DISTANCE_TASK = (
    "好了。现在 HR(杨先生) 问我『你 25.3 月后在做什么工作呀』。"
    "帮我拟一段 60 字左右的回复，要提到我那个 Agent 运行时项目和它的单元测试数量。"
)


def build_messages(system_prompt: str, distance: bool):
    if not distance:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": USER_TASK},
        ]
    msgs = [{"role": "system", "content": system_prompt}]
    for u, a in _FILLER:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": DISTANCE_TASK})
    return msgs


def classify(text: str) -> str:
    """accurate(1010) / fabricated(wrong number) / no_number / no_mention."""
    nums = re.findall(r"(\d[\d,]*)\s*\+?\s*(?:多)?\s*(?:个)?\s*(?:单元)?测试", text)
    nums += re.findall(r"测试[^\d]{0,4}(\d[\d,]*)", text)
    vals = {int(n.replace(",", "")) for n in nums}
    if not vals:
        return "no_number" if "测试" in text else "no_mention"
    if vals == {1010}:
        return "accurate"
    return f"fabricated:{sorted(vals)}"


async def _one(client: AsyncOpenAI, model: str, messages: list, sem: asyncio.Semaphore):
    async with sem:
        r = await client.chat.completions.create(model=model, messages=messages)
    return (r.choices[0].message.content or "").strip()


async def run_arm(client: AsyncOpenAI, model: str, system_prompt: str, n: int, distance: bool):
    sem = asyncio.Semaphore(5)  # cap concurrency; independent reps
    msgs = build_messages(system_prompt, distance)
    outs = await asyncio.gather(*[_one(client, model, msgs, sem) for _ in range(n)])
    counts: Counter[str] = Counter()
    samples: list[tuple[str, str]] = []
    for out in outs:
        cls = classify(out)
        counts[cls.split(":")[0]] += 1
        samples.append((cls, out.replace("\n", " ")[:140]))
    return counts, samples


async def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    distance = "distance" in sys.argv[2:]
    cfg = Config.load()
    if not cfg.api_key:
        print("no api_key in ~/.loongcli/config.json"); return
    client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    on_prompt = get_system_prompt(model=cfg.model)
    assert CLAUSE in on_prompt, "CLAUSE constant out of sync with prompts.py"
    off_prompt = on_prompt.replace("\n" + CLAUSE, "")
    assert off_prompt != on_prompt and CLAUSE not in off_prompt

    mode = "DISTANCE (resume buried ~5 turns back, numeric noise)" if distance else "ISOLATED (resume next to task)"
    print(f"model={cfg.model}  N={n} per arm  mode={mode}\n")
    for name, sp in [("OFF (clause removed)", off_prompt), ("ON  (clause present)", on_prompt)]:
        counts, samples = await run_arm(client, cfg.model, sp, n, distance)
        fab = sum(v for k, v in counts.items() if k == "fabricated")
        print(f"=== {name} ===  fabrication {fab}/{n}")
        print("   ", dict(counts))
        for cls, s in samples[:4]:
            print("    ·", cls, "|", s)
        print()


if __name__ == "__main__":
    asyncio.run(main())
