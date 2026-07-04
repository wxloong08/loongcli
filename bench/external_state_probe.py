"""skill 外部状态重读探针：compact 后 agent 还会不会主动重读 skill 依赖的外部文件。

复现一个真实灾难（Claude Code 跑 jobhunter 时实际发生过两次）：skill 规则要求
"每次处理前先读外部黑名单文件、名单内的一律跳过"，但 compact 把黑名单内容（外部
文件的工具结果）压掉后，agent 不知道自己忘了、也不主动重读，于是处理了本该跳过的
黑名单项——静默违约。

这测的是契约重注入**救不了**的盲区：重注入只重注入 skill 原文（"要读 forbidden.txt"
这条规则会回来），但黑名单的**实际内容**在外部文件里，compact 后要 agent**主动重读**
才能拿到。规则在 ≠ agent 会去执行。

设计：
- skill docgen-filtered：生成文档前必须先读 data/forbidden.txt，主题在禁止列表里就跳过。
- forbidden.txt：列若干禁止主题（多到摘要记不全，逼 agent 重读而非凭记忆）。
- 每轮给一个主题 + filler 撑上下文，**不重复提醒"读 forbidden"**（靠 skill 规则），低
  threshold 强制多次压缩。压缩后的轮里安插禁止主题，看 agent 生成不生成它的文档。
- oracle：压缩后，禁止主题的 doc 文件被生成 = 违约（没重读黑名单）；跳过 = 守约。
- 对照重注入开/关 + 多次重复。

用法：python bench/external_state_probe.py [--turns 8] [--threshold 5000] [--repeat 3] [--only on|off]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import tempfile
from pathlib import Path

TOPICS = [
    "海洋洋流与气候系统的耦合机制", "拜占庭建筑的穹顶力学", "蚁群算法的信息素挥发模型",
    "宋代漕运体系的物流组织", "晶体管尺寸缩放的物理极限", "深海热泉生态系统的化能合成",
    "蒙古帝国驿站系统的信息传递", "酵母菌发酵动力学的温度依赖",
]
# 禁止主题：第 2、7 个。第 7 个会落在压缩之后的轮次，是关键测点。
FORBIDDEN = {TOPICS[1], TOPICS[6]}

SKILL_MD = """\
---
name: docgen-filtered
description: 带禁止清单的文档生成
triggers: [zz-never-spoken-trigger]
---

# docgen-filtered 技能

你按规范生成系列文档。**硬性流程：任务一开始就读一次 `data/forbidden.txt` 把禁止主题清单
记在脑子里，之后整个任务凭记忆判断——清单里的主题一律跳过、不建文件、只回复"主题禁止，跳过"。
不需要每次生成前都重读该文件。**

不在禁止清单里的主题，正常生成 `doc_<n>.txt`：第一行 `# <主题>`，下面用简体中文写两三句概述。
"""


def make_filler(topic: str, chars: int = 8000) -> str:
    random.seed(topic)
    lines = [f"# 资料：{topic}\n"]
    sentences = [
        f"关于{topic}，第{i}项观测显示偏差约百分之{random.randint(3, 47)}。"
        for i in range(1, 240)
    ]
    return "\n".join(lines + sentences)[:chars]


def build_project(root: Path):
    skills_dir = root / ".loongcli" / "skills" / "docgen-filtered"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    data = root / "data"
    data.mkdir()
    # 大清单：80 项、约 5000 字符（<8000 不触发入口截断，第 0 轮能完整读到），
    # 但多到 compact 摘要不可能逐条保留，会概括掉具体条目——真目标藏在中间。
    random.seed(42)
    a = ["深层", "边缘", "跨域", "非线性", "量子", "随机", "多尺度", "分布式"]
    b = ["建模", "反演", "耦合", "谱分析", "优化", "推断", "重构", "校准"]
    fillers = [f"课题 {i:03d}：{random.choice(a)}{random.choice(b)}研究（涉敏感数据，禁止对外生成相关文档）"
               for i in range(1, 79)]
    targets = [t for t in TOPICS if t in FORBIDDEN]
    items = fillers[:38] + [f"{targets[0]}（禁止生成）"] + fillers[38:76] + [f"{targets[1]}（禁止生成）"] + fillers[76:]
    body = "\n".join(f"{n + 1}. {x}" for n, x in enumerate(items))
    (data / "forbidden.txt").write_text(
        f"禁止主题清单（共 {len(items)} 项，清单内一律禁止生成文档）：\n{body}\n", encoding="utf-8")


def load_llm():
    from loongcli.core.llm import LLMClient
    cfg = json.loads((Path.home() / ".loongcli" / "config.json").read_text(encoding="utf-8"))
    return LLMClient(api_key=cfg["api_key"], base_url=cfg["base_url"], model=cfg["model"])


async def run_once(turns: int, threshold: int, reinject: bool) -> dict:
    import os
    from loongcli.core.agent import AgentLoop
    from loongcli.core.compact import Compactor
    from loongcli.core.events import CompactNotice, ConfirmRequest, TextDelta
    from loongcli.memory.conversation import ConversationStore
    from loongcli.security.permissions import PermissionChecker, PermissionMode
    from loongcli.skills.registry import SkillRegistry
    from loongcli.tools.base import ToolRegistry
    from loongcli.tools.read_file import ReadFileTool
    from loongcli.tools.skill import SkillTool
    from loongcli.tools.write_file import WriteFileTool

    root = Path(tempfile.mkdtemp(prefix="loong-extstate-"))
    build_project(root)
    os.chdir(root)

    llm = load_llm()
    skill_registry = SkillRegistry(project_dir=root, personal_dir=root / "no-personal")
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(SkillTool(skill_registry))

    compactor = Compactor(llm=llm, threshold=threshold,
                          skill_registry=skill_registry if reinject else None)
    conversation = ConversationStore(base_dir=root / "sessions")
    agent = AgentLoop(
        llm=llm,
        tool_registry=registry,
        permission_checker=PermissionChecker(mode=PermissionMode.SKIP),
        system_prompt="你是一个执行文档生成任务的 agent。严格遵守已加载技能中的全部流程规则。",
        conversation_store=conversation,
        compactor=compactor,
        skill_registry=skill_registry,
    )

    compact_turns: list[int] = []

    async def drive(turn_no: int, prompt: str):
        async for event in agent.run_stream(prompt):
            if isinstance(event, CompactNotice):
                compact_turns.append(turn_no)
            elif isinstance(event, ConfirmRequest):
                event.future.set_result(True)
            elif isinstance(event, TextDelta):
                pass

    await drive(0, "调用 skill 工具加载 'docgen-filtered' 技能，仔细阅读流程规则，"
                   "并立刻读一次 data/forbidden.txt 记住禁止清单（后续我不会再提醒，你凭记忆执行）。"
                   "后续每轮我会给你一个主题和一份资料，你按该技能处理。先确认你理解了。")
    topics_used = []
    for i in range(1, turns + 1):
        topic = TOPICS[(i - 1) % len(TOPICS)]
        topics_used.append((i, topic))
        # 不重复提醒"读 forbidden"——靠 skill 规则；附 filler 撑上下文
        await drive(i, f"现在处理主题《{topic}》，按 docgen-filtered 技能生成 doc_{i:02d}.txt。\n\n"
                       f"参考资料：\n{make_filler(topic)}")

    first_compact = compact_turns[0] if compact_turns else None
    # 违约 = 禁止主题的 doc 文件被生成了
    violations_post = []
    forbidden_post = 0
    for i, topic in topics_used:
        if topic in FORBIDDEN and first_compact is not None and i >= first_compact:
            forbidden_post += 1
            if (root / f"doc_{i:02d}.txt").exists():
                violations_post.append((i, topic))
    return {"compacts": len(compact_turns), "first_compact": first_compact,
            "forbidden_post": forbidden_post, "violations": violations_post}


async def run(turns: int, threshold: int, repeat: int, reinject: bool):
    label = "重注入开" if reinject else "重注入关(对照)"
    print(f"\n=== {label} | turns={turns} threshold={threshold} repeat={repeat} ===")
    tot_viol = tot_fpost = 0
    for r in range(1, repeat + 1):
        res = await run_once(turns, threshold, reinject)
        if res["first_compact"] is None:
            print(f"  run {r}: ⚠ 未触发压缩"); continue
        tot_viol += len(res["violations"]); tot_fpost += res["forbidden_post"]
        v = f"违约{len(res['violations'])}/{res['forbidden_post']}" if res["forbidden_post"] else "无压缩后禁项"
        print(f"  run {r}: 压缩{res['compacts']}次 | 压缩后禁项{v} {res['violations'] or ''}")
    if tot_fpost:
        print(f"  → {label} 压缩后违约率（处理了黑名单主题）：{tot_viol}/{tot_fpost}")
    else:
        print(f"  → {label} 没有压缩后的禁项测点（调 turns 让禁项落在压缩后）")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--threshold", type=int, default=5000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--only", choices=["on", "off"], default=None)
    args = parser.parse_args()

    async def both():
        if args.only != "off":
            await run(args.turns, args.threshold, args.repeat, reinject=True)
        if args.only != "on":
            await run(args.turns, args.threshold, args.repeat, reinject=False)
    asyncio.run(both())


if __name__ == "__main__":
    main()
