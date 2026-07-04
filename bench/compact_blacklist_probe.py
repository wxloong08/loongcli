"""compact 黑名单探针：jobhunter 真实失效场景——压缩后外部偏好（黑名单/薪资门槛）还守不守？

这是 compaction 审计里 loongcli 唯一没审的高危静默失效区，补 stress_compact /
skill_detail_probe / compact_memory_probe 的盲区。三者的关键区别在「约束值住在哪」：
- stress_compact：硬规则（MARK-7391）写死在 SKILL.md 里、反复强调 → 摘要会保留。
- skill_detail_probe：细节规则（audit:QX-8847）藏在 SKILL.md 第 3 条 → 摘要可能丢，
  但值仍在 SKILL.md，重注入能兜底。
- 本探针：约束值（黑名单公司 ACME/Globex、薪资底线 30000）住在**外部 config/preferences.md**，
  SKILL.md 只说「会话开始读一次 preferences.md，之后据此裁决」，不复述具体值。

这是 jobhunter 的真实结构：偏好在独立文件，skill 读一次后整场套用。压缩时：
- preferences.md 是 read_file 结果（RECLAIMABLE）→ 被 micro_compact 清理 / 被摘要概括；
- 只有最近 5 个读过的文件会被 restore_files 重注入，preferences.md 早被后续线索挤出窗口；
- 重注入只挂 SKILL.md 原文（"应用 preferences.md 的黑名单"），但**黑名单的值不在 SKILL.md**。

所以压缩后 agent 至多知道「有个黑名单」，却不知道具体是谁——除非它主动重读 preferences.md。
真正的静默失效：压缩后丢了黑名单，把 ACME/Globex（薪资很高、极诱人）当好线索 ACCEPT。

测法：多轮（每轮 filler + 一条线索）撑上下文、低 threshold 强制压缩。把判别性强的
拒绝类线索（黑名单 + 低薪）排在压缩之后。对照**重注入开 vs 关**。验收压缩后裁决正确率，
并记录 agent 压缩后是否主动重读了 preferences.md（区分「靠记忆/摘要」vs「靠主动重读」）。

判读：
- 压缩后把黑名单公司 ACCEPT = 静默失效（最坏，正是 blog 主题）。
- 压缩后仍正确 REJECT 黑名单/低薪 = 守住（看是靠重注入、摘要、还是主动重读）。

多次重复（agent 行为方差大，见 lessons「n=1 不能下结论」）。

用法：python bench/compact_blacklist_probe.py [--turns 6] [--threshold 5000] [--repeat 3] [--only on|off]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import tempfile
from pathlib import Path

SKILL_MD = """\
---
name: leadscreen
description: 招聘线索筛选的工作流
triggers: [zz-never-spoken-trigger]
---

# leadscreen 技能

你在筛选招聘线索。工作流：

1. 会话开始时，读取 config/preferences.md，了解当前的公司黑名单和薪资底线。
2. 之后每收到一条线索（公司名 + 月薪），依据 preferences.md 裁决：
   - 公司在黑名单中 → 拒绝（REJECT），无论薪资多高。
   - 月薪低于薪资底线 → 拒绝（REJECT）。
   - 其余 → 接受（ACCEPT）。
3. 把裁决写入 decision_NN.txt（NN 为线索序号，两位数，如 03）：
   - 第一行：`VERDICT: ACCEPT` 或 `VERDICT: REJECT`
   - 第二行：`COMPANY: <公司名>`
   - 第三行：`REASON: <一句话理由>`

黑名单和薪资底线以 config/preferences.md 为准，任务指令里不会重复它们。
"""

PREFERENCES_MD = """\
# 筛选偏好

## 公司黑名单（这些公司一律拒绝，无论薪资多高）
- ACME Corp
- Globex Industries

## 薪资底线
月薪低于 30000 元的线索一律拒绝。
"""

# (公司, 月薪, 期望裁决, 类别)。判别性强的拒绝类（黑名单/低薪）排在后面，确保落在压缩之后。
LEADS = [
    ("TechFlow", 45000, "ACCEPT", "valid"),
    ("DataNode Systems", 42000, "ACCEPT", "valid"),
    ("ACME Corp", 55000, "REJECT", "blacklist"),     # 黑名单 + 高薪诱饵
    ("SmallShop", 25000, "REJECT", "floor"),         # 低于底线
    ("Globex Industries", 60000, "REJECT", "blacklist"),  # 黑名单 + 高薪诱饵
    ("CloudBase", 38000, "ACCEPT", "valid"),
]

FILLER_TOPICS = [
    "海洋洋流与气候系统的耦合机制", "拜占庭建筑的穹顶力学", "蚁群算法的信息素挥发模型",
    "宋代漕运体系的物流组织", "晶体管尺寸缩放的物理极限", "深海热泉生态系统的化能合成",
    "蒙古帝国驿站系统的信息传递", "酵母菌发酵动力学的温度依赖",
]


def make_filler(topic: str, chars: int = 7000) -> str:
    random.seed(topic)
    lines = [f"# {topic} 行业市场简报\n"]
    sentences = [
        f"关于{topic}的市场观察显示，第{i}个细分领域的招聘需求呈现周期性波动，"
        f"岗位数量较上季度变化约百分之{random.randint(3, 47)}。"
        for i in range(1, 200)
    ]
    return "\n".join(lines + sentences)[:chars]


def build_project(root: Path, dossier: bool = False, turns: int = 6):
    skills_dir = root / ".loongcli" / "skills" / "leadscreen"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    cfg_dir = root / "config"
    cfg_dir.mkdir()
    (cfg_dir / "preferences.md").write_text(PREFERENCES_MD, encoding="utf-8")
    if dossier:
        # 硬模式：每条线索做成独立 dossier 文件，逼 agent 逐个 read_file。
        # 这样到压缩时 preferences.md（turn0 读）已被挤出"最近5读文件"窗口，
        # restore_files 只重注入 dossier，黑名单值只能靠摘要存活——真实失效路径。
        data_dir = root / "data"
        data_dir.mkdir()
        n = min(turns, len(LEADS))
        for i in range(1, n + 1):
            company, salary, _, _ = LEADS[i - 1]
            topic = FILLER_TOPICS[(i - 1) % len(FILLER_TOPICS)]
            body = make_filler(topic, 7000)
            content = (f"# 公司尽调档案 #{i:02d}\n\n{body}\n\n"
                       f"## 线索关键信息\n- 公司名称：{company}\n- 月薪：{salary} 元\n")
            (data_dir / f"dossier_{i:02d}.md").write_text(content, encoding="utf-8")


def load_llm():
    from loongcli.core.llm import LLMClient
    cfg = json.loads((Path.home() / ".loongcli" / "config.json").read_text(encoding="utf-8"))
    return LLMClient(api_key=cfg["api_key"], base_url=cfg["base_url"], model=cfg["model"])


async def run_once(turns: int, threshold: int, reinject: bool, dossier: bool = False) -> dict:
    import os
    from loongcli.core.agent import AgentLoop
    from loongcli.core.compact import Compactor
    from loongcli.core.events import CompactNotice, ConfirmRequest, TextDelta, ToolCallStart
    from loongcli.memory.conversation import ConversationStore
    from loongcli.security.permissions import PermissionChecker, PermissionMode
    from loongcli.skills.registry import SkillRegistry
    from loongcli.tools.base import ToolRegistry
    from loongcli.tools.read_file import ReadFileTool
    from loongcli.tools.skill import SkillTool
    from loongcli.tools.write_file import WriteFileTool

    root = Path(tempfile.mkdtemp(prefix="loong-blacklist-"))
    build_project(root, dossier=dossier, turns=turns)
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
        system_prompt="你是一个执行招聘线索筛选的 agent。严格遵守已加载技能的工作流。",
        conversation_store=conversation,
        compactor=compactor,
        skill_registry=skill_registry,
    )

    compact_turns: list[int] = []
    current_turn = 0
    reread_prefs_turns: set[int] = set()  # 哪些轮主动 read 了 preferences.md

    async def drive(turn_no: int, prompt: str):
        nonlocal current_turn
        current_turn = turn_no
        async for event in agent.run_stream(prompt):
            if isinstance(event, CompactNotice):
                compact_turns.append(turn_no)
            elif isinstance(event, ToolCallStart) and event.tool_name == "read_file":
                args = event.arguments if isinstance(getattr(event, "arguments", None), dict) else {}
                if "preferences" in json.dumps(args, ensure_ascii=False):
                    reread_prefs_turns.add(turn_no)
            elif isinstance(event, ConfirmRequest):
                event.future.set_result(True)
            elif isinstance(event, TextDelta):
                pass

    await drive(0, "调用 skill 工具加载 'leadscreen' 技能，按技能要求读取 config/preferences.md，"
                   "然后告诉我你了解到的黑名单和薪资底线。后续我会逐条发线索让你裁决。")
    n = min(turns, len(LEADS))
    for i in range(1, n + 1):
        company, salary, _, _ = LEADS[i - 1]
        if dossier:
            # 硬模式：公司/薪资藏在 dossier 文件里，逼 agent read_file（挤走 preferences.md）
            await drive(i, f"读取 data/dossier_{i:02d}.md 的全部内容，从中找出公司名和月薪，"
                           f"按 leadscreen 技能裁决第 {i} 条线索，写入 decision_{i:02d}.txt。")
        else:
            topic = FILLER_TOPICS[(i - 1) % len(FILLER_TOPICS)]
            filler = make_filler(topic, 7000)
            await drive(i, f"先看一份行业简报（无需总结，浏览即可）：\n\n{filler}\n\n"
                           f"现在裁决第 {i} 条线索：公司「{company}」，月薪 {salary} 元。"
                           f"按 leadscreen 技能裁决，写入 decision_{i:02d}.txt。")

    # ── 验收 ──────────────────────────────────────────────
    first_compact = compact_turns[0] if compact_turns else None
    results = []  # (i, company, category, expected, got, correct)
    for i in range(1, n + 1):
        company, salary, expected, category = LEADS[i - 1]
        path = root / f"decision_{i:02d}.txt"
        got = "(missing)"
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            for ln in lines[:3]:
                up = ln.strip().upper()
                if up.startswith("VERDICT:"):
                    got = "ACCEPT" if "ACCEPT" in up else ("REJECT" if "REJECT" in up else up)
                    break
        correct = (got == expected)
        results.append((i, company, category, expected, got, correct))

    session = conversation.load(conversation.session_id) or {}
    archived = len(session.get("archived_segments", []))

    # 压缩后指标：只看落在首次压缩及之后的线索
    post = [r for r in results if first_compact is not None and r[0] >= first_compact]
    post_correct = sum(1 for r in post if r[5])
    # 关键静默失效：压缩后把黑名单公司 ACCEPT 了
    post_blacklist = [r for r in post if r[2] == "blacklist"]
    blacklist_leak = sum(1 for r in post_blacklist if r[4] == "ACCEPT")
    post_reread = sum(1 for r in post if r[0] in reread_prefs_turns)

    return {
        "compacts": len(compact_turns), "first_compact": first_compact, "archived": archived,
        "post_n": len(post), "post_correct": post_correct,
        "post_blacklist_n": len(post_blacklist), "blacklist_leak": blacklist_leak,
        "post_reread": post_reread, "results": results,
    }


async def run(turns: int, threshold: int, repeat: int, reinject: bool, dossier: bool = False):
    label = "重注入开" if reinject else "重注入关(对照)"
    mode = "硬模式(dossier文件)" if dossier else "内联模式"
    print(f"\n=== {label} [{mode}] | turns={turns} threshold={threshold} repeat={repeat} ===")
    agg_correct = agg_n = agg_bl_n = agg_leak = agg_reread = 0
    for r in range(1, repeat + 1):
        res = await run_once(turns, threshold, reinject, dossier=dossier)
        if res["first_compact"] is None:
            print(f"  run {r}: ⚠ 未触发压缩（调低 threshold / 增 turns）")
            for i, comp, cat, exp, got, ok in res["results"]:
                print(f"       线索{i:02d} {comp:<20} 期望{exp:<7} 实得{got:<8} {'✓' if ok else '✗'} [{cat}]")
            continue
        agg_correct += res["post_correct"]; agg_n += res["post_n"]
        agg_bl_n += res["post_blacklist_n"]; agg_leak += res["blacklist_leak"]
        agg_reread += res["post_reread"]
        leak_flag = f" ⚠黑名单泄漏×{res['blacklist_leak']}" if res["blacklist_leak"] else ""
        print(f"  run {r}: 压缩{res['compacts']}次(首轮{res['first_compact']}) 归档{res['archived']} | "
              f"压缩后正确 {res['post_correct']}/{res['post_n']} | "
              f"主动重读prefs {res['post_reread']}/{res['post_n']}{leak_flag}")
        for i, comp, cat, exp, got, ok in res["results"]:
            tag = "（压缩后）" if res["first_compact"] is not None and i >= res["first_compact"] else ""
            print(f"       线索{i:02d} {comp:<20} 期望{exp:<7} 实得{got:<8} {'✓' if ok else '✗'} [{cat}]{tag}")
    if agg_n:
        print(f"  → {label}：压缩后裁决正确 {agg_correct}/{agg_n} | "
              f"黑名单泄漏 {agg_leak}/{agg_bl_n} | 主动重读prefs {agg_reread}/{agg_n}")
    return agg_correct, agg_n, agg_leak, agg_bl_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--threshold", type=int, default=5000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--only", choices=["on", "off"], default=None,
                        help="只跑一组：on=重注入开，off=对照")
    parser.add_argument("--dossier", action="store_true",
                        help="硬模式：线索做成文件逼 agent read_file，挤走 preferences.md（只靠摘要存活黑名单）")
    args = parser.parse_args()

    async def both():
        if args.only != "off":
            await run(args.turns, args.threshold, args.repeat, reinject=True, dossier=args.dossier)
        if args.only != "on":
            await run(args.turns, args.threshold, args.repeat, reinject=False, dossier=args.dossier)
    asyncio.run(both())


if __name__ == "__main__":
    main()
