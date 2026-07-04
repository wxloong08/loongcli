"""skill 第 5 层探针：细节型契约在压缩后是否保持 + 契约重注入的价值。

这是 stress_compact 的姊妹探针，补它的已知盲区。区别在 skill 契约的「显著度」：
- stress_compact：MARKER 是被反复强调的硬规则（"硬性契约、忘记=任务失败"）——
  摘要倾向保留这种，所以重注入开/关两组都过，测不出重注入价值。
- 本探针：关键要求平铺成 8 条格式规范里**不起眼的第 3 条**（`audit: QX-8847`），
  不强调重要性——这是「细节型契约」，压缩摘要更可能把它概括掉（"遵守 docgen 格式"
  而丢掉具体第 3 条）。

测法：多轮读 filler + 生成文档（撑上下文、低 threshold 强制多次压缩），对照
**重注入开 vs 关**，看压缩后生成的文档还带不带那行 audit。
- 开守、关忘 → **测出了重注入的价值**（细节型契约靠摘要存不住，靠重注入兜底）。
- 都守 → 摘要本身就够（细节也没丢），重注入仍未被证伪。
- 都忘 → 重注入也没救住（重注入或检测有问题）。

多次重复（agent 行为方差大，见 lessons「n=1 不能下结论」）。

用法：python bench/skill_detail_probe.py [--turns 8] [--threshold 5000] [--repeat 3]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import tempfile
from pathlib import Path

AUDIT = "audit: QX-8847"  # 细节型契约：藏在 8 条规范的第 3 条，不强调

SKILL_MD = """\
---
name: docgen
description: 系列文档生成的格式规范
triggers: [zz-never-spoken-trigger]
---

# docgen 技能

你在按统一规范生成一系列文档。生成每个 .txt 文件时，遵守以下格式规范：

1. 第一行是文档标题，格式 `# <主题>`
2. 第二行留空
3. 第三行固定写 `audit: QX-8847`
4. 正文用简体中文
5. 每段不超过三句话
6. 段落之间空一行
7. 不使用 Markdown 列表符号
8. 末尾单独一行 `--- end ---`
"""

FILLER_TOPICS = [
    "海洋洋流与气候系统的耦合机制", "拜占庭建筑的穹顶力学", "蚁群算法的信息素挥发模型",
    "宋代漕运体系的物流组织", "晶体管尺寸缩放的物理极限", "深海热泉生态系统的化能合成",
    "蒙古帝国驿站系统的信息传递", "酵母菌发酵动力学的温度依赖",
]


def make_filler(topic: str, chars: int = 9000) -> str:
    random.seed(topic)
    lines = [f"# {topic}\n"]
    sentences = [
        f"关于{topic}的研究表明，第{i}个观测点的数据呈现出显著的周期性波动，"
        f"其振幅与基准值的偏差约为百分之{random.randint(3, 47)}。"
        for i in range(1, 200)
    ]
    return "\n".join(lines + sentences)[:chars]


def build_project(root: Path, turns: int):
    skills_dir = root / ".loongcli" / "skills" / "docgen"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    data_dir = root / "data"
    data_dir.mkdir()
    for i in range(1, turns + 1):
        topic = FILLER_TOPICS[(i - 1) % len(FILLER_TOPICS)]
        (data_dir / f"src_{i:02d}.md").write_text(make_filler(topic), encoding="utf-8")


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

    root = Path(tempfile.mkdtemp(prefix="loong-detail-"))
    build_project(root, turns)
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
        system_prompt="你是一个执行文档生成任务的 agent。严格遵守已加载技能中的全部格式规范。",
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

    await drive(0, "调用 skill 工具加载 'docgen' 技能并仔细阅读它的格式规范。"
                   "后续每轮我会给你一份资料，让你据此生成一个文档，全部遵守该技能规范。先确认你理解了。")
    for i in range(1, turns + 1):
        await drive(i, f"读取 data/src_{i:02d}.md 的全部内容，然后生成 doc_{i:02d}.txt，"
                       f"主题就用资料的主题，内容遵守 docgen 技能的全部格式规范。")

    # 验收：每个文档第三行是否带 audit 行
    first_compact = compact_turns[0] if compact_turns else None
    results = []
    for i in range(1, turns + 1):
        path = root / f"doc_{i:02d}.txt"
        if not path.exists():
            results.append((i, None)); continue
        lines = path.read_text(encoding="utf-8").splitlines()
        third = lines[2].strip() if len(lines) >= 3 else ""
        # 宽松匹配：audit 行出现在前 5 行任意位置都算守约（避免纠结具体行号）
        head = "\n".join(lines[:5])
        results.append((i, AUDIT in head))

    session = conversation.load(conversation.session_id) or {}
    archived = len(session.get("archived_segments", []))

    post_ok = post_n = 0
    if first_compact is not None:
        for i, ok in results:
            if i >= first_compact:
                post_n += 1
                post_ok += bool(ok)
    return {"compacts": len(compact_turns), "archived": archived,
            "first_compact": first_compact, "post_ok": post_ok, "post_n": post_n,
            "all": results}


async def run(turns: int, threshold: int, repeat: int, reinject: bool):
    label = "重注入开" if reinject else "重注入关(对照)"
    print(f"\n=== {label} | turns={turns} threshold={threshold} repeat={repeat} ===")
    tot_ok = tot_n = 0
    for r in range(1, repeat + 1):
        res = await run_once(turns, threshold, reinject)
        if res["first_compact"] is None:
            print(f"  run {r}: ⚠ 未触发压缩（调低 threshold / 增 turns）")
            continue
        tot_ok += res["post_ok"]; tot_n += res["post_n"]
        print(f"  run {r}: 压缩{res['compacts']}次 归档{res['archived']} | "
              f"压缩后守约 {res['post_ok']}/{res['post_n']}")
    if tot_n:
        print(f"  → {label} 压缩后细节契约遵守率：{tot_ok}/{tot_n}")
    return tot_ok, tot_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--threshold", type=int, default=5000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--only", choices=["on", "off"], default=None,
                        help="只跑一组：on=重注入开，off=对照")
    args = parser.parse_args()

    async def both():
        if args.only != "off":
            await run(args.turns, args.threshold, args.repeat, reinject=True)
        if args.only != "on":
            await run(args.turns, args.threshold, args.repeat, reinject=False)
    asyncio.run(both())


if __name__ == "__main__":
    main()
