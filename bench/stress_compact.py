"""compact 压力测试：多轮长对话强制触发压缩，验证压缩后行为正确性。

场景：合成技能 relay，规则写在 SKILL.md（经 skill() 工具加载，内容以工具结果
形式进入上下文）：每个 .txt 第一行必须是验证码 MARK-7391。每轮读一个大 filler
撑上下文，threshold 调低强制多次压缩。验收：压缩后文件是否仍带验证码，且
磁盘归档段数 == 压缩次数（历史保真）。

【已知局限 — 见 2026-06-13 实验结论】
本测试用的是「突出硬规则」型契约（SKILL.md 里反复强调"忘记=失败"）。实测发现
对照组（--no-reinject，关闭技能重注入）也全过——因为 skill 工具结果不在
micro_compact 的 RECLAIMABLE 清理名单里，会完整进入摘要 LLM 输入，摘要把这种
显眼的硬规则保留了下来。所以本测试当前**测的是「压缩后契约+归档保真」的回归，
不能证明技能重注入的必要性**。要测出重注入价值需要「细节型契约」（长 skill 里
不起眼的具体步骤，摘要会丢）——待后续设计。

重注入因此定位为面向未来的防御：一旦 skill 加入 micro_compact 清理、或技能原文
超出摘要保留能力、或多代压缩累积衰减，它才会真正兜底。当前 loongcli 参数下未被证伪。

用法：python bench/stress_compact.py [--turns 8] [--threshold 5000] [--no-reinject]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).parent.parent
MARKER = "MARK-7391"

SKILL_MD = """\
---
name: relay
description: 文件创建任务的契约规则
triggers: [zz-never-spoken-trigger]
---

# relay 技能

你在执行接力文件创建任务。**以下规则是硬性契约，每个文件都必须遵守：**

1. 每个创建的 .txt 文件，第一行必须是验证码 `MARK-7391`（一字不差，单独一行）
2. 第二行写文件的序号（如 `file 03`）
3. 第三行起，用一句话概括你刚读过的 filler 文件的主题

验证码只在本技能中出现，任务指令里不会重复它。忘记验证码 = 任务失败。
"""

FILLER_TOPICS = [
    "海洋洋流与气候系统的耦合机制", "拜占庭建筑的穹顶力学", "蚁群算法的信息素挥发模型",
    "宋代漕运体系的物流组织", "晶体管尺寸缩放的物理极限", "深海热泉生态系统的化能合成",
    "蒙古帝国驿站系统的信息传递", "酵母菌发酵动力学的温度依赖", "中世纪行会制度的质量管控",
    "candidate 区块链分片的跨片通信开销",
]


def make_filler(topic: str, chars: int = 9000) -> str:
    random.seed(topic)
    lines = [f"# {topic}\n"]
    sentences = [
        f"关于{topic}的研究表明，第{i}个观测点的数据呈现出显著的周期性波动，"
        f"其振幅与基准值的偏差约为百分之{random.randint(3, 47)}，"
        f"这一现象在第{random.randint(2, 19)}组对照实验中得到了重复验证。"
        for i in range(1, 200)
    ]
    text = "\n".join(lines + sentences)
    return text[:chars]


def build_project(root: Path, turns: int):
    skills_dir = root / ".loongcli" / "skills" / "relay"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    data_dir = root / "data"
    data_dir.mkdir()
    for i in range(1, turns + 1):
        topic = FILLER_TOPICS[(i - 1) % len(FILLER_TOPICS)]
        (data_dir / f"filler_{i:02d}.md").write_text(make_filler(topic), encoding="utf-8")


def load_llm():
    from loongcli.core.llm import LLMClient
    cfg_path = Path.home() / ".loongcli" / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return LLMClient(api_key=cfg["api_key"], base_url=cfg["base_url"], model=cfg["model"])


async def run(turns: int, threshold: int, reinject: bool = True) -> int:
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

    root = Path(tempfile.mkdtemp(prefix="loong-stress-"))
    build_project(root, turns)
    os.chdir(root)  # 工具的相对路径以合成项目为根

    llm = load_llm()
    skill_registry = SkillRegistry(project_dir=root, personal_dir=root / "no-personal")
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    # 技能必须走 skill() 工具加载路径：内容以工具结果形式进入上下文，
    # 压缩时会被占位符替换/摘要掉——这是 Claude Code jobhunter 故障的忠实复刻。
    # （enrich_prompt 自动触发路径会把技能注入用户消息，被摘要模板的
    # "保留所有用户消息原文"规则保护，测不出契约丢失。）
    registry.register(SkillTool(skill_registry))

    compactor = Compactor(llm=llm, threshold=threshold,
                          skill_registry=skill_registry if reinject else None)
    conversation = ConversationStore(base_dir=root / "sessions")
    agent = AgentLoop(
        llm=llm,
        tool_registry=registry,
        permission_checker=PermissionChecker(mode=PermissionMode.SKIP),
        system_prompt="你是一个执行文件任务的 agent。严格遵守已加载技能中的全部规则。",
        conversation_store=conversation,
        compactor=compactor,
        skill_registry=skill_registry,
    )

    compact_turns: list[int] = []  # 发生过压缩的轮次号

    async def drive(turn_no: int, prompt: str):
        async for event in agent.run_stream(prompt):
            if isinstance(event, CompactNotice):
                compact_turns.append(turn_no)
                print(f"    [压缩发生] {event.before} -> {event.after} 条消息")
            elif isinstance(event, ConfirmRequest):
                event.future.set_result(True)
            elif isinstance(event, TextDelta):
                pass

    mode = "实验组（技能重注入开启）" if reinject else "对照组（技能重注入关闭）"
    print(f"压力测试 [{mode}]: {turns} 轮 | threshold={threshold} tokens | 项目: {root}")
    await drive(0, "调用 skill 工具加载 'relay' 技能并仔细阅读其中的规则。"
                   "后续每轮我会让你读一个资料文件再创建一个 txt 文件，"
                   "所有文件都必须遵守该技能的全部规则。先确认你理解了规则。")
    for i in range(1, turns + 1):
        print(f"  轮次 {i}/{turns} ...")
        await drive(i, f"读取 data/filler_{i:02d}.md 的全部内容，"
                       f"然后创建 file_{i:02d}.txt，内容严格遵守 relay 技能的规则。")

    # ── 验收 ────────────────────────────────────────────────
    first_compact = compact_turns[0] if compact_turns else None
    results = []
    for i in range(1, turns + 1):
        path = root / f"file_{i:02d}.txt"
        if not path.exists():
            results.append((i, "missing"))
            continue
        first_line = path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
        results.append((i, "ok" if first_line == MARKER else f"违约: {first_line[:40]}"))

    session_data = conversation.load(conversation.session_id) or {}
    archived = len(session_data.get("archived_segments", []))

    print(f"\n{'=' * 56}")
    print(f"压缩发生 {len(compact_turns)} 次（轮次 {compact_turns}）| 磁盘归档段 {archived} 个")
    if first_compact is None:
        print("⚠ 全程未触发压缩，测试无效——调低 --threshold 或增加 --turns")
        return 2

    pre = [(i, s) for i, s in results if i < first_compact]
    post = [(i, s) for i, s in results if i >= first_compact]
    pre_ok = sum(1 for _, s in pre if s == "ok")
    post_ok = sum(1 for _, s in post if s == "ok")
    print(f"压缩前契约遵守: {pre_ok}/{len(pre)} | 压缩后: {post_ok}/{len(post)}")
    for i, s in results:
        flag = "（压缩后）" if i >= first_compact else ""
        print(f"  file_{i:02d}.txt: {s} {flag}")

    if archived != len(compact_turns):
        print(f"⚠ 归档段数({archived})与压缩次数({len(compact_turns)})不一致——历史保真可能有问题")

    passed = post and post_ok == len(post) and archived == len(compact_turns)
    print(f"\n结论: {'✓ PASS — 压缩后契约保持 + 归档保真' if passed else '✗ FAIL — 压缩破坏了契约或归档缺失'}")
    print("（注：突出硬规则型契约靠摘要本身即可存活，本测试不证明技能重注入的必要性，见模块 docstring）")
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--threshold", type=int, default=5000)
    parser.add_argument("--no-reinject", action="store_true", help="对照组：关闭技能原文重注入")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.turns, args.threshold, reinject=not args.no_reinject)))


if __name__ == "__main__":
    main()
