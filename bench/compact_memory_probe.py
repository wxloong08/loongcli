"""compact 长程记忆探针：测「第一轮埋的一次性信息，较大规模压缩后能否被 agent 访问」。

回答两个问题：
(1) 满载/较大规模压缩（把较大历史压成摘要）后，早期一次性信息保留多少？
(2) 当前机制（摘要被动保留 + search_history 主动检索）能让 agent 找回第一轮内容吗？

与 stress_compact 的关键区别：那个测「反复强调的硬规则」（摘要会保留，测不出丢失）；
这个测「只说一次、不重申的细节代号」——这才是「开头讨论的某个细节」的真实模型，
摘要更容易把它丢掉。

设计：
- 第一轮平静交代一个不可猜代号 ZEPHYR-4173（只说一次），让 agent 回个确认。
- 中间多轮读 filler + 建无关文件，把上下文撑到 threshold 触发压缩（代号被压进摘要/归档）。
- 末轮要求把代号写进文件。验收末轮文件是否带正确代号。
- 捕获末轮是否调用 search_history → 区分「摘要被动保留」vs「主动检索找回」。

对照：--no-compact（超大 threshold 不压缩）= 可达性上界（代号全程在上下文，应 100%）。
重复多次取均值（agent 行为方差极大，单次不可信——见 lessons「n=1 不能下结论」）。

用法：python bench/compact_memory_probe.py [--turns 6] [--threshold 20000] [--repeat 3] [--no-compact]
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
SECRET = "ZEPHYR-4173"  # 不可猜、只在第一轮出现一次

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
        f"其振幅与基准值的偏差约为百分之{random.randint(3, 47)}，"
        f"这一现象在第{random.randint(2, 19)}组对照实验中得到了重复验证。"
        for i in range(1, 200)
    ]
    return "\n".join(lines + sentences)[:chars]


def build_project(root: Path, turns: int):
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    for i in range(1, turns + 1):
        topic = FILLER_TOPICS[(i - 1) % len(FILLER_TOPICS)]
        (data_dir / f"filler_{i:02d}.md").write_text(make_filler(topic), encoding="utf-8")


def load_llm():
    from loongcli.core.llm import LLMClient
    cfg = json.loads((Path.home() / ".loongcli" / "config.json").read_text(encoding="utf-8"))
    return LLMClient(api_key=cfg["api_key"], base_url=cfg["base_url"], model=cfg["model"])


async def run_once(turns: int, threshold: int) -> dict:
    import os
    from loongcli.core.agent import AgentLoop
    from loongcli.core.compact import Compactor
    from loongcli.core.events import CompactNotice, ConfirmRequest, TextDelta, ToolCallStart
    from loongcli.memory.conversation import ConversationStore
    from loongcli.security.permissions import PermissionChecker, PermissionMode
    from loongcli.tools.base import ToolRegistry
    from loongcli.tools.read_file import ReadFileTool
    from loongcli.tools.write_file import WriteFileTool
    from loongcli.tools.search_history import SearchHistoryTool

    root = Path(tempfile.mkdtemp(prefix="loong-mem-"))
    os.chdir(root)  # 末轮 write deploy.txt 用

    llm = load_llm()
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    conversation = ConversationStore(base_dir=root / "sessions")
    registry.register(SearchHistoryTool(conversation))  # 给 agent 主动检索的能力

    compactor = Compactor(llm=llm, threshold=threshold)
    agent = AgentLoop(
        llm=llm,
        tool_registry=registry,
        permission_checker=PermissionChecker(mode=PermissionMode.SKIP),
        system_prompt="你是一个执行文件任务的 agent。认真记住用户交代的背景信息。",
        conversation_store=conversation,
        compactor=compactor,
    )

    compacts = 0
    used_search_in_final = False
    final_turn = False
    peak_prompt = 0

    async def drive(prompt: str):
        nonlocal compacts, used_search_in_final, peak_prompt
        async for event in agent.run_stream(prompt):
            if isinstance(event, CompactNotice):
                compacts += 1
            elif isinstance(event, ToolCallStart) and final_turn and event.tool_name == "search_history":
                used_search_in_final = True
            elif isinstance(event, ConfirmRequest):
                event.future.set_result(True)
            elif isinstance(event, TextDelta):
                pass
        peak_prompt = max(peak_prompt, agent._last_prompt_tokens)

    # 第一轮：平静交代代号（只说一次，不强调"硬规则"）
    await drive(f"背景信息：本项目的内部部署代号是 {SECRET}。后面几轮我会让你读资料、"
                f"建文件，最后一轮会用到这个代号。现在回复「已记住」即可，先别做别的。")
    # 中间轮：读 filler + 建无关文件，撑上下文触发压缩
    for i in range(1, turns + 1):
        topic = FILLER_TOPICS[(i - 1) % len(FILLER_TOPICS)]
        material = make_filler(topic, 8000)  # 直接进 user 消息（不可回收），撑大上下文
        await drive(f"下面是第 {i} 份资料，请用一句话概括它的主题（只回复概括）：\n\n{material}")
    # 末轮：用到第一轮的代号
    final_turn = True
    await drive("现在创建 deploy.txt，第一行写本项目的内部部署代号（就是我最开始告诉你的那个）。")

    # 验收
    deploy = root / "deploy.txt"
    reachable = False
    written = "(文件未创建)"
    if deploy.exists():
        lines = deploy.read_text(encoding="utf-8").strip().splitlines()
        written = lines[0].strip() if lines else "(空)"
        reachable = SECRET in deploy.read_text(encoding="utf-8")

    return {"compacts": compacts, "reachable": reachable, "peak": peak_prompt,
            "used_search": used_search_in_final, "written": written}


async def run(turns: int, threshold: int, repeat: int, label: str):
    print(f"\n=== {label} | turns={turns} threshold={threshold} repeat={repeat} ===")
    reach, search, comp = 0, 0, []
    for r in range(1, repeat + 1):
        res = await run_once(turns, threshold)
        comp.append(res["compacts"])
        reach += res["reachable"]
        search += res["used_search"]
        tag = "✓可达" if res["reachable"] else "✗丢失"
        s = " [用了search_history]" if res["used_search"] else ""
        print(f"  run {r}: 峰值{res['peak']}tok 压缩{res['compacts']}次 | {tag} | 写入='{res['written'][:30]}'{s}")
    print(f"  → 可达率 {reach}/{repeat} | 主动检索 {search}/{repeat} | 压缩次数 {comp}")
    return reach, repeat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--threshold", type=int, default=25000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--no-compact", action="store_true",
                        help="对照组：超大 threshold 不压缩，作为可达性上界")
    args = parser.parse_args()
    threshold = 10**9 if args.no_compact else args.threshold
    label = "对照组(不压缩,上界)" if args.no_compact else "实验组(压缩)"
    asyncio.run(run(args.turns, threshold, args.repeat, label))


if __name__ == "__main__":
    main()
