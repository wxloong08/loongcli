"""skill 执行能力探针：测 loongcli 能否按多阶段 skill 定义正确执行全流程。

填补 stress_compact 的盲区——那个测「压缩下单一显眼硬规则的保持」，这个测
**正常执行下的多阶段流程正确性 + 状态传递**：一个 4 阶段流水线，每阶段依赖
上一阶段的输出，oracle 客观（链式 key 传递 + 每阶段校验码 + 阶段顺序）。

关键取舍：
- 不压缩（compactor 阈值设超大），变量干净，测的是 skill 执行 harness 本身。
- 任务是简单字符串拼接，模型轻松能做对——所以失败只可能来自「没按 skill 多阶段
  流程走」（跳阶段/漏传状态/忘校验码），而不是模型算不出来。这样才测的是 B 不是 A。
- 多次重复（agent 行为方差大，见 lessons「n=1 不能下结论」）。

用法：python bench/skill_exec_probe.py [--repeat 3]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

MARKER = "PIPE-5521"          # 校验码，只在 SKILL.md 出现
INPUT_KEY = "zephyr"          # input.txt 里埋的关键词
EXPECTED_FINAL_KEY = "zephyr-processed-verified"  # 链式传递的正确终值

SKILL_MD = f"""\
---
name: pipeline
description: 四阶段数据处理流水线
triggers: [zz-never-spoken-trigger]
---

# pipeline 技能

你在执行一个四阶段流水线。**必须严格按阶段顺序 1→2→3→4 执行，每阶段依赖上一阶段的输出文件，不许跳阶段、不许并行。**

通用规则：每个阶段创建的文件，**第一行必须是** `STAGE-<n> {MARKER}`（n 是阶段号，{MARKER} 是本流水线校验码，一字不差）。校验码只在本技能出现，任务指令里不会重复。

- **阶段 1**：读 `input.txt`，取出其中 `KEY=` 后面的关键词，创建 `stage1.txt`，第二行写 `key=<关键词>`。
- **阶段 2**：读 `stage1.txt` 拿到 key，创建 `stage2.txt`，第二行写 `key=<上一阶段的key>-processed`。
- **阶段 3**：读 `stage2.txt` 拿到 key，创建 `stage3.txt`，第二行写 `key=<上一阶段的key>-verified`。
- **阶段 4**：读 `stage3.txt` 拿到 key，创建 `final.txt`，第二行写 `key=<上一阶段的key>`，第三行写 `pipeline complete`。
"""


def load_llm():
    from loongcli.core.llm import LLMClient
    cfg = json.loads((Path.home() / ".loongcli" / "config.json").read_text(encoding="utf-8"))
    return LLMClient(api_key=cfg["api_key"], base_url=cfg["base_url"], model=cfg["model"])


def check(root: Path) -> dict:
    """客观 oracle：阶段文件齐全 + 校验码 + 阶段顺序标记 + 链式 key + 完成标记。"""
    def first_two(name: str):
        p = root / name
        if not p.exists():
            return None, None
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        return (lines[0].strip() if lines else ""), (lines[1].strip() if len(lines) > 1 else "")

    stages_present = all((root / f"stage{i}.txt").exists() for i in (1, 2, 3)) and (root / "final.txt").exists()
    # 每阶段第一行校验码 + 阶段号
    marker_ok = True
    for i, name in [(1, "stage1.txt"), (2, "stage2.txt"), (3, "stage3.txt"), (4, "final.txt")]:
        head, _ = first_two(name)
        if head != f"STAGE-{i} {MARKER}":
            marker_ok = False
    # 链式 key 传递是否正确
    _, fkey = first_two("final.txt")
    chain_ok = fkey == f"key={EXPECTED_FINAL_KEY}"
    # 完成标记
    final = (root / "final.txt")
    complete_ok = final.exists() and "pipeline complete" in final.read_text(encoding="utf-8")

    overall = stages_present and marker_ok and chain_ok and complete_ok
    return {"stages": stages_present, "marker": marker_ok, "chain": chain_ok,
            "complete": complete_ok, "overall": overall, "final_key": fkey}


async def run_once() -> dict:
    import os
    from loongcli.core.agent import AgentLoop
    from loongcli.core.compact import Compactor
    from loongcli.core.events import ConfirmRequest, TextDelta
    from loongcli.memory.conversation import ConversationStore
    from loongcli.security.permissions import PermissionChecker, PermissionMode
    from loongcli.skills.registry import SkillRegistry
    from loongcli.tools.base import ToolRegistry
    from loongcli.tools.read_file import ReadFileTool
    from loongcli.tools.skill import SkillTool
    from loongcli.tools.write_file import WriteFileTool

    root = Path(tempfile.mkdtemp(prefix="loong-skill-"))
    skills_dir = root / ".loongcli" / "skills" / "pipeline"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    (root / "input.txt").write_text(f"配置项：KEY={INPUT_KEY}\n其它无关内容。\n", encoding="utf-8")
    os.chdir(root)

    llm = load_llm()
    skill_registry = SkillRegistry(project_dir=root, personal_dir=root / "no-personal")
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(SkillTool(skill_registry))

    # 不压缩：阈值设超大，测的是正常执行下的多阶段流程
    compactor = Compactor(llm=llm, threshold=10**9, skill_registry=skill_registry)
    conversation = ConversationStore(base_dir=root / "sessions")
    agent = AgentLoop(
        llm=llm,
        tool_registry=registry,
        permission_checker=PermissionChecker(mode=PermissionMode.SKIP),
        system_prompt="你是一个执行文件任务的 agent。严格遵守已加载技能中的全部规则与阶段顺序。",
        conversation_store=conversation,
        compactor=compactor,
        skill_registry=skill_registry,
    )

    steps = 0
    async for event in agent.run_stream(
        "调用 skill 工具加载 'pipeline' 技能，仔细阅读它的阶段定义，"
        "然后严格按它的四阶段流程处理 input.txt，完成全部四个阶段。"
    ):
        from loongcli.core.events import ToolCallStart
        if isinstance(event, ToolCallStart):
            steps += 1
        elif isinstance(event, ConfirmRequest):
            event.future.set_result(True)
        elif isinstance(event, TextDelta):
            pass

    res = check(root)
    res["steps"] = steps
    return res


async def run(repeat: int):
    print(f"=== skill 执行探针 | 4 阶段流水线 | repeat={repeat} ===")
    ok = 0
    for r in range(1, repeat + 1):
        res = await run_once()
        tag = "✓ 全对" if res["overall"] else "✗"
        detail = "" if res["overall"] else (
            f" [阶段齐={res['stages']} 校验码={res['marker']} "
            f"链式={res['chain']} 完成={res['complete']} final_key={res['final_key']}]"
        )
        print(f"  run {r}: {res['steps']} 步 | {tag}{detail}")
        ok += res["overall"]
    print(f"  → 多阶段全对率 {ok}/{repeat}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(run(args.repeat))


if __name__ == "__main__":
    main()
