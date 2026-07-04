"""Empirical probe: how good is automatic RecallEngine against the REAL store?

RecallEngine (agent.py pre-turn) does an LLM sideQuery: match user_input to the
one-line descriptions of all memories, pick <=5. This probe runs it against the
actual ~/.loongcli/memory store with queries that have known ground truth, and
reports hit / miss / spurious so we can see the real (not theorized) behavior.

Run:  python bench/recall_probe.py
"""
from __future__ import annotations
import asyncio
from pathlib import Path

from loongcli.core.config import Config
from loongcli.core.llm import LLMClient
from loongcli.memory.markdown_store import MarkdownMemoryStore
from loongcli.memory.recall_engine import RecallEngine

# (query, expected memory names that SHOULD surface, label)
CASES: list[tuple[str, set[str], str]] = [
    ("tracker.py 跑出来报错 invalid quote 是怎么回事",
     {"project-jobhunter-tracker-json-invalid-quote-error"},
     "直接词面命中"),
    ("我决定放弃一个职位，怎么保证以后不会又把它捞出来重复推进",
     {"project-negative-decision-persistence"},
     "语义·低词面重叠"),
    ("面试官让我讲讲什么是 agent harness，我该突出哪些点",
     {"user-expertise-harness-engineering", "user-q2-terminology-not-stages"},
     "用户画像·释义"),
    ("有没有工具能统计我每天用各个软件花了多少时间",
     {"reference-tai-windows-time-tracker"},
     "释义·中等重叠"),
    ("帮我写一个快速排序函数",
     set(),
     "无关·假阳性测试"),
    ("嗯好的，那就这样，继续吧",
     set(),
     "闲聊延续·噪声测试"),
]


async def _run_case(engine: RecallEngine, query: str) -> list[str]:
    mems = await engine.recall(query)
    return [m["name"] for m in mems]


async def main() -> None:
    cfg = Config.load()
    if not cfg.api_key:
        print("no api_key in ~/.loongcli/config.json"); return

    store = MarkdownMemoryStore(base_dir=Path.home() / ".loongcli" / "memory")
    total = len(store.list_all())
    llm = LLMClient(api_key=cfg.api_key, model="deepseek-v4-flash",
                    base_url=cfg.base_url, thinking=False)
    engine = RecallEngine(memory=store, llm=llm)

    print(f"store: {total} memories  model: deepseek-v4-flash (utility)\n")
    results = await asyncio.gather(*[_run_case(engine, q) for q, _, _ in CASES])

    hits = misses = spurious_cases = 0
    for (query, expected, label), got in zip(CASES, results):
        got_set = set(got)
        recalled = expected & got_set
        missed = expected - got_set
        extra = got_set - expected
        ok = (not missed) and (not extra if not expected else True)
        if expected:
            hits += len(recalled); misses += len(missed)
        if extra and not expected:
            spurious_cases += 1
        mark = "✓" if (expected and not missed) or (not expected and not extra) else "✗"
        print(f"[{mark}] {label}")
        print(f"     q: {query}")
        print(f"     want: {sorted(expected) or '(none)'}")
        print(f"     got : {sorted(got_set) or '(none)'}")
        if missed:
            print(f"     MISS: {sorted(missed)}")
        if extra:
            tag = "SPURIOUS" if not expected else "extra"
            print(f"     {tag}: {sorted(extra)}")
        print()

    print(f"summary: ground-truth recalled {hits} / {hits + misses}; "
          f"spurious injection on {spurious_cases} / "
          f"{sum(1 for _, e, _ in CASES if not e)} no-match queries")


if __name__ == "__main__":
    asyncio.run(main())
