"""汇总 bench/results/*.jsonl 生成对比报告（按 profile 分组）。

用法：python bench/report.py [--out bench/REPORT.md]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

BENCH_DIR = Path(__file__).parent
RESULTS_DIR = BENCH_DIR / "results"


def total_tokens(usage: dict | None) -> int | None:
    if not isinstance(usage, dict):
        return None
    if "total_tokens" in usage:
        return usage["total_tokens"]
    known = [v for k, v in usage.items() if isinstance(v, (int, float)) and "token" in k]
    return int(sum(known)) if known else None


def total_cost(cost: dict | None) -> float | None:
    if not isinstance(cost, dict):
        return None
    for key in ("total", "total_cost"):
        if isinstance(cost.get(key), (int, float)):
            return float(cost[key])
    nums = [v for v in cost.values() if isinstance(v, (int, float))]
    return sum(nums) if nums else None


def fmt(value, pattern="{:.1f}", missing="—"):
    return pattern.format(value) if value is not None else missing


def mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(BENCH_DIR / "REPORT.md"))
    args = parser.parse_args()

    rows = []
    for f in sorted(RESULTS_DIR.glob("run-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["_run"] = f.stem
                rows.append(row)

    if not rows:
        print("没有结果文件")
        return

    # 同一 profile 多次运行时取每个任务的最新一条
    latest: dict[tuple, dict] = {}
    for row in rows:
        latest[(row["profile"], row["task_id"], row["_run"])] = row

    by_profile = defaultdict(list)
    for row in latest.values():
        by_profile[row["profile"]].append(row)

    lines = ["# loong-bench 报告", ""]
    lines.append("| profile | resolved | 成功率 | 平均步数 | 平均tokens | 平均成本 | 平均耗时(s) |")
    lines.append("|---|---|---|---|---|---|---|")
    for profile, items in sorted(by_profile.items()):
        scored = [r for r in items if r["status"] in ("resolved", "failed", "cheated", "error")]
        solved = [r for r in scored if r["status"] == "resolved"]
        rate = len(solved) / len(scored) * 100 if scored else 0
        lines.append(
            f"| {profile} | {len(solved)}/{len(scored)} | {rate:.0f}% "
            f"| {fmt(mean([r.get('steps') for r in solved]))} "
            f"| {fmt(mean([total_tokens(r.get('usage')) for r in solved]), '{:.0f}')} "
            f"| {fmt(mean([total_cost(r.get('cost')) for r in solved]), '{:.4f}')} "
            f"| {fmt(mean([r.get('duration') for r in solved]))} |"
        )

    lines.append("")
    lines.append("## 任务明细")
    lines.append("")
    lines.append("| task | difficulty | " + " | ".join(sorted(by_profile)) + " |")
    lines.append("|---|---|" + "---|" * len(by_profile))
    task_ids = sorted({r["task_id"] for r in latest.values()})
    icon = {"resolved": "✓", "failed": "✗", "cheated": "⚠改测试", "error": "⚠错误",
            "invalid": "无效", None: "—"}
    for tid in task_ids:
        cells = []
        diff = "?"
        for profile in sorted(by_profile):
            match = [r for r in by_profile[profile] if r["task_id"] == tid]
            if match:
                diff = match[-1].get("difficulty", diff)
                status = match[-1]["status"]
                steps = match[-1].get("steps")
                cells.append(f"{icon.get(status, status)}({steps}步)" if steps is not None else icon.get(status, status))
            else:
                cells.append("—")
        lines.append(f"| {tid} | {diff} | " + " | ".join(cells) + " |")

    report = "\n".join(lines) + "\n"
    Path(args.out).write_text(report, encoding="utf-8")
    print(report)
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
