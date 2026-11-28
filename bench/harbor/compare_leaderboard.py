"""loongcli 的 Harbor 跑分 vs HF Harness Leaderboard 同模型其他外壳（todo 第 24.5 项）。

用法：python bench/harbor/compare_leaderboard.py <job 目录> [<job 目录> ...] [--replace <job 目录>] <subset.json> <榜单 trials.csv.gz>
  job 目录：harbor run -o 指定目录下的某次 job（含 <题>__<id>/ 子目录）；主跑 + 补跑可一起传入，合并统计
  --replace：该目录里的题整题替换其他目录里的同名题（配置修正后整题重跑用；可重复）
  subset.json / trials.csv.gz：见 D:/dogfood/tbench/（24.2 产出）

口径与榜单一致：基础设施异常（无 reward，或判卷没跑成）不计分，单列；通过 = reward ≥ 1。
榜单每个外壳×题都补满 3 次计分（基础设施失败重跑），不足 3 次的题会列出缺几次，按此补跑。
置信区间：按题聚类自助法（题与题难度差异大，逐次独立假设会把区间算窄）。
"""
from __future__ import annotations

import io
import json
import random
import re
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import pandas as pd


_PROVIDER_FAIL = re.compile(r"Error code: (402|429|5\d\d)|Connection|timed out|Timeout", re.I)


def _event_lines(t: Path):
    """一次尝试的事件流行：旧版适配器打成 agent/loongcli-projects.tgz，新版直接写在 agent/loongcli-projects/ 下。"""
    tgz = t / "agent" / "loongcli-projects.tgz"
    if tgz.is_file():
        with tarfile.open(tgz) as tf:
            for m in tf.getmembers():
                if m.name.endswith(".events.jsonl"):
                    yield from io.TextIOWrapper(tf.extractfile(m), encoding="utf-8")
    for f in (t / "agent" / "loongcli-projects").rglob("*.events.jsonl"):
        with f.open(encoding="utf-8") as fh:
            yield from fh


def _trial_rows(job: Path) -> list[dict]:
    rows = []
    for t in sorted(p for p in job.iterdir() if p.is_dir() and "__" in p.name):
        task = t.name.split("__")[0]
        reward_f = t / "verifier" / "reward.txt"
        reward = float(reward_f.read_text().strip()) if reward_f.is_file() and reward_f.read_text().strip() else None
        # 取最后一行含 Error 的（异常文件末行常是 "stderr: None"，看不出原因）
        exc_lines = (t / "exception.txt").read_text(errors="replace").strip().splitlines() if (t / "exception.txt").is_file() else []
        exc = [x for x in exc_lines if "Error" in x][-1:] or exc_lines[-1:]
        # 判卷没跑成（判卷脚本自己装 uv 时断网 → 没跑 pytest、没产出 ctrf.json，却照样写了 reward 0）：
        # 按榜单原则「grading could not run → infra，重跑」计为基础设施异常，而不是 agent 未过
        ctrf = t / "verifier" / "ctrf.json"
        if reward is not None and not (ctrf.is_file() and ctrf.stat().st_size > 0):
            out_f = t / "verifier" / "test-stdout.txt"
            why = [x for x in out_f.read_text(errors="replace").splitlines() if x.startswith("curl:") or "not found" in x][:1] \
                if out_f.is_file() else []
            exc, reward = [f"判卷未跑成（无 ctrf.json）：{why[0] if why else '?'}"], None
        calls = prompt = hit = out = 0
        for line in _event_lines(t):
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("type") == "llm_call":
                # 服务商故障（余额不足、限流、5xx、连不上）会直接结束本轮；旧版 loongcli 照常退出 0、照样判卷。
                # 按榜单原则「provider failure → infra，重跑」计为基础设施异常，不管判卷结果。
                # 400 之类（如上下文超长）是外壳自己的问题，照常算 agent 未过，不算进来
                if reward is not None and _PROVIDER_FAIL.search(e.get("error") or ""):
                    exc, reward = [f"模型调用失败：{e['error'][:120]}"], None
                calls += 1
                prompt += e.get("prompt_tokens", 0)
                hit += e.get("cache_hit", 0)
                out += e.get("completion_tokens", 0)
        rows.append({"task": task, "trial": t.name, "reward": reward, "exception": exc[0] if exc else "",
                     "calls": calls, "tokens": prompt + out, "cache_hit": hit, "output": out})
    return rows


def _cluster_ci(per_task: dict[str, list[int]], n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    rnd = random.Random(seed)
    tasks = list(per_task)
    stats = []
    for _ in range(n_boot):
        pick = [per_task[rnd.choice(tasks)] for _ in tasks]
        flat = [x for xs in pick for x in xs]
        stats.append(sum(flat) / len(flat))
    stats.sort()
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot)]


def main() -> int:
    args = sys.argv[1:]
    # --replace <job 目录>：该目录里出现的题，整题替换其他目录里的同名题（配置修正后整题重跑，不挑结果）
    replace_jobs = [Path(args[i + 1]) for i, a in enumerate(args) if a == "--replace"]
    args = [a for i, a in enumerate(args) if a != "--replace" and (i == 0 or args[i - 1] != "--replace")]
    if len(args) < 3:
        print(__doc__)
        return 2
    *job_args, subset_arg, lb_arg = args
    subset_path, lb_path = Path(subset_arg), Path(lb_arg)
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    tasks = subset["hard"] + subset["easy"]
    base = [r for j in job_args for r in _trial_rows(Path(j))]
    repl = [r for j in replace_jobs for r in _trial_rows(j)]
    replaced = {r["task"] for r in repl}
    if replaced:
        print(f"整题替换：{' '.join(sorted(replaced))}（丢弃其他目录里这些题的 "
              f"{sum(r['task'] in replaced for r in base)} 次尝试）")
    rows = pd.DataFrame([r for r in base if r["task"] not in replaced] + repl)
    scored = rows[rows.reward.notna()].copy()
    scored["ok"] = (scored.reward >= 1).astype(int)
    infra = rows[rows.reward.isna()]

    per_task = {t: scored[scored.task == t].ok.tolist() for t in tasks if (scored.task == t).any()}
    rate = scored.ok.mean()
    lo, hi = _cluster_ci(per_task)
    print(f"loongcli：{scored.ok.sum()}/{len(scored)} = {rate:.1%}（95% 题聚类区间 {lo:.1%}–{hi:.1%}）；"
          f"基础设施异常 {len(infra)} 次不计分")
    print(f"每次尝试：模型调用中位 {scored.calls.median():.0f} 次，token 中位 {scored.tokens.median()/1e6:.2f}M，"
          f"缓存命中率 {scored.cache_hit.sum()/max(1, (scored.tokens - scored.output).sum()):.0%}")

    lb = pd.read_csv(lb_path, low_memory=False)
    lb = lb[lb.model.str.contains("DeepSeek-V4.1-Flash") & (lb.benchmark == "terminal-bench-2.1")
            & (lb.effort == subset.get("effort", "high")) & lb.task.isin(tasks) & lb.verdict.isin(["pass", "fail"])]
    lb = lb.assign(ok=(lb.verdict == "pass").astype(int))
    board = lb.groupby("harness").agg(rate=("ok", "mean"), calls=("calls", "median"), tokens=("tokens", "median"))
    board.loc["loongcli"] = [rate, scored.calls.median(), scored.tokens.median()]
    print("\n同一子集、同模型同档位（榜单 vs loongcli）：")
    print(board.sort_values("rate", ascending=False).assign(rate=lambda d: (d.rate * 100).round(1),
                                                            tokens=lambda d: (d.tokens / 1e6).round(2)).to_string())

    lb_task = lb.groupby("task").ok.mean()
    print("\n逐题（loongcli 通过/次数 · 榜单平均通过率）：")
    for t in tasks:
        xs = per_task.get(t, [])
        flag = "  ← 明显低于榜单" if xs and sum(xs) / len(xs) + 0.34 < lb_task.get(t, 0) else ""
        print(f"  {t:<30} {sum(xs)}/{len(xs)}   榜单 {lb_task.get(t, float('nan')):.0%}{flag}")
    # 某题计分次数不足时，它在 loongcli 这边权重变小甚至消失，榜单那边仍按 20 题算——比较会偏
    want = 3
    short = {t: want - len(per_task.get(t, [])) for t in tasks if len(per_task.get(t, [])) < want}
    if short:
        print(f"\n计分不足 {want} 次的题（需补跑）：")
        for n in sorted(set(short.values())):
            print(f"  缺 {n} 次：{' '.join(t for t, m in short.items() if m == n)}")
    if len(infra):
        print("\n基础设施异常：")
        for _, r in infra.iterrows():
            print(f"  {r.trial}: {r.exception[:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
