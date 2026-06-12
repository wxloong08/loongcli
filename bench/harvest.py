"""从 git 历史挖掘评测候选任务（SWE-bench 方法论）。

候选标准：一个 commit 同时修改了 loongcli/ 下的代码和 tests/ 下的测试。
对每个候选提取「仅测试」的 patch——评测时基线 = 父 commit + 测试 patch，
目标测试此时应当失败（fail-to-pass），agent 的任务是改代码让它通过。

用法：
    python bench/harvest.py [--branch dev] [--limit 200]

输出：
    bench/patches/<sha>.patch   每个候选的测试 patch
    bench/candidates.jsonl      候选清单（人工筛选后写 prompt 进 tasks.jsonl）
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

BENCH_DIR = Path(__file__).parent
REPO = BENCH_DIR.parent
PATCHES_DIR = BENCH_DIR / "patches"


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def changed_files(sha: str) -> list[str]:
    out = git("diff-tree", "--no-commit-id", "--name-only", "-r", sha)
    return [line for line in out.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default="dev")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    PATCHES_DIR.mkdir(exist_ok=True)

    # %x09 由 git 展开为 tab，避免在 Windows 命令行参数里携带特殊字符
    log = git("log", args.branch, f"--max-count={args.limit}",
              "--no-merges", "--format=%H%x09%s")
    candidates = []

    for line in log.splitlines():
        if "\t" not in line:
            continue
        sha, subject = line.split("\t", 1)
        files = changed_files(sha)
        code_files = [f for f in files if f.startswith("loongcli/") and f.endswith(".py")]
        test_files = [f for f in files if f.startswith("tests/") and f.endswith(".py")]
        if not code_files or not test_files:
            continue

        # 必须有父 commit 才能构造基线
        try:
            parent = git("rev-parse", f"{sha}^").strip()
        except RuntimeError:
            continue

        test_patch = git("diff", parent, sha, "--", "tests/")
        if not test_patch.strip():
            continue

        short = sha[:10]
        (PATCHES_DIR / f"{short}.patch").write_text(test_patch, encoding="utf-8")

        code_diff_stat = git("diff", "--stat", parent, sha, "--", "loongcli/").strip()
        candidates.append({
            "id": short,
            "sha": sha,
            "parent": parent,
            "subject": subject,
            "code_files": code_files,
            "test_files": test_files,
            "code_diff_lines": code_diff_stat.splitlines()[-1] if code_diff_stat else "",
        })

    out_path = BENCH_DIR / "candidates.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"找到 {len(candidates)} 个候选（代码+测试同时变更），已写入 {out_path}")
    for c in candidates:
        print(f"  {c['id']}  {c['subject'][:70]}")
        print(f"             代码: {len(c['code_files'])} 文件 | {c['code_diff_lines']}")


if __name__ == "__main__":
    main()
