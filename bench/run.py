"""loong-bench 跑分器。

每个任务的流程：
1. 在 bench/worktrees/<id> 创建指向 base commit 的独立 git worktree
2. 应用「仅测试」patch 并提交为基线（之后可用 git 检测 agent 是否改了测试）
3. fail-to-pass 预检：目标测试必须失败，否则任务标记 invalid
4. 用临时 HOME 运行 loongcli 非交互模式（隔离记忆/会话，防止真实记忆泄题）
5. 评分：目标测试通过 = resolved；改动了 tests/ = cheated（判负）
6. 记录 成功率/步数/token/成本/耗时 到 bench/results/

用法：
    python bench/run.py [--profile NAME] [--only id1,id2] [--keep-worktrees]
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

BENCH_DIR = Path(__file__).parent
REPO = BENCH_DIR.parent
PATCHES_DIR = BENCH_DIR / "patches"
WORKTREES_DIR = BENCH_DIR / "worktrees"
RESULTS_DIR = BENCH_DIR / "results"
PYTHON = sys.executable

AGENT_TIMEOUT = 900
TEST_TIMEOUT = 300


def run_cmd(cmd: list[str], cwd: Path, timeout: int, env: dict | None = None):
    return subprocess.run(
        cmd, cwd=str(cwd), timeout=timeout, env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        stdin=subprocess.DEVNULL,
    )


def git(*args: str, cwd: Path = REPO):
    result = run_cmd(["git", *args], cwd=cwd, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def make_isolated_home() -> Path:
    """临时 HOME：复制真实 config.json（保留 API key 和 profiles），
    不带记忆/会话/MCP，保证评测不被真实记忆污染、也不污染真实数据。"""
    home = Path(tempfile.mkdtemp(prefix="loongbench-home-"))
    cfg_dir = home / ".loongcli"
    cfg_dir.mkdir(parents=True)
    real_cfg = Path.home() / ".loongcli" / "config.json"
    if real_cfg.exists():
        cfg = json.loads(real_cfg.read_text(encoding="utf-8"))
        cfg.pop("mcp_servers", None)  # MCP 与评测无关，避免连接开销
        cfg.pop("mcpServers", None)
        (cfg_dir / "config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return home


def agent_env(home: Path) -> dict:
    import os
    env = dict(os.environ)
    env["USERPROFILE"] = str(home)
    env["HOME"] = str(home)
    return env


def setup_worktree(task: dict) -> Path:
    wt = WORKTREES_DIR / task["id"]
    if wt.exists():
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)],
                       cwd=str(REPO), capture_output=True)
        shutil.rmtree(wt, ignore_errors=True)
    WORKTREES_DIR.mkdir(parents=True, exist_ok=True)
    git("worktree", "add", "--detach", str(wt), task["base"])
    git("apply", str((PATCHES_DIR / f"{task['fix'][:10]}.patch").resolve()), cwd=wt)
    git("add", "-A", cwd=wt)
    git("-c", "user.name=bench", "-c", "user.email=bench@local",
        "commit", "-m", "bench baseline: base + test patch", cwd=wt)
    return wt


def run_tests(task: dict, wt: Path) -> bool:
    """返回目标测试是否通过。"""
    argv = shlex.split(task["test_cmd"])
    argv[0] = PYTHON  # "python" → 当前 venv 解释器；cwd=worktree 保证导入 worktree 代码
    try:
        result = run_cmd(argv, cwd=wt, timeout=TEST_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def run_agent(task: dict, wt: Path, profile: str | None, home: Path) -> dict:
    # -P (safe path)：把 cwd 从 sys.path 移除，agent 进程经 editable finder 导入
    # 主仓库的「当前」harness（而不是 worktree 里被测的旧代码）。该标志不传染
    # 子进程——agent 内部运行 pytest 时 sys.path[0]=cwd 仍指向 worktree，测的是任务代码。
    cmd = [PYTHON, "-P", "-m", "loongcli", task["prompt"],
           "--output-format", "json", "--verbose", "--dangerously-skip-permissions"]
    if profile:
        cmd += ["--profile", profile]

    start = time.time()
    try:
        result = run_cmd(cmd, cwd=wt, timeout=AGENT_TIMEOUT, env=agent_env(home))
        duration = time.time() - start
    except subprocess.TimeoutExpired:
        return {"error": "agent timeout", "duration": AGENT_TIMEOUT}

    out = {"duration": round(duration, 1), "exit_code": result.returncode}
    try:
        payload = json.loads(result.stdout)
        out["steps"] = len(payload.get("tool_calls", []))
        out["usage"] = payload.get("usage")
        out["cost"] = payload.get("cost")
    except json.JSONDecodeError:
        out["error"] = f"无法解析 agent 输出: {result.stdout[:200]} | stderr: {result.stderr[-300:]}"
    return out


def tests_modified(wt: Path) -> bool:
    """agent 是否动了 tests/（相对基线 commit，含未跟踪新文件）。"""
    out = run_cmd(["git", "status", "--porcelain", "--", "tests/"], cwd=wt, timeout=60)
    return bool(out.stdout.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None, help="loongcli model profile")
    parser.add_argument("--tasks", default=str(BENCH_DIR / "tasks.jsonl"))
    parser.add_argument("--only", default=None, help="逗号分隔的任务 id 子集")
    parser.add_argument("--keep-worktrees", action="store_true")
    args = parser.parse_args()

    tasks = [json.loads(line) for line in Path(args.tasks).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.only:
        wanted = set(args.only.split(","))
        tasks = [t for t in tasks if t["id"] in wanted]

    RESULTS_DIR.mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    label = args.profile or "default"
    results_path = RESULTS_DIR / f"run-{run_id}-{label}.jsonl"

    home = make_isolated_home()
    print(f"loong-bench: {len(tasks)} 题 | profile={label} | 隔离 HOME={home}")

    resolved_count = 0
    for i, task in enumerate(tasks, 1):
        row = {"task_id": task["id"], "profile": label, "difficulty": task.get("difficulty", "?")}
        print(f"\n[{i}/{len(tasks)}] {task['id']} ({row['difficulty']}) ...", flush=True)
        try:
            wt = setup_worktree(task)

            if run_tests(task, wt):
                row["status"] = "invalid"
                print("  ✗ 预检失败：基线上目标测试已通过，任务无效")
            else:
                agent_out = run_agent(task, wt, args.profile, home)
                row.update(agent_out)
                if "error" in agent_out:
                    row["status"] = "error"
                    print(f"  ✗ agent 错误: {agent_out['error'][:120]}")
                elif tests_modified(wt):
                    row["status"] = "cheated"
                    print("  ✗ 判负：agent 修改了 tests/")
                elif run_tests(task, wt):
                    row["status"] = "resolved"
                    resolved_count += 1
                    print(f"  ✓ resolved | {row.get('steps', '?')} 步 | {row.get('duration', '?')}s")
                else:
                    row["status"] = "failed"
                    print(f"  ✗ 未解决 | {row.get('steps', '?')} 步 | {row.get('duration', '?')}s")

            if not args.keep_worktrees:
                subprocess.run(["git", "worktree", "remove", "--force", str(wt)],
                               cwd=str(REPO), capture_output=True)
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)[:300]
            print(f"  ✗ 异常: {exc}")

        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    shutil.rmtree(home, ignore_errors=True)
    scored = [t for t in tasks]
    print(f"\n完成: {resolved_count}/{len(scored)} resolved | 结果: {results_path}")


if __name__ == "__main__":
    main()
