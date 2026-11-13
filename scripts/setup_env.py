#!/usr/bin/env python3
"""一键准备 loongcli 运行环境。

为什么需要这个脚本（而不是一条 pip install）：
1. **新机器装不上**——依赖里少声明一个包，本地靠传递依赖活着，换台机器就
   ModuleNotFoundError。现在 pyproject 已修好，并且有回归测试守着。
2. **国内源不一定能用**——2026-11 实测清华源整站 403（根路径和包索引一起拒），
   而用户机器的 pip 全局配置很可能正指向它，报错还长得像"这个包不存在"。
   所以这里显式探测、显式指定 index，不看用户的全局配置。
3. **要能验证装完真能跑**——装成功 ≠ 能 import。装完立刻 import 一次，
   把"装是装了但拉起来就崩"挡在用户动手之前。

用法（在仓库根目录）：
    python scripts/setup_env.py

可选参数：
    --venv PATH        虚拟环境位置（默认 <仓库>/.venv）
    --index URL        只用指定源，跳过自动探测
    --no-venv          不建虚拟环境，直接装进当前 Python
    --dry-run          只探测源，不安装

脚本只用标准库：它要能在"什么都还没装"的机器上跑起来。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MIN_PYTHON = (3, 12)

# 按顺序探测，第一个可用的就用。
# 清华放在首位（国内传统首选、用户习惯），但它 2026-11 实测整站 403，
# 所以后面必须备着几个——清华一挂就直接落官方源的话，国内可能慢到装不完。
CANDIDATE_INDEXES = [
    ("清华 TUNA", "https://pypi.tuna.tsinghua.edu.cn/simple"),
    ("中科大 USTC", "https://mirrors.ustc.edu.cn/pypi/simple"),
    ("腾讯云", "https://mirrors.cloud.tencent.com/pypi/simple"),
    ("阿里云", "https://mirrors.aliyun.com/pypi/simple"),
    ("官方 PyPI", "https://pypi.org/simple"),
]

# 探测用这个包：够小、每个源都必然有、路径稳定。
PROBE_PACKAGE = "setuptools"

TIMEOUT = 8


def log(msg: str) -> None:
    print(msg, flush=True)


def venv_python(venv_dir: Path) -> Path:
    """虚拟环境里的解释器路径，Windows 和 POSIX 不同。"""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def probe_index(name: str, url: str) -> tuple[bool, str, float]:
    """探测一个源是否可用。返回 (可用, 说明, 耗时秒)。"""
    probe_url = f"{url.rstrip('/')}/{PROBE_PACKAGE}/"
    started = time.monotonic()
    request = urllib.request.Request(
        probe_url,
        headers={"User-Agent": "loongcli-setup/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            status = response.status
            # 只读一点点，确认真的能拿到内容，而不是一个空 200。
            body = response.read(512)
        elapsed = time.monotonic() - started
        if status != 200:
            return False, f"HTTP {status}", elapsed
        if b"<html" in body[:100].lower() and b"href" not in body.lower():
            # 有些镜像被封时会返回一个 200 的拦截页，不能当作可用。
            return False, "返回了拦截页而非包索引", elapsed
        return True, f"HTTP {status}", elapsed
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}", time.monotonic() - started
    except urllib.error.URLError as exc:
        return False, f"连接失败({exc.reason})", time.monotonic() - started
    except Exception as exc:  # 探测失败不该让脚本崩掉
        return False, f"{type(exc).__name__}: {exc}", time.monotonic() - started


def pick_index() -> str | None:
    """按候选顺序探测，返回第一个可用源的 URL。"""
    log("探测 Python 包索引：")
    for name, url in CANDIDATE_INDEXES:
        ok, detail, elapsed = probe_index(name, url)
        mark = "可用" if ok else "不可用"
        log(f"  [{mark:4}] {name:12} {elapsed:5.2f}s  {url}")
        if ok:
            log(f"\n选用：{name} ({url})\n")
            return url
    return None


def run(cmd: list[str], capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=capture,
    )


def ensure_python_version() -> bool:
    if sys.version_info >= MIN_PYTHON:
        return True
    want = ".".join(map(str, MIN_PYTHON))
    have = ".".join(map(str, sys.version_info[:3]))
    log(f"错误：需要 Python {want} 或更高，当前是 {have}。")
    log("如果机器上装了多个 Python，用对应版本重跑本脚本。")
    return False


def create_venv(venv_dir: Path) -> bool:
    if venv_python(venv_dir).exists():
        log(f"复用已有虚拟环境：{venv_dir}")
        return True
    log(f"创建虚拟环境：{venv_dir}")
    result = run([sys.executable, "-m", "venv", str(venv_dir)])
    if result.returncode != 0:
        log("错误：创建虚拟环境失败。")
        return False
    return True


def install(python: str, index: str) -> bool:
    # 显式 -i + --disable-pip-version-check：不让用户那份可能已经坏掉的全局 pip
    # 配置（比如指向一个 403 的镜像）掺和进来。
    cmd = [
        python, "-m", "pip", "install", "-e", ".",
        "--index-url", index,
        "--disable-pip-version-check",
    ]
    log(f"\n安装依赖（源：{index}）……")
    log("  " + " ".join(cmd))
    result = run(cmd)
    if result.returncode != 0:
        log("\n错误：依赖安装失败。")
        log("排查建议：")
        log("  1. 确认网络能访问该源（可加 --index 手动换源）")
        log("  2. 若报 setuptools 相关错误，多半是源不可用而非包不存在")
        log("  3. 可重跑本脚本，它会重新探测源")
        return False
    return True


def verify(python: str) -> bool:
    """装完必须验证能真的 import——装成功不等于能跑。"""
    log("\n验证：")
    checks = [
        ("import httpx", [python, "-c", "import httpx"]),
        ("import mcp 的 streamablehttp_client", [
            python, "-c",
            "from mcp.client.streamable_http import streamablehttp_client",
        ]),
        ("import loongcli.main", [python, "-c", "import loongcli.main"]),
    ]
    all_ok = True
    for label, cmd in checks:
        result = run(cmd, capture=True)
        if result.returncode == 0:
            log(f"  [通过] {label}")
        else:
            all_ok = False
            log(f"  [失败] {label}")
            err = (result.stderr or "").strip().splitlines()
            if err:
                log(f"         {err[-1]}")
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="一键准备 loongcli 运行环境")
    parser.add_argument("--venv", default=None, help="虚拟环境路径（默认 <仓库>/.venv）")
    parser.add_argument("--index", default=None, help="只用指定源，跳过自动探测")
    parser.add_argument("--no-venv", action="store_true", help="不建虚拟环境")
    parser.add_argument("--dry-run", action="store_true", help="只探测源，不安装")
    args = parser.parse_args()

    if not (REPO_ROOT / "pyproject.toml").exists():
        log(f"错误：{REPO_ROOT} 下没有 pyproject.toml，请在本仓库根目录运行。")
        return 2

    if not ensure_python_version():
        return 2

    if args.index:
        index = args.index
        log(f"使用指定源：{index}\n")
    else:
        index = pick_index()
        if not index:
            log("错误：所有候选源都不可用。")
            log("可手动指定一个能用的源重跑：--index https://<你的镜像>/simple")
            return 1

    if args.dry_run:
        log("--dry-run：只探测，不安装。")
        return 0

    venv_dir = Path(args.venv).expanduser() if args.venv else REPO_ROOT / ".venv"
    if not args.no_venv:
        if not create_venv(venv_dir):
            return 1
        python = str(venv_python(venv_dir))
    else:
        python = sys.executable

    if not install(python, index):
        return 1

    if not verify(python):
        log("\n环境装上了但验证未通过，请把上面的报错贴给维护者。")
        return 1

    log("\n完成。启动方式：")
    if args.no_venv:
        log("  loongcli")
    elif os.name == "nt":
        log(f"  {venv_dir}\\Scripts\\loongcli.exe")
    else:
        log(f"  {venv_dir}/bin/loongcli")
    return 0


if __name__ == "__main__":
    sys.exit(main())
