"""shell 权限闸的真实命令回归集：从历史会话收集真实 shell 命令，记下每条的权限判定，改代码后逐条对比。

为什么要它：permissions.py 两个能绕过确认的漏洞（白名单按字符串前缀匹配；拆命令不认 & 与换行）都是拿真实命令
实测才暴露的，手写的单测想不到这些写法。数据放 tasks/eval/（私有目录，push-master 双闸剔除——命令里有 ssh 主机、
IP、本机路径），永远不进 GitHub。

用法：
    python scripts/shell_corpus.py harvest   # 扫会话，新命令按当前判定打标签；已有条目的标签不动
    python scripts/shell_corpus.py check     # 当前判定与记录的期望逐条对比；有差异退出码 1
    python scripts/shell_corpus.py accept    # 人工确认差异合理后，把当前判定写成新的期望
    python scripts/shell_corpus.py audit     # 列出「免确认」且含写操作字样、尚未审查的命令，供人工逐条看

字段：cmd 原文；source loongcli / claude-code；default DEFAULT 模式判定（ALLOW/CONFIRM/DENY）；
read_only 规划阶段是否放行；reviewed 免确认的条目是否已人工确认只读；note 审查备注。
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loongcli.memory.conversation import _projects_root  # noqa: E402
from loongcli.security.permissions import PermissionChecker, PermissionMode  # noqa: E402

CORPUS = ROOT / "tasks" / "eval" / "shell_corpus.jsonl"
# 免确认的命令里出现这些字样就值得人工看一眼（多数只是 echo/grep 的参数，但漏洞就藏在这里）
_SUSPICIOUS = re.compile(
    r"\b(rm|mv|cp|mkdir|touch|tee|install|uninstall|del|kill|chmod|chown|curl|wget|python|node|npm|"
    r"checkout|commit|push|reset|merge|rebase|clean|apply|stash|Remove-Item|Set-Content|Out-File|New-Item)\b"
    r"|>|-i\b|--output|-o\b",
    re.IGNORECASE,
)


def _checker() -> PermissionChecker:
    return PermissionChecker(PermissionMode.DEFAULT)


def judge(checker: PermissionChecker, cmd: str) -> dict:
    return {
        "default": checker.check_tool("shell", {"command": cmd})[0].name,
        "read_only": checker.is_read_only_command(cmd),
    }


def load() -> list[dict]:
    if not CORPUS.exists():
        return []
    return [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(entries: list[dict]) -> None:
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    CORPUS.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8")


def _session_commands() -> dict[str, str]:
    """历史会话（主会话 + 子代理 trace）里的全部 shell 命令 → 来源。loongcli 的 shell 与 Claude Code 迁移会话的 Bash 都收。"""
    root = str(_projects_root())
    found: dict[str, str] = {}
    for f in sorted(glob.glob(root + "/*/sessions/*.json") + glob.glob(root + "/*/tasks/*.json")):
        try:
            data = json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for m in (data.get("messages") if isinstance(data, dict) else None) or []:
            for tc in m.get("tool_calls") or []:
                name = tc.get("function", {}).get("name")
                if name not in ("shell", "Bash"):
                    continue
                try:
                    cmd = json.loads(tc["function"].get("arguments") or "{}").get("command", "")
                except (ValueError, TypeError):
                    continue
                if cmd:
                    found.setdefault(cmd, "loongcli" if name == "shell" else "claude-code")
    return found


def harvest() -> int:
    entries = load()
    known = {e["cmd"] for e in entries}
    checker = _checker()
    added = 0
    for cmd, source in _session_commands().items():
        if cmd in known:
            continue
        entries.append({"cmd": cmd, "source": source, **judge(checker, cmd), "reviewed": False, "note": ""})
        added += 1
    save(entries)
    print(f"新增 {added} 条，共 {len(entries)} 条 → {CORPUS}")
    return 0


def diff(entries: list[dict]) -> tuple[list, list]:
    """返回（新放行，新拦下）：新放行 = 原本要确认/不算只读、现在免确认/算只读——危险方向。"""
    checker = _checker()
    loosened, tightened = [], []
    rank = {"DENY": 0, "CONFIRM": 1, "ALLOW": 2}
    for e in entries:
        now = judge(checker, e["cmd"])
        if now == {"default": e["default"], "read_only": e["read_only"]}:
            continue
        looser = rank[now["default"]] > rank[e["default"]] or (now["read_only"] and not e["read_only"])
        (loosened if looser else tightened).append((e, now))
    return loosened, tightened


def check() -> int:
    entries = load()
    if not entries:
        print(f"没有数据：{CORPUS}（先跑 harvest）")
        return 0
    loosened, tightened = diff(entries)
    for title, rows in (("新放行（危险方向，逐条确认）", loosened), ("新拦下（摩擦方向）", tightened)):
        if rows:
            print(f"== {title}：{len(rows)} 条")
            for e, now in rows[:50]:
                print(f"  {e['default']}/{e['read_only']} → {now['default']}/{now['read_only']} | {e['cmd'][:150]!r}")
    print(f"共 {len(entries)} 条，差异 {len(loosened) + len(tightened)} 条")
    return 1 if (loosened or tightened) else 0


def accept() -> int:
    entries = load()
    checker = _checker()
    changed = 0
    for e in entries:
        now = judge(checker, e["cmd"])
        if now != {"default": e["default"], "read_only": e["read_only"]}:
            e.update(now)
            e["reviewed"] = False  # 判定变了，旧的审查结论作废
            changed += 1
    save(entries)
    print(f"更新期望 {changed} 条")
    return 0


def audit() -> int:
    rows = [e for e in load() if e["default"] == "ALLOW" and not e["reviewed"] and _SUSPICIOUS.search(e["cmd"])]
    for i, e in enumerate(rows):
        print(f"[{i}] {e['cmd'][:300]!r}")
    print(f"待审查 {len(rows)} 条（免确认且含写操作字样、未审查）")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    actions = {"harvest": harvest, "check": check, "accept": accept, "audit": audit}
    if len(sys.argv) != 2 or sys.argv[1] not in actions:
        print(__doc__)
        sys.exit(2)
    sys.exit(actions[sys.argv[1]]())
