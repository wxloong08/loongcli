"""真实命令回归集：历史会话里的 5000+ 条 shell 命令，权限判定不得悄悄变化。

数据在 tasks/eval/shell_corpus.jsonl（私有目录，不进 master / GitHub）——不存在时整体跳过。
判定变了：先看 `python scripts/shell_corpus.py check` 的报告，「新放行」逐条确认确实只读，
再 `python scripts/shell_corpus.py accept` 更新期望。
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "tasks" / "eval" / "shell_corpus.jsonl"

pytestmark = pytest.mark.skipif(not CORPUS.exists(), reason="真实命令语料是私有数据，只在 dev 工作区存在")


def _corpus_module():
    spec = importlib.util.spec_from_file_location("shell_corpus", ROOT / "scripts" / "shell_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_verdict_changes_on_real_commands():
    mod = _corpus_module()
    loosened, tightened = mod.diff(mod.load())
    assert not loosened, "新放行（危险方向）：\n" + "\n".join(e["cmd"][:200] for e, _ in loosened[:20])
    assert not tightened, "新拦下（摩擦方向）：\n" + "\n".join(e["cmd"][:200] for e, _ in tightened[:20])


def test_every_auto_allowed_command_was_reviewed():
    """免确认的命令每一条都要经人工确认只读——漏洞就藏在「免确认」里。"""
    mod = _corpus_module()
    unreviewed = [e["cmd"] for e in mod.load() if e["default"] == "ALLOW" and not e["reviewed"]]
    assert not unreviewed, "未审查的免确认命令：\n" + "\n".join(c[:200] for c in unreviewed[:20])
