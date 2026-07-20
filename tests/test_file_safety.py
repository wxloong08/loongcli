"""文件安全四件套的单测：写前语法校验、读后写硬闸、checkpoint 会话级留存。

背景见 tasks/todo.md（2026-07-16 批）：事故根因是「凭记忆重写整文件 +
写前无备份」，防御必须是确定性代码闸门，不靠 LLM 自觉。
"""
import pytest
from pathlib import Path

from loongcli.tools.syntax_check import check_syntax
from loongcli.tools.write_file import WriteFileTool
from loongcli.tools.edit_file import EditFileTool
from loongcli.tools.base import ToolRegistry
from loongcli.core.agent import AgentLoop
import loongcli.core.checkpoint as checkpoint_mod
from loongcli.core.checkpoint import CheckpointManager


# ---------- 写前语法校验 ----------

class TestCheckSyntax:
    def test_valid_json(self):
        assert check_syntax("a.json", '{"k": 1}') is None

    def test_invalid_json(self):
        err = check_syntax("a.json", '{"k": 1,}')
        assert err and "JSON" in err

    def test_jsonc_conventions_skipped(self):
        # 带注释的 JSONC 是合法用法，严格解析会误杀
        assert check_syntax("tsconfig.json", '// c\n{"k": 1,}') is None
        assert check_syntax(r"proj\.vscode\settings.json", '// c\n{}') is None
        assert check_syntax("a.jsonc", '// c\n{}') is None

    def test_python(self):
        assert check_syntax("a.py", "x = 1\n") is None
        err = check_syntax("a.py", "def f(:\n")
        assert err and "Python" in err

    def test_toml_and_yaml(self):
        assert check_syntax("a.toml", 'k = "v"') is None
        assert check_syntax("a.toml", "k = = 1") is not None
        assert check_syntax("a.yaml", "k: v\n") is None
        assert check_syntax("a.yaml", "k: [1, 2\n") is not None
        # 多文档 YAML（--- 分隔）合法
        assert check_syntax("a.yml", "---\na: 1\n---\nb: 2\n") is None

    def test_unknown_extensions_pass(self):
        assert check_syntax("a.md", "{broken json —无所谓") is None
        assert check_syntax("a.txt", "def f(:") is None


class TestWriteFileValidation:
    async def test_invalid_json_rejected_not_created(self, tmp_path):
        target = tmp_path / "cfg.json"
        result = await WriteFileTool().execute(path=str(target), content='{"k": 1,}')
        assert "拒绝写入" in result
        assert not target.exists()

    async def test_invalid_overwrite_rejected_unchanged(self, tmp_path):
        target = tmp_path / "cfg.json"
        target.write_text('{"ok": true}', encoding="utf-8")
        result = await WriteFileTool().execute(path=str(target), content="{bad")
        assert "拒绝写入" in result
        assert target.read_text(encoding="utf-8") == '{"ok": true}'

    async def test_valid_json_written(self, tmp_path):
        target = tmp_path / "cfg.json"
        result = await WriteFileTool().execute(path=str(target), content='{"k": 1}')
        assert "成功" in result
        assert target.exists()


class TestEditFileValidation:
    async def test_breaking_valid_py_rejected(self, tmp_path):
        target = tmp_path / "m.py"
        target.write_text("x = 1\n", encoding="utf-8")
        result = await EditFileTool().execute(
            path=str(target), old_string="x = 1", new_string="x = (",
        )
        assert "拒绝写入" in result
        assert target.read_text(encoding="utf-8") == "x = 1\n"

    async def test_repairing_broken_json_allowed(self, tmp_path):
        # 原内容本就非法 → 放行任意编辑，避免锁死修复路径
        target = tmp_path / "b.json"
        target.write_text('{"k": 1,}', encoding="utf-8")
        result = await EditFileTool().execute(
            path=str(target), old_string='1,}', new_string='1}',
        )
        assert "成功" in result
        assert target.read_text(encoding="utf-8") == '{"k": 1}'

    async def test_valid_to_valid_edit_passes(self, tmp_path):
        target = tmp_path / "m.py"
        target.write_text("x = 1\n", encoding="utf-8")
        result = await EditFileTool().execute(
            path=str(target), old_string="x = 1", new_string="x = 2",
        )
        assert "成功" in result
        assert target.read_text(encoding="utf-8") == "x = 2\n"


# ---------- 读后写硬闸（agent 层） ----------

def _make_agent() -> AgentLoop:
    # __init__ 只存引用不调用，哑对象即可
    return AgentLoop(llm=object(), tool_registry=ToolRegistry(), permission_checker=object())


class TestReadBeforeWriteGate:
    def test_blind_write_existing_blocked(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("原内容", encoding="utf-8")
        agent = _make_agent()
        err = agent._check_write_gate("write_file", {"path": str(f)})
        assert err and "未读取过" in err

    def test_write_after_read_allowed(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("原内容", encoding="utf-8")
        agent = _make_agent()
        agent._record_file_state(str(f))
        assert agent._check_write_gate("write_file", {"path": str(f)}) is None
        assert agent._check_write_gate("edit_file", {"path": str(f)}) is None

    def test_external_modification_blocked(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("原内容", encoding="utf-8")
        agent = _make_agent()
        agent._record_file_state(str(f))
        f.write_text("外部改动后的内容——更长以保证 size 变化", encoding="utf-8")
        err = agent._check_write_gate("write_file", {"path": str(f)})
        assert err and "外部修改" in err

    def test_new_file_allowed(self, tmp_path):
        agent = _make_agent()
        assert agent._check_write_gate("write_file", {"path": str(tmp_path / "new.txt")}) is None

    def test_non_modify_tools_pass(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("x", encoding="utf-8")
        agent = _make_agent()
        assert agent._check_write_gate("read_file", {"path": str(f)}) is None
        assert agent._check_write_gate("shell", {"command": "ls"}) is None

    def test_case_insensitive_key(self, tmp_path):
        # Windows 路径大小写不敏感：读时小写、写时大写应视为同一文件
        f = tmp_path / "a.txt"
        f.write_text("x", encoding="utf-8")
        agent = _make_agent()
        agent._record_file_state(str(f).lower())
        assert agent._check_write_gate("write_file", {"path": str(f).upper()}) is None


# ---------- checkpoint 会话级留存 + restore(keep=True) ----------

@pytest.fixture
def ckpt_mgr(tmp_path, monkeypatch):
    """备份目录重定向到临时目录，防止测试触碰真实 ~/.loongcli/checkpoints/。"""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return CheckpointManager(cwd=tmp_path)


class TestCheckpointRetention:
    def test_list_returns_metadata_newest_first(self, tmp_path, ckpt_mgr):
        f = tmp_path / "a.txt"
        f.write_text("v1", encoding="utf-8")
        id1 = ckpt_mgr.save([str(f)], label="第一次")
        f.write_text("v2", encoding="utf-8")
        id2 = ckpt_mgr.save([str(f)])
        snaps = ckpt_mgr.list_checkpoints()
        assert [s["id"] for s in snaps] == [id2, id1]
        assert snaps[1]["label"] == "第一次"
        assert snaps[0]["files"] == [str(f)]
        assert snaps[0]["created_at"] is not None

    def test_restore_keep_preserves_snapshot(self, tmp_path, ckpt_mgr):
        f = tmp_path / "a.txt"
        f.write_text("v1", encoding="utf-8")
        ckpt_id = ckpt_mgr.save([str(f)])
        f.write_text("v2", encoding="utf-8")

        assert ckpt_mgr.restore(ckpt_id, keep=True)
        assert f.read_text(encoding="utf-8") == "v1"
        # 快照仍在，可重复回滚
        assert any(s["id"] == ckpt_id for s in ckpt_mgr.list_checkpoints())
        f.write_text("v3", encoding="utf-8")
        assert ckpt_mgr.restore(ckpt_id, keep=True)
        assert f.read_text(encoding="utf-8") == "v1"

    def test_restore_default_removes_snapshot(self, tmp_path, ckpt_mgr):
        f = tmp_path / "a.txt"
        f.write_text("v1", encoding="utf-8")
        ckpt_id = ckpt_mgr.save([str(f)])
        f.write_text("v2", encoding="utf-8")
        assert ckpt_mgr.restore(ckpt_id)
        assert f.read_text(encoding="utf-8") == "v1"
        assert not any(s["id"] == ckpt_id for s in ckpt_mgr.list_checkpoints())

    def test_max_checkpoints_evicts_oldest(self, tmp_path, ckpt_mgr, monkeypatch):
        monkeypatch.setattr(checkpoint_mod, "MAX_CHECKPOINTS", 3)
        f = tmp_path / "a.txt"
        ids = []
        for i in range(4):
            f.write_text(f"v{i}", encoding="utf-8")
            ids.append(ckpt_mgr.save([str(f)]))
        remaining = [s["id"] for s in ckpt_mgr.list_checkpoints()]
        assert ids[0] not in remaining
        assert set(ids[1:]) == set(remaining)
