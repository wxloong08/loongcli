from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from loongcli.core.checkpoint import CheckpointManager, MODIFY_TOOLS, MAX_CHECKPOINTS


@pytest.fixture
def ckpt_mgr(tmp_path: Path) -> CheckpointManager:
    return CheckpointManager(cwd=tmp_path)


# --- Save / Restore ---


def test_save_restore(ckpt_mgr: CheckpointManager, tmp_path: Path):
    (tmp_path / "bar.py").write_text("original")
    ckpt_id = ckpt_mgr.save(["bar.py"])
    assert ckpt_id is not None

    (tmp_path / "bar.py").write_text("modified")
    assert ckpt_mgr.restore(ckpt_id)
    assert (tmp_path / "bar.py").read_text() == "original"


def test_save_does_not_remove_file(ckpt_mgr: CheckpointManager, tmp_path: Path):
    """Regression: save() must NOT remove files from the working tree."""
    (tmp_path / "foo.py").write_text("content")
    ckpt_mgr.save(["foo.py"])
    assert (tmp_path / "foo.py").exists()
    assert (tmp_path / "foo.py").read_text() == "content"


def test_save_absolute_path(ckpt_mgr: CheckpointManager, tmp_path: Path):
    """Absolute paths should work correctly."""
    (tmp_path / "foo.py").write_text("original")
    abs_path = str(tmp_path / "foo.py")
    ckpt_id = ckpt_mgr.save([abs_path])
    assert ckpt_id is not None

    (tmp_path / "foo.py").write_text("modified")
    assert ckpt_mgr.restore(ckpt_id)
    assert (tmp_path / "foo.py").read_text() == "original"


def test_save_multiple_files(ckpt_mgr: CheckpointManager, tmp_path: Path):
    (tmp_path / "a.py").write_text("a")
    (tmp_path / "b.py").write_text("b")
    ckpt_id = ckpt_mgr.save(["a.py", "b.py"])

    (tmp_path / "a.py").write_text("aa")
    (tmp_path / "b.py").write_text("bb")
    assert ckpt_mgr.restore(ckpt_id)
    assert (tmp_path / "a.py").read_text() == "a"
    assert (tmp_path / "b.py").read_text() == "b"


def test_save_same_basename_no_collision(ckpt_mgr: CheckpointManager, tmp_path: Path):
    """Two files sharing a basename must back up and restore independently."""
    (tmp_path / "pkg_a").mkdir()
    (tmp_path / "pkg_b").mkdir()
    (tmp_path / "pkg_a" / "__init__.py").write_text("a-original")
    (tmp_path / "pkg_b" / "__init__.py").write_text("b-original")

    ckpt_id = ckpt_mgr.save(["pkg_a/__init__.py", "pkg_b/__init__.py"])

    (tmp_path / "pkg_a" / "__init__.py").write_text("a-changed")
    (tmp_path / "pkg_b" / "__init__.py").write_text("b-changed")

    assert ckpt_mgr.restore(ckpt_id)
    assert (tmp_path / "pkg_a" / "__init__.py").read_text() == "a-original"
    assert (tmp_path / "pkg_b" / "__init__.py").read_text() == "b-original"


def test_restore_recreates_deleted_file(ckpt_mgr: CheckpointManager, tmp_path: Path):
    (tmp_path / "x.py").write_text("content")
    ckpt_id = ckpt_mgr.save(["x.py"])
    (tmp_path / "x.py").unlink()
    assert ckpt_mgr.restore(ckpt_id)
    assert (tmp_path / "x.py").read_text() == "content"


def test_save_empty_list_returns_none(ckpt_mgr: CheckpointManager):
    assert ckpt_mgr.save([]) is None


def test_save_nonexistent_file_returns_none(ckpt_mgr: CheckpointManager):
    assert ckpt_mgr.save(["nonexistent.py"]) is None


def test_restore_nonexistent(ckpt_mgr: CheckpointManager):
    assert not ckpt_mgr.restore("does-not-exist")


# --- Discard ---


def test_discard_removes_checkpoint(ckpt_mgr: CheckpointManager, tmp_path: Path):
    (tmp_path / "x.py").write_text("x")
    ckpt_id = ckpt_mgr.save(["x.py"])
    assert ckpt_mgr.discard(ckpt_id)
    assert ckpt_id not in ckpt_mgr.list_checkpoints()


def test_discard_nonexistent(ckpt_mgr: CheckpointManager):
    assert not ckpt_mgr.discard("ghost")


def test_discard_cleans_backup_dir(ckpt_mgr: CheckpointManager, tmp_path: Path):
    (tmp_path / "y.py").write_text("y")
    ckpt_id = ckpt_mgr.save(["y.py"])
    backup_dir = ckpt_mgr._backup_dir / ckpt_id
    assert backup_dir.exists()
    ckpt_mgr.discard(ckpt_id)
    assert not backup_dir.exists()


# --- Cleanup ---


def test_cleanup_old_checkpoints(ckpt_mgr: CheckpointManager, tmp_path: Path):
    for i in range(MAX_CHECKPOINTS + 10):
        (tmp_path / f"f{i}.py").write_text(f"v{i}")
        ckpt_mgr.save([f"f{i}.py"])
    assert len(ckpt_mgr.list_checkpoints()) <= MAX_CHECKPOINTS


def test_list_checkpoints_empty(ckpt_mgr: CheckpointManager):
    assert ckpt_mgr.list_checkpoints() == []


# --- No git dependency ---


def test_no_git_dependency(ckpt_mgr: CheckpointManager, tmp_path: Path):
    """Checkpoint works in a directory without git."""
    (tmp_path / "foo.py").write_text("content")
    ckpt_id = ckpt_mgr.save(["foo.py"])
    assert ckpt_id is not None
    (tmp_path / "foo.py").write_text("changed")
    assert ckpt_mgr.restore(ckpt_id)
    assert (tmp_path / "foo.py").read_text() == "content"


# --- MODIFY_TOOLS constant ---


def test_modify_tools_constant():
    assert "write_file" in MODIFY_TOOLS
    assert "edit_file" in MODIFY_TOOLS
    assert "shell" not in MODIFY_TOOLS


# --- AgentLoop integration ---

from loongcli.core.agent import AgentLoop, AgentServices
from loongcli.core.llm import LLMClient
from loongcli.tools.base import ToolRegistry
from loongcli.security.permissions import PermissionChecker
from loongcli.core.events import AgentDone


def _make_chunk(content=None, finish_reason=None):
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    delta = MagicMock(spec=[])
    delta.content = content
    delta.tool_calls = None
    chunk.choices[0].delta = delta
    chunk.choices[0].finish_reason = finish_reason
    chunk.usage = None
    return chunk


@pytest.mark.asyncio
async def test_agent_no_checkpoint_without_tool_calls():
    llm = LLMClient(api_key="test")
    ckpt = MagicMock()
    ckpt.save = MagicMock(return_value="ckpt-1")
    ckpt.discard = MagicMock()

    agent = AgentLoop(
        llm=llm,
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(),
        system_prompt="test",
        services=AgentServices(checkpoint_manager=ckpt),
    )

    async def mock_stream(**kwargs):
        yield _make_chunk(content="Hello!")
        yield _make_chunk(finish_reason="stop")

    llm.chat_stream = mock_stream

    events = []
    async for event in agent.run_stream("hi"):
        events.append(event)

    assert any(isinstance(e, AgentDone) for e in events)
    ckpt.save.assert_not_called()
    ckpt.discard.assert_not_called()


@pytest.mark.asyncio
async def test_agent_extracts_file_args():
    assert AgentLoop._extract_file_args("write_file", {"file_path": "foo.py"}) == ["foo.py"]
    assert AgentLoop._extract_file_args("edit_file", {"file_path": "bar.py"}) == ["bar.py"]
    assert AgentLoop._extract_file_args("edit_file", {"path": "baz.py"}) == ["baz.py"]
    assert AgentLoop._extract_file_args("shell", {"command": "git diff"}) == []
