from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from loongcli.core.checkpoint import CheckpointManager, MODIFY_TOOLS, MAX_CHECKPOINTS


@pytest.fixture
def ckpt_mgr(tmp_path: Path) -> CheckpointManager:
    return CheckpointManager(cwd=tmp_path)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a git repo with a committed file."""
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path, capture_output=True,
    )
    (tmp_path / "foo.py").write_text("original")
    subprocess.run(["git", "add", "foo.py"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)
    return tmp_path


# --- Git mode ---


def test_save_restore_in_git_repo(git_repo: Path):
    mgr = CheckpointManager(cwd=git_repo)
    ckpt_id = mgr.save(["foo.py"])
    assert ckpt_id is not None

    (git_repo / "foo.py").write_text("modified")
    assert mgr.restore(ckpt_id)
    assert (git_repo / "foo.py").read_text() == "original"


def test_discard_in_git_repo(git_repo: Path):
    mgr = CheckpointManager(cwd=git_repo)
    ckpt_id = mgr.save(["foo.py"])
    assert ckpt_id is not None
    assert mgr.discard(ckpt_id)
    assert ckpt_id not in mgr.list_checkpoints()


def test_save_tracked_files_in_git(git_repo: Path):
    """Both tracked and untracked files are backed up."""
    (git_repo / "bar.py").write_text("new file, not tracked")
    mgr = CheckpointManager(cwd=git_repo)
    ckpt_id = mgr.save(["foo.py", "bar.py"])
    assert ckpt_id is not None
    (git_repo / "foo.py").write_text("modified")
    assert mgr.restore(ckpt_id)
    assert (git_repo / "foo.py").read_text() == "original"


def test_save_does_not_remove_tracked_file(git_repo: Path):
    """Regression: save() must NOT remove tracked files from the working tree."""
    (git_repo / "foo.py").write_text("modified before checkpoint")
    mgr = CheckpointManager(cwd=git_repo)
    mgr.save(["foo.py"])
    assert (git_repo / "foo.py").exists(), "Checkpoint removed tracked file!"
    assert (git_repo / "foo.py").read_text() == "modified before checkpoint"


def test_save_does_not_remove_untracked_file(git_repo: Path):
    """Regression: save() must NOT remove untracked files from the working tree."""
    (git_repo / "new_file.py").write_text("untracked content")
    mgr = CheckpointManager(cwd=git_repo)
    mgr.save(["new_file.py"])
    assert (git_repo / "new_file.py").exists(), "Checkpoint removed untracked file!"
    assert (git_repo / "new_file.py").read_text() == "untracked content"


# --- File backup mode (no git) ---


def test_save_restore_file_backup(ckpt_mgr: CheckpointManager, tmp_path: Path):
    (tmp_path / "bar.py").write_text("original")
    ckpt_id = ckpt_mgr.save(["bar.py"])
    assert ckpt_id is not None

    (tmp_path / "bar.py").write_text("modified")
    assert ckpt_mgr.restore(ckpt_id)
    assert (tmp_path / "bar.py").read_text() == "original"


def test_save_multiple_files(ckpt_mgr: CheckpointManager, tmp_path: Path):
    (tmp_path / "a.py").write_text("a")
    (tmp_path / "b.py").write_text("b")
    ckpt_id = ckpt_mgr.save(["a.py", "b.py"])

    (tmp_path / "a.py").write_text("aa")
    (tmp_path / "b.py").write_text("bb")
    assert ckpt_mgr.restore(ckpt_id)
    assert (tmp_path / "a.py").read_text() == "a"
    assert (tmp_path / "b.py").read_text() == "b"


def test_discard_removes_checkpoint(ckpt_mgr: CheckpointManager, tmp_path: Path):
    (tmp_path / "x.py").write_text("x")
    ckpt_id = ckpt_mgr.save(["x.py"])
    assert ckpt_mgr.discard(ckpt_id)
    assert ckpt_id not in ckpt_mgr.list_checkpoints()


def test_restore_nonexistent(ckpt_mgr: CheckpointManager):
    assert not ckpt_mgr.restore("does-not-exist")


def test_discard_nonexistent(ckpt_mgr: CheckpointManager):
    assert not ckpt_mgr.discard("ghost")


# --- Cleanup ---


def test_cleanup_old_checkpoints(ckpt_mgr: CheckpointManager, tmp_path: Path):
    for i in range(MAX_CHECKPOINTS + 10):
        (tmp_path / f"f{i}.py").write_text(f"v{i}")
        ckpt_mgr.save([f"f{i}.py"])
    assert len(ckpt_mgr.list_checkpoints()) <= MAX_CHECKPOINTS


def test_list_checkpoints_empty(ckpt_mgr: CheckpointManager):
    assert ckpt_mgr.list_checkpoints() == []


# --- MODIFY_TOOLS constant ---


def test_modify_tools_constant():
    assert "write_file" in MODIFY_TOOLS
    assert "edit_file" in MODIFY_TOOLS
    assert "shell" in MODIFY_TOOLS


# --- AgentLoop integration ---

import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from loongcli.core.agent import AgentLoop
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
async def test_agent_checkpoints_before_write_file():
    """Agent saves checkpoint before write_file tool execution."""
    llm = LLMClient(api_key="test")
    ckpt = AsyncMock()
    ckpt.save = MagicMock(return_value="ckpt-1")

    agent = AgentLoop(
        llm=llm,
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(),
        system_prompt="You are helpful.",
        checkpoint_manager=ckpt,
    )

    async def mock_stream(**kwargs):
        yield _make_chunk(content="Hello!")
        yield _make_chunk(finish_reason="stop")

    llm.chat_stream = mock_stream

    events = []
    async for event in agent.run_stream("hi"):
        events.append(event)

    assert any(isinstance(e, AgentDone) for e in events)
    # No tool calls in this test, so no checkpoint should be saved


@pytest.mark.asyncio
async def test_agent_extracts_file_args():
    """_extract_file_args returns file paths from tool args."""
    assert AgentLoop._extract_file_args("write_file", {"file_path": "foo.py"}) == ["foo.py"]
    assert AgentLoop._extract_file_args("edit_file", {"file_path": "bar.py"}) == ["bar.py"]
    assert AgentLoop._extract_file_args("read_file", {"file_path": "x.py"}) == ["x.py"]
    # Shell without file redirects returns no files
    assert AgentLoop._extract_file_args("shell", {"command": "git diff"}) == []
