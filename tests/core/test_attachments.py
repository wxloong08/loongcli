import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from loongcli.core.attachments import (
    build_attachments,
    extract_recent_files,
    POST_COMPACT_MAX_FILES,
)


class TestExtractRecentFiles:
    def test_extracts_read_file_paths(self):
        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "file content"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc2", "function": {"name": "read_file", "arguments": '{"path": "b.py"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc2", "content": "file content 2"},
        ]
        paths = extract_recent_files(msgs)
        assert paths == ["b.py", "a.py"]  # most recent first

    def test_deduplicates(self):
        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "v1"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc2", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc2", "content": "v2"},
        ]
        paths = extract_recent_files(msgs)
        assert paths == ["a.py"]

    def test_limits_to_max(self):
        msgs = []
        for i in range(10):
            msgs.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": f"tc{i}", "function": {"name": "read_file", "arguments": f'{{"path": "f{i}.py"}}'}}
            ]})
            msgs.append({"role": "tool", "tool_call_id": f"tc{i}", "content": f"c{i}"})
        paths = extract_recent_files(msgs)
        assert len(paths) == POST_COMPACT_MAX_FILES

    def test_ignores_non_read_tools(self):
        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "function": {"name": "grep", "arguments": '{"pattern": "foo"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "grep result"},
        ]
        paths = extract_recent_files(msgs)
        assert paths == []


class TestBuildAttachments:
    def test_empty_when_no_context(self):
        msgs = build_attachments([], plan_store=None, task_manager=None)
        assert msgs == []

    def test_includes_plan_status(self):
        plan_store = MagicMock()
        plan_store.format_for_prompt.return_value = "**修复 bug** (2/5 完成)"
        msgs = build_attachments([], plan_store=plan_store, task_manager=None)
        content = msgs[0]["content"]
        assert "计划进度" in content
        assert "修复 bug" in content

    def test_includes_task_status(self):
        from loongcli.core.task import TaskManager, Task, TaskStatus
        tm = TaskManager()
        task = Task(id="abc123", prompt="search docs", status=TaskStatus.RUNNING)
        tm.register(task)
        msgs = build_attachments([], plan_store=None, task_manager=tm)
        content = msgs[0]["content"]
        assert "子任务" in content
        assert "abc123" in content
