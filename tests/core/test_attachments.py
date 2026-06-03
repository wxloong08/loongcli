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


class TestRestoreFiles:
    def test_reads_existing_file(self, tmp_path):
        from loongcli.core.attachments import restore_files
        f = tmp_path / "hello.py"
        f.write_text("print('hi')", encoding="utf-8")
        result = restore_files([str(f)])
        assert "hello.py" in result
        assert "print('hi')" in result

    def test_truncates_large_file(self, tmp_path):
        from loongcli.core.attachments import restore_files, POST_COMPACT_MAX_CHARS_PER_FILE
        f = tmp_path / "big.py"
        f.write_text("x" * (POST_COMPACT_MAX_CHARS_PER_FILE + 500), encoding="utf-8")
        result = restore_files([str(f)])
        assert "文件已截断" in result

    def test_skips_missing_file(self, tmp_path):
        from loongcli.core.attachments import restore_files
        result = restore_files([str(tmp_path / "nonexistent.py")])
        assert result == ""

    def test_budget_exhaustion(self, tmp_path, monkeypatch):
        import loongcli.core.attachments as mod
        monkeypatch.setattr(mod, "POST_COMPACT_CHAR_BUDGET", 1000)
        f1 = tmp_path / "f1.py"
        f1.write_text("a" * 900, encoding="utf-8")
        f2 = tmp_path / "f2.py"
        f2.write_text("b" * 500, encoding="utf-8")
        result = mod.restore_files([str(f1), str(f2)])
        assert "f1.py" in result
        assert "f2.py" not in result

    def test_empty_paths(self):
        from loongcli.core.attachments import restore_files
        assert restore_files([]) == ""


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
