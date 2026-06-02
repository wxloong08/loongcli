import pytest
import sys

from loongcli.hooks.manager import (
    HookManager, HookEvent, HookConfig, HookResult,
)


class TestHookConfig:
    def test_wildcard_matches_all(self):
        h = HookConfig(matcher="*", command="echo ok")
        assert h.matches("shell") is True
        assert h.matches("glob") is True
        assert h.matches(None) is True

    def test_empty_matcher_matches_all(self):
        h = HookConfig(matcher="", command="echo ok")
        assert h.matches("shell") is True

    def test_exact_match(self):
        h = HookConfig(matcher="shell", command="echo ok")
        assert h.matches("shell") is True
        assert h.matches("glob") is False

    def test_pipe_separated(self):
        h = HookConfig(matcher="shell|write_file", command="echo ok")
        assert h.matches("shell") is True
        assert h.matches("write_file") is True
        assert h.matches("glob") is False


class TestHookManagerFromConfig:
    def test_empty_config(self):
        m = HookManager.from_config({})
        assert not m.has_hooks(HookEvent.PRE_TOOL_USE)

    def test_valid_config(self):
        m = HookManager.from_config({
            "PreToolUse": [
                {"matcher": "shell", "command": "echo pre"},
            ],
            "PostToolUse": [
                {"matcher": "*", "command": "echo post"},
            ],
        })
        assert m.has_hooks(HookEvent.PRE_TOOL_USE)
        assert m.has_hooks(HookEvent.POST_TOOL_USE)
        assert not m.has_hooks(HookEvent.SESSION_START)

    def test_unknown_event_ignored(self):
        m = HookManager.from_config({
            "UnknownEvent": [{"command": "echo bad"}],
        })
        for event in HookEvent:
            assert not m.has_hooks(event)

    def test_missing_command_ignored(self):
        m = HookManager.from_config({
            "PreToolUse": [{"matcher": "shell"}],
        })
        assert not m.has_hooks(HookEvent.PRE_TOOL_USE)

    def test_custom_timeout(self):
        m = HookManager.from_config({
            "PreToolUse": [
                {"matcher": "*", "command": "echo ok", "timeout": 5},
            ],
        })
        assert m._hooks[HookEvent.PRE_TOOL_USE][0].timeout == 5.0


class TestHookManagerRun:
    @pytest.mark.asyncio
    async def test_no_hooks_returns_empty(self):
        m = HookManager()
        result = await m.run(HookEvent.PRE_TOOL_USE, {}, tool_name="shell")
        assert not result.blocked
        assert result.output == ""

    @pytest.mark.asyncio
    async def test_successful_hook(self):
        m = HookManager()
        m.add(HookEvent.PRE_TOOL_USE, HookConfig(
            matcher="*",
            command=f"{sys.executable} -c \"print('hello from hook')\"",
        ))
        result = await m.run(
            HookEvent.PRE_TOOL_USE,
            {"tool": "shell"},
            tool_name="shell",
        )
        assert not result.blocked
        assert "hello from hook" in result.output

    @pytest.mark.asyncio
    async def test_blocking_hook(self):
        m = HookManager()
        m.add(HookEvent.PRE_TOOL_USE, HookConfig(
            matcher="shell",
            command=f"{sys.executable} -c \"import sys; print('blocked'); sys.exit(2)\"",
        ))
        result = await m.run(
            HookEvent.PRE_TOOL_USE,
            {"tool": "shell"},
            tool_name="shell",
        )
        assert result.blocked
        assert "blocked" in result.reason

    @pytest.mark.asyncio
    async def test_non_matching_hook_skipped(self):
        m = HookManager()
        m.add(HookEvent.PRE_TOOL_USE, HookConfig(
            matcher="shell",
            command=f"{sys.executable} -c \"print('should not run')\"",
        ))
        result = await m.run(
            HookEvent.PRE_TOOL_USE,
            {"tool": "glob"},
            tool_name="glob",
        )
        assert not result.blocked
        assert result.output == ""

    @pytest.mark.asyncio
    async def test_hook_receives_stdin(self):
        m = HookManager()
        m.add(HookEvent.POST_TOOL_USE, HookConfig(
            matcher="*",
            command=f"{sys.executable} -c \"import sys,json; d=json.load(sys.stdin); print(d['tool'])\"",
        ))
        result = await m.run(
            HookEvent.POST_TOOL_USE,
            {"tool": "grep", "result": "found"},
            tool_name="grep",
        )
        assert result.output == "grep"

    @pytest.mark.asyncio
    async def test_hook_timeout(self):
        m = HookManager()
        m.add(HookEvent.PRE_TOOL_USE, HookConfig(
            matcher="*",
            command=f"{sys.executable} -c \"import time; time.sleep(10)\"",
            timeout=0.5,
        ))
        result = await m.run(
            HookEvent.PRE_TOOL_USE,
            {},
            tool_name="shell",
        )
        assert not result.blocked
        assert "timed out" in result.output.lower()

    @pytest.mark.asyncio
    async def test_multiple_hooks_run_in_order(self):
        m = HookManager()
        m.add(HookEvent.POST_TOOL_USE, HookConfig(
            matcher="*",
            command=f"{sys.executable} -c \"print('first')\"",
        ))
        m.add(HookEvent.POST_TOOL_USE, HookConfig(
            matcher="*",
            command=f"{sys.executable} -c \"print('second')\"",
        ))
        result = await m.run(
            HookEvent.POST_TOOL_USE,
            {},
            tool_name="shell",
        )
        assert "first" in result.output
        assert "second" in result.output

    @pytest.mark.asyncio
    async def test_blocking_stops_remaining(self):
        m = HookManager()
        m.add(HookEvent.PRE_TOOL_USE, HookConfig(
            matcher="*",
            command=f"{sys.executable} -c \"import sys; print('block'); sys.exit(2)\"",
        ))
        m.add(HookEvent.PRE_TOOL_USE, HookConfig(
            matcher="*",
            command=f"{sys.executable} -c \"print('should not run')\"",
        ))
        result = await m.run(
            HookEvent.PRE_TOOL_USE,
            {},
            tool_name="shell",
        )
        assert result.blocked
        assert "should not run" not in result.output
