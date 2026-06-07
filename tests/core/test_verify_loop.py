from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from loongcli.core.verify_loop import (
    VerifyState,
    build_verify_prompt,
    detect_test_failure,
    is_test_command,
    MAX_VERIFY_ROUNDS,
)


class TestVerifyState:
    def test_initial_state(self):
        vs = VerifyState()
        assert not vs.is_active
        assert not vs.is_exhausted
        assert vs.round == 0

    def test_start_sets_round_one(self):
        vs = VerifyState()
        vs.start(["foo.py"], "pytest")
        assert vs.is_active
        assert not vs.is_exhausted
        assert vs.round == 1
        assert vs.changed_files == ["foo.py"]
        assert vs.test_command == "pytest"

    def test_exhausted_after_max_rounds(self):
        vs = VerifyState()
        vs.start(["foo.py"], "pytest")
        vs.round = MAX_VERIFY_ROUNDS + 1
        assert vs.is_exhausted
        assert not vs.is_active

    def test_reset_clears_all(self):
        vs = VerifyState()
        vs.start(["foo.py"], "pytest")
        vs.round = 3
        vs.last_error = "failed"
        vs.reset()
        assert vs.round == 0
        assert not vs.is_active
        assert vs.changed_files == []
        assert vs.test_command is None
        assert vs.last_error is None


class TestBuildVerifyPrompt:
    def test_round_1_with_test_command(self):
        prompt = build_verify_prompt(
            changed_files=["src/foo.py", "src/bar.py"],
            test_command="pytest",
            round_number=1,
        )
        assert "src/foo.py" in prompt
        assert "src/bar.py" in prompt
        assert "pytest" in prompt
        assert "验证" in prompt

    def test_round_1_without_test_command(self):
        prompt = build_verify_prompt(
            changed_files=["src/foo.py"],
            test_command=None,
            round_number=1,
        )
        assert "src/foo.py" in prompt
        assert "验证" in prompt or "test" in prompt.lower()

    def test_round_2_with_error(self):
        prompt = build_verify_prompt(
            changed_files=["src/foo.py"],
            test_command="pytest",
            round_number=2,
            last_error="AssertionError: expected 5, got 3",
        )
        assert "AssertionError" in prompt
        assert "重试" in prompt or "retry" in prompt.lower()

    def test_round_3_last_attempt(self):
        prompt = build_verify_prompt(
            changed_files=["src/foo.py"],
            test_command="pytest",
            round_number=3,
            last_error="test failed",
        )
        assert "重试" in prompt or "retry" in prompt.lower()

    def test_long_error_truncated(self):
        long_error = "x" * 5000
        prompt = build_verify_prompt(
            changed_files=["src/foo.py"],
            test_command="pytest",
            round_number=2,
            last_error=long_error,
        )
        assert len(long_error) > len(prompt)
        # The error in the prompt should be truncated to ~3000 chars
        error_in_prompt = prompt.count("x")
        assert error_in_prompt <= 3100  # allow small overhead


class TestIsTestCommand:
    @pytest.mark.parametrize("cmd", [
        "pytest",
        "pytest tests/",
        "pytest -x tests/core/test_verify_loop.py",
        "python -m pytest",
        "python -m pytest --tb=short",
        "npm test",
        "npm test -- --coverage",
        "go test ./...",
        "cargo test",
        "make test",
    ])
    def test_recognized(self, cmd):
        assert is_test_command(cmd)

    @pytest.mark.parametrize("cmd", [
        "python main.py",
        "ls -la",
        "git status",
        "pip install pytest",
        "echo 'test'",
        "cat test_file.py",
        "npm install",
        "npm run build",
        "go build",
        "cargo build",
        "make build",
    ])
    def test_not_recognized(self, cmd):
        assert not is_test_command(cmd)

    def test_case_insensitive(self):
        assert is_test_command("PYTEST")
        assert is_test_command("NPM TEST")

    def test_empty_string(self):
        assert not is_test_command("")


class TestDetectTestFailure:
    def test_exit_code_0_is_not_failure(self):
        output = "[exit code: 0]\n15 passed in 0.17s"
        assert not detect_test_failure(output)

    def test_exit_code_1_all_passed_is_not_failure(self):
        """PowerShell stderr false positive: exit code 1 but all tests pass."""
        output = (
            "[exit code: 1 — 一般错误]\n"
            "================= 980 passed, 3 warnings in 90.70s =================="
        )
        assert not detect_test_failure(output)

    def test_exit_code_1_with_failures_is_failure(self):
        output = (
            "[exit code: 1 — 一般错误]\n"
            "================= 3 failed, 977 passed in 90.70s =================="
        )
        assert detect_test_failure(output)

    def test_exit_code_1_no_pytest_output_is_failure(self):
        output = "[exit code: 1 — 一般错误]\nSyntaxError: invalid syntax"
        assert detect_test_failure(output)

    def test_no_exit_code_is_not_failure(self):
        output = "some random shell output"
        assert not detect_test_failure(output)

    def test_exit_code_0_passed_and_failed_zero(self):
        output = "[exit code: 0]\n5 passed, 0 failed"
        assert not detect_test_failure(output)

    def test_exit_code_1_passed_and_0_failed_is_not_failure(self):
        output = "[exit code: 1 — 一般错误]\n22 passed, 0 failed in 0.45s"
        assert not detect_test_failure(output)

    def test_exit_code_2_no_tests_collected(self):
        output = "[exit code: 2 — 错误]\nno tests ran"
        assert detect_test_failure(output)


class TestVerifyStateExhausted:
    def test_exhausted_is_not_active(self):
        vs = VerifyState()
        vs.start(["foo.py"], "pytest")
        vs.round = MAX_VERIFY_ROUNDS + 1
        assert vs.is_exhausted
        assert not vs.is_active

    def test_round_at_max_is_still_active(self):
        vs = VerifyState()
        vs.start(["foo.py"], "pytest")
        vs.round = MAX_VERIFY_ROUNDS
        assert vs.is_active
        assert not vs.is_exhausted

    def test_round_beyond_max_is_exhausted(self):
        vs = VerifyState()
        vs.start(["foo.py"], "pytest")
        vs.round = MAX_VERIFY_ROUNDS + 1
        assert vs.is_exhausted


# --- AgentLoop integration ---

@pytest.mark.asyncio
async def test_agent_verify_loop_no_file_changes():
    """No file modifications — no verify loop triggered."""
    from loongcli.core.agent import AgentLoop
    from loongcli.core.llm import LLMClient
    from loongcli.tools.base import ToolRegistry
    from loongcli.security.permissions import PermissionChecker
    from loongcli.core.events import AgentDone

    llm = LLMClient(api_key="test")

    async def mock_stream(**kwargs):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        delta = MagicMock(spec=[])
        delta.content = "Hello!"
        delta.tool_calls = None
        chunk.choices[0].delta = delta
        chunk.choices[0].finish_reason = "stop"
        chunk.usage = None
        yield chunk

    llm.chat_stream = mock_stream

    agent = AgentLoop(
        llm=llm,
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(),
        system_prompt="You are helpful.",
    )

    events = []
    async for event in agent.run_stream("hi"):
        events.append(event)

    assert any(isinstance(e, AgentDone) for e in events)
    # No file changes, verify state should not be active
    assert not agent._verify_state.is_active


@pytest.mark.asyncio
async def test_verify_state_not_active_initially():
    """VerifyState should be reset at agent creation and after each run."""
    from loongcli.core.agent import AgentLoop
    from loongcli.core.llm import LLMClient
    from loongcli.tools.base import ToolRegistry
    from loongcli.security.permissions import PermissionChecker

    agent = AgentLoop(
        llm=LLMClient(api_key="test"),
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(),
        system_prompt="test",
    )
    assert not agent._verify_state.is_active
    assert agent._verify_state.round == 0


@pytest.mark.asyncio
async def test_verify_exhausted_emits_warning():
    """When verify is exhausted after max retries, AgentDone should contain a warning."""
    from loongcli.core.agent import AgentLoop
    from loongcli.core.llm import LLMClient
    from loongcli.tools.base import ToolRegistry
    from loongcli.security.permissions import PermissionChecker
    from loongcli.core.events import AgentDone

    call_count = 0
    llm = LLMClient(api_key="test")

    async def mock_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        delta = MagicMock(spec=[])
        delta.content = "Tests still failing."
        delta.tool_calls = None
        chunk.choices[0].delta = delta
        chunk.choices[0].finish_reason = "stop"
        chunk.usage = None
        yield chunk

    llm.chat_stream = mock_stream

    agent = AgentLoop(
        llm=llm,
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(),
        system_prompt="test",
    )

    # Patch reset so we can pre-set verify state before the loop runs
    original_reset = agent._verify_state.reset
    first_reset = [True]

    def patched_reset():
        if first_reset[0]:
            first_reset[0] = False
            original_reset()
            # After reset, set verify to exhausted state
            agent._verify_state.start(["foo.py"], "pytest")
            agent._verify_state.round = MAX_VERIFY_ROUNDS + 1
            agent._verify_state.test_failed = True
        else:
            original_reset()

    agent._verify_state.reset = patched_reset

    events = []
    async for event in agent.run_stream("hi"):
        events.append(event)

    done_events = [e for e in events if isinstance(e, AgentDone)]
    assert len(done_events) == 1
    assert "验证失败" in done_events[0].content
