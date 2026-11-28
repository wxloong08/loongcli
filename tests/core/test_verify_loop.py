from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from loongcli.core.verify_loop import (
    build_verify_prompt,
    detect_test_failure,
    filter_code_files,
    is_test_command,
)


class TestBuildVerifyPrompt:
    def test_with_test_command(self):
        prompt = build_verify_prompt(changed_files=["src/foo.py", "src/bar.py"], test_command="pytest")
        assert prompt.startswith("## 验证")  # TUI 按这个前缀渲染成系统提示
        assert "src/foo.py" in prompt and "src/bar.py" in prompt
        assert "pytest" in prompt

    def test_without_test_command(self):
        prompt = build_verify_prompt(changed_files=["src/foo.py"], test_command=None)
        assert prompt.startswith("## 验证")
        assert "src/foo.py" in prompt
        assert "未检测到项目测试命令" in prompt

    def test_lists_files_not_as_failing_tests(self):
        """回归（Harbor 基线 cancel-async）：旧措辞「收尾前还没有通过的测试:」后面列的是改动文件，
        模型把它读成这些测试没通过。"""
        for cmd in ("pytest", None):
            prompt = build_verify_prompt(changed_files=["a.py"], test_command=cmd)
            assert "没有通过的测试" not in prompt
            assert "改动的文件" in prompt

    def test_no_retry_wording(self):
        """只提醒一次：提示里不再有轮次、重试次数之类的说法。"""
        for cmd in ("pytest", None):
            prompt = build_verify_prompt(changed_files=["a.py"], test_command=cmd)
            for wording in ("最多", "重试", "剩余", "第 2 轮"):
                assert wording not in prompt


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
        # unittest：解释器常是带路径的 venv python.exe 或 shell 变量（2026-11-15 dogfooding 真机命令）
        "python -m unittest discover -s tests",
        "py -m unittest tests.test_storage -v",
        ".work/tools/voice-input-venv/Scripts/python.exe -m unittest discover -s tests -v 2>&1 | tail -20",
        '"$VENV/Scripts/python.exe" -m unittest discover -s tests -t .',
        "VENV=.work/tools/venv/Scripts/python.exe && $VENV -m unittest discover -s tests -t . -v",
        "cd /d/proj && python -m pytest tests/",
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
        "python -m pip install pytest",
        ".work/tools/venv/Scripts/python.exe -m voice_input --diagnose",
        'powershell -NoProfile -Command "Get-Process VoiceInput"',
        "grep -m 1 unittest notes.txt",
        "python -m unittestx",
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

    # --- 总结行优先于退出码：测试命令常接 `| tail`，管道退出码恒为 0 ---

    def test_piped_unittest_failure_without_exit_code_is_failure(self):
        output = "test_a (tests.test_x.T.test_a) ... FAIL\n\nRan 3 tests in 0.010s\n\nFAILED (failures=1)"
        assert detect_test_failure(output)

    def test_piped_unittest_errors_is_failure(self):
        output = "Ran 5 tests in 0.2s\n\nFAILED (failures=1, errors=2, skipped=1)"
        assert detect_test_failure(output)

    def test_unittest_ok_is_not_failure(self):
        assert not detect_test_failure("Ran 33 tests in 2.069s\n\nOK")
        assert not detect_test_failure("Ran 49 tests in 3.592s\n\nOK (skipped=1)")

    def test_unittest_ok_with_exit_code_1_is_not_failure(self):
        """PowerShell 下 stderr 误报退出码 1，但 unittest 总结行是 OK。"""
        assert not detect_test_failure("[exit code: 1 — 一般错误]\nRan 4 tests in 0.1s\n\nOK")

    def test_unittest_no_tests_ran_is_failure(self):
        assert detect_test_failure("Ran 0 tests in 0.000s\n\nNO TESTS RAN")

    def test_piped_pytest_failure_without_exit_code_is_failure(self):
        assert detect_test_failure("==== 1 failed, 12 passed in 0.31s ====")

    def test_pytest_error_count_is_failure(self):
        assert detect_test_failure("[exit code: 0]\n==== 1 error in 0.05s ====")

    def test_pytest_xfailed_not_counted_as_failed(self):
        assert not detect_test_failure("==== 10 passed, 2 xfailed in 1.0s ====")

    def test_summary_cut_off_falls_back_to_exit_code(self):
        """输出被 head 截掉、看不到总结行时，只能退回看退出码（这是已知盲区）。"""
        assert not detect_test_failure("test_a ... ok\ntest_b ... ok")
        assert detect_test_failure("[exit code: 1]\ntest_a ... ok")


class TestFilterCodeFiles:
    def test_keeps_python_files(self):
        assert filter_code_files(["src/main.py", "lib/utils.py"]) == ["src/main.py", "lib/utils.py"]

    def test_keeps_js_ts_files(self):
        result = filter_code_files(["app.js", "index.ts", "Component.tsx"])
        assert len(result) == 3

    def test_removes_markdown(self):
        assert filter_code_files(["README.md", "docs/guide.md"]) == []

    def test_removes_config_files(self):
        assert filter_code_files(["config.yaml", "settings.json", ".env"]) == []

    def test_mixed_files(self):
        files = ["src/main.py", "README.md", "tests/test_foo.py", "docs/notes.txt"]
        result = filter_code_files(files)
        assert result == ["src/main.py", "tests/test_foo.py"]

    def test_empty_list(self):
        assert filter_code_files([]) == []

    def test_case_insensitive_extension(self):
        assert filter_code_files(["Main.PY", "app.JS"]) == ["Main.PY", "app.JS"]

    def test_go_rust_java(self):
        result = filter_code_files(["main.go", "lib.rs", "App.java"])
        assert len(result) == 3

    def test_shell_scripts(self):
        assert filter_code_files(["deploy.sh", "build.bash"]) == ["deploy.sh", "build.bash"]


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
    # 没改文件就不提醒验证
    assert not agent._verify_nudged
    assert not [m for m in agent.messages if str(m.get("content", "")).startswith("## 验证")]


@pytest.mark.asyncio
async def test_verify_not_nudged_initially():
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
    assert agent._verify_nudged is False


# --- 验证轮端到端：按脚本驱动真实 AgentLoop，假 shell 按命令依次返回预设输出 ---

def _text_step(content):
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = content
    chunk.choices[0].delta.tool_calls = None
    chunk.choices[0].finish_reason = "stop"
    chunk.usage = None
    return [chunk]


def _tool_step(call_id, name, arguments):
    tc_delta = MagicMock()
    tc_delta.index = 0
    tc_delta.id = call_id
    tc_delta.type = "function"
    tc_delta.function = MagicMock()
    tc_delta.function.name = name
    tc_delta.function.arguments = json.dumps(arguments)

    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = None
    chunk1.choices[0].delta.tool_calls = [tc_delta]
    chunk1.choices[0].finish_reason = None
    chunk1.usage = None

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = None
    chunk2.choices[0].delta.tool_calls = None
    chunk2.choices[0].finish_reason = "tool_calls"
    chunk2.usage = None
    return [chunk1, chunk2]


async def _run_scripted(steps, shell_outputs):
    """steps：每次模型调用返回的 chunk 列表；shell_outputs：命令 → 依次返回的输出列表。

    脚本用完后模型只回一句「（脚本结束）」，不抛异常——多出来的重试由断言判定，而不是让测试崩在别处。
    """
    from loongcli.core.agent import AgentLoop
    from loongcli.core.llm import LLMClient
    from loongcli.security.permissions import PermissionChecker, PermissionMode
    from loongcli.tools.base import Tool, ToolRegistry

    llm = LLMClient(api_key="test")
    remaining = list(steps)

    async def mock_stream(**kwargs):
        for c in (remaining.pop(0) if remaining else _text_step("（脚本结束）")):
            yield c

    llm.chat_stream = mock_stream
    holder: list = []

    class FakeEdit(Tool):
        name = "fake_edit"
        description = "模拟改了一个代码文件"
        parameters = {"type": "object", "properties": {}}

        async def execute(self) -> str:
            holder[0]._files_modified_this_turn.append("src/a.py")
            return "成功"

    class FakeWrite(Tool):
        # 用真实工具名，触发 agent 里按 write_file 处理的逻辑；路径指向不存在的文件，先读后写闸门放行
        name = "write_file"
        description = "模拟写文件"
        parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

        async def execute(self, path: str, **kwargs) -> str:
            holder[0]._files_modified_this_turn.append(path)
            return "成功"

    class FakeShell(Tool):
        name = "shell"
        description = "按命令依次返回预设输出"
        parameters = {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }

        async def execute(self, command: str, **kwargs) -> str:
            return shell_outputs[command].pop(0)

    registry = ToolRegistry()
    registry.register(FakeEdit())
    registry.register(FakeWrite())
    registry.register(FakeShell())
    agent = AgentLoop(
        llm=llm,
        tool_registry=registry,
        permission_checker=PermissionChecker(mode=PermissionMode.SKIP),
    )
    holder.append(agent)
    async for _ in agent.run_stream("改代码"):
        pass
    return agent


def _verify_prompts(agent):
    return [
        m["content"] for m in agent.messages
        if m["role"] == "user" and isinstance(m.get("content"), str) and m["content"].startswith("## 验证")
    ]


@pytest.mark.asyncio
async def test_e2e_edit_without_test_nudges_exactly_once():
    """jobhunter 的形状：改了代码没测就收尾 → 提醒一次；提醒后模型又改代码、仍没测就收尾，也不再追问。"""
    agent = await _run_scripted(
        [
            _tool_step("c1", "fake_edit", {}),
            _text_step("改完了"),
            _tool_step("c2", "fake_edit", {}),
            _text_step("又改了一处，不测了"),
        ],
        {},
    )
    assert len(_verify_prompts(agent)) == 1
    assert agent.messages[-1]["content"] == "又改了一处，不测了"  # 提醒后的那一轮直接收尾


@pytest.mark.asyncio
async def test_e2e_piped_passing_test_after_edit_no_nudge():
    """dogfood 三轮的形状：改完代码跑过接管道的 unittest 且通过 → 不提醒（此前因认不出 unittest 每轮都多提醒一次）。"""
    unit = ".work/tools/venv/Scripts/python.exe -m unittest discover -s tests -v 2>&1 | tail -20"
    agent = await _run_scripted(
        [
            _tool_step("c1", "fake_edit", {}),
            _tool_step("c2", "shell", {"command": unit}),
            _text_step("测试通过，完成"),
        ],
        {unit: ["test_a ... ok\n\nRan 33 tests in 2.069s\n\nOK"]},
    )
    assert _verify_prompts(agent) == []


@pytest.mark.asyncio
async def test_e2e_piped_failing_test_counts_as_not_tested():
    """测试接了 `| tail`（退出码 0）但失败就收尾：失败不算「测过」，提醒一次。"""
    unit = "python -m unittest discover -s tests 2>&1 | tail -3"
    agent = await _run_scripted(
        [
            _tool_step("c1", "fake_edit", {}),
            _tool_step("c2", "shell", {"command": unit}),
            _text_step("完成"),
            _text_step("确认还有一个失败，原因是……"),
        ],
        {unit: ["Ran 3 tests in 0.1s\n\nFAILED (failures=1)"]},
    )
    assert len(_verify_prompts(agent)) == 1


@pytest.mark.asyncio
async def test_e2e_non_test_failure_after_passing_test_no_nudge():
    """exec-3 的形状：测试通过后又跑了一条返回 1 的非测试命令 → 不提醒、不重试。"""
    unit = "$VENV -m unittest discover -s tests 2>&1 | tail -5"
    ps = 'powershell -NoProfile -Command "Get-Process VoiceInput"'
    agent = await _run_scripted(
        [
            _tool_step("c1", "fake_edit", {}),
            _tool_step("c2", "shell", {"command": unit}),
            _tool_step("c3", "shell", {"command": ps}),
            _text_step("验证通过"),
        ],
        {unit: ["Ran 3 tests in 0.1s\n\nOK"], ps: ["[exit code: 1 — 一般错误]\n"]},
    )
    assert _verify_prompts(agent) == []


_NOWHERE = "zz_verify_test_nonexistent"  # 不存在的目录：假 write_file 写这里不会碰真实文件，先读后写闸门也放行


@pytest.mark.asyncio
async def test_e2e_doc_edit_after_passing_test_no_nudge():
    """dogfood exec-2/3 的形状：改代码 → 测试通过 → 把证据写进 todo.md → 收尾。写文档不作废「已测过」。"""
    unit = "python -m unittest discover -s tests 2>&1 | tail -5"
    agent = await _run_scripted(
        [
            _tool_step("c1", "write_file", {"path": f"{_NOWHERE}/src/a.py"}),
            _tool_step("c2", "shell", {"command": unit}),
            _tool_step("c3", "write_file", {"path": f"{_NOWHERE}/tasks/todo.md"}),
            _text_step("完成，证据已写入 todo"),
        ],
        {unit: ["Ran 3 tests in 0.1s\n\nOK"]},
    )
    assert _verify_prompts(agent) == []


@pytest.mark.asyncio
async def test_e2e_code_edit_after_passing_test_nudges():
    """测试通过后又改了代码就收尾：这次改动没测过，提醒一次。"""
    unit = "python -m unittest discover -s tests"
    agent = await _run_scripted(
        [
            _tool_step("c1", "write_file", {"path": f"{_NOWHERE}/src/a.py"}),
            _tool_step("c2", "shell", {"command": unit}),
            _tool_step("c3", "write_file", {"path": f"{_NOWHERE}/src/b.py"}),
            _text_step("完成"),
        ],
        {unit: ["Ran 3 tests in 0.1s\n\nOK"]},
    )
    assert len(_verify_prompts(agent)) == 1

    def test_scratch_files_in_temp_dir_excluded(self, tmp_path, monkeypatch):
        """回归（Harbor 基线 protein-assembly）：临时目录里、工作目录外的草稿脚本不算交付物。"""
        import tempfile
        from pathlib import Path
        project = tmp_path / "proj"
        project.mkdir()
        monkeypatch.chdir(project)
        scratch = str(Path(tempfile.gettempdir()) / "loongcli-scratch-probe" / "design.py")
        assert filter_code_files([scratch, "src/main.py"]) == ["src/main.py"]

    def test_project_inside_temp_dir_still_counted(self, tmp_path, monkeypatch):
        """工作目录本身在临时目录里（如 tmp_path）时，项目里的文件照常算。"""
        monkeypatch.chdir(tmp_path)
        inside = str(tmp_path / "pkg" / "mod.py")
        assert filter_code_files([inside, "rel.py"]) == [inside, "rel.py"]
