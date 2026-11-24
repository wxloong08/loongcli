import asyncio

import pytest

from loongcli.security.auto_review import AutoReviewer, MAX_ARG_CHARS


class FakeLLM:
    """按队列返回预设回复；元素是异常就抛出，是 'hang' 就挂住（测超时）。"""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def chat(self, prompt: str) -> str:
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if reply == "hang":
            await asyncio.sleep(10)
        return reply


ALLOW = '{"verdict": "allow", "category": "普通开发", "reason": "在 venv 里装依赖"}'
BLOCK = '{"verdict": "block", "category": "桌面与外设", "reason": "用 keybd_event 合成全局 Ctrl 按键"}'


@pytest.mark.asyncio
async def test_allow():
    r = await AutoReviewer(FakeLLM(ALLOW)).review("shell", {"command": "pip install x"})
    assert r.allowed and r.category == "普通开发" and not r.failed


@pytest.mark.asyncio
async def test_block_with_reason():
    r = await AutoReviewer(FakeLLM(BLOCK)).review("shell", {"command": "python hookprobe.py"})
    assert not r.allowed and "keybd_event" in r.reason and not r.failed


@pytest.mark.asyncio
async def test_json_wrapped_in_text_still_parsed():
    r = await AutoReviewer(FakeLLM("判定如下：\n```json\n" + ALLOW + "\n```")).review("shell", {"command": "pytest"})
    assert r.allowed


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [
    "好的，可以执行",                                     # 非 JSON
    '{"verdict": "maybe", "reason": "x"}',                 # verdict 不在取值内
    '{"verdict": "allow", ',                               # 半截 JSON
])
async def test_unparseable_fails_closed(reply):
    r = await AutoReviewer(FakeLLM(reply)).review("shell", {"command": "x"})
    assert not r.allowed and r.failed


@pytest.mark.asyncio
async def test_exception_fails_closed():
    r = await AutoReviewer(FakeLLM(RuntimeError("503"))).review("shell", {"command": "x"})
    assert not r.allowed and r.failed and "RuntimeError" in r.reason


@pytest.mark.asyncio
async def test_timeout_fails_closed():
    r = await AutoReviewer(FakeLLM("hang"), timeout=0.05).review("shell", {"command": "x"})
    assert not r.allowed and r.failed and "超时" in r.reason


@pytest.mark.asyncio
async def test_same_args_reviewed_once():
    llm = FakeLLM(ALLOW)
    rv = AutoReviewer(llm)
    await rv.review("shell", {"command": "pytest -q"})
    second = await rv.review("shell", {"command": "pytest -q"})
    assert second.allowed and second.cached
    assert len(llm.prompts) == 1


@pytest.mark.asyncio
async def test_block_also_cached():
    llm = FakeLLM(BLOCK)
    rv = AutoReviewer(llm)
    await rv.review("shell", {"command": "python hookprobe.py"})
    second = await rv.review("shell", {"command": "python hookprobe.py"})
    assert not second.allowed and len(llm.prompts) == 1


@pytest.mark.asyncio
async def test_failure_not_cached():
    # 一次超时/报错不能永久拦住这条命令：下次重新审
    llm = FakeLLM(RuntimeError("503"), ALLOW)
    rv = AutoReviewer(llm)
    first = await rv.review("shell", {"command": "pytest"})
    second = await rv.review("shell", {"command": "pytest"})
    assert first.failed and second.allowed and len(llm.prompts) == 2


@pytest.mark.asyncio
async def test_different_args_not_shared():
    llm = FakeLLM(ALLOW, BLOCK)
    rv = AutoReviewer(llm)
    await rv.review("shell", {"command": "pytest"})
    r = await rv.review("shell", {"command": "python hookprobe.py"})
    assert not r.allowed and len(llm.prompts) == 2


def test_prompt_contains_full_command_and_context():
    script = "cat > /tmp/p.py <<'EOF'\nuser32.keybd_event(0x11, 0, 0, 0)\nEOF\npython /tmp/p.py"
    p = AutoReviewer(None).build_prompt("shell", {"command": script}, goal="只做设计", step="核实环境", cwd="D:/x")
    assert "keybd_event(0x11" in p and "只做设计" in p and "核实环境" in p and "D:/x" in p


def test_long_args_clipped():
    p = AutoReviewer(None).build_prompt("shell", {"command": "x" * (MAX_ARG_CHARS + 500)})
    assert "已截断" in p and "x" * (MAX_ARG_CHARS + 1) not in p


def test_files_included_in_prompt():
    p = AutoReviewer(None).build_prompt("shell", {"command": "python tools/probe.py"},
                                        files={"tools/probe.py": "import pyautogui\npyautogui.press('ctrl')"})
    assert "tools/probe.py" in p and "pyautogui.press" in p


# ── 21.4 命令里运行的项目内脚本一并送审 ──

from loongcli.security.auto_review import referenced_scripts, MAX_SCRIPT_CHARS  # noqa: E402


def _proj(tmp_path):
    root = tmp_path / "proj"
    (root / "tools").mkdir(parents=True)
    (root / "tools" / "probe.py").write_text("import ctypes\nctypes.windll.user32.keybd_event(0x11, 0, 0, 0)\n", encoding="utf-8")
    return root


def test_project_script_content_included(tmp_path):
    root = _proj(tmp_path)
    files = referenced_scripts("shell", {"command": "python tools/probe.py --x"}, root, cwd=root)
    assert files and "keybd_event" in files["tools/probe.py"]


def test_outside_project_and_missing_not_read(tmp_path):
    root = _proj(tmp_path)
    outside = tmp_path / "elsewhere.py"
    outside.write_text("secret = 1", encoding="utf-8")
    cmd = f"python {outside.as_posix()} && python tools/nope.py"
    assert referenced_scripts("shell", {"command": cmd}, root, cwd=root) is None


def test_at_most_two_scripts_and_clipped(tmp_path):
    root = _proj(tmp_path)
    for n in ("a", "b", "c"):
        (root / f"{n}.sh").write_text("x" * (MAX_SCRIPT_CHARS + 10), encoding="utf-8")
    files = referenced_scripts("shell", {"command": "bash a.sh; bash b.sh; bash c.sh"}, root, cwd=root)
    assert list(files) == ["a.sh", "b.sh"]
    assert "已截断" in files["a.sh"]


def test_non_shell_tool_ignored(tmp_path):
    root = _proj(tmp_path)
    assert referenced_scripts("write_file", {"path": "tools/probe.py"}, root, cwd=root) is None


@pytest.mark.asyncio
async def test_reviewer_receives_script_content(tmp_path):
    root = _proj(tmp_path)
    llm = FakeLLM(BLOCK)
    files = referenced_scripts("shell", {"command": "python tools/probe.py"}, root, cwd=root)
    await AutoReviewer(llm).review("shell", {"command": "python tools/probe.py"}, files=files)
    assert "keybd_event(0x11" in llm.prompts[0]
