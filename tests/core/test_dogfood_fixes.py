"""2026-11 dogfooding（jobhunter 四个会话）暴露的 harness 问题的回归测试。

每个测试对应 tasks/todo.md 那一节里的一条：回合内逐迭代落盘、迭代上限与工具预算对齐、
撞迭代兜底时说明入历史、进程内跨日标记、视觉提示归因、验证提示去重。
（控制台隔离见 tests/tools/test_shell_console_isolation.py；MCP 落盘见
test_tool_result_manager.py。）
"""
import json
from datetime import date, datetime, timedelta

import pytest

from loongcli.core.agent import (
    AgentLoop, AgentServices, DEFAULT_MAX_ITERATIONS, MAX_TOOL_CALLS_PER_RUN,
    NO_VISION_MSG, NO_VISION_READ_MSG,
)
from loongcli.core.events import AgentDone
from loongcli.core.llm import LLMClient
from loongcli.core.resume_gap import DATE_ROLLOVER_MARKER
from loongcli.memory.conversation import ConversationStore
from loongcli.security.permissions import PermissionChecker
from loongcli.tools.base import Tool, ToolRegistry
from tests.core.test_agent import EchoTool, _make_text_chunks, _make_tool_call_chunks


def _agent(llm, tools=(), **kwargs) -> AgentLoop:
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    return AgentLoop(
        llm=llm, tool_registry=registry, permission_checker=PermissionChecker(), **kwargs,
    )


def _scripted_llm(script):
    """script: 每次 LLM 调用依次取一项——chunk 列表，或一个要抛出的异常实例。"""
    llm = LLMClient(api_key="test")
    calls = {"n": 0}

    async def mock_stream(**kwargs):
        step = script[min(calls["n"], len(script) - 1)]
        calls["n"] += 1
        if isinstance(step, BaseException):
            raise step
        for c in step:
            yield c

    llm.chat_stream = mock_stream
    llm.calls = calls
    return llm


# ── 回合内逐迭代落盘 ─────────────────────────────────────────────────────────

class _ProcessDied(BaseException):
    """模拟进程在回合中途消失（冻结/被杀）：BaseException 穿透 run_stream 里的
    `except Exception`，等价于「后面的代码再也没机会执行」。"""


@pytest.mark.asyncio
async def test_turn_is_persisted_after_each_iteration(tmp_path):
    """第 2 个迭代的 LLM 调用时进程消失：磁盘上必须已有用户消息 + 第 1 迭代的完整工具往返。

    事故（会话 b379c5b3aaa0）：TUI 冻结在第 3 次工具调用，此前只在回合收尾落盘，
    含 git stash + rebase 的整个回合不在会话文件里。"""
    store = ConversationStore(base_dir=tmp_path)
    llm = _scripted_llm([
        _make_tool_call_chunks("call_1", "echo", {"text": "副作用已发生"}),
        _ProcessDied(),
    ])
    agent = _agent(llm, [EchoTool()], services=AgentServices(conversation_store=store))

    with pytest.raises(_ProcessDied):
        async for _ in agent.run_stream("做点有副作用的事"):
            pass

    saved = json.loads(store.session_path.read_text(encoding="utf-8"))["messages"]
    roles = [m["role"] for m in saved]
    assert roles == ["user", "assistant", "tool"], roles
    assert saved[0]["content"] == "做点有副作用的事"
    assert saved[2]["content"] == "echo: 副作用已发生"
    # 落盘的中间态对 API 合法：每个 tool_call 都有对应的 tool 结果，没有孤儿
    call_ids = {tc["id"] for tc in saved[1]["tool_calls"]}
    assert call_ids == {m["tool_call_id"] for m in saved if m["role"] == "tool"}


@pytest.mark.asyncio
async def test_mid_turn_persist_failure_does_not_abort_turn():
    """回合内落盘是保险不是闸：磁盘写失败只记日志，回合照常跑完。"""
    class _FlakyStore:
        def __init__(self):
            self.saves = 0

        def save(self, messages):
            self.saves += 1
            if self.saves == 1:  # 第一次（回合内那次）失败
                raise OSError("disk full")

    store = _FlakyStore()
    llm = _scripted_llm([
        _make_tool_call_chunks("call_1", "echo", {"text": "x"}),
        _make_text_chunks(["完成"]),
    ])
    agent = _agent(llm, [EchoTool()], services=AgentServices(conversation_store=store))

    events = [e async for e in agent.run_stream("go")]

    done = [e for e in events if isinstance(e, AgentDone)]
    assert done and done[0].content == "完成"
    assert store.saves == 2  # 回合内 1 次（失败）+ 收尾 1 次（成功）


# ── 迭代上限与工具预算对齐 ───────────────────────────────────────────────────

def test_main_agent_iteration_cap_aligned_with_tool_budget():
    """默认迭代上限 = 工具预算。此前 50 < 200：一轮≈一次调用，迭代上限永远先触发，
    工具预算那条「收走工具 → 文字总结」的优雅收场形同虚设（四个会话首轮 4/4 撞 50）。"""
    assert DEFAULT_MAX_ITERATIONS == MAX_TOOL_CALLS_PER_RUN
    agent = _agent(LLMClient(api_key="test"))
    assert agent.max_iterations == DEFAULT_MAX_ITERATIONS


@pytest.mark.asyncio
async def test_turn_longer_than_old_cap_is_not_cut_off():
    """真实工作量（看 boss 新消息 ≈70 轮）不再被截停：60 轮单工具迭代后正常收尾。"""
    script = [
        _make_tool_call_chunks(f"call_{i}", "echo", {"text": f"step {i}"})
        for i in range(60)
    ] + [_make_text_chunks(["全部做完"])]
    agent = _agent(_scripted_llm(script), [EchoTool()])

    events = [e async for e in agent.run_stream("长任务")]

    done = [e for e in events if isinstance(e, AgentDone)]
    assert done[-1].content == "全部做完"
    assert "迭代上限" not in done[-1].content


@pytest.mark.asyncio
async def test_iteration_cap_notice_is_written_to_history(tmp_path):
    """撞迭代兜底：说明必须进历史并落盘——否则历史停在 tool 结果上，用户说「继续」后
    模型不知道上一回合是被截停的。提示里要告诉用户可以直接继续。"""
    store = ConversationStore(base_dir=tmp_path)
    script = [
        _make_tool_call_chunks(f"call_{i}", "echo", {"text": f"step {i}"}) for i in range(5)
    ]
    agent = _agent(
        _scripted_llm(script), [EchoTool()], max_iterations=3,
        services=AgentServices(conversation_store=store),
    )

    events = [e async for e in agent.run_stream("loop")]

    done = [e for e in events if isinstance(e, AgentDone)]
    assert "迭代上限" in done[0].content and "继续" in done[0].content
    assert agent.messages[-1] == {"role": "assistant", "content": done[0].content}
    saved = json.loads(store.session_path.read_text(encoding="utf-8"))["messages"]
    assert saved[-1]["content"] == done[0].content


# ── 进程内跨日标记 ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_date_rollover_marker_injected_before_user_message():
    """上一回合在昨天、这一回合在今天：用户消息之前出现一对「日期变更」标记。

    事故（会话 0eb18513fbee）：11-10 下午开的会话聊到 11-11 凌晨，模型仍按系统提示里
    11-10 的日期快照推算，把 11-12 的安排说成「今天下午」。"""
    agent = _agent(_scripted_llm([_make_text_chunks(["ok"])]), system_prompt="sys")
    agent._last_turn_date = date.today() - timedelta(days=1)

    [e async for e in agent.run_stream("会议是哪天？")]

    contents = [(m["role"], m["content"]) for m in agent.messages]
    idx = next(i for i, (_, c) in enumerate(contents) if c == "会议是哪天？")
    marker_role, marker_text = contents[idx - 2]
    assert marker_role == "user" and marker_text.startswith(DATE_ROLLOVER_MARKER)
    assert datetime.now().strftime("%Y-%m-%d") in marker_text
    assert contents[idx - 1][0] == "assistant"  # 角色交替不被破坏
    assert agent.messages[0] == {"role": "system", "content": "sys"}  # 系统提示原样（不破缓存）
    assert agent._last_turn_date == date.today()


@pytest.mark.asyncio
async def test_no_date_marker_on_first_turn_or_same_day():
    """本进程第一回合（系统提示日期新鲜）与同日后续回合都不插标记——只为跨日付上下文。"""
    agent = _agent(_scripted_llm([_make_text_chunks(["ok"])]))

    [e async for e in agent.run_stream("第一句")]
    [e async for e in agent.run_stream("第二句")]

    assert not [
        m for m in agent.messages
        if isinstance(m.get("content"), str) and m["content"].startswith(DATE_ROLLOVER_MARKER)
    ]


# ── 视觉提示归因 ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("msg", [NO_VISION_MSG, NO_VISION_READ_MSG])
def test_no_vision_messages_blame_config_not_model(msg):
    """门控是 harness 按配置做的，提示必须把归因落在配置开关上，并明说「不代表模型
    不支持」。旧措辞「当前模型未开启视觉能力」被模型转述成「模型没有视觉能力」。"""
    assert "vision" in msg and "配置" in msg
    assert "不代表模型不支持" in msg
    assert "当前模型未开启视觉能力" not in msg


def test_no_vision_read_msg_tells_model_how_to_relay():
    assert "不要说成模型没有视觉能力" in NO_VISION_READ_MSG


# ── 验证提示文件去重 ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_verify_prompt_lists_each_modified_file_once():
    """同一文件本回合改 3 次，验证提示只列 1 次、计数同步（真机：setup_env.py 列了 3 遍）。"""
    holder = {}

    class EditTool(Tool):
        name = "fake_edit"
        description = "pretend to edit a file"
        parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

        async def execute(self, path: str) -> str:
            holder["agent"]._files_modified_this_turn.append(path)
            return "edited"

    script = [
        _make_tool_call_chunks("c1", "fake_edit", {"path": "scripts/setup_env.py"}),
        _make_tool_call_chunks("c2", "fake_edit", {"path": "scripts/setup_env.py"}),
        _make_tool_call_chunks("c3", "fake_edit", {"path": "scripts/setup_env.py"}),
        _make_tool_call_chunks("c4", "fake_edit", {"path": "loongcli/core/agent.py"}),
        _make_text_chunks(["改完了"]),      # → 触发验证提示
        _make_text_chunks(["验证通过"]),
    ]
    agent = _agent(_scripted_llm(script), [EditTool()])
    holder["agent"] = agent

    [e async for e in agent.run_stream("改脚本")]

    prompts = [
        m["content"] for m in agent.messages
        if m["role"] == "user" and isinstance(m["content"], str) and m["content"].startswith("## 验证")
    ]
    assert len(prompts) == 1
    assert "本轮改动了 2 个代码文件" in prompts[0]
    assert prompts[0].count("scripts/setup_env.py") == 1
    assert prompts[0].count("loongcli/core/agent.py") == 1
