import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from loongcli.core.telemetry import EventLogger, NULL_TELEMETRY, _MAX_CONSECUTIVE_FAILURES
from loongcli.core.agent import AgentLoop, AgentServices
from loongcli.core.llm import LLMClient
from loongcli.tools.base import Tool, ToolRegistry
from loongcli.security.permissions import PermissionChecker


class TestEventLogger:
    def test_emit_writes_parseable_jsonl(self, tmp_path):
        log = EventLogger(tmp_path / "e.jsonl")
        log.emit("llm_call", model="deepseek-v4-flash", prompt_tokens=100)
        log.emit("tool_exec", name="read_file", duration_ms=12)

        lines = (tmp_path / "e.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["type"] == "llm_call"
        assert first["model"] == "deepseek-v4-flash"
        assert first["prompt_tokens"] == 100
        assert "ts" in first
        assert json.loads(lines[1])["name"] == "read_file"

    def test_disabled_writes_nothing(self, tmp_path):
        path = tmp_path / "e.jsonl"
        log = EventLogger(path, enabled=False)
        log.emit("llm_call")
        assert not path.exists()

    def test_null_telemetry_is_safe(self):
        NULL_TELEMETRY.emit("anything", key="value")
        assert NULL_TELEMETRY.enabled is False

    def test_unserializable_field_degrades_to_str(self, tmp_path):
        log = EventLogger(tmp_path / "e.jsonl")
        log.emit("event", path=Path("D:/some/where"))
        record = json.loads((tmp_path / "e.jsonl").read_text(encoding="utf-8"))
        assert isinstance(record["path"], str)

    def test_write_failure_silent_then_self_disable(self, tmp_path):
        # path 指向一个目录：open(..., "a") 必然 OSError
        log = EventLogger(tmp_path)
        for _ in range(_MAX_CONSECUTIVE_FAILURES):
            log.emit("event")  # 不抛即为静默
        assert log.enabled is False

    def test_failure_counter_resets_on_success(self, tmp_path):
        log = EventLogger(tmp_path / "e.jsonl")
        log._failures = _MAX_CONSECUTIVE_FAILURES - 1
        log.emit("event")
        assert log._failures == 0
        assert log.enabled is True

    def test_for_session_derives_path(self, tmp_path):
        class FakeStore:
            base_dir = tmp_path
            session_id = "abc123"

        log = EventLogger.for_session(FakeStore())
        assert log.enabled
        assert log.path == tmp_path / "abc123.events.jsonl"

    def test_for_session_none_store_disabled(self):
        log = EventLogger.for_session(None)
        assert log.enabled is False
        log.emit("event")  # 空转不抛

    def test_for_session_bad_store_disabled(self):
        log = EventLogger.for_session(object())
        assert log.enabled is False

    def test_unicode_not_escaped(self, tmp_path):
        log = EventLogger(tmp_path / "e.jsonl")
        log.emit("event", text="中文内容")
        raw = (tmp_path / "e.jsonl").read_text(encoding="utf-8")
        assert "中文内容" in raw


# ── 集成：AgentLoop 埋点 ──

class _EchoTool(Tool):
    name = "echo"
    description = "Echoes input"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def execute(self, text: str) -> str:
        return f"echo: {text}"


def _text_chunks(texts):
    chunks = []
    for t in texts:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = t
        chunk.choices[0].delta.tool_calls = None
        chunk.choices[0].finish_reason = None
        chunk.usage = None
        chunks.append(chunk)
    if chunks:
        chunks[-1].choices[0].finish_reason = "stop"
    return chunks


def _tool_call_chunks(tool_id, tool_name, arguments):
    tc_delta = MagicMock()
    tc_delta.index = 0
    tc_delta.id = tool_id
    tc_delta.type = "function"
    tc_delta.function = MagicMock()
    tc_delta.function.name = tool_name
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


def _read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _make_agent(llm, tmp_path, registry=None) -> tuple[AgentLoop, Path]:
    events_path = tmp_path / "e.jsonl"
    agent = AgentLoop(
        llm=llm,
        tool_registry=registry or ToolRegistry(),
        permission_checker=PermissionChecker(),
        services=AgentServices(telemetry=EventLogger(events_path)),
    )
    return agent, events_path


class TestAgentTelemetry:
    @pytest.mark.asyncio
    async def test_text_turn_event_sequence(self, tmp_path):
        llm = LLMClient(api_key="test")
        chunks = _text_chunks(["Hello"])

        async def mock_stream(**kwargs):
            for c in chunks:
                yield c

        llm.chat_stream = mock_stream
        agent, events_path = _make_agent(llm, tmp_path)

        async for _ in agent.run_stream("hi"):
            pass

        types = [(e["type"], e.get("phase")) for e in _read_events(events_path)]
        assert types == [("turn", "start"), ("llm_call", None), ("turn", "end")]

    @pytest.mark.asyncio
    async def test_usage_none_yields_zero_fields(self, tmp_path):
        llm = LLMClient(api_key="test")
        chunks = _text_chunks(["ok"])

        async def mock_stream(**kwargs):
            for c in chunks:
                yield c

        llm.chat_stream = mock_stream
        agent, events_path = _make_agent(llm, tmp_path)

        async for _ in agent.run_stream("hi"):
            pass

        llm_call = next(e for e in _read_events(events_path) if e["type"] == "llm_call")
        assert llm_call["prompt_tokens"] == 0
        assert llm_call["cache_hit"] == 0
        assert llm_call["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_tool_exec_event(self, tmp_path):
        llm = LLMClient(api_key="test")
        tool_chunks = _tool_call_chunks("c1", "echo", {"text": "world"})
        text_chunks = _text_chunks(["Done"])
        call_count = 0

        async def mock_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            for c in (tool_chunks if call_count == 1 else text_chunks):
                yield c

        llm.chat_stream = mock_stream
        registry = ToolRegistry()
        registry.register(_EchoTool())
        agent, events_path = _make_agent(llm, tmp_path, registry)

        async for _ in agent.run_stream("run echo"):
            pass

        events = _read_events(events_path)
        tool_events = [e for e in events if e["type"] == "tool_exec"]
        assert len(tool_events) == 1
        assert tool_events[0]["name"] == "echo"
        assert tool_events[0]["error"] is False
        assert tool_events[0]["truncated"] is False
        assert tool_events[0]["result_chars"] == len("echo: world")
        # turn end 记录本轮工具调用数
        turn_end = next(e for e in events if e["type"] == "turn" and e["phase"] == "end")
        assert turn_end["tool_calls"] == 1
        assert turn_end["aborted"] is False

    @pytest.mark.asyncio
    async def test_no_telemetry_service_uses_null(self):
        llm = LLMClient(api_key="test")
        agent = AgentLoop(
            llm=llm, tool_registry=ToolRegistry(),
            permission_checker=PermissionChecker(),
        )
        assert agent.telemetry is NULL_TELEMETRY

    @pytest.mark.asyncio
    async def test_llm_error_emits_error_event(self, tmp_path):
        llm = LLMClient(api_key="test")

        async def mock_stream(**kwargs):
            raise RuntimeError("boom")
            yield  # pragma: no cover

        llm.chat_stream = mock_stream
        agent, events_path = _make_agent(llm, tmp_path)

        async for _ in agent.run_stream("hi"):
            pass

        events = _read_events(events_path)
        llm_call = next(e for e in events if e["type"] == "llm_call")
        assert "boom" in llm_call["error"]
        # turn end 仍然到达（finally 保证）
        assert events[-1] == {**events[-1], "type": "turn", "phase": "end"}
