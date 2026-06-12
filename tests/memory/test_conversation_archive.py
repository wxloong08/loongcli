"""compact/clear 历史保真：archive_segment / full_history / save 保留归档字段。"""
from __future__ import annotations

import json

from loongcli.memory.conversation import ConversationStore


def make_store(tmp_path) -> ConversationStore:
    return ConversationStore(base_dir=tmp_path / "sessions")


def msgs(*contents: str, role: str = "user") -> list[dict]:
    return [{"role": role, "content": c} for c in contents]


class TestArchiveSegment:
    def test_archive_then_save_preserves_original(self, tmp_path):
        """核心场景：compact 前归档 → 压缩版覆写 messages → 原始历史仍在归档段。"""
        store = make_store(tmp_path)
        original = msgs("第一轮", "第二轮", "第三轮")
        store.save(original)

        store.archive_segment(original, reason="auto-compact")
        compacted = [{"role": "user", "content": "[对话历史摘要] ..."}]
        store.save(compacted)

        data = json.loads(store.session_path.read_text(encoding="utf-8"))
        assert data["messages"] == compacted
        assert len(data["archived_segments"]) == 1
        seg = data["archived_segments"][0]
        assert seg["reason"] == "auto-compact"
        assert seg["message_count"] == 3
        assert seg["messages"] == original

    def test_multiple_compactions_accumulate_segments(self, tmp_path):
        store = make_store(tmp_path)
        first = msgs("a", "b")
        store.save(first)
        store.archive_segment(first, reason="auto-compact")
        second = msgs("摘要1", "c")
        store.save(second)
        store.archive_segment(second, reason="clear")
        store.save([])

        data = json.loads(store.session_path.read_text(encoding="utf-8"))
        assert [s["reason"] for s in data["archived_segments"]] == ["auto-compact", "clear"]

    def test_empty_messages_not_archived(self, tmp_path):
        store = make_store(tmp_path)
        store.archive_segment([], reason="clear")
        assert not store.session_path.exists()

    def test_archive_before_first_save(self, tmp_path):
        """会话文件还不存在时归档也不能崩。"""
        store = make_store(tmp_path)
        store.archive_segment(msgs("x"), reason="auto-compact")
        data = json.loads(store.session_path.read_text(encoding="utf-8"))
        assert len(data["archived_segments"]) == 1


class TestFullHistory:
    def test_concatenates_segments_and_current(self, tmp_path):
        store = make_store(tmp_path)
        original = msgs("远古消息甲", "远古消息乙")
        store.save(original)
        store.archive_segment(original)
        store.save(msgs("摘要", "新消息"))

        history = store.full_history()
        contents = [m["content"] for m in history]
        assert contents == ["远古消息甲", "远古消息乙", "摘要", "新消息"]

    def test_no_session_returns_empty(self, tmp_path):
        store = make_store(tmp_path)
        assert store.full_history() == []

    def test_no_segments_returns_current(self, tmp_path):
        store = make_store(tmp_path)
        store.save(msgs("唯一消息"))
        assert [m["content"] for m in store.full_history()] == ["唯一消息"]


class TestSavePreservesFields:
    def test_save_keeps_compact_messages_and_state(self, tmp_path):
        """save() 不再整体覆写文件，已有的 compact_messages/structured_state 要保留。"""
        store = make_store(tmp_path)
        store.save(msgs("a"))
        store.save_compact([{"role": "user", "content": "压缩"}], structured_state={"k": "v"})
        store.save(msgs("a", "b"))

        data = json.loads(store.session_path.read_text(encoding="utf-8"))
        assert data["compact_messages"] == [{"role": "user", "content": "压缩"}]
        assert data["structured_state"] == {"k": "v"}
        assert len(data["messages"]) == 2
