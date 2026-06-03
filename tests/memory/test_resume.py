import json
import pytest
from pathlib import Path

from loongcli.memory.conversation import ConversationStore


@pytest.fixture
def store(tmp_path):
    return ConversationStore(base_dir=tmp_path)


def test_resume_loads_messages(store):
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    store.save(messages)
    sid = store.session_id

    store2 = ConversationStore(base_dir=store.base_dir)
    restored = store2.resume(sid)

    assert restored is not None
    assert len(restored) == 3
    assert restored[0]["role"] == "system"
    assert restored[1]["content"] == "hello"
    assert store2.session_id == sid


def test_resume_nonexistent(store):
    result = store.resume("nonexistent")
    assert result is None


def test_resume_preserves_session_id(store):
    messages = [{"role": "user", "content": "test"}]
    store.save(messages)
    sid = store.session_id

    store2 = ConversationStore(base_dir=store.base_dir)
    original_sid = store2.session_id
    assert original_sid != sid

    store2.resume(sid)
    assert store2.session_id == sid


def test_resume_then_save_appends(store):
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
    ]
    store.save(messages)
    sid = store.session_id

    store2 = ConversationStore(base_dir=store.base_dir)
    restored = store2.resume(sid)
    restored.append({"role": "user", "content": "second"})
    store2.save(restored)

    data = store2.load(sid)
    assert len(data["messages"]) == 3
    assert data["messages"][-1]["content"] == "second"


# ---------- Structured state tests ----------


def test_save_compact_with_structured_state(store):
    """save_compact with structured_state persists both into the session JSON."""
    messages = [{"role": "user", "content": "hello"}]
    store.save(messages)
    sid = store.session_id

    compact_msgs = [{"role": "user", "content": "summary"}]
    store.save_compact(compact_msgs, structured_state={
        "summary": "User asked about Python.",
        "recent_files": ["a.py", "b.py"],
        "plan_id": "plan123",
        "active_tasks": [{"id": "t1", "prompt": "do stuff"}],
    })

    data = json.loads((store.base_dir / f"{sid}.json").read_text(encoding="utf-8"))
    assert data["compact_messages"] == compact_msgs
    ss = data["structured_state"]
    assert ss["summary"] == "User asked about Python."
    assert ss["recent_files"] == ["a.py", "b.py"]
    assert ss["plan_id"] == "plan123"
    assert ss["active_tasks"] == [{"id": "t1", "prompt": "do stuff"}]
    assert "messages" in data


def test_save_compact_without_structured_state(store):
    """save_compact without structured_state only saves compact_messages."""
    messages = [{"role": "user", "content": "hello"}]
    store.save(messages)
    sid = store.session_id

    store.save_compact([{"role": "user", "content": "compact"}])

    data = json.loads((store.base_dir / f"{sid}.json").read_text(encoding="utf-8"))
    assert "compact_messages" in data
    assert "structured_state" not in data


def test_resume_structured_returns_state(store):
    """resume_structured returns structured_state dict when it exists."""
    messages = [{"role": "user", "content": "hello"}]
    store.save(messages)
    sid = store.session_id

    store.save_compact([{"role": "user", "content": "compact"}], structured_state={
        "summary": "Summary text",
        "recent_files": ["x.py"],
        "plan_id": None,
        "active_tasks": [],
    })

    store2 = ConversationStore(base_dir=store.base_dir)
    result = store2.resume_structured(sid)

    assert result is not None
    assert result["summary"] == "Summary text"
    assert result["recent_files"] == ["x.py"]
    assert result["plan_id"] is None
    assert result["active_tasks"] == []


def test_resume_structured_returns_none_without_state(store):
    """resume_structured returns None when structured_state is absent."""
    messages = [{"role": "user", "content": "hello"}]
    store.save(messages)
    sid = store.session_id

    store2 = ConversationStore(base_dir=store.base_dir)
    result = store2.resume_structured(sid)
    assert result is None


def test_resume_structured_sets_session_id(store):
    """resume_structured sets session_id and _meta like resume does."""
    messages = [{"role": "user", "content": "hello"}]
    store.save(messages)
    sid = store.session_id

    store.save_compact([{"role": "user", "content": "compact"}], structured_state={
        "summary": "test",
        "recent_files": [],
        "plan_id": None,
        "active_tasks": [],
    })

    store2 = ConversationStore(base_dir=store.base_dir)
    original_sid = store2.session_id
    assert original_sid != sid

    store2.resume_structured(sid)
    assert store2.session_id == sid
    assert store2._meta["session_id"] == sid


def test_resume_structured_nonexistent(store):
    """resume_structured returns None for a non-existent session."""
    result = store.resume_structured("does_not_exist_999")
    assert result is None
