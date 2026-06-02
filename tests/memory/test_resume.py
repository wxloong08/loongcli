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
