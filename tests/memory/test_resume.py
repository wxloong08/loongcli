import json
import time

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


# ---------- 胶囊新鲜度 + 恢复归档（2026-07-17 两洞回归） ----------


def test_stale_capsule_falls_back_to_raw(store):
    """干净退出→continue 干活→崩溃退出：胶囊比 raw 旧，必须降级 raw 而非时间旅行。"""
    store.save([{"role": "user", "content": "day1"}])
    sid = store.session_id
    store.save_compact(
        [{"role": "user", "content": "day1 摘要"}],
        structured_state={"summary": "day1"},
    )
    time.sleep(0.002)  # 保证 updated_at 严格晚于 capsule_saved_at
    day2 = [
        {"role": "user", "content": "day1 摘要"},
        {"role": "user", "content": "day2 新工作"},
    ]
    store.save(day2)  # 模拟 continue 后每轮落盘，随后崩溃（没有新胶囊）

    store2 = ConversationStore(base_dir=store.base_dir)
    assert store2.resume_structured(sid) is None  # 过期胶囊不加载
    assert store2.resume(sid) == day2  # 降级到最新 raw


def test_fresh_capsule_still_preferred(store):
    """干净退出后立即 continue：胶囊比 raw 新，正常走结构化恢复。"""
    store.save([{"role": "user", "content": "干活"}])
    sid = store.session_id
    store.save_compact([{"role": "user", "content": "摘要"}], structured_state={"summary": "s"})

    store2 = ConversationStore(base_dir=store.base_dir)
    assert store2.resume_structured(sid) == {"summary": "s"}


def test_legacy_capsule_without_timestamp_still_loads(store):
    """升级前的会话文件没有 capsule_saved_at——按不过期处理，维持原行为。"""
    store.save([{"role": "user", "content": "旧"}])
    sid = store.session_id
    store.save_compact([{"role": "user", "content": "摘要"}], structured_state={"summary": "s"})
    path = store.base_dir / f"{sid}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["capsule_saved_at"]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    store2 = ConversationStore(base_dir=store.base_dir)
    assert store2.resume_structured(sid) is not None


def test_resume_structured_archives_raw_tail(store):
    """恢复重建上下文前先归档旧段 raw——否则恢复后第一次 save() 就把尾巴覆盖丢失。"""
    day1 = [{"role": "user", "content": "day1 原始细节"}]
    store.save(day1)
    sid = store.session_id
    store.save_compact([{"role": "user", "content": "摘要"}], structured_state={"summary": "s"})

    store2 = ConversationStore(base_dir=store.base_dir)
    store2.resume_structured(sid)
    store2.save([{"role": "user", "content": "[摘要注入]"}])  # 恢复后第一轮落盘覆写 messages

    history = store2.full_history()
    assert any(m["content"] == "day1 原始细节" for m in history)
    data = store2.load(sid)
    assert data["archived_segments"][-1]["reason"] == "resume"


def test_resume_compact_archives_raw_tail(store):
    """compact 降级路径同样先归档 raw。"""
    day1 = [{"role": "user", "content": "raw 尾巴"}]
    store.save(day1)
    sid = store.session_id
    store.save_compact([{"role": "user", "content": "压缩版"}])

    store2 = ConversationStore(base_dir=store.base_dir)
    restored = store2.resume(sid)
    assert restored == [{"role": "user", "content": "压缩版"}]
    data = store2.load(sid)
    assert data["archived_segments"][-1]["messages"] == day1


def test_repeated_resume_does_not_duplicate_archive(store):
    """无新轮次的连续 resume 不重复归档同一内容。"""
    store.save([{"role": "user", "content": "内容"}])
    sid = store.session_id
    store.save_compact([{"role": "user", "content": "摘要"}], structured_state={"summary": "s"})

    for _ in range(3):
        s = ConversationStore(base_dir=store.base_dir)
        s.resume_structured(sid)

    data = store.load(sid)
    assert len(data["archived_segments"]) == 1
