"""Claude Code 迁移：记忆 frontmatter 解析、会话块格式翻译、配对修复、幂等。"""
import json

import pytest

from loongcli.migrate_cc import (
    cc_slug,
    convert_session,
    import_memories,
    import_sessions,
    parse_cc_memory,
    _repair_pairing,
)
from loongcli.memory.markdown_store import MarkdownMemoryStore
from loongcli.memory.memory_router import MemoryRouter


def test_cc_slug_keeps_double_dashes(tmp_path):
    # CC 规则：冒号/斜杠全替换为 '-' 且不折叠（D: → 'D--'），与 loongcli 折叠规则不同
    slug = cc_slug(tmp_path)
    assert ":" not in slug and "\\" not in slug and "/" not in slug


# ── 记忆 ─────────────────────────────────────────────────────────

CC_MEMORY = """---
name: test-fact
description: 一句话摘要
metadata:
  type: feedback
---

事实正文。

**Why:** 原因。
"""


def test_parse_cc_memory_nested_type(tmp_path):
    f = tmp_path / "test-fact.md"
    f.write_text(CC_MEMORY, encoding="utf-8")
    parsed = parse_cc_memory(f)
    assert parsed["name"] == "test-fact"
    assert parsed["description"] == "一句话摘要"
    assert parsed["type"] == "feedback"  # 嵌在 metadata: 块下
    assert "事实正文" in parsed["content"]


def test_parse_cc_memory_no_frontmatter(tmp_path):
    f = tmp_path / "raw-note.md"
    f.write_text("裸正文，无 frontmatter", encoding="utf-8")
    parsed = parse_cc_memory(f)
    assert parsed["name"] == "raw-note"
    assert parsed["type"] == "project"
    assert "裸正文" in parsed["content"]


def test_import_memories_idempotent(tmp_path):
    cc_dir = tmp_path / "cc"
    (cc_dir / "memory").mkdir(parents=True)
    (cc_dir / "memory" / "test-fact.md").write_text(CC_MEMORY, encoding="utf-8")
    (cc_dir / "memory" / "MEMORY.md").write_text("- 索引行", encoding="utf-8")

    store = MarkdownMemoryStore(base_dir=tmp_path / "loong_mem")
    stats = import_memories(cc_dir, store)
    assert stats == {"imported": 1, "skipped": 0, "failed": 0}  # MEMORY.md 索引不算
    assert (store.base_dir / "test-fact.md").exists()

    stats2 = import_memories(cc_dir, store)
    assert stats2 == {"imported": 0, "skipped": 1, "failed": 0}  # 重复导入幂等


CC_USER_MEMORY = """---
name: who-i-am
description: 用户身份背景
metadata:
  type: user
---

用户是 Python 开发者。
"""


def test_import_memories_routes_by_type(tmp_path):
    # router 注入时按 type 分库：user → 全局，其余 → 项目库
    cc_dir = tmp_path / "cc"
    (cc_dir / "memory").mkdir(parents=True)
    (cc_dir / "memory" / "test-fact.md").write_text(CC_MEMORY, encoding="utf-8")
    (cc_dir / "memory" / "who-i-am.md").write_text(CC_USER_MEMORY, encoding="utf-8")

    router = MemoryRouter(
        global_dir=tmp_path / "loong_mem",
        project_dir=tmp_path / "projects" / "D-proj" / "memory",
    )
    stats = import_memories(cc_dir, router)
    assert stats == {"imported": 2, "skipped": 0, "failed": 0}
    assert (router.global_store.base_dir / "who-i-am.md").exists()
    assert (router.project_store.base_dir / "test-fact.md").exists()
    assert not (router.global_store.base_dir / "test-fact.md").exists()

    stats2 = import_memories(cc_dir, router)
    assert stats2 == {"imported": 0, "skipped": 2, "failed": 0}  # 两库幂等


# ── 会话 ─────────────────────────────────────────────────────────

def _cc_line(type_, content, **extra):
    d = {"type": type_, "message": {"role": type_, "content": content}}
    d.update(extra)
    return json.dumps(d, ensure_ascii=False)


def _write_jsonl(path, lines):
    path.write_text("\n".join(lines), encoding="utf-8")


def test_convert_session_full_roundtrip(tmp_path):
    lines = [
        json.dumps({"type": "summary", "summary": "标题行"}),
        _cc_line("user", "帮我跑测试"),
        _cc_line("assistant", [
            {"type": "thinking", "thinking": "内部推理，应被丢弃"},
            {"type": "text", "text": "好的，执行"},
            {"type": "tool_use", "id": "t1", "name": "shell", "input": {"command": "pytest"}},
        ]),
        _cc_line("user", [
            {"type": "tool_result", "tool_use_id": "t1", "content": "3 passed"},
        ]),
        _cc_line("assistant", [{"type": "text", "text": "全部通过"}]),
        _cc_line("user", "meta 行应跳过", isMeta=True),
        _cc_line("user", "sidechain 应跳过", isSidechain=True),
    ]
    f = tmp_path / "abc.jsonl"
    _write_jsonl(f, lines)

    msgs = convert_session(f)
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "shell"
    assert json.loads(msgs[1]["tool_calls"][0]["function"]["arguments"]) == {"command": "pytest"}
    assert msgs[2]["tool_call_id"] == "t1"
    assert "3 passed" in msgs[2]["content"]
    assert "内部推理" not in json.dumps(msgs, ensure_ascii=False)  # thinking 丢弃
    assert "跳过" not in json.dumps(msgs, ensure_ascii=False)      # meta/sidechain 丢弃


def test_repair_pairing_unclosed_and_orphan():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "x", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "y", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "a", "content": "ok"},
        # b 未回（原会话中断）；下面还有一个无主孤儿
        {"role": "user", "content": "继续"},
        {"role": "tool", "tool_call_id": "ghost", "content": "孤儿"},
    ]
    out = _repair_pairing(msgs)
    ids = [m.get("tool_call_id") for m in out if m["role"] == "tool"]
    assert "b" in ids and "ghost" not in ids  # 未回补占位；孤儿丢弃
    b_msg = next(m for m in out if m.get("tool_call_id") == "b")
    assert "中断" in b_msg["content"]
    # 占位必须插在下一条非 tool 消息之前（紧跟所属 assistant 的回结果区）
    assert out.index(b_msg) < out.index(next(m for m in out if m.get("content") == "继续"))


def test_import_sessions_idempotent_and_meta(tmp_path):
    cc_dir = tmp_path / "cc"
    cc_dir.mkdir()
    _write_jsonl(cc_dir / "deadbeef-1234.jsonl", [
        _cc_line("user", "问题"),
        _cc_line("assistant", [{"type": "text", "text": "回答"}]),
    ])
    _write_jsonl(cc_dir / "empty-file.jsonl", [json.dumps({"type": "summary"})])

    out_dir = tmp_path / "sessions"
    stats = import_sessions(cc_dir, sessions_dir=out_dir)
    assert stats["imported"] == 1 and stats["empty"] == 1

    sid = "deadbeef-1234".replace("-", "")[:12]
    data = json.loads((out_dir / f"{sid}.json").read_text(encoding="utf-8"))
    assert data["meta"]["origin"] == "claude-code"
    assert data["meta"]["cc_session_id"] == "deadbeef-1234"
    assert [m["role"] for m in data["messages"]] == ["user", "assistant"]

    stats2 = import_sessions(cc_dir, sessions_dir=out_dir)
    assert stats2["imported"] == 0 and stats2["skipped"] == 1  # 幂等
