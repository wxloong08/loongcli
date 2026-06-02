import pytest
from pathlib import Path
from loongcli.memory.store import MemoryStore, MemoryEntry, MEMORY_TYPES, PRIORITY_ORDER


@pytest.fixture
def store(tmp_path):
    return MemoryStore(path=tmp_path / "kv.json")


def test_set_and_get(store):
    store.set("git", "repo", "loongcli")
    assert store.get("git", "repo") == "loongcli"


def test_get_missing(store):
    assert store.get("nope", "nope") is None


def test_delete_key(store):
    store.set("git", "repo", "loongcli")
    store.set("git", "branch", "main")
    assert store.delete("git", "repo") is True
    assert store.get("git", "repo") is None
    assert store.get("git", "branch") == "main"


def test_delete_category(store):
    store.set("git", "repo", "loongcli")
    store.set("git", "branch", "main")
    assert store.delete("git") is True
    assert store.list_categories() == {}


def test_delete_nonexistent(store):
    assert store.delete("nope") is False
    assert store.delete("nope", "nope") is False


def test_list_categories(store):
    store.set("git", "repo", "loongcli")
    store.set("git", "branch", "main")
    store.set("deploy", "target", "prod")
    assert store.list_categories() == {"git": 2, "deploy": 1}


def test_summary(store):
    assert store.summary() == ""
    store.set("git", "repo", "loongcli")
    store.set("deploy", "target", "prod")
    s = store.summary()
    assert "git(1)" in s
    assert "deploy(1)" in s


def test_get_all(store):
    store.set("git", "repo", "loongcli")
    store.set("git", "branch", "main")
    assert store.get_all("git") == {"repo": "loongcli", "branch": "main"}
    assert store.get_all("nope") == {}


def test_persistence(tmp_path):
    path = tmp_path / "kv.json"
    s1 = MemoryStore(path=path)
    s1.set("git", "repo", "loongcli")
    s2 = MemoryStore(path=path)
    assert s2.get("git", "repo") == "loongcli"


def test_delete_last_key_removes_category(store):
    store.set("git", "repo", "loongcli")
    store.delete("git", "repo")
    assert "git" not in store.list_categories()


# --- Metadata tests ---

def test_entry_has_metadata(store):
    store.set("prefs", "lang", "Python", type="user")
    entry = store.get_entry("prefs", "lang")
    assert entry is not None
    assert entry.value == "Python"
    assert entry.type == "user"
    assert entry.created_at is not None
    assert entry.updated_at is not None


def test_entry_type_default(store):
    store.set("misc", "key", "val")
    entry = store.get_entry("misc", "key")
    assert entry.type == "project"


def test_entry_type_invalid_falls_back(store):
    store.set("misc", "key", "val", type="invalid_type")
    entry = store.get_entry("misc", "key")
    assert entry.type == "project"


def test_update_preserves_created_at(store):
    store.set("prefs", "lang", "Python", type="user")
    entry1 = store.get_entry("prefs", "lang")
    created = entry1.created_at

    store.set("prefs", "lang", "Rust", type="user")
    entry2 = store.get_entry("prefs", "lang")
    assert entry2.value == "Rust"
    assert entry2.created_at == created
    assert entry2.updated_at >= created


def test_persistence_with_metadata(tmp_path):
    path = tmp_path / "kv.json"
    s1 = MemoryStore(path=path)
    s1.set("prefs", "style", "concise", type="feedback")

    s2 = MemoryStore(path=path)
    entry = s2.get_entry("prefs", "style")
    assert entry.value == "concise"
    assert entry.type == "feedback"
    assert entry.created_at is not None


def test_backward_compat_plain_string(tmp_path):
    import json
    path = tmp_path / "kv.json"
    path.write_text(json.dumps({
        "git": {"repo": "loongcli", "branch": "main"},
    }), encoding="utf-8")

    store = MemoryStore(path=path)
    assert store.get("git", "repo") == "loongcli"
    assert store.get("git", "branch") == "main"
    entry = store.get_entry("git", "repo")
    assert entry.type == "project"


def test_all_entries(store):
    store.set("a", "k1", "v1", type="user")
    store.set("b", "k2", "v2", type="feedback")
    entries = store.all_entries()
    assert len(entries) == 2
    cats = {cat for cat, _, _ in entries}
    assert cats == {"a", "b"}


def test_format_for_prompt_empty(store):
    assert store.format_for_prompt() == ""


def test_format_for_prompt_priority_order(store):
    store.set("prefs", "style", "concise", type="reference")
    store.set("corrections", "no-comments", "don't add comments", type="feedback")
    store.set("role", "job", "backend engineer", type="user")

    text = store.format_for_prompt()
    lines = text.strip().split("\n")
    assert "[feedback]" in lines[0]
    assert "[user]" in lines[1]
    assert "[reference]" in lines[2]


def test_format_for_prompt_truncation(store):
    for i in range(100):
        store.set("bulk", f"key-{i}", "x" * 100, type="project")

    text = store.format_for_prompt(max_chars=500)
    assert "已截断" in text
    assert len(text) < 600
