from __future__ import annotations

import pytest
from pathlib import Path
from loongcli.memory.markdown_store import MarkdownMemoryStore


@pytest.fixture
def store(tmp_path: Path) -> MarkdownMemoryStore:
    return MarkdownMemoryStore(base_dir=tmp_path)


# ---- 1. save creates file ----

def test_save_creates_md_file(store: MarkdownMemoryStore, tmp_path: Path):
    store.save("my-note", "A test note", "project", "Hello world")
    assert (tmp_path / "my-note.md").exists()


# ---- 2. save writes frontmatter ----

def test_save_writes_frontmatter(store: MarkdownMemoryStore, tmp_path: Path):
    store.save("my-note", "A test note", "project", "Hello world")
    content = (tmp_path / "my-note.md").read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert "name: my-note" in content
    assert "description: A test note" in content
    assert "type: project" in content
    assert "created_at:" in content
    assert "updated_at:" in content
    assert "Hello world" in content


# ---- 3. load round-trip ----

def test_load_returns_memory(store: MarkdownMemoryStore):
    store.save("roundtrip", "desc here", "user", "Body text")
    mem = store.load("roundtrip")
    assert mem is not None
    assert mem["name"] == "roundtrip"
    assert mem["description"] == "desc here"
    assert mem["type"] == "user"
    assert mem["content"] == "Body text"
    assert "created_at" in mem
    assert "updated_at" in mem


# ---- 4. load nonexistent ----

def test_load_nonexistent(store: MarkdownMemoryStore):
    assert store.load("does-not-exist") is None


# ---- 5. update overwrites content, preserves created_at ----

def test_update_overwrites_content(store: MarkdownMemoryStore):
    store.save("updatable", "first version", "project", "Content v1")
    mem1 = store.load("updatable")
    created = mem1["created_at"]

    store.save("updatable", "second version", "project", "Content v2")
    mem2 = store.load("updatable")
    assert mem2["content"] == "Content v2"
    assert mem2["description"] == "second version"
    assert mem2["created_at"] == created
    assert mem2["updated_at"] >= created


# ---- 6. delete ----

def test_delete(store: MarkdownMemoryStore, tmp_path: Path):
    store.save("to-delete", "will be deleted", "project", "Bye")
    assert store.delete("to-delete") is True
    assert not (tmp_path / "to-delete.md").exists()
    assert store.load("to-delete") is None


def test_delete_nonexistent(store: MarkdownMemoryStore):
    assert store.delete("ghost") is False


# ---- 7. list_all ----

def test_list_all(store: MarkdownMemoryStore):
    store.save("note-a", "desc a", "user", "aaa")
    store.save("note-b", "desc b", "project", "bbb")
    store.save("note-c", "desc c", "feedback", "ccc")
    items = store.list_all()
    names = {item["name"] for item in items}
    assert names == {"note-a", "note-b", "note-c"}
    assert len(items) == 3


# ---- 8. list by type ----

def test_list_by_type(store: MarkdownMemoryStore):
    store.save("note-a", "desc a", "user", "aaa")
    store.save("note-b", "desc b", "project", "bbb")
    store.save("note-c", "desc c", "user", "ccc")
    users = store.list_all(type_filter="user")
    assert len(users) == 2
    assert all(item["type"] == "user" for item in users)


# ---- 9. invalid type defaults to project ----

def test_invalid_type_defaults_to_project(store: MarkdownMemoryStore):
    store.save("bad-type", "desc", "banana", "content")
    mem = store.load("bad-type")
    assert mem["type"] == "project"


# ---- 10. name sanitization ----

def test_name_sanitization(store: MarkdownMemoryStore, tmp_path: Path):
    returned = store.save("hello world!@#$%", "desc", "project", "body")
    assert returned == "hello-world"
    assert (tmp_path / "hello-world.md").exists()
    mem = store.load("hello-world")
    assert mem is not None
    assert mem["content"] == "body"


def test_name_sanitization_unicode(store: MarkdownMemoryStore, tmp_path: Path):
    returned = store.save("my/file\\name:test", "desc", "project", "body")
    # slashes, backslashes, colons replaced with dashes
    assert "/" not in returned
    assert "\\" not in returned
    assert ":" not in returned
    assert (tmp_path / f"{returned}.md").exists()


# ---- 11. index rebuilt on save ----

def test_index_rebuilt_on_save(store: MarkdownMemoryStore, tmp_path: Path):
    store.save("indexed", "my description", "project", "content")
    index_path = tmp_path / "MEMORY.md"
    assert index_path.exists()
    index_text = index_path.read_text(encoding="utf-8")
    assert "indexed" in index_text
    assert "my description" in index_text


def test_index_rebuilt_on_delete(store: MarkdownMemoryStore, tmp_path: Path):
    store.save("keep", "keeper", "project", "content")
    store.save("remove", "remover", "project", "content")
    store.delete("remove")
    index_text = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    assert "keep" in index_text
    assert "remove" not in index_text


def test_index_line_capped_to_150(store: MarkdownMemoryStore, tmp_path: Path):
    store.save("my-memory", "x" * 5000, "project", "body")
    index_text = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    lines = index_text.strip().split("\n")
    for line in lines:
        assert len(line) <= 150


# ---- 12. get_index empty ----

def test_get_index_empty(store: MarkdownMemoryStore):
    assert store.get_index() == ""


# ---- 13. get_index truncation ----

def test_get_index_truncation(store: MarkdownMemoryStore):
    for i in range(100):
        store.save(f"note-{i:03d}", f"description for note {i} " + "x" * 50, "project", "body")
    index = store.get_index(max_bytes=500)
    raw_bytes = index.encode("utf-8")
    assert len(raw_bytes) <= 500 + 300
    assert "索引已截断" in index


# ---- Extra edge cases ----

def test_save_returns_sanitized_name(store: MarkdownMemoryStore):
    result = store.save("Simple-Name_123", "desc", "project", "body")
    assert result == "Simple-Name_123"


def test_content_with_frontmatter_delimiters(store: MarkdownMemoryStore):
    """Content containing --- should not break parsing."""
    store.save("tricky", "desc", "project", "Line 1\n---\nLine 2\n---\nLine 3")
    mem = store.load("tricky")
    assert mem is not None
    assert "Line 1" in mem["content"]
    assert "Line 3" in mem["content"]


def test_description_with_colon(store: MarkdownMemoryStore):
    """Descriptions containing colons should survive round-trip."""
    store.save("colon-test", "key: value style desc", "project", "body")
    mem = store.load("colon-test")
    assert mem["description"] == "key: value style desc"


def test_list_all_excludes_memory_index(store: MarkdownMemoryStore, tmp_path: Path):
    """MEMORY.md should never appear in list_all results."""
    store.save("real-note", "desc", "project", "body")
    items = store.list_all()
    names = {item["name"] for item in items}
    assert "MEMORY" not in names
    assert "real-note" in names


def test_persistence_across_instances(tmp_path: Path):
    """Data written by one instance is readable by another."""
    s1 = MarkdownMemoryStore(base_dir=tmp_path)
    s1.save("persist", "persistent note", "feedback", "important content")

    s2 = MarkdownMemoryStore(base_dir=tmp_path)
    mem = s2.load("persist")
    assert mem is not None
    assert mem["content"] == "important content"
    assert mem["type"] == "feedback"
