import json
import pytest
from pathlib import Path
from loongcli.memory.migrate import migrate_kv_to_markdown
from loongcli.memory.markdown_store import MarkdownMemoryStore


@pytest.fixture
def old_kv(tmp_path):
    kv_path = tmp_path / "kv.json"
    data = {
        "preferences": {
            "lang": {
                "value": "Python is my main language",
                "type": "user",
                "created_at": "2026-05-30T10:00:00+00:00",
                "updated_at": "2026-06-01T08:00:00+00:00",
            },
            "style": {
                "value": "I prefer concise answers",
                "type": "feedback",
                "created_at": "2026-05-31T10:00:00+00:00",
                "updated_at": "2026-05-31T10:00:00+00:00",
            },
        },
        "project": {
            "db-choice": {
                "value": "Using SQLite for simplicity",
                "type": "project",
                "created_at": "2026-06-01T12:00:00+00:00",
                "updated_at": "2026-06-01T12:00:00+00:00",
            },
        },
    }
    kv_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return tmp_path


def test_migration_creates_md_files(old_kv):
    count = migrate_kv_to_markdown(old_kv)
    assert count == 3
    store = MarkdownMemoryStore(base_dir=old_kv)
    assert store.load("preferences-lang") is not None
    assert store.load("preferences-style") is not None
    assert store.load("project-db-choice") is not None


def test_migration_preserves_metadata(old_kv):
    migrate_kv_to_markdown(old_kv)
    store = MarkdownMemoryStore(base_dir=old_kv)
    mem = store.load("preferences-lang")
    assert mem["type"] == "user"
    assert mem["created_at"] == "2026-05-30T10:00:00+00:00"
    assert "Python is my main language" in mem["content"]


def test_migration_renames_kv_json(old_kv):
    migrate_kv_to_markdown(old_kv)
    assert not (old_kv / "kv.json").exists()
    assert (old_kv / "kv.json.bak").exists()


def test_migration_idempotent(old_kv):
    count1 = migrate_kv_to_markdown(old_kv)
    assert count1 == 3
    count2 = migrate_kv_to_markdown(old_kv)
    assert count2 == 0  # kv.json already renamed to .bak


def test_migration_no_kv_file(tmp_path):
    count = migrate_kv_to_markdown(tmp_path)
    assert count == 0


def test_migration_builds_index(old_kv):
    migrate_kv_to_markdown(old_kv)
    index = (old_kv / "MEMORY.md").read_text(encoding="utf-8")
    assert "preferences-lang" in index
    assert "preferences-style" in index


def test_migration_plain_string_values(tmp_path):
    kv_path = tmp_path / "kv.json"
    data = {"git": {"repo": "loongcli", "branch": "main"}}
    kv_path.write_text(json.dumps(data), encoding="utf-8")
    count = migrate_kv_to_markdown(tmp_path)
    assert count == 2
    store = MarkdownMemoryStore(base_dir=tmp_path)
    mem = store.load("git-repo")
    assert mem is not None
    assert "loongcli" in mem["content"]
