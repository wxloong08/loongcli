"""Tests for _memory_section with MarkdownMemoryStore."""

import pytest
from pathlib import Path
from loongcli.core.prompts import _memory_section


@pytest.fixture
def md_store(tmp_path):
    from loongcli.memory.markdown_store import MarkdownMemoryStore

    store = MarkdownMemoryStore(base_dir=tmp_path)
    store.save(
        name="user-role",
        description="Python developer",
        type="user",
        content="Senior Python dev",
    )
    store.save(
        name="no-comments",
        description="Don't add comments",
        type="feedback",
        content="Stop commenting",
    )
    return store


def test_memory_section_includes_index(md_store):
    section = _memory_section(md_store)
    assert "# 记忆系统" in section
    assert "记忆索引" in section
    assert "user-role" in section
    assert "no-comments" in section


def test_memory_section_empty_store(tmp_path):
    from loongcli.memory.markdown_store import MarkdownMemoryStore

    store = MarkdownMemoryStore(base_dir=tmp_path)
    section = _memory_section(store)
    assert "# 记忆系统" in section
    assert "记忆索引" not in section


def test_memory_section_truncation(tmp_path):
    """Large number of memories doesn't blow up prompt."""
    from loongcli.memory.markdown_store import MarkdownMemoryStore

    store = MarkdownMemoryStore(base_dir=tmp_path)
    # 198 entries × ~160 bytes/line ≈ 31KB → triggers 25KB byte cap
    for i in range(198):
        store.save(
            name=f"mem-{i:04d}",
            description=f"Memory entry {i:04d} — detailed description with padding to ensure line length hits cap " + "x" * 80,
            type="project",
            content=f"Content body for memory {i}. " * 5,
        )
    section = _memory_section(store)
    assert "# 记忆系统" in section
    assert "记忆索引" in section
    # 200 entries with max 150-char lines ≈ 30KB → triggers 25KB byte cap
    assert "索引已截断" in section
    assert len(section) < 35000
