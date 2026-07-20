from __future__ import annotations

import pytest
from pathlib import Path

from loongcli.memory.markdown_store import MarkdownMemoryStore
from loongcli.memory.memory_router import MemoryRouter


@pytest.fixture
def global_dir(tmp_path: Path) -> Path:
    return tmp_path / "memory"


@pytest.fixture
def projects_root(tmp_path: Path) -> Path:
    return tmp_path / "projects"


@pytest.fixture
def project_dir(projects_root: Path) -> Path:
    return projects_root / "D-proj-a" / "memory"


@pytest.fixture
def router(global_dir: Path, project_dir: Path) -> MemoryRouter:
    return MemoryRouter(global_dir=global_dir, project_dir=project_dir)


def _make_other_project(projects_root: Path, slug: str, n: int) -> Path:
    """在指定 slug 下建一个含 n 条记忆的项目库，返回其 memory 目录。"""
    store = MarkdownMemoryStore(projects_root / slug / "memory")
    for i in range(n):
        store.save(f"{slug}-note-{i}", f"desc {slug} x{i}", "project", "body")
    return store.base_dir


def _unique_desc(i: int, n_tokens: int = 20) -> str:
    """token 互不相交的长描述——绕开 store 的 Jaccard 去重合并。"""
    return " ".join(f"t{i}w{j}" for j in range(n_tokens))


# ---- save：按 type 落对库 ----

def test_save_user_goes_global(router: MemoryRouter, global_dir: Path, project_dir: Path):
    router.save("who-am-i", "user profile", "user", "Python dev")
    assert (global_dir / "who-am-i.md").exists()
    assert not (project_dir / "who-am-i.md").exists()


@pytest.mark.parametrize("mem_type", ["project", "feedback", "reference"])
def test_save_non_user_goes_project(
    router: MemoryRouter, global_dir: Path, project_dir: Path, mem_type: str
):
    router.save(f"note-{mem_type}", "some fact", mem_type, "body")
    assert (project_dir / f"note-{mem_type}.md").exists()
    assert not (global_dir / f"note-{mem_type}.md").exists()


def test_save_unknown_type_goes_project(router: MemoryRouter, project_dir: Path):
    # store 会把非法 type 归一化为 project，路由也应落项目库
    router.save("weird", "unknown type", "banana", "body")
    mem = router.load("weird")
    assert mem is not None
    assert mem["type"] == "project"
    assert (project_dir / "weird.md").exists()


def test_save_returns_sanitized_name(router: MemoryRouter):
    assert router.save("my note!", "desc", "project", "body") == "my-note"


def test_save_type_change_moves_project_to_global(
    router: MemoryRouter, global_dir: Path, project_dir: Path
):
    # web 编辑改 type 的场景：name 即身份，换 type 即换库，旧库副本清掉
    router.save("fact", "some fact", "project", "v1")
    router.save("fact", "some fact", "user", "v2")
    assert (global_dir / "fact.md").exists()
    assert not (project_dir / "fact.md").exists()
    assert router.load("fact")["content"] == "v2"


def test_save_type_change_moves_global_to_project(
    router: MemoryRouter, global_dir: Path, project_dir: Path
):
    router.save("fact", "some fact", "user", "v1")
    router.save("fact", "some fact", "feedback", "v2")
    assert (project_dir / "fact.md").exists()
    assert not (global_dir / "fact.md").exists()
    assert router.load("fact")["content"] == "v2"


# ---- load：项目优先，回落全局 ----

def test_load_project_shadows_global(router: MemoryRouter):
    router.global_store.save("dup", "global copy", "user", "GLOBAL")
    router.project_store.save("dup", "project copy", "project", "PROJECT")
    mem = router.load("dup")
    assert mem is not None
    assert mem["content"] == "PROJECT"


def test_load_falls_back_to_global(router: MemoryRouter):
    router.save("g-only", "global fact", "user", "body")
    mem = router.load("g-only")
    assert mem is not None
    assert mem["description"] == "global fact"


def test_load_missing_returns_none(router: MemoryRouter):
    assert router.load("nope") is None


def test_load_carries_scope(router: MemoryRouter):
    router.save("g-fact", "global fact", "user", "body")
    router.save("p-fact", "project fact", "project", "body")
    assert router.load("g-fact")["scope"] == "global"
    assert router.load("p-fact")["scope"] == "project"


# ---- delete：项目优先，回落全局 ----

def test_delete_project_first_leaves_global(router: MemoryRouter):
    router.global_store.save("dup", "global copy", "user", "GLOBAL")
    router.project_store.save("dup", "project copy", "project", "PROJECT")
    assert router.delete("dup") is True
    mem = router.load("dup")  # 项目副本删掉后回落到全局副本
    assert mem is not None
    assert mem["content"] == "GLOBAL"


def test_delete_falls_back_to_global(router: MemoryRouter):
    router.save("g-only", "global fact", "user", "body")
    assert router.delete("g-only") is True
    assert router.load("g-only") is None


def test_delete_missing_returns_false(router: MemoryRouter):
    assert router.delete("nope") is False


# ---- list_all：合并当前项目 + 全局，不含其他项目 ----

def test_list_all_merges_with_scope(router: MemoryRouter):
    router.save("g-fact", "global fact", "user", "body")
    router.save("p-fact", "project fact", "project", "body")
    entries = {e["name"]: e for e in router.list_all()}
    assert entries["g-fact"]["scope"] == "global"
    assert entries["p-fact"]["scope"] == "project"


def test_list_all_excludes_other_projects(router: MemoryRouter, projects_root: Path):
    _make_other_project(projects_root, "D-proj-b", 2)
    router.save("mine", "my fact", "project", "body")
    names = {e["name"] for e in router.list_all()}
    assert names == {"mine"}


def test_list_all_type_filter_spans_both(router: MemoryRouter):
    router.save("g-fact", "global fact", "user", "body")
    router.save("p-fact", "project fact", "feedback", "body")
    assert {e["name"] for e in router.list_all(type_filter="user")} == {"g-fact"}
    assert {e["name"] for e in router.list_all(type_filter="feedback")} == {"p-fact"}


def test_list_all_name_collision_project_shadows(router: MemoryRouter):
    router.global_store.save("dup", "global copy", "user", "GLOBAL")
    router.project_store.save("dup", "project copy", "project", "PROJECT")
    matches = [e for e in router.list_all() if e["name"] == "dup"]
    assert len(matches) == 1
    assert matches[0]["scope"] == "project"


# ---- get_index：三段拼装 ----

def test_get_index_three_sections(router: MemoryRouter, projects_root: Path):
    other = _make_other_project(projects_root, "D-proj-b", 3)
    router.save("g-fact", "global user fact", "user", "body")
    router.save("p-fact", "current project fact", "project", "body")

    index = router.get_index()
    assert "### 全局记忆" in index
    assert "global user fact" in index
    assert "### 当前项目记忆" in index
    assert "current project fact" in index
    # 指针段：slug + 条数 + MEMORY.md 绝对路径
    assert "### 其他项目记忆" in index
    assert "D-proj-b：3 条" in index
    assert str(other / "MEMORY.md") in index


def test_get_index_pointer_excludes_current_project(router: MemoryRouter):
    router.save("p-fact", "current project fact", "project", "body")
    index = router.get_index()
    assert "### 其他项目记忆" not in index
    assert "D-proj-a" not in index


def test_get_index_skips_empty_other_projects(router: MemoryRouter, projects_root: Path):
    # 有 memory 目录但无索引的项目不产生指针行
    (projects_root / "D-proj-empty" / "memory").mkdir(parents=True)
    router.save("p-fact", "current project fact", "project", "body")
    assert "D-proj-empty" not in router.get_index()


def test_get_index_all_empty_returns_empty(router: MemoryRouter):
    assert router.get_index() == ""


def test_get_index_respects_max_bytes(router: MemoryRouter, projects_root: Path):
    # 两库各塞 90 条长描述条目（超各自段预算，触发段内截断），加若干指针项目
    for i in range(90):
        router.save(f"g-{i}", _unique_desc(i), "user", "body")
        router.save(f"p-{i}", _unique_desc(1000 + i), "project", "body")
    for k in range(5):
        _make_other_project(projects_root, f"D-proj-x{k}", 2)

    index = router.get_index(max_bytes=25_000)
    assert len(index.encode("utf-8")) <= 25_000
    # 截断后三段仍齐全
    assert "### 全局记忆" in index
    assert "### 其他项目记忆" in index
    assert "### 当前项目记忆" in index


# ---- index_is_complete：两段与门 ----

def test_index_complete_when_both_small(router: MemoryRouter):
    router.save("g-fact", "global fact", "user", "body")
    router.save("p-fact", "project fact", "project", "body")
    assert router.index_is_complete() is True


def test_index_incomplete_when_global_oversized(router: MemoryRouter):
    # 90 条 ×~150 字节/行 ≈ 13.5K，超过全局段预算（默认 25K 时 ~11.5K）
    for i in range(90):
        router.save(f"g-{i}", _unique_desc(i), "user", "body")
    router.save("p-fact", "project fact", "project", "body")
    assert router.index_is_complete() is False


def test_index_incomplete_when_project_oversized(router: MemoryRouter):
    router.save("g-fact", "global fact", "user", "body")
    for i in range(90):
        router.save(f"p-{i}", _unique_desc(i), "project", "body")
    assert router.index_is_complete() is False


def test_index_complete_true_when_both_empty(router: MemoryRouter):
    assert router.index_is_complete() is True
