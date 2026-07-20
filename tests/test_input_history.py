"""持久输入历史（方向键翻项目历史消息）的单测。

背景：此前 PromptSession 用 InMemoryHistory，只活在进程内——新会话方向键
历史为空。改为项目级 FileHistory + 首次一次性回填历史会话的用户消息。
"""
import pytest
from pathlib import Path

from loongcli.memory.conversation import ConversationStore, input_history_path
from loongcli.tui.app import TUI


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    return tmp_path


def test_input_history_path_per_project(fake_home):
    path = input_history_path()
    # 与 sessions 平级、按项目 slug 隔离
    assert path.name == "input_history"
    assert path.parent == ConversationStore().base_dir.parent


def test_backfill_from_past_sessions(fake_home):
    store = ConversationStore()
    store.save([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "第一条历史输入"},
        {"role": "assistant", "content": "回复"},
        {"role": "user", "content": "[compact-boundary] 内部消息应被过滤"},
        {"role": "user", "content": "第二条历史输入"},
    ])

    tui = TUI()
    tui._backfill_input_history(store)

    path = input_history_path()
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "第一条历史输入" in text
    assert "第二条历史输入" in text
    assert "内部消息" not in text


def test_backfill_noop_when_file_exists(fake_home):
    store = ConversationStore()
    store.save([{"role": "user", "content": "历史输入"}])

    tui = TUI()
    tui._backfill_input_history(store)
    first = input_history_path().read_text(encoding="utf-8")
    # 再跑一次不应重复追加（正常输入由 FileHistory 实时落盘）
    tui._backfill_input_history(store)
    assert input_history_path().read_text(encoding="utf-8") == first


def test_backfill_without_store_is_safe(fake_home):
    TUI()._backfill_input_history(None)
    assert not input_history_path().exists()
