from loongcli.memory.conversation import ConversationStore, _project_sessions_dir, _path_to_slug


def test_path_to_slug(tmp_path):
    proj = tmp_path / "myproject"
    proj.mkdir()
    slug = _path_to_slug(proj)
    assert "myproject" in slug
    assert ":" not in slug
    assert "\\" not in slug


def test_project_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    dir_a = tmp_path / "project_a"
    dir_b = tmp_path / "project_b"
    dir_a.mkdir()
    dir_b.mkdir()

    path_a = _project_sessions_dir(dir_a)
    path_b = _project_sessions_dir(dir_b)
    assert path_a != path_b
    assert "projects" in str(path_a)

    cs_a = ConversationStore(project_dir=dir_a)
    cs_a.save([{"role": "user", "content": "in project A"}])

    cs_b = ConversationStore(project_dir=dir_b)
    cs_b.save([{"role": "user", "content": "in project B"}])

    assert len(cs_a.list_sessions()) == 1
    assert len(cs_b.list_sessions()) == 1
    assert cs_a.list_sessions()[0]["title"] == "in project A"
    assert cs_b.list_sessions()[0]["title"] == "in project B"


def test_save_and_load(tmp_path):
    cs = ConversationStore(base_dir=tmp_path)
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "hi there"},
    ]
    cs.save(messages)

    data = cs.load(cs.session_id)
    assert data is not None
    assert data["messages"] == messages
    assert data["meta"]["title"] == "hello world"
    assert data["meta"]["turn_count"] == 1


def test_list_sessions(tmp_path):
    cs1 = ConversationStore(base_dir=tmp_path)
    cs1.save([{"role": "user", "content": "first"}])
    cs2 = ConversationStore(base_dir=tmp_path)
    cs2.save([{"role": "user", "content": "second"}])

    sessions = cs2.list_sessions()
    assert len(sessions) == 2


def test_load_nonexistent(tmp_path):
    cs = ConversationStore(base_dir=tmp_path)
    assert cs.load("nonexistent") is None


def test_save_image_session_json_has_no_base64(tmp_path):
    """带图消息落盘：session JSON 只含 loongimg:// 引用，不含 base64（防膨胀）。"""
    from loongcli.core import messages as messages_mod
    from loongcli.core.messages import store_image

    p = tmp_path / "shot.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 1024)
    ref = store_image(str(p))

    store = ConversationStore(base_dir=tmp_path / "sessions")
    store.save([{"role": "user", "content": [
        {"type": "text", "text": "看图"},
        {"type": "image_url", "image_url": {"url": ref}},
    ]}])

    raw = store.session_path.read_text(encoding="utf-8")
    assert "data:image" not in raw
    assert "loongimg://" in raw
    # 往返一致：引用原样恢复
    loaded = store.load(store.session_id)
    assert loaded["messages"][0]["content"][1]["image_url"]["url"] == ref


def test_load_corrupt_file_returns_none(tmp_path):
    """损坏的会话 JSON 不应让 load/resume 抛异常，优雅降级为 None。"""
    store = ConversationStore(base_dir=tmp_path / "sessions")
    store.save([{"role": "user", "content": "hi"}])
    # 模拟非原子写崩溃残留：文件被截断成半个 JSON
    store.session_path.write_text('{"meta": {"session_id": "x"', encoding="utf-8")
    assert store.load(store.session_id) is None
    assert store.resume(store.session_id) is None
    assert store.resume_structured(store.session_id) is None


def test_save_is_atomic_no_tmp_left(tmp_path):
    """save 走 os.replace 原子落盘，成功后目录里不残留 .tmp。"""
    sessions = tmp_path / "sessions"
    store = ConversationStore(base_dir=sessions)
    store.save([{"role": "user", "content": "hi"}])
    store.save([{"role": "user", "content": "hi again"}])
    assert store.session_path.exists()
    leftover = [p.name for p in sessions.iterdir() if p.name.endswith(".tmp")]
    assert leftover == []
    # 内容仍可正常读回
    assert store.load(store.session_id)["messages"][-1]["content"] == "hi again"
