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
