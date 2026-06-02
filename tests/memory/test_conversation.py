from loongcli.memory.conversation import ConversationStore


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
