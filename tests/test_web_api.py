"""Web API 测试：记忆 CRUD 全链路、会话只读、安全校验、路径穿越。"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from loongcli.memory.conversation import list_all_projects
from loongcli.memory.markdown_store import MarkdownMemoryStore
from loongcli.memory.memory_router import MemoryRouter
from loongcli.web.api import WebAPI
from loongcli.web.server import create_server, server_url


@pytest.fixture
def api(tmp_path):
    memory_dir = tmp_path / "memory"
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    return WebAPI(
        memory_store=MarkdownMemoryStore(memory_dir),
        projects_root=projects_root,
        current_dir=tmp_path,
    )


@pytest.fixture
def router_api(tmp_path):
    """MemoryRouter 背书的 WebAPI——运行时真实装配（main.py web / TUI /web）。"""
    projects_root = tmp_path / "projects"
    return WebAPI(
        memory_store=MemoryRouter(
            global_dir=tmp_path / "memory",
            project_dir=projects_root / "D-current" / "memory",
        ),
        projects_root=projects_root,
        current_dir=tmp_path,
    )


def make_session(projects_root, slug, session_id, title="测试会话", messages=None):
    sessions_dir = projects_root / slug / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "meta": {
            "session_id": session_id,
            "title": title,
            "created_at": "2026-06-12T00:00:00+00:00",
            "updated_at": "2026-06-12T00:00:00+00:00",
            "turn_count": 1,
        },
        "messages": messages or [{"role": "user", "content": "你好"}],
    }
    path = sessions_dir / f"{session_id}.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


# ── 记忆 CRUD ─────────────────────────────────────────────


class TestMemoryCRUD:
    def test_full_lifecycle(self, api):
        # 创建
        status, data = api.create_memory({
            "name": "test-topic", "description": "测试主题的描述",
            "type": "project", "content": "# 内容\n\n正文",
        })
        assert status == 201
        assert data["name"] == "test-topic"
        assert data["merged"] is False

        # 列表
        status, data = api.list_memories()
        assert status == 200
        assert len(data["memories"]) == 1

        # 详情
        status, data = api.get_memory("test-topic")
        assert status == 200
        assert data["content"] == "# 内容\n\n正文"

        # 更新
        status, data = api.update_memory("test-topic", {
            "description": "更新后的描述内容完全不同避免去重",
            "type": "feedback", "content": "新内容",
        })
        assert status == 200
        status, data = api.get_memory("test-topic")
        assert data["type"] == "feedback"
        assert data["content"] == "新内容"

        # 删除
        status, data = api.delete_memory("test-topic")
        assert status == 200
        status, _ = api.get_memory("test-topic")
        assert status == 404

    def test_delete_missing_returns_404(self, api):
        status, _ = api.delete_memory("not-exist")
        assert status == 404

    def test_update_missing_returns_404(self, api):
        status, _ = api.update_memory("not-exist", {
            "description": "x", "type": "project", "content": "y",
        })
        assert status == 404

    def test_dedup_merges_to_existing(self, api):
        api.create_memory({
            "name": "original", "description": "DeepSeek 缓存机制的工作原理",
            "type": "project", "content": "v1",
        })
        status, data = api.create_memory({
            "name": "duplicate", "description": "DeepSeek 缓存机制工作原理",
            "type": "project", "content": "v2",
        })
        assert status == 201
        assert data["name"] == "original"
        assert data["merged"] is True

    def test_type_filter(self, api):
        api.create_memory({"name": "a", "description": "用户偏好相关", "type": "user", "content": "x"})
        api.create_memory({"name": "b", "description": "项目进展相关", "type": "project", "content": "x"})
        status, data = api.list_memories("user")
        assert status == 200
        assert [m["name"] for m in data["memories"]] == ["a"]

    @pytest.mark.parametrize("name", ["../evil", "a b", "中文名", "a/b", ""])
    def test_invalid_name_rejected(self, api, name):
        status, _ = api.get_memory(name)
        assert status == 400
        status, _ = api.create_memory({
            "name": name, "description": "desc", "type": "project", "content": "c",
        })
        assert status == 400
        status, _ = api.delete_memory(name)
        assert status == 400

    def test_invalid_body_rejected(self, api):
        cases = [
            {"name": "x", "description": "", "type": "project", "content": "c"},
            {"name": "x", "description": "d", "type": "project", "content": "  "},
            {"name": "x", "description": "d", "type": "bogus", "content": "c"},
            {"name": "x", "description": "d" * 501, "type": "project", "content": "c"},
        ]
        for body in cases:
            status, _ = api.create_memory(body)
            assert status == 400, body

    def test_invalid_type_filter_rejected(self, api):
        status, _ = api.list_memories("bogus")
        assert status == 400


class TestMemoryCRUDWithRouter:
    """两层路由下的 CRUD：user 落全局库、其余落项目库，list/get 带 scope。"""

    def test_create_routes_by_type(self, router_api):
        router_api.create_memory({
            "name": "u1", "description": "用户身份与背景", "type": "user", "content": "x",
        })
        router_api.create_memory({
            "name": "p1", "description": "项目具体决策记录", "type": "project", "content": "x",
        })
        router = router_api.memory
        assert (router.global_store.base_dir / "u1.md").exists()
        assert (router.project_store.base_dir / "p1.md").exists()

    def test_list_and_get_carry_scope(self, router_api):
        router_api.create_memory({
            "name": "u1", "description": "用户身份与背景", "type": "user", "content": "x",
        })
        router_api.create_memory({
            "name": "p1", "description": "项目具体决策记录", "type": "project", "content": "x",
        })
        _, data = router_api.list_memories()
        scopes = {m["name"]: m["scope"] for m in data["memories"]}
        assert scopes == {"u1": "global", "p1": "project"}
        _, entry = router_api.get_memory("u1")
        assert entry["scope"] == "global"

    def test_update_type_change_moves_scope(self, router_api):
        router_api.create_memory({
            "name": "fact", "description": "某个事实的描述", "type": "project", "content": "v1",
        })
        status, _ = router_api.update_memory("fact", {
            "description": "某个事实的描述", "type": "user", "content": "v2",
        })
        assert status == 200
        router = router_api.memory
        assert (router.global_store.base_dir / "fact.md").exists()
        assert not (router.project_store.base_dir / "fact.md").exists()
        _, entry = router_api.get_memory("fact")
        assert entry["scope"] == "global"
        assert entry["content"] == "v2"

    def test_delete_falls_back_to_global(self, router_api):
        router_api.create_memory({
            "name": "u1", "description": "用户身份与背景", "type": "user", "content": "x",
        })
        status, _ = router_api.delete_memory("u1")
        assert status == 200
        status, _ = router_api.get_memory("u1")
        assert status == 404


# ── 会话（只读） ──────────────────────────────────────────


class TestSessions:
    def test_projects_listing_and_order(self, api):
        import os
        import time
        p1 = make_session(api.projects_root, "proj-old", "a" * 12)
        p2 = make_session(api.projects_root, "proj-new", "b" * 12)
        old = time.time() - 3600
        os.utime(p1, (old, old))

        status, data = api.get_projects()
        assert status == 200
        assert [p["slug"] for p in data["projects"]] == ["proj-new", "proj-old"]
        assert data["projects"][0]["session_count"] == 1
        assert "current" in data

    def test_empty_projects_skipped(self, api):
        (api.projects_root / "empty-proj" / "sessions").mkdir(parents=True)
        (api.projects_root / "no-sessions-dir").mkdir()
        assert list_all_projects(api.projects_root) == []

    def test_session_list_and_detail(self, api):
        make_session(api.projects_root, "proj", "c" * 12, title="标题甲")
        status, data = api.get_sessions("proj")
        assert status == 200
        assert data["sessions"][0]["title"] == "标题甲"
        assert data["sessions"][0]["file_size"] > 0

        status, data = api.get_session("proj", "c" * 12)
        assert status == 200
        assert data["messages"][0]["content"] == "你好"

    def test_corrupt_json_skipped(self, api):
        make_session(api.projects_root, "proj", "d" * 12)
        bad = api.projects_root / "proj" / "sessions" / ("e" * 12 + ".json")
        bad.write_text("{broken", encoding="utf-8")

        status, data = api.get_sessions("proj")
        assert status == 200
        assert len(data["sessions"]) == 1
        status, _ = api.get_session("proj", "e" * 12)
        assert status == 404

    @pytest.mark.parametrize("session_id", [
        "../../etc/passwd", "ABCDEF123456", "abc",  # 路径穿越 / 大写 / 太短
        "a" * 11, "a" * 37,                         # 低于 12 位 / 超过 36 位
        "a" * 12 + "/x",                            # 含路径分隔符
    ])
    def test_invalid_session_id_rejected(self, api, session_id):
        status, _ = api.get_session("proj", session_id)
        assert status == 400

    @pytest.mark.parametrize("slug", ["..", "../other", "a b", "中文"])
    def test_invalid_slug_rejected(self, api, slug):
        status, _ = api.get_sessions(slug)
        assert status == 400
        status, _ = api.get_session(slug, "a" * 12)
        assert status == 400

    def test_missing_session_404(self, api):
        status, _ = api.get_session("nope", "f" * 12)
        assert status == 404


# ── HTTP 层：路由 / 只读 / 路径穿越 ───────────────────────


@pytest.fixture
def http_server(api):
    server = create_server(api=api, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server_url(server).rstrip("/")
    server.shutdown()
    server.server_close()


def request(url, method="GET", body=None, headers=None):
    merged = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, method=method,
                                 data=json.dumps(body).encode() if body else None,
                                 headers=merged)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class TestHTTPServer:
    def test_index_served(self, http_server):
        status, body = request(f"{http_server}/")
        assert status == 200
        assert b"loongcli" in body

    def test_static_traversal_blocked(self, http_server):
        for path in ["/static/../api.py", "/static/..%2f..%2fapi.py", "/static/../../config.json"]:
            status, _ = request(f"{http_server}{path}")
            assert status == 404, path

    def test_unknown_extension_blocked(self, http_server):
        status, _ = request(f"{http_server}/static/no-such.py")
        assert status == 404

    def test_sessions_write_methods_405(self, http_server, api):
        make_session(api.projects_root, "proj", "9" * 12)
        for method in ["POST", "PUT", "DELETE"]:
            status, _ = request(f"{http_server}/api/projects/proj/sessions/{'9' * 12}", method=method)
            assert status == 405, method
        # GET 正常
        status, _ = request(f"{http_server}/api/projects/proj/sessions/{'9' * 12}")
        assert status == 200

    def test_memory_crud_over_http(self, http_server):
        status, body = request(f"{http_server}/api/memories", method="POST", body={
            "name": "http-test", "description": "经由 HTTP 创建", "type": "project", "content": "正文",
        })
        assert status == 201
        assert json.loads(body)["name"] == "http-test"

        status, body = request(f"{http_server}/api/memories/http-test")
        assert status == 200
        assert json.loads(body)["content"] == "正文"

        status, _ = request(f"{http_server}/api/memories/http-test", method="DELETE")
        assert status == 200

    def test_invalid_json_body_400(self, http_server):
        req = urllib.request.Request(f"{http_server}/api/memories", method="POST",
                                     data=b"{not json", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                status = resp.status
        except urllib.error.HTTPError as e:
            status = e.code
        assert status == 400

    def test_unknown_api_404(self, http_server):
        status, _ = request(f"{http_server}/api/bogus")
        assert status == 404


class TestCrossOriginGuards:
    """localhost 不是信任边界：恶意网页可跨源 fetch 本服务，必须在服务端拦截。"""

    BODY = {"name": "csrf-test", "description": "跨源写入测试", "type": "project", "content": "x"}

    def test_post_without_json_content_type_rejected(self, http_server):
        # HTML 表单可跨源发 text/plain POST 且不触发 preflight，必须 415
        status, _ = request(f"{http_server}/api/memories", method="POST",
                            body=self.BODY, headers={"Content-Type": "text/plain"})
        assert status == 415

    def test_write_with_evil_origin_rejected(self, http_server):
        status, _ = request(f"{http_server}/api/memories", method="POST",
                            body=self.BODY, headers={"Origin": "http://evil.example.com"})
        assert status == 403
        status, _ = request(f"{http_server}/api/memories/whatever", method="DELETE",
                            headers={"Origin": "http://evil.example.com"})
        assert status == 403

    def test_write_with_own_origin_allowed(self, http_server):
        origin = http_server  # http://127.0.0.1:<port>
        status, body = request(f"{http_server}/api/memories", method="POST",
                               body=self.BODY, headers={"Origin": origin})
        assert status == 201
        name = json.loads(body)["name"]
        status, _ = request(f"{http_server}/api/memories/{name}", method="DELETE",
                            headers={"Origin": origin})
        assert status == 200

    def test_evil_host_rejected(self, http_server):
        # DNS rebinding：Host 不是 127.0.0.1/localhost 一律拒绝（含只读 GET）。
        # urllib 不允许覆盖 Host 头，用裸 socket 构造请求。
        import socket
        port = int(http_server.rsplit(":", 1)[1])

        def raw_get(path, host):
            with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                sock.sendall(f"GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode())
                return sock.recv(64).decode(errors="replace").split(" ")[1]

        assert raw_get("/api/projects", "evil.example.com") == "403"
        assert raw_get("/", "evil.example.com:8765") == "403"
        assert raw_get("/api/projects", "localhost") == "200"
