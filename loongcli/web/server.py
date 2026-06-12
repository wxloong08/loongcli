"""本地 Web 服务：只监听 127.0.0.1，路由分发到 WebAPI。"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from loongcli.web.api import WebAPI

_STATIC_DIR = Path(__file__).parent / "static"
_MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}
_MAX_BODY_BYTES = 1_000_000
_ALLOWED_HOSTS = ("127.0.0.1", "localhost")

DEFAULT_PORT = 8765
_PORT_RETRY = 20


class _Handler(BaseHTTPRequestHandler):
    api: WebAPI  # 由 create_server 注入到子类属性

    server_version = "loongcli-web"

    # ── 响应辅助 ─────────────────────────────────────────────

    def _send_json(self, status: int, data: dict):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_static(self, rel_path: str):
        target = (_STATIC_DIR / rel_path).resolve()
        static_root = _STATIC_DIR.resolve()
        if not target.is_relative_to(static_root) or not target.is_file():
            self._send_json(404, {"error": "not found"})
            return
        mime = _MIME_TYPES.get(target.suffix.lower())
        if mime is None:
            self._send_json(404, {"error": "not found"})
            return
        payload = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return None
        if length <= 0 or length > _MAX_BODY_BYTES:
            return None
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def log_message(self, format, *args):  # noqa: A002 — 基类签名
        pass  # 静默访问日志，避免污染 TUI 输出

    # ── 跨源防护 ─────────────────────────────────────────────
    # localhost 不是信任边界：恶意网页可以跨源 fetch 本服务。
    # Host 校验挡 DNS rebinding；写方法上 Origin 校验 + 强制 JSON
    # Content-Type（使跨源写必触发 preflight，而本服务不发 CORS 头）。

    def _host_allowed(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in _ALLOWED_HOSTS

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True  # 非浏览器客户端（curl 等）
        hostname = urlparse(origin).hostname
        return hostname is not None and hostname.lower() in _ALLOWED_HOSTS

    def _guard_write(self, has_body: bool) -> tuple[int, dict] | None:
        if not self._origin_allowed():
            return 403, {"error": "跨源请求被拒绝"}
        if has_body:
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                return 415, {"error": "Content-Type 必须是 application/json"}
        return None

    # ── 路由 ─────────────────────────────────────────────────

    def _route(self, method: str):
        if not self._host_allowed():
            self._send_json(403, {"error": "非法的 Host"})
            return
        if method != "GET":
            if bad := self._guard_write(has_body=method in ("POST", "PUT")):
                self._send_json(*bad)
                return
        parsed = urlparse(self.path)
        parts = [unquote(p) for p in parsed.path.split("/") if p]
        query = parse_qs(parsed.query)

        try:
            status, data = self._dispatch(method, parts, query)
        except Exception as exc:  # 兜底，避免线程内未捕获异常断连
            status, data = 500, {"error": f"内部错误: {exc}"}
        self._send_json(status, data)

    def _dispatch(self, method: str, parts: list[str], query: dict) -> tuple[int, dict]:
        # /api/projects[/{slug}/sessions[/{id}]] — 只读，只允许 GET
        if parts[:2] == ["api", "projects"]:
            if method != "GET":
                return 405, {"error": "会话接口只读"}
            if len(parts) == 2:
                return self.api.get_projects()
            if len(parts) == 4 and parts[3] == "sessions":
                limit = int(query.get("limit", ["50"])[0]) if query.get("limit", ["50"])[0].isdigit() else 50
                return self.api.get_sessions(parts[2], limit=limit)
            if len(parts) == 5 and parts[3] == "sessions":
                return self.api.get_session(parts[2], parts[4])
            return 404, {"error": "not found"}

        # /api/memories[/{name}]
        if parts[:2] == ["api", "memories"]:
            if len(parts) == 2:
                if method == "GET":
                    type_filter = query.get("type", [None])[0]
                    return self.api.list_memories(type_filter)
                if method == "POST":
                    body = self._read_body()
                    if body is None:
                        return 400, {"error": "请求体不是合法 JSON"}
                    return self.api.create_memory(body)
                return 405, {"error": "method not allowed"}
            if len(parts) == 3:
                name = parts[2]
                if method == "GET":
                    return self.api.get_memory(name)
                if method == "PUT":
                    body = self._read_body()
                    if body is None:
                        return 400, {"error": "请求体不是合法 JSON"}
                    return self.api.update_memory(name, body)
                if method == "DELETE":
                    return self.api.delete_memory(name)
                return 405, {"error": "method not allowed"}

        return 404, {"error": "not found"}

    # ── HTTP 方法入口 ────────────────────────────────────────

    def do_GET(self):
        if not self._host_allowed():
            self._send_json(403, {"error": "非法的 Host"})
            return
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self._send_static("index.html")
        elif path.startswith("/static/"):
            self._send_static(unquote(path[len("/static/"):]))
        elif path.startswith("/api/"):
            self._route("GET")
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        self._route("POST")

    def do_PUT(self):
        self._route("PUT")

    def do_DELETE(self):
        self._route("DELETE")


def create_server(api: WebAPI | None = None, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """创建只绑定 127.0.0.1 的服务。端口被占用时向上递增重试。"""
    api = api or WebAPI()
    handler = type("BoundHandler", (_Handler,), {"api": api})
    last_err: OSError | None = None
    for candidate in range(port, port + _PORT_RETRY + 1):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), handler)
            server.daemon_threads = True
            return server
        except OSError as exc:
            last_err = exc
    raise OSError(f"端口 {port}-{port + _PORT_RETRY} 全部被占用") from last_err


def server_url(server: ThreadingHTTPServer) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}/"
