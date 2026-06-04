from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class LSPError(Exception):
    def __init__(self, error: dict):
        self.code = error.get("code", -1)
        self.message = error.get("message", "Unknown LSP error")
        super().__init__(self.message)


class JsonRpcClient:
    """Async JSON-RPC 2.0 client over subprocess stdio (Content-Length framing)."""

    def __init__(self, process: asyncio.subprocess.Process):
        self._process = process
        self._request_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._alive = True
        self._notification_handler: Callable[[str, dict], None] | None = None

    @property
    def alive(self) -> bool:
        return self._alive and self._process.returncode is None

    def on_notification(self, handler: Callable[[str, dict], None]) -> None:
        self._notification_handler = handler

    async def start(self):
        self._reader_task = asyncio.create_task(self._read_loop())
        self._reader_task.add_done_callback(self._on_reader_done)

    def _on_reader_done(self, task: asyncio.Task) -> None:
        self._alive = False
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(LSPError({"message": "LSP server disconnected"}))
        self._pending.clear()

    async def request(self, method: str, params: dict, timeout: float = 30) -> dict:
        self._request_id += 1
        rid = self._request_id
        msg = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        future = asyncio.get_running_loop().create_future()
        self._pending[rid] = future
        await self._send(msg)
        return await asyncio.wait_for(future, timeout=timeout)

    async def notify(self, method: str, params: dict):
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        await self._send(msg)

    async def _send(self, msg: dict):
        body = json.dumps(msg).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self._process.stdin.write(header + body)
        await self._process.stdin.drain()

    async def _read_loop(self):
        reader = self._process.stdout
        while True:
            headers: dict[str, str] = {}
            while True:
                line = await reader.readline()
                if not line:
                    return
                decoded = line.decode("ascii", errors="replace").strip()
                if not decoded:
                    break
                if ":" in decoded:
                    key, value = decoded.split(":", 1)
                    headers[key.strip()] = value.strip()

            length_str = headers.get("Content-Length")
            if not length_str:
                continue
            content_length = int(length_str)
            body = await reader.readexactly(content_length)
            msg = json.loads(body)

            if "id" in msg and msg["id"] in self._pending:
                future = self._pending.pop(msg["id"])
                if "error" in msg:
                    future.set_exception(LSPError(msg["error"]))
                else:
                    future.set_result(msg.get("result"))
            elif "method" in msg and "id" not in msg:
                if self._notification_handler:
                    try:
                        self._notification_handler(msg["method"], msg.get("params", {}))
                    except Exception:
                        logger.debug("Notification handler error", exc_info=True)

    async def shutdown(self):
        if not self.alive:
            return
        try:
            await self.request("shutdown", {}, timeout=5)
            await self.notify("exit", {})
        except Exception:
            pass
        self._alive = False
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            self._process.kill()
        except ProcessLookupError:
            pass
