from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from loongcli.lsp.client import JsonRpcClient, LSPError


def _encode_message(msg: dict) -> bytes:
    """Encode a JSON-RPC message with Content-Length framing."""
    body = json.dumps(msg).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _encode_message_multi_header(msg: dict) -> bytes:
    """Encode with multiple headers (like some LSP servers do)."""
    body = json.dumps(msg).encode("utf-8")
    header = (
        f"Content-Length: {len(body)}\r\n"
        f"Content-Type: application/vscode-jsonrpc; charset=utf-8\r\n"
        f"\r\n"
    ).encode("ascii")
    return header + body


class FakeStreamReader:
    """Simulates asyncio.StreamReader for testing."""

    def __init__(self, data: bytes = b""):
        self._buffer = bytearray(data)

    def feed(self, data: bytes):
        self._buffer.extend(data)

    async def readline(self) -> bytes:
        while b"\n" not in self._buffer:
            if not self._buffer:
                return b""
            await asyncio.sleep(0)
        idx = self._buffer.index(b"\n") + 1
        line = bytes(self._buffer[:idx])
        del self._buffer[:idx]
        return line

    async def readexactly(self, n: int) -> bytes:
        while len(self._buffer) < n:
            await asyncio.sleep(0)
        data = bytes(self._buffer[:n])
        del self._buffer[:n]
        return data


class FakeStreamWriter:
    def __init__(self):
        self.data = bytearray()

    def write(self, data: bytes):
        self.data.extend(data)

    async def drain(self):
        pass


def _make_process(reader_data: bytes = b""):
    process = MagicMock()
    process.returncode = None
    process.stdout = FakeStreamReader(reader_data)
    process.stdin = FakeStreamWriter()
    process.kill = MagicMock()
    return process


class TestJsonRpcClient:

    @pytest.mark.asyncio
    async def test_send_request(self):
        response = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        data = _encode_message(response)
        process = _make_process(data)
        client = JsonRpcClient(process)
        await client.start()

        result = await client.request("initialize", {"processId": 1})
        assert result == {"capabilities": {}}

        # Verify sent data
        sent = bytes(process.stdin.data)
        assert b"Content-Length:" in sent
        assert b'"method": "initialize"' in sent

    @pytest.mark.asyncio
    async def test_send_notification(self):
        process = _make_process()
        client = JsonRpcClient(process)
        # Don't start reader — just test send
        await client.notify("initialized", {})

        sent = bytes(process.stdin.data)
        assert b'"method": "initialized"' in sent
        assert b'"id"' not in sent

    @pytest.mark.asyncio
    async def test_multi_header_parsing(self):
        response = {"jsonrpc": "2.0", "id": 1, "result": "ok"}
        data = _encode_message_multi_header(response)
        process = _make_process(data)
        client = JsonRpcClient(process)
        await client.start()

        result = await client.request("test", {})
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_error_response(self):
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32601, "message": "Method not found"},
        }
        data = _encode_message(response)
        process = _make_process(data)
        client = JsonRpcClient(process)
        await client.start()

        with pytest.raises(LSPError) as exc_info:
            await client.request("unknown", {})
        assert "Method not found" in str(exc_info.value)
        assert exc_info.value.code == -32601

    @pytest.mark.asyncio
    async def test_notification_callback(self):
        notif = {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {"uri": "file:///test.py", "diagnostics": []},
        }
        data = _encode_message(notif)
        process = _make_process(data)
        client = JsonRpcClient(process)

        received = []
        client.on_notification(lambda method, params: received.append((method, params)))

        await client.start()
        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0][0] == "textDocument/publishDiagnostics"
        assert received[0][1]["uri"] == "file:///test.py"

    @pytest.mark.asyncio
    async def test_server_request_gets_method_not_found_reply(self):
        # 服务器发起的请求（带 method 且带 id）——client 必须回响应，否则服务器阻塞
        server_req = {
            "jsonrpc": "2.0",
            "id": "server-1",
            "method": "workspace/configuration",
            "params": {"items": []},
        }
        data = _encode_message(server_req)
        process = _make_process(data)
        client = JsonRpcClient(process)
        await client.start()
        await asyncio.sleep(0.1)

        sent = bytes(process.stdin.data).decode("utf-8")
        assert '"id": "server-1"' in sent
        assert "-32601" in sent
        assert '"error"' in sent

    @pytest.mark.asyncio
    async def test_request_timeout(self):
        # Use a reader that blocks forever (never returns empty)
        class BlockingReader:
            async def readline(self) -> bytes:
                await asyncio.sleep(10)
                return b""

            async def readexactly(self, n: int) -> bytes:
                await asyncio.sleep(10)
                return b""

        process = _make_process(b"")
        process.stdout = BlockingReader()
        client = JsonRpcClient(process)
        await client.start()

        with pytest.raises(asyncio.TimeoutError):
            await client.request("slow", {}, timeout=0.1)

    @pytest.mark.asyncio
    async def test_alive_property(self):
        process = _make_process(b"")
        process.returncode = None
        client = JsonRpcClient(process)
        assert client.alive is True

        process.returncode = 1
        assert client.alive is False

    @pytest.mark.asyncio
    async def test_alive_after_disconnect(self):
        process = _make_process(b"")
        client = JsonRpcClient(process)
        await client.start()
        await asyncio.sleep(0.1)
        # Reader exits because buffer is empty -> alive should become False
        assert client.alive is False

    @pytest.mark.asyncio
    async def test_shutdown(self):
        # Response to shutdown request
        response = {"jsonrpc": "2.0", "id": 1, "result": None}
        data = _encode_message(response)
        process = _make_process(data)
        client = JsonRpcClient(process)
        await client.start()

        await client.shutdown()
        assert client.alive is False

    @pytest.mark.asyncio
    async def test_shutdown_already_dead(self):
        process = _make_process(b"")
        process.returncode = 1
        client = JsonRpcClient(process)
        client._alive = False
        await client.shutdown()  # Should not raise

    @pytest.mark.asyncio
    async def test_pending_futures_resolved_on_disconnect(self):
        process = _make_process(b"")
        client = JsonRpcClient(process)
        await client.start()

        # Start a request that will never get a response
        task = asyncio.create_task(client.request("test", {}, timeout=5))
        await asyncio.sleep(0.1)

        # Reader exits, pending futures should be resolved with error
        with pytest.raises(LSPError, match="disconnected"):
            await task


class TestLSPError:

    def test_fields(self):
        err = LSPError({"code": -32600, "message": "Invalid Request"})
        assert err.code == -32600
        assert err.message == "Invalid Request"
        assert "Invalid Request" in str(err)

    def test_defaults(self):
        err = LSPError({})
        assert err.code == -1
        assert err.message == "Unknown LSP error"
