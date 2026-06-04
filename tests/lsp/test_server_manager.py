from __future__ import annotations

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from loongcli.lsp.server_manager import (
    LSPServerManager,
    EXTENSION_MAP,
    KNOWN_SERVERS,
)
from loongcli.lsp.protocol import file_uri


class TestDetectLanguage:

    def test_python(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        assert mgr.detect_language("main.py") == "python"

    def test_typescript(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        assert mgr.detect_language("app.tsx") == "typescript"
        assert mgr.detect_language("app.ts") == "typescript"

    def test_javascript(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        assert mgr.detect_language("index.js") == "javascript"
        assert mgr.detect_language("component.jsx") == "javascript"

    def test_go(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        assert mgr.detect_language("main.go") == "go"

    def test_rust(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        assert mgr.detect_language("lib.rs") == "rust"

    def test_c_cpp(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        assert mgr.detect_language("main.c") == "c"
        assert mgr.detect_language("util.h") == "c"
        assert mgr.detect_language("app.cpp") == "cpp"
        assert mgr.detect_language("app.cc") == "cpp"

    def test_unknown(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        assert mgr.detect_language("data.csv") is None
        assert mgr.detect_language("Makefile") is None

    def test_case_insensitive(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        assert mgr.detect_language("Main.PY") == "python"
        assert mgr.detect_language("app.TS") == "typescript"


class TestExtensionMapCompleteness:

    def test_all_languages_have_servers(self):
        languages = set(EXTENSION_MAP.values())
        for lang in languages:
            assert lang in KNOWN_SERVERS, f"{lang} has no server entry"


class TestDiagnosticsCache:

    def test_empty_initially(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        assert mgr.get_diagnostics("file:///test.py") == []

    def test_handle_notification(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        mgr._handle_notification("textDocument/publishDiagnostics", {
            "uri": "file:///test.py",
            "diagnostics": [{"message": "error", "severity": 1}],
        })
        diags = mgr.get_diagnostics("file:///test.py")
        assert len(diags) == 1
        assert diags[0]["message"] == "error"

    def test_update_diagnostics(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        mgr._handle_notification("textDocument/publishDiagnostics", {
            "uri": "file:///test.py",
            "diagnostics": [{"message": "error1"}],
        })
        mgr._handle_notification("textDocument/publishDiagnostics", {
            "uri": "file:///test.py",
            "diagnostics": [{"message": "error2"}, {"message": "error3"}],
        })
        diags = mgr.get_diagnostics("file:///test.py")
        assert len(diags) == 2

    def test_ignores_other_notifications(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        mgr._handle_notification("window/logMessage", {"message": "hi"})
        assert mgr.get_diagnostics("file:///test.py") == []


class TestInvalidateDoc:

    @pytest.mark.asyncio
    async def test_invalidate_removes_from_opened(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        f = tmp_path / "test.py"
        f.write_text("x = 1")
        uri = file_uri(str(f))
        mgr._opened_docs.add(uri)
        assert uri in mgr._opened_docs

        await mgr.invalidate_doc(str(f))
        assert uri not in mgr._opened_docs

    @pytest.mark.asyncio
    async def test_invalidate_nonexistent_is_safe(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        await mgr.invalidate_doc("nonexistent.py")  # Should not raise

    @pytest.mark.asyncio
    async def test_invalidate_sends_didclose(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        f = tmp_path / "test.py"
        f.write_text("x = 1")
        uri = file_uri(str(f))
        mgr._opened_docs.add(uri)

        fake_client = MagicMock()
        fake_client.alive = True
        fake_client.notify = AsyncMock()
        mgr._clients["python"] = fake_client

        await mgr.invalidate_doc(str(f))

        fake_client.notify.assert_called_once_with(
            "textDocument/didClose",
            {"textDocument": {"uri": uri}},
        )

    @pytest.mark.asyncio
    async def test_invalidate_skips_if_not_opened(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        fake_client = MagicMock()
        fake_client.alive = True
        fake_client.notify = AsyncMock()
        mgr._clients["python"] = fake_client

        await mgr.invalidate_doc("test.py")
        fake_client.notify.assert_not_called()


class TestWorkspaceRoot:

    def test_property(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        assert mgr.workspace_root == tmp_path


class TestDocVersionTracking:

    @pytest.mark.asyncio
    async def test_version_increments_on_reopen(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        f = tmp_path / "test.py"
        f.write_text("x = 1")
        uri = file_uri(str(f))

        client = MagicMock()
        client.alive = True
        client.notify = AsyncMock()

        await mgr._ensure_doc_open(client, str(f))
        assert mgr._doc_versions[uri] == 1

        mgr._opened_docs.discard(uri)  # simulate invalidate
        await mgr._ensure_doc_open(client, str(f))
        assert mgr._doc_versions[uri] == 2


class TestSuggestInstall:

    def test_known_language(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        assert "pip install" in mgr.suggest_install("python")
        assert "npm" in mgr.suggest_install("typescript")
        assert "gopls" in mgr.suggest_install("go")

    def test_unknown_language(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        result = mgr.suggest_install("cobol")
        assert "language server" in result


class TestGetClient:

    @pytest.mark.asyncio
    async def test_unsupported_language_returns_none(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        result = await mgr.get_client("data.csv")
        assert result is None

    @pytest.mark.asyncio
    async def test_no_server_available(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        with patch("shutil.which", return_value=None):
            result = await mgr.get_client("test.py")
        assert result is None

    @pytest.mark.asyncio
    async def test_dead_client_gets_removed(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        fake_client = MagicMock()
        fake_client.alive = False
        mgr._clients["python"] = fake_client

        with patch("shutil.which", return_value=None):
            result = await mgr.get_client("test.py")
        assert result is None
        assert "python" not in mgr._clients

    @pytest.mark.asyncio
    async def test_cached_client_reused(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        f = tmp_path / "test.py"
        f.write_text("x = 1")

        fake_client = MagicMock()
        fake_client.alive = True
        fake_client.notify = AsyncMock()
        mgr._clients["python"] = fake_client

        result = await mgr.get_client(str(f))
        assert result is fake_client


class TestShutdownAll:

    @pytest.mark.asyncio
    async def test_shutdown_clears_state(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        fake_client = MagicMock()
        fake_client.shutdown = AsyncMock()
        mgr._clients["python"] = fake_client
        mgr._opened_docs.add("file:///test.py")
        mgr._diagnostics["file:///test.py"] = [{"msg": "err"}]

        await mgr.shutdown_all()

        assert len(mgr._clients) == 0
        assert len(mgr._opened_docs) == 0
        assert len(mgr._diagnostics) == 0
        fake_client.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_empty_is_safe(self, tmp_path):
        mgr = LSPServerManager(tmp_path)
        await mgr.shutdown_all()  # Should not raise
