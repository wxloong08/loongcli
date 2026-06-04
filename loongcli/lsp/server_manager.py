from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from loongcli.lsp.client import JsonRpcClient
from loongcli.lsp.protocol import file_uri

logger = logging.getLogger(__name__)

EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".lua": "lua",
    ".zig": "zig",
}

KNOWN_SERVERS: dict[str, list[list[str]]] = {
    "python":     [["pylsp"], ["pyright-langserver", "--stdio"]],
    "typescript": [["typescript-language-server", "--stdio"]],
    "javascript": [["typescript-language-server", "--stdio"]],
    "go":         [["gopls", "serve"]],
    "rust":       [["rust-analyzer"]],
    "c":          [["clangd"]],
    "cpp":        [["clangd"]],
    "java":       [["jdtls"]],
    "ruby":       [["solargraph", "stdio"]],
    "php":        [["phpactor", "language-server"]],
    "csharp":     [["OmniSharp", "--languageserver"]],
    "lua":        [["lua-language-server"]],
    "zig":        [["zls"]],
}

_INIT_TIMEOUT = 60
_SUGGEST_INSTALL: dict[str, str] = {
    "python": "pip install python-lsp-server",
    "typescript": "npm i -g typescript-language-server typescript",
    "javascript": "npm i -g typescript-language-server typescript",
    "go": "go install golang.org/x/tools/gopls@latest",
    "rust": "rustup component add rust-analyzer",
    "c": "apt install clangd / brew install llvm",
    "cpp": "apt install clangd / brew install llvm",
    "lua": "brew install lua-language-server",
}


class LSPServerManager:
    def __init__(self, workspace_root: Path):
        self._workspace_root = workspace_root
        self._clients: dict[str, JsonRpcClient] = {}
        self._opened_docs: set[str] = set()
        self._doc_versions: dict[str, int] = {}
        self._diagnostics: dict[str, list[dict]] = {}

    @property
    def workspace_root(self) -> Path:
        return self._workspace_root

    def detect_language(self, file_path: str) -> str | None:
        ext = Path(file_path).suffix.lower()
        return EXTENSION_MAP.get(ext)

    def get_diagnostics(self, uri: str) -> list[dict]:
        return self._diagnostics.get(uri, [])

    async def invalidate_doc(self, file_path: str) -> None:
        """Send didClose and mark stale so next LSP access re-opens."""
        uri = file_uri(file_path)
        if uri not in self._opened_docs:
            return
        self._opened_docs.discard(uri)
        lang = self.detect_language(file_path)
        if lang and lang in self._clients:
            client = self._clients[lang]
            if client.alive:
                try:
                    await client.notify("textDocument/didClose", {
                        "textDocument": {"uri": uri},
                    })
                except Exception:
                    pass

    async def get_client(self, file_path: str) -> JsonRpcClient | None:
        lang = self.detect_language(file_path)
        if not lang:
            return None

        if lang in self._clients:
            client = self._clients[lang]
            if not client.alive:
                logger.info("LSP server for %s is dead, restarting", lang)
                del self._clients[lang]
            else:
                await self._ensure_doc_open(client, file_path)
                return client

        client = await self._start_server(lang)
        if client:
            await self._ensure_doc_open(client, file_path)
        return client

    def get_client_for_language(self, language: str) -> JsonRpcClient | None:
        client = self._clients.get(language)
        if client and not client.alive:
            del self._clients[language]
            return None
        return client

    async def ensure_server(self, language: str) -> JsonRpcClient | None:
        existing = self.get_client_for_language(language)
        if existing:
            return existing
        return await self._start_server(language)

    async def _start_server(self, language: str) -> JsonRpcClient | None:
        candidates = KNOWN_SERVERS.get(language, [])
        for cmd in candidates:
            if not shutil.which(cmd[0]):
                continue
            try:
                kwargs: dict = dict(
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                if sys.platform == "win32":
                    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                process = await asyncio.create_subprocess_exec(*cmd, **kwargs)
                client = JsonRpcClient(process)
                client.on_notification(self._handle_notification)
                await client.start()
                await self._initialize(client)
                self._clients[language] = client
                logger.info("Started LSP server for %s: %s", language, cmd)
                return client
            except Exception as e:
                logger.warning("Failed to start LSP server %s: %s", cmd, e)
                continue
        return None

    async def _initialize(self, client: JsonRpcClient):
        result = await client.request("initialize", {
            "processId": os.getpid(),
            "rootUri": self._workspace_root.as_uri(),
            "capabilities": {
                "textDocument": {
                    "definition": {"dynamicRegistration": False},
                    "references": {"dynamicRegistration": False},
                    "hover": {"contentFormat": ["plaintext", "markdown"]},
                    "documentSymbol": {"dynamicRegistration": False},
                    "publishDiagnostics": {"relatedInformation": True},
                },
                "workspace": {
                    "symbol": {"dynamicRegistration": False},
                },
            },
        }, timeout=_INIT_TIMEOUT)
        await client.notify("initialized", {})
        return result

    async def _ensure_doc_open(self, client: JsonRpcClient, file_path: str):
        uri = file_uri(file_path)
        if uri in self._opened_docs:
            return
        try:
            text = await asyncio.to_thread(
                Path(file_path).read_text, encoding="utf-8", errors="replace",
            )
        except Exception as e:
            logger.warning("Cannot read %s for LSP: %s", file_path, e)
            return
        lang_id = self.detect_language(file_path) or "plaintext"
        version = self._doc_versions.get(uri, 0) + 1
        self._doc_versions[uri] = version
        await client.notify("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": lang_id,
                "version": version,
                "text": text,
            },
        })
        self._opened_docs.add(uri)

    def _handle_notification(self, method: str, params: dict) -> None:
        if method == "textDocument/publishDiagnostics":
            uri = params.get("uri", "")
            self._diagnostics[uri] = params.get("diagnostics", [])

    def suggest_install(self, language: str) -> str:
        return _SUGGEST_INSTALL.get(language, f"Search for '{language} language server'")

    async def shutdown_all(self):
        tasks = [client.shutdown() for client in self._clients.values()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._clients.clear()
        self._opened_docs.clear()
        self._doc_versions.clear()
        self._diagnostics.clear()
