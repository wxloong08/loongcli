from __future__ import annotations

import asyncio
from pathlib import Path

from loongcli.tools.base import Tool, ToolRegistry
from loongcli.lsp.server_manager import LSPServerManager
from loongcli.lsp.protocol import (
    file_uri,
    make_text_document_position,
    format_locations,
    format_hover,
    format_symbols,
    format_diagnostics,
)


class LspGotoDefinitionTool(Tool):
    name = "lsp_goto_definition"
    description = (
        "Jump to a symbol's definition. Resolves imports, inheritance, and interface "
        "implementations — more precise than grep for navigating code."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file containing the symbol",
            },
            "line": {
                "type": "integer",
                "description": "Line number (1-based)",
            },
            "column": {
                "type": "integer",
                "description": "Column number (1-based)",
            },
        },
        "required": ["file_path", "line", "column"],
    }

    def __init__(self, manager: LSPServerManager):
        self._manager = manager

    async def execute(self, file_path: str, line: int, column: int) -> str:
        client = await self._manager.get_client(file_path)
        if client is None:
            return _no_server_msg(self._manager, file_path)
        params = make_text_document_position(file_path, line, column)
        try:
            result = await client.request("textDocument/definition", params)
        except Exception as e:
            return f"LSP error: {e}"
        return format_locations(result, self._manager.workspace_root)


class LspFindReferencesTool(Tool):
    name = "lsp_find_references"
    description = (
        "Find all references to a symbol across the workspace. "
        "Includes call sites, type references, and the declaration."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file containing the symbol",
            },
            "line": {
                "type": "integer",
                "description": "Line number (1-based)",
            },
            "column": {
                "type": "integer",
                "description": "Column number (1-based)",
            },
            "include_declaration": {
                "type": "boolean",
                "description": "Include the declaration itself (default: true)",
            },
        },
        "required": ["file_path", "line", "column"],
    }

    def __init__(self, manager: LSPServerManager):
        self._manager = manager

    async def execute(
        self, file_path: str, line: int, column: int,
        include_declaration: bool = True,
    ) -> str:
        client = await self._manager.get_client(file_path)
        if client is None:
            return _no_server_msg(self._manager, file_path)
        params = make_text_document_position(file_path, line, column)
        params["context"] = {"includeDeclaration": include_declaration}
        try:
            result = await client.request("textDocument/references", params)
        except Exception as e:
            return f"LSP error: {e}"
        return format_locations(result, self._manager.workspace_root)


class LspSymbolSearchTool(Tool):
    name = "lsp_symbol_search"
    description = (
        "Search for symbols (classes, functions, variables) by name across the "
        "workspace. Fuzzy-matches symbol names in all indexed files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Symbol name or partial name to search for",
            },
            "language": {
                "type": "string",
                "description": "Limit search to this language (optional)",
            },
        },
        "required": ["query"],
    }

    def __init__(self, manager: LSPServerManager):
        self._manager = manager

    async def execute(self, query: str, language: str | None = None) -> str:
        if language:
            client = await self._manager.ensure_server(language)
            if client is None:
                return f"No LSP server available for {language}. Install: {self._manager.suggest_install(language)}"
            try:
                result = await client.request("workspace/symbol", {"query": query})
            except Exception as e:
                return f"LSP error: {e}"
            return format_symbols(result, self._manager.workspace_root)

        async def _query_one(client):
            try:
                return await client.request("workspace/symbol", {"query": query})
            except Exception:
                return None

        live_clients = [c for c in self._manager._clients.values() if c.alive]
        if not live_clients:
            return f"No symbols matching '{query}' found. Ensure an LSP server is running (use any lsp_ tool on a file first)."
        raw = await asyncio.gather(*[_query_one(c) for c in live_clients])
        results: list[dict] = []
        for r in raw:
            if r:
                results.extend(r)
        if not results:
            return f"No symbols matching '{query}' found. Ensure an LSP server is running (use any lsp_ tool on a file first)."
        return format_symbols(results, self._manager.workspace_root)


class LspHoverTool(Tool):
    name = "lsp_hover"
    description = (
        "Get type information and documentation for a symbol. "
        "Shows type signature, docstring, and parameter info."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file containing the symbol",
            },
            "line": {
                "type": "integer",
                "description": "Line number (1-based)",
            },
            "column": {
                "type": "integer",
                "description": "Column number (1-based)",
            },
        },
        "required": ["file_path", "line", "column"],
    }

    def __init__(self, manager: LSPServerManager):
        self._manager = manager

    async def execute(self, file_path: str, line: int, column: int) -> str:
        client = await self._manager.get_client(file_path)
        if client is None:
            return _no_server_msg(self._manager, file_path)
        params = make_text_document_position(file_path, line, column)
        try:
            result = await client.request("textDocument/hover", params)
        except Exception as e:
            return f"LSP error: {e}"
        return format_hover(result)


class LspDiagnosticsTool(Tool):
    name = "lsp_diagnostics"
    description = (
        "Get diagnostic information (errors, warnings, hints) for a file. "
        "Diagnostics are collected from the LSP server's real-time analysis."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to check",
            },
        },
        "required": ["file_path"],
    }

    def __init__(self, manager: LSPServerManager):
        self._manager = manager

    async def execute(self, file_path: str) -> str:
        client = await self._manager.get_client(file_path)
        if client is None:
            return _no_server_msg(self._manager, file_path)
        uri = file_uri(file_path)
        diags = self._manager.get_diagnostics(uri)
        if not diags:
            # Server may not have pushed diagnostics yet; wait briefly
            for _ in range(5):
                await asyncio.sleep(0.2)
                diags = self._manager.get_diagnostics(uri)
                if diags:
                    break
        return format_diagnostics(diags, file_path)


def _no_server_msg(manager: LSPServerManager, file_path: str) -> str:
    lang = manager.detect_language(file_path)
    if not lang:
        ext = Path(file_path).suffix
        return f"LSP not supported for '{ext}' files"
    suggestion = manager.suggest_install(lang)
    return f"No LSP server found for {lang}. Install: {suggestion}"


LSP_TOOL_NAMES = frozenset({
    "lsp_goto_definition",
    "lsp_find_references",
    "lsp_symbol_search",
    "lsp_hover",
    "lsp_diagnostics",
})


def register_lsp_tools(registry: ToolRegistry, manager: LSPServerManager) -> None:
    registry.register(LspGotoDefinitionTool(manager))
    registry.register(LspFindReferencesTool(manager))
    registry.register(LspSymbolSearchTool(manager))
    registry.register(LspHoverTool(manager))
    registry.register(LspDiagnosticsTool(manager))
