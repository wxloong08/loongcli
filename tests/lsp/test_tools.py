from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from loongcli.lsp.tools import (
    LspGotoDefinitionTool,
    LspFindReferencesTool,
    LspSymbolSearchTool,
    LspHoverTool,
    LspDiagnosticsTool,
    LSP_TOOL_NAMES,
    register_lsp_tools,
)
from loongcli.lsp.server_manager import LSPServerManager
from loongcli.tools.base import ToolRegistry


def _make_manager(tmp_path: Path) -> LSPServerManager:
    return LSPServerManager(tmp_path)


def _mock_client():
    client = MagicMock()
    client.alive = True
    client.request = AsyncMock()
    client.notify = AsyncMock()
    return client


class TestLspGotoDefinition:

    @pytest.mark.asyncio
    async def test_no_server(self, tmp_path):
        mgr = _make_manager(tmp_path)
        tool = LspGotoDefinitionTool(mgr)
        result = await tool.execute(file_path="test.csv", line=1, column=1)
        assert "not supported" in result

    @pytest.mark.asyncio
    async def test_no_server_for_language(self, tmp_path):
        mgr = _make_manager(tmp_path)
        tool = LspGotoDefinitionTool(mgr)
        with patch("shutil.which", return_value=None):
            result = await tool.execute(file_path="test.py", line=1, column=1)
        assert "No LSP server" in result
        assert "pip install" in result

    @pytest.mark.asyncio
    async def test_success(self, tmp_path):
        mgr = _make_manager(tmp_path)
        client = _mock_client()
        client.request.return_value = {
            "uri": f"file:///{tmp_path}/target.py",
            "range": {"start": {"line": 9, "character": 0}},
        }
        mgr._clients["python"] = client

        f = tmp_path / "test.py"
        f.write_text("import foo")

        tool = LspGotoDefinitionTool(mgr)
        result = await tool.execute(file_path=str(f), line=1, column=8)
        assert "target.py:10:1" in result

    @pytest.mark.asyncio
    async def test_lsp_error(self, tmp_path):
        mgr = _make_manager(tmp_path)
        client = _mock_client()
        client.request.side_effect = Exception("Server crashed")
        mgr._clients["python"] = client

        f = tmp_path / "test.py"
        f.write_text("x = 1")

        tool = LspGotoDefinitionTool(mgr)
        result = await tool.execute(file_path=str(f), line=1, column=1)
        assert "LSP error" in result

    @pytest.mark.asyncio
    async def test_no_results(self, tmp_path):
        mgr = _make_manager(tmp_path)
        client = _mock_client()
        client.request.return_value = []
        mgr._clients["python"] = client

        f = tmp_path / "test.py"
        f.write_text("x = 1")

        tool = LspGotoDefinitionTool(mgr)
        result = await tool.execute(file_path=str(f), line=1, column=1)
        assert "No results" in result


class TestLspFindReferences:

    @pytest.mark.asyncio
    async def test_success(self, tmp_path):
        mgr = _make_manager(tmp_path)
        client = _mock_client()
        client.request.return_value = [
            {"uri": f"file:///{tmp_path}/a.py", "range": {"start": {"line": 0, "character": 0}}},
            {"uri": f"file:///{tmp_path}/b.py", "range": {"start": {"line": 5, "character": 3}}},
        ]
        mgr._clients["python"] = client

        f = tmp_path / "test.py"
        f.write_text("def foo(): pass")

        tool = LspFindReferencesTool(mgr)
        result = await tool.execute(file_path=str(f), line=1, column=5)
        assert "a.py" in result
        assert "b.py" in result

    @pytest.mark.asyncio
    async def test_include_declaration_param(self, tmp_path):
        mgr = _make_manager(tmp_path)
        client = _mock_client()
        client.request.return_value = []
        mgr._clients["python"] = client

        f = tmp_path / "test.py"
        f.write_text("x = 1")

        tool = LspFindReferencesTool(mgr)
        await tool.execute(file_path=str(f), line=1, column=1, include_declaration=False)

        call_args = client.request.call_args
        assert call_args[0][1]["context"]["includeDeclaration"] is False


class TestLspSymbolSearch:

    @pytest.mark.asyncio
    async def test_with_language(self, tmp_path):
        mgr = _make_manager(tmp_path)
        client = _mock_client()
        client.request.return_value = [{
            "name": "MyClass",
            "kind": 5,
            "location": {
                "uri": f"file:///{tmp_path}/models.py",
                "range": {"start": {"line": 0, "character": 0}},
            },
        }]
        mgr._clients["python"] = client

        tool = LspSymbolSearchTool(mgr)
        result = await tool.execute(query="MyClass", language="python")
        assert "MyClass" in result
        assert "Class" in result

    @pytest.mark.asyncio
    async def test_no_server_for_language(self, tmp_path):
        mgr = _make_manager(tmp_path)
        tool = LspSymbolSearchTool(mgr)
        with patch("shutil.which", return_value=None):
            result = await tool.execute(query="Foo", language="rust")
        assert "No LSP server" in result

    @pytest.mark.asyncio
    async def test_across_all_servers(self, tmp_path):
        mgr = _make_manager(tmp_path)
        py_client = _mock_client()
        py_client.request.return_value = [
            {"name": "PyClass", "kind": 5, "location": {
                "uri": f"file:///{tmp_path}/py.py",
                "range": {"start": {"line": 0, "character": 0}},
            }},
        ]
        ts_client = _mock_client()
        ts_client.request.return_value = [
            {"name": "TsClass", "kind": 5, "location": {
                "uri": f"file:///{tmp_path}/ts.ts",
                "range": {"start": {"line": 0, "character": 0}},
            }},
        ]
        mgr._clients["python"] = py_client
        mgr._clients["typescript"] = ts_client

        tool = LspSymbolSearchTool(mgr)
        result = await tool.execute(query="Class")
        assert "PyClass" in result
        assert "TsClass" in result

    @pytest.mark.asyncio
    async def test_no_servers_running(self, tmp_path):
        mgr = _make_manager(tmp_path)
        tool = LspSymbolSearchTool(mgr)
        result = await tool.execute(query="Foo")
        assert "No symbols" in result
        assert "Ensure an LSP server" in result


class TestLspHover:

    @pytest.mark.asyncio
    async def test_success(self, tmp_path):
        mgr = _make_manager(tmp_path)
        client = _mock_client()
        client.request.return_value = {
            "contents": {"kind": "markdown", "value": "```python\ndef foo(x: int) -> str\n```"},
        }
        mgr._clients["python"] = client

        f = tmp_path / "test.py"
        f.write_text("def foo(x): pass")

        tool = LspHoverTool(mgr)
        result = await tool.execute(file_path=str(f), line=1, column=5)
        assert "def foo" in result

    @pytest.mark.asyncio
    async def test_no_hover(self, tmp_path):
        mgr = _make_manager(tmp_path)
        client = _mock_client()
        client.request.return_value = None
        mgr._clients["python"] = client

        f = tmp_path / "test.py"
        f.write_text("x = 1")

        tool = LspHoverTool(mgr)
        result = await tool.execute(file_path=str(f), line=1, column=1)
        assert "No hover" in result


class TestLspDiagnostics:

    @pytest.mark.asyncio
    async def test_with_diagnostics(self, tmp_path):
        mgr = _make_manager(tmp_path)
        client = _mock_client()
        mgr._clients["python"] = client

        f = tmp_path / "test.py"
        f.write_text("x = undefined_var")
        uri = f.resolve().as_uri()
        mgr._diagnostics[uri] = [{
            "severity": 1,
            "message": "Undefined name 'undefined_var'",
            "range": {"start": {"line": 0, "character": 4}},
            "source": "pyflakes",
        }]

        tool = LspDiagnosticsTool(mgr)
        result = await tool.execute(file_path=str(f))
        assert "Error" in result
        assert "undefined_var" in result

    @pytest.mark.asyncio
    async def test_no_diagnostics(self, tmp_path):
        mgr = _make_manager(tmp_path)
        client = _mock_client()
        mgr._clients["python"] = client

        f = tmp_path / "test.py"
        f.write_text("x = 1")

        tool = LspDiagnosticsTool(mgr)
        result = await tool.execute(file_path=str(f))
        assert "No diagnostics" in result

    @pytest.mark.asyncio
    async def test_no_server(self, tmp_path):
        mgr = _make_manager(tmp_path)
        tool = LspDiagnosticsTool(mgr)
        result = await tool.execute(file_path="test.unknown")
        assert "not supported" in result


class TestToolConstants:

    def test_lsp_tool_names(self):
        assert "lsp_goto_definition" in LSP_TOOL_NAMES
        assert "lsp_find_references" in LSP_TOOL_NAMES
        assert "lsp_symbol_search" in LSP_TOOL_NAMES
        assert "lsp_hover" in LSP_TOOL_NAMES
        assert "lsp_diagnostics" in LSP_TOOL_NAMES
        assert len(LSP_TOOL_NAMES) == 5


class TestRegisterLspTools:

    def test_registers_all_five(self, tmp_path):
        registry = ToolRegistry()
        mgr = _make_manager(tmp_path)
        register_lsp_tools(registry, mgr)

        schemas = registry.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        for tool_name in LSP_TOOL_NAMES:
            assert tool_name in names

    def test_tool_schemas_valid(self, tmp_path):
        registry = ToolRegistry()
        mgr = _make_manager(tmp_path)
        register_lsp_tools(registry, mgr)

        for schema in registry.get_tool_schemas():
            func = schema["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            params = func["parameters"]
            assert params["type"] == "object"
            assert "required" in params
