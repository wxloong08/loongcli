from __future__ import annotations

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from loongcli.lsp.protocol import (
    file_uri,
    uri_to_path,
    make_position,
    make_text_document_position,
    format_location,
    format_locations,
    format_hover,
    format_symbols,
    format_diagnostics,
)


class TestFileUri:

    def test_absolute_path(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("hello")
        uri = file_uri(str(f))
        assert uri.startswith("file:///")
        assert "test.py" in uri

    def test_path_object(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("hello")
        uri = file_uri(f)
        assert uri.startswith("file:///")


class TestUriToPath:

    def test_unix_style(self):
        with patch("os.name", "posix"):
            path = uri_to_path("file:///home/user/test.py")
            assert path == "/home/user/test.py"

    def test_windows_style(self):
        with patch("os.name", "nt"):
            path = uri_to_path("file:///C:/Users/test.py")
            assert path == "C:/Users/test.py"

    def test_windows_no_strip_for_posix(self):
        with patch("os.name", "posix"):
            path = uri_to_path("file:///C:/Users/test.py")
            assert path == "/C:/Users/test.py"

    def test_encoded_spaces(self):
        with patch("os.name", "posix"):
            path = uri_to_path("file:///home/user/my%20project/test.py")
            assert path == "/home/user/my project/test.py"


class TestMakePosition:

    def test_one_based_to_zero_based(self):
        pos = make_position(10, 5)
        assert pos == {"line": 9, "character": 4}

    def test_min_zero(self):
        pos = make_position(0, 0)
        assert pos == {"line": 0, "character": 0}

    def test_first_line_first_col(self):
        pos = make_position(1, 1)
        assert pos == {"line": 0, "character": 0}


class TestMakeTextDocumentPosition:

    def test_structure(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("hello")
        result = make_text_document_position(str(f), 5, 3)
        assert "textDocument" in result
        assert "position" in result
        assert result["position"]["line"] == 4
        assert result["position"]["character"] == 2


class TestFormatLocation:

    def test_basic(self, tmp_path):
        loc = {
            "uri": f"file:///{tmp_path}/src/main.py",
            "range": {"start": {"line": 9, "character": 4}},
        }
        result = format_location(loc, tmp_path)
        assert "main.py" in result
        assert ":10:" in result
        assert ":5" in result

    def test_target_uri(self, tmp_path):
        loc = {
            "targetUri": f"file:///{tmp_path}/lib.py",
            "targetRange": {"start": {"line": 0, "character": 0}},
        }
        result = format_location(loc, tmp_path)
        assert "lib.py" in result
        assert ":1:1" in result

    def test_no_range(self, tmp_path):
        loc = {"uri": f"file:///{tmp_path}/test.py"}
        result = format_location(loc, tmp_path)
        assert ":1:1" in result


class TestFormatLocations:

    def test_empty(self, tmp_path):
        assert format_locations(None, tmp_path) == "No results found"
        assert format_locations([], tmp_path) == "No results found"

    def test_single_dict(self, tmp_path):
        loc = {
            "uri": f"file:///{tmp_path}/a.py",
            "range": {"start": {"line": 0, "character": 0}},
        }
        result = format_locations(loc, tmp_path)
        assert "a.py:1:1" in result

    def test_multiple(self, tmp_path):
        locs = [
            {"uri": f"file:///{tmp_path}/a.py", "range": {"start": {"line": 0, "character": 0}}},
            {"uri": f"file:///{tmp_path}/b.py", "range": {"start": {"line": 5, "character": 2}}},
        ]
        result = format_locations(locs, tmp_path)
        assert "a.py" in result
        assert "b.py" in result

    def test_max_results(self, tmp_path):
        locs = [
            {"uri": f"file:///{tmp_path}/f{i}.py", "range": {"start": {"line": 0, "character": 0}}}
            for i in range(35)
        ]
        result = format_locations(locs, tmp_path, max_results=30)
        assert "5 more" in result


class TestFormatHover:

    def test_none(self):
        assert format_hover(None) == "No hover information available"

    def test_string_contents(self):
        result = format_hover({"contents": "int"})
        assert result == "int"

    def test_dict_contents(self):
        result = format_hover({"contents": {"kind": "markdown", "value": "```python\ndef foo()```"}})
        assert "def foo()" in result

    def test_list_contents(self):
        result = format_hover({"contents": [
            {"value": "class Foo"},
            "A test class",
        ]})
        assert "class Foo" in result
        assert "A test class" in result

    def test_empty_contents(self):
        result = format_hover({"contents": ""})
        assert result == "No hover information available"


class TestFormatSymbols:

    def test_none(self, tmp_path):
        assert format_symbols(None, tmp_path) == "No symbols found"
        assert format_symbols([], tmp_path) == "No symbols found"

    def test_basic(self, tmp_path):
        syms = [{
            "name": "MyClass",
            "kind": 5,  # Class
            "location": {
                "uri": f"file:///{tmp_path}/models.py",
                "range": {"start": {"line": 10, "character": 0}},
            },
        }]
        result = format_symbols(syms, tmp_path)
        assert "Class" in result
        assert "MyClass" in result
        assert "models.py" in result

    def test_with_container(self, tmp_path):
        syms = [{
            "name": "process",
            "kind": 6,  # Method
            "containerName": "Worker",
            "location": {
                "uri": f"file:///{tmp_path}/worker.py",
                "range": {"start": {"line": 5, "character": 0}},
            },
        }]
        result = format_symbols(syms, tmp_path)
        assert "Method" in result
        assert "Worker" in result

    def test_max_results(self, tmp_path):
        syms = [
            {
                "name": f"sym{i}",
                "kind": 12,
                "location": {
                    "uri": f"file:///{tmp_path}/f.py",
                    "range": {"start": {"line": i, "character": 0}},
                },
            }
            for i in range(35)
        ]
        result = format_symbols(syms, tmp_path, max_results=30)
        assert "5 more" in result


class TestFormatDiagnostics:

    def test_empty(self):
        result = format_diagnostics([], "test.py")
        assert "No diagnostics" in result

    def test_basic(self):
        diags = [{
            "severity": 1,
            "message": "Undefined variable 'x'",
            "range": {"start": {"line": 5, "character": 10}},
            "source": "pylsp",
        }]
        result = format_diagnostics(diags, "test.py")
        assert "Error" in result
        assert "Undefined variable" in result
        assert "L6:11" in result
        assert "[pylsp]" in result

    def test_multiple_severities(self):
        diags = [
            {"severity": 1, "message": "err", "range": {"start": {"line": 0, "character": 0}}},
            {"severity": 2, "message": "warn", "range": {"start": {"line": 1, "character": 0}}},
            {"severity": 3, "message": "info", "range": {"start": {"line": 2, "character": 0}}},
            {"severity": 4, "message": "hint", "range": {"start": {"line": 3, "character": 0}}},
        ]
        result = format_diagnostics(diags, "test.py")
        assert "Error" in result
        assert "Warning" in result
        assert "Info" in result
        assert "Hint" in result
