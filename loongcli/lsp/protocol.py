from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import unquote, urlparse


def file_uri(path: str | Path) -> str:
    return Path(path).resolve().as_uri()


def uri_to_path(uri: str) -> str:
    parsed = urlparse(uri)
    path = unquote(parsed.path)
    if os.name == "nt" and len(path) > 2 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return path


def make_position(line: int, column: int) -> dict:
    """Convert 1-based line/col to LSP 0-based Position."""
    return {"line": max(0, line - 1), "character": max(0, column - 1)}


def make_text_document_position(file_path: str, line: int, column: int) -> dict:
    return {
        "textDocument": {"uri": file_uri(file_path)},
        "position": make_position(line, column),
    }


def format_location(loc: dict, workspace_root: Path) -> str:
    uri = loc.get("uri", loc.get("targetUri", ""))
    path = uri_to_path(uri)
    try:
        path = str(Path(path).relative_to(workspace_root))
    except ValueError:
        pass
    range_ = loc.get("range", loc.get("targetRange", {}))
    start = range_.get("start", {})
    line = start.get("line", 0) + 1
    col = start.get("character", 0) + 1
    return f"{path}:{line}:{col}"


def format_locations(locations, workspace_root: Path, max_results: int = 30) -> str:
    if not locations:
        return "No results found"
    if isinstance(locations, dict):
        locations = [locations]
    lines = []
    for loc in locations[:max_results]:
        lines.append(format_location(loc, workspace_root))
    if len(locations) > max_results:
        lines.append(f"... and {len(locations) - max_results} more")
    return "\n".join(lines)


def format_hover(result: dict | None) -> str:
    if not result:
        return "No hover information available"
    contents = result.get("contents")
    if not contents:
        return "No hover information available"
    if isinstance(contents, str):
        return contents
    if isinstance(contents, dict):
        return contents.get("value", str(contents))
    if isinstance(contents, list):
        parts = []
        for item in contents:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("value", str(item)))
        return "\n\n".join(parts)
    return str(contents)


def format_symbols(symbols: list[dict] | None, workspace_root: Path, max_results: int = 30) -> str:
    if not symbols:
        return "No symbols found"
    lines = []
    kind_names = _symbol_kind_names()
    for sym in symbols[:max_results]:
        name = sym.get("name", "?")
        kind = kind_names.get(sym.get("kind", 0), "?")
        container = sym.get("containerName", "")
        loc = sym.get("location", {})
        if loc:
            path = uri_to_path(loc.get("uri", ""))
            try:
                path = str(Path(path).relative_to(workspace_root))
            except ValueError:
                pass
            start = loc.get("range", {}).get("start", {})
            line = start.get("line", 0) + 1
            lines.append(f"  {kind} {name} — {path}:{line}" + (f" ({container})" if container else ""))
        else:
            lines.append(f"  {kind} {name}" + (f" ({container})" if container else ""))
    if len(symbols) > max_results:
        lines.append(f"  ... and {len(symbols) - max_results} more")
    return "\n".join(lines)


def format_diagnostics(diagnostics: list[dict], file_path: str) -> str:
    if not diagnostics:
        return f"No diagnostics for {file_path}"
    severity_names = {1: "Error", 2: "Warning", 3: "Info", 4: "Hint"}
    lines = []
    for d in diagnostics:
        sev = severity_names.get(d.get("severity", 4), "?")
        msg = d.get("message", "")
        start = d.get("range", {}).get("start", {})
        line = start.get("line", 0) + 1
        col = start.get("character", 0) + 1
        source = d.get("source", "")
        prefix = f"[{source}] " if source else ""
        lines.append(f"  {sev} L{line}:{col} — {prefix}{msg}")
    return "\n".join(lines)


def _symbol_kind_names() -> dict[int, str]:
    return {
        1: "File", 2: "Module", 3: "Namespace", 4: "Package",
        5: "Class", 6: "Method", 7: "Property", 8: "Field",
        9: "Constructor", 10: "Enum", 11: "Interface", 12: "Function",
        13: "Variable", 14: "Constant", 15: "String", 16: "Number",
        17: "Boolean", 18: "Array", 19: "Object", 20: "Key",
        21: "Null", 22: "EnumMember", 23: "Struct", 24: "Event",
        25: "Operator", 26: "TypeParameter",
    }
