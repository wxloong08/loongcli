from __future__ import annotations
from pathlib import Path
from loongcli.tools.base import Tool

_ENCODING_CHAIN = ["utf-8", "utf-8-sig", "gbk", "cp936", "latin-1"]
_MAX_FILE_MB = 20
_MAX_LINES_SKIP = 100_000
_ENCODING_SAMPLE_BYTES = 4096


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a file's contents. Returns text with line numbers."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file",
            },
            "offset": {
                "type": "integer",
                "description": "Start reading from this line number (1-based). Default: 1",
            },
            "limit": {
                "type": "integer",
                "description": "Max number of lines to read. Default: 2000",
            },
        },
        "required": ["path"],
    }

    async def execute(self, path: str, offset: int = 1, limit: int = 2000) -> str:
        try:
            p = Path(path)
            if not p.exists():
                return f"错误：文件不存在 '{path}'"
            if not p.is_file():
                return f"错误：'{path}' 不是文件"

            size_mb = p.stat().st_size / (1024 * 1024)
            if size_mb > _MAX_FILE_MB:
                return (
                    f"错误：文件过大（{size_mb:.1f} MB，超过 {_MAX_FILE_MB} MB 上限）。"
                    f"请使用 offset/limit 分段读取，或通过 Shell 工具处理。"
                )

            head = p.read_bytes()[: _ENCODING_SAMPLE_BYTES]
            _, enc = _decode_with_fallback(head)

            start_line = max(1, offset)
            end_line = start_line + limit - 1

            numbered = []
            total_lines = 0
            has_more = False
            with open(p, encoding=enc) as f:
                for i, line in enumerate(f, 1):
                    total_lines = i
                    if i < start_line:
                        if i > _MAX_LINES_SKIP:
                            return (
                                f"错误：文件超过 {_MAX_LINES_SKIP} 行，"
                                f"无法跳转到第 {start_line} 行。请减小 offset。"
                            )
                        continue
                    if i > end_line:
                        has_more = True
                        break
                    numbered.append(f"{i}\t{line.rstrip('\n\r')}")

            parts = []
            if enc != "utf-8":
                parts.append(f"[编码：{enc}]")
            if start_line > 1 or has_more:
                shown_end = end_line if has_more else total_lines
                total_str = f"{total_lines}+" if has_more else str(total_lines)
                parts.append(f"[第 {start_line}-{shown_end} 行 / 共 {total_str} 行]")
            if parts:
                parts.append("")
            parts.append("\n".join(numbered))
            return "\n".join(parts)
        except Exception as e:
            return f"错误：{e}"


def _decode_with_fallback(raw: bytes) -> tuple[str, str]:
    """Try decoding raw bytes with fallback chain.
    Returns (decoded_text, encoding_used).
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "utf-8-sig"

    for enc in _ENCODING_CHAIN:
        try:
            return raw.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("latin-1"), "latin-1"
