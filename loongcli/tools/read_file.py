from __future__ import annotations
from pathlib import Path
from loongcli.core.messages import detect_image_mime, make_image_sentinel, store_image
from loongcli.tools.base import Tool
from loongcli.tools.errors import ToolError

_ENCODING_CHAIN = ["utf-8", "utf-8-sig", "gbk", "cp936", "latin-1"]
_MAX_FILE_MB = 20
_MAX_LINES_SKIP = 100_000
_ENCODING_SAMPLE_BYTES = 4096
# 单次返回的字符预算（约 1 万 token）。超出就在整行边界分页，并告诉模型下一页的 offset。
# 此前 read_file 只限行数、交给入口通用截断（>8000 字符 → 头 1400 + 尾 600）：dogfood p3-1 里
# 53 次读文件 7 次被截，全是 8.8K–13.6K 字符的中等文件，丢掉约 80%；模型缩小范围重读仍超 8000 又被截，
# 每次重读多花一次模型调用（整段上下文重发）。读文件幂等、是模型明确要的内容，该分页而不是掐头去尾。
READ_CHAR_BUDGET = 40_000


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a file's contents with line numbers. Returns up to `limit` lines and at most about "
        f"{READ_CHAR_BUDGET} characters per call; for a longer file the header shows which lines were "
        "returned and the offset to continue from."
    )
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

            # 图片分支：存入图片库返回 sentinel，由 agent 循环转成带图 user 消息注入
            # （设计 §4）。放在 20MB 文本上限前——图片有自己的 10MB 上限与错误文案，
            # "用 offset/limit 分段读"对图片没有意义。offset/limit 对图片忽略。
            mime = detect_image_mime(str(p))
            if mime:
                try:
                    return make_image_sentinel([(store_image(str(p)), mime)])
                except (ValueError, OSError) as e:
                    return f"错误：{e}"

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
            chars = 0
            last_shown = start_line - 1
            page_full = False  # 本页已到行数或字符上限，之后只数总行数
            hit_char_budget = False
            with open(p, encoding=enc) as f:
                for i, line in enumerate(f, 1):
                    total_lines = i  # 读到底，给出准确总行数（文件已限 20MB）
                    if i < start_line:
                        if i > _MAX_LINES_SKIP:
                            return (
                                f"错误：文件超过 {_MAX_LINES_SKIP} 行，"
                                f"无法跳转到第 {start_line} 行。请减小 offset。"
                            )
                        continue
                    if page_full or i > end_line:
                        page_full = True
                        continue
                    entry = f"{i}\t{line.rstrip(chr(10) + chr(13))}"
                    if chars + len(entry) + 1 > READ_CHAR_BUDGET:
                        hit_char_budget = page_full = True
                        if numbered:
                            continue
                        # 单行就超预算（压缩过的 JS、单行 JSON）：截取这一行，否则永远读不出任何内容
                        entry = (
                            entry[:READ_CHAR_BUDGET]
                            + f" …[该行过长，只显示前 {READ_CHAR_BUDGET} 字符，原行 {len(entry)} 字符；完整内容请用 shell 处理]"
                        )
                    numbered.append(entry)
                    chars += len(entry) + 1
                    last_shown = i

            has_more = last_shown < total_lines
            parts = []
            if enc != "utf-8":
                parts.append(f"[编码：{enc}]")
            if start_line > 1 or has_more:
                parts.append(f"[第 {start_line}-{last_shown} 行 / 共 {total_lines} 行]")
            if has_more:
                why = f"本次已到约 {READ_CHAR_BUDGET} 字符上限" if hit_char_budget else "本次已到行数上限"
                parts.append(f"[未读完（{why}）：继续读用 offset={last_shown + 1}]")
            if parts:
                parts.append("")
            parts.append("\n".join(numbered))
            return "\n".join(parts)
        except PermissionError:
            raise ToolError(f"权限不足，无法读取 '{path}'", retryable=True, retry_after=1.0)
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
        except UnicodeDecodeError as e:
            # Sample may truncate a multi-byte UTF-8 char at the boundary
            if enc == "utf-8" and e.start > len(raw) - 4:
                return raw[:e.start].decode(enc), enc
            continue
        except LookupError:
            continue
    return raw.decode("latin-1"), "latin-1"
