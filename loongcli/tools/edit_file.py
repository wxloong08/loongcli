from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

from loongcli.tools.base import Tool

FUZZY_AUTO_THRESHOLD = 0.85
FUZZY_CANDIDATE_MIN = 0.6
FUZZY_AMBIGUITY_GAP = 0.1
MAX_SCAN_LINES = 5000
MAX_OLD_STRING_LEN = 1000


class EditFileTool(Tool):
    name = "edit_file"
    description = "Edit a file by replacing an exact string match. The old_string must appear exactly once in the file."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit",
            },
            "old_string": {
                "type": "string",
                "description": "The exact string to find and replace (must be unique in the file)",
            },
            "new_string": {
                "type": "string",
                "description": "The replacement string",
            },
        },
        "required": ["path", "old_string", "new_string"],
    }

    async def execute(self, path: str, old_string: str, new_string: str) -> str:
        try:
            p = Path(path)
            if not p.exists():
                return f"错误：文件不存在 '{path}'"
            if not p.is_file():
                return f"错误：'{path}' 不是文件"

            content = p.read_text(encoding="utf-8")

            result = self._try_exact(content, p, path, old_string, new_string)
            if result:
                return result

            old_stripped = old_string.strip()
            if old_stripped != old_string:
                result = self._try_exact(content, p, path, old_stripped, new_string)
                if result:
                    return result

            old_normalized = _normalize_whitespace(old_string)
            if old_normalized not in (old_string, old_stripped):
                result = self._try_exact(content, p, path, old_normalized, new_string)
                if result:
                    return result

            return self._fuzzy_replace(content, p, path, old_string, new_string)

        except Exception as e:
            return f"错误：{e}"

    def _try_exact(self, content: str, p: Path, path: str, old: str, new: str) -> str | None:
        """Try exact replacement. Returns result on success/error, None if not found."""
        count = content.count(old)
        if count == 0:
            return None
        if count > 1:
            return (
                f"错误：目标字符串有 {count} 处匹配（多处匹配），"
                f"需要唯一匹配。请提供更多上下文使其唯一。"
            )
        new_content = content.replace(old, new, 1)
        p.write_text(new_content, encoding="utf-8")
        return f"成功编辑 {path}"

    def _fuzzy_replace(self, content: str, p: Path, path: str, old_string: str, new_string: str) -> str:
        """Fuzzy match fallback after exact/strip/normalize all failed."""
        if len(old_string) > MAX_OLD_STRING_LEN:
            return self._closest_lines_error(content, old_string)

        file_lines = content.splitlines(keepends=True)
        if len(file_lines) > MAX_SCAN_LINES:
            file_lines = file_lines[:MAX_SCAN_LINES]

        matches = _fuzzy_locate(file_lines, old_string)
        if not matches:
            return self._closest_lines_error(content, old_string)

        best = matches[0]
        _, best_ratio, best_start, best_end = best
        second_ratio = matches[1][1] if len(matches) >= 2 else 0.0
        is_ambiguous = (best_ratio - second_ratio) < FUZZY_AMBIGUITY_GAP

        if best_ratio > FUZZY_AUTO_THRESHOLD and not is_ambiguous:
            # Preserve trailing newline if old_string didn't include one
            start, end = best_start, best_end
            matched_text = content[start:end]
            if matched_text.endswith("\n") and not old_string.endswith("\n"):
                end -= len(matched_text) - len(matched_text.rstrip("\n"))
            new_content = content[:start] + new_string + content[end:]
            p.write_text(new_content, encoding="utf-8")
            return f"成功编辑 {path}（fuzzy matched, 相似度 {best_ratio:.0%}）"

        if best_ratio >= FUZZY_CANDIDATE_MIN:
            matched_text = content[best_start:best_end]
            lines = []
            lines.append(f"未找到精确匹配。文件中有疑似目标（相似度 {best_ratio:.0%}）：")
            lines.append(f"<<<文件中的内容>>>\n{matched_text}<<<")
            if is_ambiguous and len(matches) >= 2:
                alt_text = content[matches[1][2]:matches[1][3]]
                alt_ratio = matches[1][1]
                lines.append(f"\n另有疑似候选（相似度 {alt_ratio:.0%}）：")
                lines.append(f"<<<\n{alt_text}>>>")
            lines.append("请确认这是否是你要替换的内容，或提供更精确的 old_string。")
            lines.append("（注：文件未被修改）")
            return "\n".join(lines)

        return self._closest_lines_error(content, old_string)

    def _closest_lines_error(self, content: str, old_string: str) -> str:
        """Build error with top-3 most similar lines."""
        lines = content.splitlines()
        if not lines:
            return "错误：未找到目标字符串（文件为空）"

        scored = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            ratio = SequenceMatcher(None, line, old_string).ratio()
            scored.append((ratio, i + 1, line[:120]))
        scored.sort(reverse=True)
        top3 = scored[:3]

        if not top3:
            return "错误：未找到目标字符串"

        lines_msg = "\n".join(
            f"  第 {idx} 行 (相似度 {ratio:.0%}): {line}"
            for ratio, idx, line in top3
        )
        return (
            f"错误：未找到目标字符串。文件中最相似的行：\n"
            f"{lines_msg}\n"
            f"请检查 old_string 是否与文件内容一致（注意空格、缩进、引号）。"
        )


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def _fuzzy_locate(file_lines: list[str], old_string: str) -> list[tuple[str, float, int, int]]:
    """Find best matching regions in file_lines for old_string.
    Returns up to 2 matches sorted by ratio desc: [(text, ratio, start_char, end_char), ...].
    """
    old_lines = old_string.splitlines(keepends=True)
    if len(old_lines) == 1:
        return _fuzzy_locate_single_line(file_lines, old_string)
    return _fuzzy_locate_multi_line(file_lines, old_lines, old_string)


def _prefix_offsets(file_lines: list[str]) -> list[int]:
    """Precompute cumulative character offsets for O(1) lookup."""
    offsets = [0]
    for line in file_lines:
        offsets.append(offsets[-1] + len(line))
    return offsets


def _fuzzy_locate_single_line(file_lines: list[str], target: str) -> list[tuple[str, float, int, int]]:
    """Find best matching single line in file."""
    offsets = _prefix_offsets(file_lines)
    scored = []
    for i, line in enumerate(file_lines):
        ratio = SequenceMatcher(None, line, target).ratio()
        if ratio >= FUZZY_CANDIDATE_MIN:
            scored.append((line, ratio, offsets[i], offsets[i] + len(line)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:2]


def _fuzzy_locate_multi_line(
    file_lines: list[str], old_lines: list[str], old_string: str
) -> list[tuple[str, float, int, int]]:
    """Find best matching N-line window in file."""
    n = len(old_lines)
    if n > len(file_lines):
        return []

    offsets = _prefix_offsets(file_lines)

    cands_per_line = []
    for ol in old_lines:
        cands = []
        for i, fl in enumerate(file_lines):
            ratio = SequenceMatcher(None, fl, ol).ratio()
            if ratio >= FUZZY_CANDIDATE_MIN:
                cands.append((i, ratio))
        if not cands:
            return []
        cands_per_line.append(cands)

    windows = []
    for start_idx, start_ratio in cands_per_line[0]:
        chain = [(start_idx, start_ratio)]
        cur = start_idx
        for line_idx in range(1, n):
            best = None
            for idx, ratio in cands_per_line[line_idx]:
                if idx > cur and idx - cur <= 3:
                    if best is None or ratio > best[1]:
                        best = (idx, ratio)
            if best is None:
                break
            chain.append(best)
            cur = best[0]

        if len(chain) == n:
            first = chain[0][0]
            last = chain[-1][0] + 1
            window_text = "".join(file_lines[first:last])
            ratio = SequenceMatcher(None, window_text, old_string).ratio()
            if ratio >= FUZZY_CANDIDATE_MIN:
                windows.append((window_text, ratio, offsets[first], offsets[last]))

    windows.sort(key=lambda x: x[1], reverse=True)
    return windows[:2]
