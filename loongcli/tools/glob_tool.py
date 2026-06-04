from __future__ import annotations
from difflib import SequenceMatcher
from pathlib import Path
from loongcli.tools.base import Tool

_MAX_FILES = 200
_MAX_SCAN = 10_000
_MAX_LIST_ITEMS = 50
_SIMILAR_THRESHOLD = 0.5


class GlobTool(Tool):
    name = "glob"
    description = "Find files matching a glob pattern. Returns matching file paths."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern (e.g. '**/*.py', 'src/*.ts')",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in. Default: current working directory",
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, pattern: str, path: str = ".") -> str:
        try:
            base = Path(path).resolve()
            if not base.is_dir():
                return f"错误：'{path}' 不是目录"

            files: list[Path] = []
            scan_count = 0

            for match in sorted(base.glob(pattern)):
                scan_count += 1
                if scan_count > _MAX_SCAN:
                    break
                if match.is_file():
                    files.append(match)

            if not files:
                return self._zero_result_report(base, pattern, scan_count)

            lines = []
            for f in files[:_MAX_FILES]:
                try:
                    rel = f.relative_to(base)
                except ValueError:
                    rel = f
                lines.append(str(rel))

            result = "\n".join(lines)
            if len(files) > _MAX_FILES:
                result += f"\n... 共 {len(files)} 个文件，仅显示前 {_MAX_FILES} 个"
            if scan_count > _MAX_SCAN:
                result += f"\n... 扫描已截断（超过 {_MAX_SCAN} 条记录），请缩小搜索范围"
            return result
        except Exception as e:
            return f"错误：{e}"

    def _zero_result_report(self, base: Path, pattern: str, scan_count: int) -> str:
        """Build diagnostic report when glob finds nothing."""
        parts = [f"未找到匹配 '{pattern}' 的文件"]

        issues = _check_pattern_issues(pattern)
        if issues:
            parts.append("\n可能的模式问题：")
            parts.extend(f"  - {i}" for i in issues)

        snapshot = _dir_snapshot(base)
        if snapshot:
            parts.append(f"\n目录 '{base}' 的内容：")
            parts.append(snapshot)

        sim = _suggest_similar_dir(base, pattern)
        if sim:
            parts.append(sim)

        if scan_count > _MAX_SCAN:
            parts.append(f"\n... 扫描已截断（超过 {_MAX_SCAN} 条），请缩小范围")

        return "\n".join(parts)


def _check_pattern_issues(pattern: str) -> list[str]:
    """Check for common glob pattern mistakes."""
    issues = []
    has_wildcard = any(c in pattern for c in "*?[")
    if not has_wildcard:
        issues.append(
            f"模式中无通配符（* ? []），当前将精确匹配 '{pattern}'。"
            f"如需搜索含该字符串的文件，请尝试 '*{pattern}*'"
        )
    if "\\" in pattern:
        issues.append("使用了反斜杠 '\\'，glob 使用正斜杠 '/' 作为路径分隔符")
    if pattern.endswith("/"):
        issues.append("模式以 '/' 结尾（匹配目录而非文件），如需找目录下的文件请加 '*' 或 '**/*'")
    return issues


def _dir_snapshot(base: Path) -> str:
    """Return a short listing of first-level contents (no recursion)."""
    try:
        items = sorted(base.iterdir())[:_MAX_LIST_ITEMS]
    except PermissionError:
        return ""

    dirs = []
    other_files = []
    for p in items:
        if p.is_dir():
            dirs.append(f"  [{p.name}/]")
        else:
            other_files.append(p.name)

    lines = []
    if dirs:
        shown = dirs[:20]
        if len(dirs) > 20:
            shown.append(f"  ... 共 {len(dirs)} 个子目录")
        lines.append(f"子目录：")
        lines.extend(shown)
    if other_files:
        shown = other_files[:20]
        if len(other_files) > 20:
            shown.append(f"  ... 共 {len(other_files)} 个文件")
        lines.append(f"文件：")
        lines.extend(f"  {f}" for f in shown)
    if not lines:
        return "  (空目录)"
    return "\n".join(lines)


def _suggest_similar_dir(base: Path, pattern: str) -> str:
    """If pattern has a non-existent directory prefix, suggest similar names."""
    pattern = pattern.replace("\\", "/")
    # Split into parts, stop at first wildcard
    parts = []
    for p in pattern.split("/"):
        if any(c in p for c in "*?[]"):
            break
        if p and p != ".":
            parts.append(p)

    if not parts:
        return ""

    # Walk to find where it diverges
    current = base
    for i, part in enumerate(parts):
        candidate = current / part
        if candidate.exists() and candidate.is_dir():
            current = candidate
            continue
        # Diverged — find similar names in current
        try:
            siblings = [p.name for p in current.iterdir() if p.is_dir()]
        except PermissionError:
            return ""
        if not siblings:
            return f"\n目录 '{current}' 下无任何子目录"
        similar = _find_similar(part, siblings)
        if similar:
            prefix = "/".join(parts[:i])
            lines = [f"\n未找到目录 '{part}'，你是否要找："]
            for s in similar:
                full = f"{prefix}/{s}" if prefix else s
                lines.append(f"  {full}/")
            return "\n".join(lines)
        return f"\n目录 '{part}' 不存在（在 '{current}' 下）"

    return ""


def _find_similar(target: str, candidates: list[str]) -> list[str]:
    """Return candidate names with similarity >= threshold, sorted best first."""
    scored = []
    for c in candidates:
        ratio = SequenceMatcher(None, target.lower(), c.lower()).ratio()
        if ratio >= _SIMILAR_THRESHOLD:
            scored.append((ratio, c))
    scored.sort(reverse=True)
    return [c for _, c in scored[:3]]
