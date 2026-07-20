"""一次性迁移：全局记忆库拆两层——全局只留 user，其余按来源项目归位。

归位判定（按优先级）：
1. type=user → 留全局。
2. vps 拆分 6 条（source=split-from-vps-infrastructure / correction-20260719，
   拆分发生在 CC 迁入之后，CC 对照表里没有）→ 硬归 D-skills-vps-picker。
3. 文件名命中某个 CC 项目 memory 的 frontmatter name（migrate_cc 按 frontmatter
   name 落盘，CC 文件名与 name 大量不一致，对照必须用 name 不能用文件名）→ 归
   该 CC 项目对应的 loongcli slug（CC slug 折叠连字符）。多项目重名 → 人工清单。
4. source_session 是 auto-extract:<id> / memorize:<id> → 反查
   ~/.loongcli/projects/*/sessions/<id>.json 定位项目；唯一命中才归位。
5. 都不中 → 留全局 + 打印人工清单。

安全：先整库备份 ~/.loongcli/memory.bak-<ts>；移动=先写项目库成功再删全局，
中断不丢；目标已存在同名则跳过（幂等）。跑完重建所有涉及库的索引，
并打印各库统计 + 库内语义重复对清单（供人工合并）。

用法：python scripts/migrate_memory_split.py [--dry-run]
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loongcli.memory.markdown_store import MarkdownMemoryStore, _similarity  # noqa: E402

GLOBAL_DIR = Path.home() / ".loongcli" / "memory"
PROJECTS_ROOT = Path.home() / ".loongcli" / "projects"
CC_ROOT = Path.home() / ".claude" / "projects"

# vps 基础设施记忆 2026-07-19 从单条拆成 6 条，晚于 CC 迁入，名字不在对照表里
VPS_SPLIT = {
    "vps-access-map", "vps-fleet-roster", "vps-incidents-lessons",
    "vps-ops-archive", "vps-perf-baseline", "vps-proxy-topology",
}
VPS_SLUG = "D-skills-vps-picker"

# 库内重复对报告阈值：低于去重线（0.7）也报，抓措辞变体（如 avoid-circled-numbers
# ↔ feedback-no-circled-symbols 这类 Jaccard 不到 0.7 的语义重复）
_DUP_REPORT_THRESHOLD = 0.5


def _parse_meta(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    for key in ("name", "description", "type", "source_session"):
        m = re.search(rf"^{key}:\s*(.+)$", text, re.M)
        if m:
            meta[key] = m.group(1).strip().strip('"')
    return meta


def _cc_name_map() -> dict[str, set[str]]:
    """CC 各项目 memory 的 frontmatter name → {loongcli slug,...}"""
    mapping: dict[str, set[str]] = defaultdict(set)
    if not CC_ROOT.is_dir():
        return mapping
    for cc_dir in sorted(CC_ROOT.iterdir()):
        mem = cc_dir / "memory"
        if not mem.is_dir():
            continue
        slug = re.sub(r"-+", "-", cc_dir.name)  # CC slug 折叠连字符即 loongcli slug
        for f in mem.glob("*.md"):
            if f.name == "MEMORY.md":
                continue
            name = _parse_meta(f).get("name", f.stem)
            mapping[name].add(slug)
    return mapping


def _session_project(session_id: str) -> str | None:
    hits = {p.parent.parent.name for p in PROJECTS_ROOT.glob(f"*/sessions/{session_id}.json")}
    return hits.pop() if len(hits) == 1 else None


def _resolve_target(stem: str, meta: dict[str, str], cc_map: dict[str, set[str]]) -> tuple[str | None, str]:
    """返回 (目标 slug 或 None, 判定理由)。None = 留全局。"""
    if meta.get("type", "project") == "user":
        return None, "user 留全局"
    if stem in VPS_SPLIT:
        return VPS_SLUG, "vps 拆分硬归"
    slugs = cc_map.get(stem, set())
    if len(slugs) == 1:
        return next(iter(slugs)), "CC 名字对照"
    if len(slugs) > 1:
        return None, f"多项目重名 {sorted(slugs)} → 人工"
    src = meta.get("source_session", "")
    m = re.match(r"(?:auto-extract|memorize):([0-9a-f]{12,36})$", src)
    if m:
        slug = _session_project(m.group(1))
        if slug:
            return slug, f"session 反查 {m.group(1)[:12]}"
        return None, f"session {m.group(1)[:12]} 无法定位 → 人工"
    return None, "无归位依据 → 人工"


def _move(src: Path, target_dir: Path, dry_run: bool) -> str:
    """先写项目库成功再删全局；目标同名已存在则跳过（幂等）。"""
    dest = target_dir / src.name
    if dest.exists():
        return "目标已存在，跳过（全局副本保留，见人工清单）"
    if dry_run:
        return "将移动"
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    if dest.read_bytes() != src.read_bytes():
        dest.unlink()
        raise RuntimeError(f"写入校验失败: {dest}")
    src.unlink()
    return "已移动"


def _dup_pairs(lib_dir: Path) -> list[tuple[str, str, float]]:
    entries = []
    for f in sorted(lib_dir.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        meta = _parse_meta(f)
        entries.append((f.stem, meta.get("description", "")))
    pairs = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            score = _similarity(entries[i][1], entries[j][1])
            if score >= _DUP_REPORT_THRESHOLD:
                pairs.append((entries[i][0], entries[j][0], score))
    return sorted(pairs, key=lambda p: -p[2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只打印归位计划，不动文件")
    args = parser.parse_args()

    if not GLOBAL_DIR.is_dir():
        sys.exit(f"全局记忆库不存在: {GLOBAL_DIR}")

    if not args.dry_run:
        backup = GLOBAL_DIR.with_name(f"memory.bak-{int(time.time())}")
        shutil.copytree(GLOBAL_DIR, backup)
        print(f"✔ 已备份全局库 → {backup}")

    cc_map = _cc_name_map()
    moves: dict[str, list[str]] = defaultdict(list)  # slug -> [stem...]
    manual: list[tuple[str, str]] = []               # (stem, 理由)
    stay_user = 0

    for f in sorted(GLOBAL_DIR.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        meta = _parse_meta(f)
        target, reason = _resolve_target(f.stem, meta, cc_map)
        if target is None:
            if reason == "user 留全局":
                stay_user += 1
            else:
                manual.append((f.stem, reason))
            continue
        status = _move(f, PROJECTS_ROOT / target / "memory", args.dry_run)
        if status.startswith("目标已存在"):
            manual.append((f.stem, status))
        else:
            moves[target].append(f.stem)
        print(f"  {f.stem} → {target}  [{reason}] {status}")

    # 重建索引：全局 + 所有涉及的项目库
    if not args.dry_run:
        MarkdownMemoryStore(GLOBAL_DIR)._rebuild_index()
        for slug in moves:
            MarkdownMemoryStore(PROJECTS_ROOT / slug / "memory")._rebuild_index()

    print("\n== 归位统计 ==")
    print(f"  全局保留: user {stay_user} 条 + 未归位 {len(manual)} 条")
    for slug in sorted(moves):
        lib = PROJECTS_ROOT / slug / "memory"
        on_disk = len(list(lib.glob("*.md"))) - int((lib / "MEMORY.md").exists()) if lib.is_dir() else 0
        print(f"  {slug}: 本次移入 {len(moves[slug])} 条（库内现有 {on_disk} 条）")

    if manual:
        print("\n== 人工清单（留在全局库，需手动裁决） ==")
        for stem, reason in manual:
            print(f"  {stem}: {reason}")

    print("\n== 库内语义重复对（≥%.1f，建议人工合并） ==" % _DUP_REPORT_THRESHOLD)
    libs = [("全局", GLOBAL_DIR)] + [(slug, PROJECTS_ROOT / slug / "memory") for slug in sorted(moves)]
    found = False
    for label, lib_dir in libs:
        if not lib_dir.is_dir():
            continue
        for a, b, score in _dup_pairs(lib_dir):
            print(f"  [{label}] {a} ↔ {b}  ({score:.2f})")
            found = True
    if not found:
        print("  （无）")

    if args.dry_run:
        print("\n（dry-run：未动任何文件）")


if __name__ == "__main__":
    main()
