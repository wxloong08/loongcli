"""Claude Code 迁移：把 CC 的会话（JSONL）与记忆（.md）导入 loongcli。

- 记忆层近同构（一条记忆一个 .md + frontmatter + MEMORY.md 索引），文件级映射。
- 会话层核心是格式翻译：Anthropic 块格式（text/tool_use/tool_result/thinking）
  → OpenAI messages（content / assistant.tool_calls / role=tool 配对）。
- CLAUDE.md 不自动搬——它是指令不是记忆，提示用户手动并入 LOONG.md。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from loongcli.memory.conversation import ConversationStore, _project_sessions_dir, project_memory_dir
from loongcli.memory.markdown_store import MarkdownMemoryStore
from loongcli.memory.memory_router import MemoryRouter


# ── 定位 ─────────────────────────────────────────────────────────

def cc_slug(project_dir: Path | None = None) -> str:
    """CC 的项目目录 slug：冒号/斜杠全部替换为 '-'，不折叠连字符
    （D:\\HKUDS\\loongcli → D--HKUDS-loongcli，与 loongcli 自己的折叠规则不同）。"""
    p = (project_dir or Path.cwd()).resolve()
    return re.sub(r"[:\\/]", "-", str(p))


def find_cc_project_dir(project_dir: Path | None = None) -> Path | None:
    d = Path.home() / ".claude" / "projects" / cc_slug(project_dir)
    return d if d.is_dir() else None


# ── 记忆导入 ─────────────────────────────────────────────────────

def parse_cc_memory(path: Path) -> dict | None:
    """解析 CC 记忆文件：frontmatter（name/description/metadata.type）+ 正文。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
    if not m:
        # 无 frontmatter：整个文件当正文，名字取文件名
        return {"name": path.stem, "description": "", "type": "project", "content": text.strip()}
    front, body = m.group(1), m.group(2)

    def _field(key: str) -> str:
        fm = re.search(rf"^{key}:\s*(.+?)\s*$", front, re.MULTILINE)
        return fm.group(1).strip() if fm else ""

    # type 嵌在 metadata: 块下（缩进行），全文匹配即可——frontmatter 里只有一处 type
    tm = re.search(r"^\s+type:\s*(\S+)", front, re.MULTILINE)
    return {
        "name": _field("name") or path.stem,
        "description": _field("description"),
        "type": tm.group(1) if tm else "project",
        "content": body.strip(),
    }


def import_memories(
    cc_dir: Path,
    store: MarkdownMemoryStore | MemoryRouter | None = None,
    overwrite: bool = False,
) -> dict:
    """导入 <cc_dir>/memory/*.md（跳过 MEMORY.md 索引）。同名默认跳过（幂等）。

    默认走 MemoryRouter 按 type 路由：user → 全局库，其余 → 当前项目库
    （CC 数据本就按项目隔离，导入不再把项目记忆倒进全局）。"""
    store = store or MemoryRouter(
        global_dir=Path.home() / ".loongcli" / "memory",
        project_dir=project_memory_dir(),
    )
    stats = {"imported": 0, "skipped": 0, "failed": 0}
    mem_dir = cc_dir / "memory"
    if not mem_dir.is_dir():
        return stats
    for f in sorted(mem_dir.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        parsed = parse_cc_memory(f)
        if not parsed or not parsed["content"]:
            stats["failed"] += 1
            continue
        # load 而非 base_dir 文件探测：store/router 鸭子兼容（router 查项目+全局两库）
        if store.load(parsed["name"]) is not None and not overwrite:
            stats["skipped"] += 1
            continue
        # CC 记忆可能有 _sanitize_name 不接受的纯符号/全数字名字（如 "---"），
        # 不死磕——计入失败、继续下一条，不崩掉整个迁移
        try:
            store.save(
                name=parsed["name"],
                description=parsed["description"],
                type=parsed["type"],
                content=parsed["content"],
                source="claude-code",
            )
            stats["imported"] += 1
        except (ValueError, OSError):
            stats["failed"] += 1
    return stats


# ── 会话转换 ─────────────────────────────────────────────────────

def _block_text(content) -> str:
    """tool_result 的 content 可能是字符串或块列表——拍平为纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    parts.append(b.get("text", ""))
                elif b.get("type") == "image":
                    parts.append("[图片]")
        return "\n".join(parts)
    return ""


def convert_session(jsonl_path: Path) -> list[dict]:
    """CC JSONL → OpenAI messages。丢弃 meta/summary/sidechain/thinking；
    tool_use → assistant.tool_calls；tool_result → role=tool。"""
    raw: list[dict] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") not in ("user", "assistant"):
            continue
        if d.get("isMeta") or d.get("isSidechain"):
            continue
        msg = d.get("message") or {}
        content = msg.get("content")

        if d["type"] == "user":
            if isinstance(content, str):
                if content.strip():
                    raw.append({"role": "user", "content": content})
                continue
            texts = []
            for b in content or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_result":
                    raw.append({
                        "role": "tool",
                        "tool_call_id": b.get("tool_use_id", ""),
                        "content": _block_text(b.get("content")) or "(空结果)",
                    })
                elif b.get("type") == "text":
                    texts.append(b.get("text", ""))
                elif b.get("type") == "image":
                    texts.append("[图片]")
            text = "\n".join(t for t in texts if t.strip())
            if text.strip():
                raw.append({"role": "user", "content": text})
        else:  # assistant
            texts = []
            tool_calls = []
            for b in content or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    texts.append(b.get("text", ""))
                elif b.get("type") == "tool_use":
                    tool_calls.append({
                        "id": b.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": b.get("name", ""),
                            "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False),
                        },
                    })
                # thinking 块丢弃：内部推理不属于可回放对话
            out: dict = {"role": "assistant", "content": "\n".join(texts).strip()}
            if tool_calls:
                out["tool_calls"] = tool_calls
            if out["content"] or tool_calls:
                raw.append(out)

    return _repair_pairing(raw)


def _repair_pairing(messages: list[dict]) -> list[dict]:
    """tool_calls/tool 严格配对修复——与 abort_turn 同一契约：
    孤儿 tool 消息丢弃；未回齐的 tool_calls 补占位，否则 API 校验 400。"""
    out: list[dict] = []
    open_ids: list[str] = []

    def _flush_open():
        for tc_id in open_ids:
            out.append({
                "role": "tool", "tool_call_id": tc_id,
                "content": "[导入时缺失结果，原会话在此中断]",
            })
        open_ids.clear()

    for m in messages:
        if m["role"] == "tool":
            if m.get("tool_call_id") in open_ids:
                open_ids.remove(m["tool_call_id"])
                out.append(m)
            # 孤儿 tool 结果：无对应 tool_calls，丢弃
            continue
        _flush_open()
        out.append(m)
        if m["role"] == "assistant":
            open_ids.extend(tc["id"] for tc in m.get("tool_calls") or [])
    _flush_open()
    return out


def import_sessions(
    cc_dir: Path,
    sessions_dir: Path | None = None,
) -> dict:
    """导入 <cc_dir>/*.jsonl。已导入的（同名目标存在）跳过，幂等。"""
    sessions_dir = sessions_dir or _project_sessions_dir()
    stats = {"imported": 0, "skipped": 0, "empty": 0}
    for f in sorted(cc_dir.glob("*.jsonl")):
        # CC 的 session ID 是 uuid4-hex（含连字符的 36 位 UUID），
        # 去连字符后完整保留为 loongcli session_id——不截断，溯源不需要猜。
        sid = f.stem.replace("-", "")
        if (sessions_dir / f"{sid}.json").exists():
            stats["skipped"] += 1
            continue
        messages = convert_session(f)
        if not messages:
            stats["empty"] += 1
            continue
        store = ConversationStore(
            base_dir=sessions_dir,
            session_id=sid,
            extra_meta={"origin": "claude-code", "cc_session_id": f.stem},
        )
        store.save(messages)
        stats["imported"] += 1
    return stats


# ── 入口 ─────────────────────────────────────────────────────────

def run_migration(console, project_dir: Path | None = None, assume_yes: bool = False) -> int:
    """预览 → 确认 → 执行 → 报告。返回退出码。"""
    cc_dir = find_cc_project_dir(project_dir)
    if cc_dir is None:
        console.print(f"[yellow]未找到当前项目的 Claude Code 数据目录"
                      f"（~/.claude/projects/{cc_slug(project_dir)}）[/yellow]")
        return 1

    session_files = list(cc_dir.glob("*.jsonl"))
    mem_dir = cc_dir / "memory"
    memory_files = [f for f in mem_dir.glob("*.md") if f.name != "MEMORY.md"] if mem_dir.is_dir() else []
    console.print(f"发现 Claude Code 数据：[bold]{len(session_files)}[/bold] 个会话，"
                  f"[bold]{len(memory_files)}[/bold] 条记忆（{cc_dir}）")
    if not session_files and not memory_files:
        console.print("[yellow]没有可迁移的内容[/yellow]")
        return 0

    if not assume_yes:
        answer = input("导入到 loongcli？已存在的同名条目会跳过 [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            console.print("[yellow]已取消，未写入任何内容[/yellow]")
            return 0

    mem_stats = import_memories(cc_dir)
    sess_stats = import_sessions(cc_dir)
    mem_extra = f" / 失败 {mem_stats['failed']}" if mem_stats["failed"] else ""
    sess_extra = f" / 空 {sess_stats['empty']}" if sess_stats["empty"] else ""
    console.print(
        f"[green]迁移完成[/green] — 记忆：导入 {mem_stats['imported']}"
        f" / 跳过 {mem_stats['skipped']}{mem_extra}；"
        f"会话：导入 {sess_stats['imported']} / 跳过 {sess_stats['skipped']}{sess_extra}"
    )
    if (Path.cwd() / "CLAUDE.md").exists():
        console.print("[dim]提示：CLAUDE.md 是项目指令而非记忆，未自动迁移——"
                      "如需生效请手动并入 LOONG.md[/dim]")
    return 0
