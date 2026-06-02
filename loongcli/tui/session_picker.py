from __future__ import annotations
import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from loongcli.memory.conversation import ConversationStore


def _relative_time(iso_str: str) -> str:
    if not iso_str:
        return "?"
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return "刚刚"
        if seconds < 3600:
            return f"{seconds // 60} 分钟前"
        if seconds < 86400:
            return f"{seconds // 3600} 小时前"
        days = seconds // 86400
        if days < 30:
            return f"{days} 天前"
        if days < 365:
            return f"{days // 30} 个月前"
        return f"{days // 365} 年前"
    except (ValueError, OSError):
        return "?"


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes}B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    return f"{size_bytes / (1024 * 1024):.1f}MB"


@dataclass
class PickerState:
    sessions: list[dict] = field(default_factory=list)
    cursor: int = 0
    search: str = ""
    scroll_offset: int = 0
    preview_id: str | None = None
    preview_lines: list[str] = field(default_factory=list)
    page_size: int = 8

    @property
    def filtered(self) -> list[dict]:
        if not self.search:
            return self.sessions
        q = self.search.lower()
        return [s for s in self.sessions if q in s.get("title", "").lower()]

    def move(self, delta: int):
        items = self.filtered
        if not items:
            return
        self.cursor = max(0, min(len(items) - 1, self.cursor + delta))
        if self.cursor < self.scroll_offset:
            self.scroll_offset = self.cursor
        elif self.cursor >= self.scroll_offset + self.page_size:
            self.scroll_offset = self.cursor - self.page_size + 1

    def page_move(self, direction: int):
        self.move(direction * self.page_size)

    def selected(self) -> dict | None:
        items = self.filtered
        if 0 <= self.cursor < len(items):
            return items[self.cursor]
        return None

    def reset_cursor(self):
        self.cursor = 0
        self.scroll_offset = 0
        self.preview_id = None
        self.preview_lines = []


def render_picker(state: PickerState) -> Panel:
    items = state.filtered
    total = len(state.sessions)
    showing = len(items)

    lines = Text()

    search_display = state.search if state.search else ""
    lines.append("  ⌕ ", style="cyan")
    if search_display:
        lines.append(search_display, style="bold white")
    else:
        lines.append("输入搜索...", style="dim")
    lines.append("\n\n")

    if not items:
        lines.append("  没有匹配的会话\n", style="dim")
    else:
        visible = items[state.scroll_offset:state.scroll_offset + state.page_size]

        if state.scroll_offset > 0:
            lines.append("  ↑ 更多\n", style="dim")

        for i, meta in enumerate(visible):
            idx = state.scroll_offset + i
            is_selected = idx == state.cursor
            prefix = " > " if is_selected else "   "
            title = meta.get("title", "(无标题)").replace("\n", " ").strip()
            if len(title) > 50:
                title = title[:50] + "..."

            style = "bold cyan" if is_selected else ""
            lines.append(prefix, style=style)
            lines.append(f"{title}\n", style=style)

            time_str = _relative_time(meta.get("updated_at", ""))
            turns = meta.get("turn_count", "?")
            size = _format_size(meta.get("file_size", 0))
            detail = f"   {time_str} · {turns} 轮 · {size}\n"
            lines.append(detail, style="dim")

            if state.preview_id == meta.get("session_id") and state.preview_lines:
                for pl in state.preview_lines:
                    lines.append(f"     {pl}\n", style="dim italic")

            if i < len(visible) - 1:
                lines.append("\n")

        if state.scroll_offset + state.page_size < len(items):
            lines.append("\n  ↓ 更多", style="dim")

    title = f"恢复会话 ({showing}/{total})" if state.search else f"恢复会话 ({total})"
    footer = "↑↓ 导航 · Enter 选择 · Space 预览 · Esc 取消 · 输入搜索"

    return Panel(
        lines,
        title=title,
        subtitle=footer,
        subtitle_align="center",
        border_style="cyan",
        padding=(0, 1),
    )


async def _read_key() -> str:
    if sys.platform == "win32":
        import msvcrt
        loop = asyncio.get_running_loop()
        ch = await loop.run_in_executor(None, msvcrt.getwch)
        if ch in ("\x00", "\xe0"):
            ch2 = await loop.run_in_executor(None, msvcrt.getwch)
            mapping = {"H": "up", "P": "down", "I": "pageup", "Q": "pagedown"}
            return mapping.get(ch2, "")
        if ch == "\r":
            return "enter"
        if ch == "\x1b":
            return "escape"
        if ch == " ":
            return "space"
        if ch == "\x08":
            return "backspace"
        return ch
    else:
        import tty
        import termios
        loop = asyncio.get_running_loop()
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = await loop.run_in_executor(None, lambda: sys.stdin.read(1))
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        if ch == "\x1b":
            ch2 = await loop.run_in_executor(None, lambda: sys.stdin.read(1))
            if ch2 == "[":
                ch3 = await loop.run_in_executor(None, lambda: sys.stdin.read(1))
                mapping = {"A": "up", "B": "down", "5": "pageup", "6": "pagedown"}
                if ch3 in ("5", "6"):
                    await loop.run_in_executor(None, lambda: sys.stdin.read(1))
                return mapping.get(ch3, "")
            return "escape"
        if ch == "\r" or ch == "\n":
            return "enter"
        if ch == " ":
            return "space"
        if ch == "\x7f" or ch == "\x08":
            return "backspace"
        if ch == "\x03":
            return "escape"
        return ch


class SessionPicker:
    def __init__(self, console: Console, conversation: ConversationStore):
        self.console = console
        self.conversation = conversation

    def _load_preview(self, session_id: str) -> list[str]:
        data = self.conversation.load(session_id)
        if not data:
            return ["(无法加载)"]
        messages = data.get("messages", [])
        lines = []
        for m in messages:
            if m.get("role") == "user":
                text = m.get("content", "")
                preview = text if len(text) <= 60 else text[:60] + "..."
                lines.append(f"● {preview}")
                if len(lines) >= 5:
                    break
        return lines or ["(无用户消息)"]

    async def pick(self) -> str | None:
        sessions = self.conversation.list_sessions(limit=50)
        if not sessions:
            self.console.print("[yellow]没有历史会话[/yellow]")
            return None

        state = PickerState(sessions=sessions)

        with Live(render_picker(state), console=self.console, refresh_per_second=12, screen=False) as live:
            while True:
                key = await _read_key()

                if key == "up":
                    state.move(-1)
                elif key == "down":
                    state.move(1)
                elif key == "pageup":
                    state.page_move(-1)
                elif key == "pagedown":
                    state.page_move(1)
                elif key == "enter":
                    sel = state.selected()
                    if sel:
                        return sel["session_id"]
                elif key == "escape":
                    return None
                elif key == "space":
                    sel = state.selected()
                    if sel:
                        sid = sel["session_id"]
                        if state.preview_id == sid:
                            state.preview_id = None
                            state.preview_lines = []
                        else:
                            state.preview_id = sid
                            state.preview_lines = self._load_preview(sid)
                elif key == "backspace":
                    if state.search:
                        state.search = state.search[:-1]
                        state.reset_cursor()
                elif len(key) == 1 and key.isprintable():
                    state.search += key
                    state.reset_cursor()
                else:
                    continue

                live.update(render_picker(state))
