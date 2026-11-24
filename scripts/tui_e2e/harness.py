"""TUI 端到端回归框架：ConPTY 驱动真实的 loongcli + 本地假模型服务（14f）。

为什么不用人工/截图：闪烁、抢焦点都是几十毫秒级的时序问题，人手和截图都抓不到、也复现不稳。
Windows Terminal 自己也是经 ConPTY 拿输出，这里录到的字节流就是 WT 收到的。

- FakeLLM：OpenAI 兼容 /chat/completions。主 agent 请求（带 system 消息）按剧本队列逐个应答；
  副调用（意图判断、摘要等，只有一条 user 消息）按规则应答。不花钱、确定性。
- Tui：pywinpty 起 `python -m loongcli`，USERPROFILE/LOCALAPPDATA 指向临时目录（不碰真实配置、记忆、会话），
  后台线程持续读输出喂给 pyte 还原屏幕，并逐块记录「输入框提示符是否在屏幕上」用于量化闪烁。
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pyte
from winpty import PtyProcess

# ── 按键 ──
ENTER, ESC, CTRL_C, CTRL_D, CTRL_J, SHIFT_TAB, TAB = "\r", "\x1b", "\x03", "\x04", "\n", "\x1b[Z", "\t"
UP, DOWN, BACKSPACE = "\x1b[A", "\x1b[B", "\x7f"


# ════════════════════════ 假模型 ════════════════════════

class Reply:
    """剧本里的一条主 agent 应答：先流式吐正文（可选），再发工具调用（可选）。"""

    def __init__(self, text: str = "", tools: list[tuple[str, dict]] | None = None,
                 chunk: int = 6, delay: float = 0.02, think: str = "", think_seconds: float = 0,
                 tool_seconds: float = 0, wait: float = 0):
        self.text, self.tools, self.chunk, self.delay = text, tools or [], chunk, delay
        self.wait = wait                    # 吐第一个 token 前先静默多少秒（模型「想」但不出思考内容）
        self.think, self.think_seconds = think, think_seconds    # 先流式吐的思考内容（reasoning_content）及耗时
        self.tool_seconds = tool_seconds    # 工具参数分多少秒流完（0 = 一块发出）；真模型写大文件时参数要流几十秒


def text(t: str, **kw) -> Reply:
    return Reply(text=t, **kw)


def tool(name: str, args: dict, say: str = "", **kw) -> Reply:
    return Reply(text=say, tools=[(name, args)], **kw)


class FakeLLM:
    def __init__(self):
        self.script: deque[Reply] = deque()
        self.main_requests: list[dict] = []     # 主 agent 收到的请求体（断言插话是否注入用）
        self.side_requests: list[dict] = []
        self.intent = "CONTINUE"                # goal 意图判断的回答
        self.default_text = "（剧本已用完）"
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def push(self, *replies: Reply) -> None:
        with self._lock:
            self.script.extend(replies)

    def close(self) -> None:
        self._server.shutdown()

    # ── HTTP ──
    def _handler(self):
        llm = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                msgs = body.get("messages") or []
                is_main = bool(msgs) and msgs[0].get("role") == "system"
                with llm._lock:
                    (llm.main_requests if is_main else llm.side_requests).append(body)
                    if is_main:
                        reply = llm.script.popleft() if llm.script else Reply(llm.default_text)
                    else:
                        prompt = json.dumps(msgs, ensure_ascii=False)
                        reply = Reply(llm.intent if "NEEDS_INPUT" in prompt else "摘要：测试会话。", delay=0)
                if body.get("stream"):
                    self._stream(reply, body)
                else:
                    self._json({"id": "x", "object": "chat.completion", "created": 0, "model": "fake",
                                "choices": [{"index": 0, "finish_reason": "stop",
                                             "message": {"role": "assistant", "content": reply.text}}],
                                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

            def _json(self, obj):
                data = json.dumps(obj, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _stream(self, reply: Reply, body: dict):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def send(obj):
                    data = ("data: " + (obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False))
                            + "\n\n").encode()
                    self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                    self.wfile.flush()

                def chunk(delta, finish=None):
                    send({"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "fake",
                          "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

                try:
                    chunk({"role": "assistant", "content": ""})
                    if reply.wait:
                        time.sleep(reply.wait)
                    if reply.think:
                        pieces = [reply.think[i:i + 4] for i in range(0, len(reply.think), 4)]
                        for piece in pieces:
                            chunk({"reasoning_content": piece})
                            time.sleep(reply.think_seconds / len(pieces))
                    t = reply.text
                    for i in range(0, len(t), max(1, reply.chunk)):
                        chunk({"content": t[i:i + reply.chunk]})
                        if reply.delay:
                            time.sleep(reply.delay)
                    for n, (name, args) in enumerate(reply.tools):
                        args = args(body) if callable(args) else args   # 参数可依赖请求内容（如上一步生成的 plan_id）
                        raw = json.dumps(args, ensure_ascii=False)
                        if not reply.tool_seconds:
                            chunk({"tool_calls": [{"index": n, "id": f"call_{time.time_ns()}_{n}", "type": "function",
                                                   "function": {"name": name, "arguments": raw}}]})
                            continue
                        # 像真模型一样：首块带 id 和名字，之后只续参数片段
                        pieces = [raw[i:i + 40] for i in range(0, len(raw), 40)]
                        chunk({"tool_calls": [{"index": n, "id": f"call_{time.time_ns()}_{n}", "type": "function",
                                               "function": {"name": name, "arguments": ""}}]})
                        for piece in pieces:
                            chunk({"tool_calls": [{"index": n, "function": {"arguments": piece}}]})
                            time.sleep(reply.tool_seconds / len(pieces))
                    chunk({}, "tool_calls" if reply.tools else "stop")
                    send({"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "fake", "choices": [],
                          "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})
                    send("[DONE]")
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass                         # 客户端中断（Esc）属正常

        return H


# ════════════════════════ 伪终端里的 loongcli ════════════════════════

PROMPT_RE = re.compile(r"^\s*(规划 )?(❯|\?) ")   # 输入框提示行（普通 ❯ / 规划模式「规划 ❯」/ 提问模式 ?）


class Tui:
    def __init__(self, llm: FakeLLM, files: dict[str, str] | None = None, rows: int = 40, cols: int = 120,
                 config_extra: dict | None = None):
        self.root = Path(tempfile.mkdtemp(prefix="loong-e2e-"))
        home, self.cwd = self.root / "home", self.root / "work"
        (home / ".loongcli").mkdir(parents=True)
        self.cwd.mkdir()
        (self.root / "local").mkdir()
        cfg = {"api_key": "fake", "model": "fake-model", "base_url": llm.base_url, "thinking": False,
               "mcpServers": {}, "hooks": {}, "providers": {}, "roles": {}, **(config_extra or {})}
        (home / ".loongcli" / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        for name, content in (files or {}).items():
            (self.cwd / name).write_text(content, encoding="utf-8")

        env = dict(os.environ)
        env.update(USERPROFILE=str(home), HOME=str(home), LOCALAPPDATA=str(self.root / "local"),
                   DSCLI_BASE_URL=llm.base_url, PYTHONIOENCODING="utf-8")
        for k in ("DEEPSEEK_API_KEY", "DSCLI_MODEL", "DSCLI_SUB_MODEL"):
            env.pop(k, None)
        self.rows, self.cols = rows, cols
        self.screen = pyte.HistoryScreen(cols, rows, history=20000)
        self.stream = pyte.Stream(self.screen)
        self.raw: list[tuple[float, str]] = []
        self.samples: list[tuple[float, int]] = []     # (时间, 提示行所在屏幕行；-1=不在屏幕上)，每读到一块输出采一次
        self._lock = threading.Lock()
        self.proc = PtyProcess.spawn([sys.executable, "-m", "loongcli"], cwd=str(self.cwd), env=env,
                                     dimensions=(rows, cols))
        self.t0 = time.monotonic()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while True:
            try:
                data = self.proc.read(65536)
            except EOFError:
                return
            except Exception:
                if not self.proc.isalive():
                    return
                continue
            if not data:
                if not self.proc.isalive():
                    return
                time.sleep(0.005)
                continue
            now = time.monotonic() - self.t0
            with self._lock:
                self.raw.append((now, data))
                self.stream.feed(data)
                self.samples.append((now, self._prompt_row()))

    # ── 屏幕 ──
    def lines(self) -> list[str]:
        with self._lock:
            return [l.rstrip() for l in self.screen.display]

    def _prompt_row(self) -> int:
        rows = [i for i, l in enumerate(self.screen.display) if PROMPT_RE.match(l)]
        return rows[-1] if rows else -1

    def text(self) -> str:
        return "\n".join(self.lines())

    def transcript(self) -> str:
        """滚动区历史 + 当前屏幕（断言「某内容最终出现过」用）。"""
        with self._lock:
            hist = ["".join(ch.data for ch in (line[x] for x in sorted(line))).rstrip()
                    for line in self.screen.history.top]
            return "\n".join(hist + [l.rstrip() for l in self.screen.display])

    def dump(self, path: Path) -> None:
        path.write_text(self.transcript(), encoding="utf-8")

    # ── 动作 ──
    def send(self, keys: str, settle: float = 0.15) -> None:
        self.proc.write(keys)
        time.sleep(settle)

    def resize(self, rows: int, cols: int, settle: float = 1.0) -> None:
        """改终端尺寸（像用户拖窗口）。ConPTY 会按新宽度重排已有内容再重画，录到的就是终端看到的样子。"""
        with self._lock:
            self.rows, self.cols = rows, cols
            self.screen.resize(rows, cols)
        self.proc.setwinsize(rows, cols)
        time.sleep(settle)

    def type(self, s: str, per_char: float = 0.02) -> None:
        for ch in s:
            self.proc.write(ch)
            time.sleep(per_char)

    def wait_for(self, pred, timeout: float = 15, what: str = "") -> bool:
        if isinstance(pred, str):
            needle = pred
            pred = lambda: needle in self.transcript()
            what = what or needle
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if pred():
                return True
            time.sleep(0.05)
        raise TimeoutError(f"等待超时：{what}\n---- 屏幕 ----\n{self.text()}")

    def wait_idle_prompt(self, timeout: float = 20) -> None:
        """等到底部是空闲的 ❯ 提示行（无转圈、无提问）。"""
        def ok():
            ls = self.lines()
            row = self._prompt_row()
            if row < 0 or ls[row].strip() not in ("❯", "规划 ❯"):
                return False
            above = [l for l in ls[max(0, row - 4):row] if l.strip()]
            return not any(l.lstrip()[:1] in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" or "⚠" in l for l in above)
        self.wait_for(ok, timeout, "空闲提示符")

    def flicker(self, t_from: float, t_to: float) -> dict:
        """[t_from, t_to] 时间窗内的输入框稳定性。

        drops：提示行从屏幕上消失的次数（擦了还没重画就被终端画出来 = 闪一下）；
        jumps：提示行往上跳的次数（动态区忽高忽低，输入框跟着上下动——屏幕满了以后它应该一直钉在底部）。"""
        with self._lock:
            win = [(t, r) for t, r in self.samples if t_from <= t <= t_to]
        rows = [r for _, r in win]
        gone = sum(1 for r in rows if r < 0)
        drops = sum(1 for a, b in zip(rows, rows[1:]) if a >= 0 > b)
        seen = [r for r in rows if r >= 0]
        jumps = sum(1 for a, b in zip(seen, seen[1:]) if b < a)
        return {"chunks": len(win), "prompt_missing": gone, "drops": drops, "jumps": jumps,
                "seconds": round(t_to - t_from, 2)}

    def now(self) -> float:
        return time.monotonic() - self.t0

    def close(self) -> None:
        try:
            if self.proc.isalive():
                self.proc.terminate(force=True)
        except Exception:
            pass
