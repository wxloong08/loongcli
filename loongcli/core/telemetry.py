"""结构化事件流（JSONL 遥测）。

每次 LLM 调用/工具执行/compact/verify/recall 追加一行结构化事件到会话旁的
{session_id}.events.jsonl——回答「这轮 cache 命中多少、哪个工具吃了时间、
verify 重试了几轮」这类问题时靠翻文件，不靠临时 print / 探针。

设计约束：
- 遥测绝不能炸主循环：任何写入异常静默吞掉；连续失败 3 次自禁用（磁盘满/
  权限坏时不再每个事件都撞一次盘）。
- 零依赖、不 import 任何 loongcli 模块——谁都能安全 import 它，无环。
- append-only：事件是日志不是状态，与会话 JSON（就地重写的状态文件）职责分离。
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

# 连续写失败达到该次数后自禁用
_MAX_CONSECUTIVE_FAILURES = 3


class EventLogger:
    def __init__(self, path: Path | None, enabled: bool = True):
        self._path = path
        self._enabled = enabled and path is not None
        self._failures = 0

    @classmethod
    def for_session(cls, store) -> "EventLogger":
        """从 ConversationStore 派生会话级事件文件（{session_id}.events.jsonl）。

        store 为 None（无持久化的测试/临时 agent）→ disabled 实例，emit 全部空转。"""
        if store is None:
            return cls(path=None, enabled=False)
        try:
            path = Path(store.base_dir) / f"{store.session_id}.events.jsonl"
        except AttributeError:
            return cls(path=None, enabled=False)
        return cls(path=path)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def path(self) -> Path | None:
        return self._path

    def emit(self, type: str, **fields) -> None:
        if not self._enabled:
            return
        try:
            record = {"ts": datetime.now(timezone.utc).isoformat(), "type": type, **fields}
            # default=str：埋点处不小心传了 Path/枚举等不可序列化值时降级成字符串，
            # 而不是让一条遥测毁掉整个 agent 轮次
            line = json.dumps(record, ensure_ascii=False, default=str)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._failures = 0
        except Exception:
            self._failures += 1
            if self._failures >= _MAX_CONSECUTIVE_FAILURES:
                self._enabled = False


# 未注入 telemetry 时的空对象：埋点处无需 None 检查
NULL_TELEMETRY = EventLogger(path=None, enabled=False)
