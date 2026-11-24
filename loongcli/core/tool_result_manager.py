"""工具结果的统一兜底：单个结果超过 HARD_CAP 就全文落盘，上下文只留头尾预览 + 文件路径。

定位是保险丝，不是调节阀——「每个工具该返回多少」由工具自己决定（read_file 按行分页、
delegate 族按任务数均摊并附 trace 指针、skill 自管上限），这里只防失控的巨型结果
（一条命令吐出几 MB 日志、某个 MCP 返回整页 HTML）把上下文撑爆。

2026-11-19 由「8000 字符一刀切 + 豁免名单」改成现在这样。旧设计用一个小阈值替所有工具决定
返回量，不合适的就开豁免（delegate、batch_delegate、task_status、wait_tasks、skill、read_file），
名单越开越长；全部 152 个历史会话里它截过的 48 次全是 8K–40K 的正常大小结果，「每次回复累计 30K」
闸从未触发，原文超 40K 的一次都没有——它防的东西没出现，截掉的却是模型要的内容。
别家同样是保险丝思路：Gemini CLI 40,000 字符、OpenHands 30,000 字符、Codex 约 1 万 token，
超限都是写盘给路径或掐中段；「请重新调用工具」这条恢复路径没有一家用（对不可重放的工具它是死循环）。
"""
from __future__ import annotations
import logging
import re
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# 高于所有自管上限（skill 内容 50K + 外层说明、read_file 40K、delegate 族 36K），正常结果碰不到。
HARD_CAP = 60_000
PREVIEW_HEAD = 3_000   # 头部：命令、输入回显、首部报错
PREVIEW_TAIL = 1_000   # 尾部：测试总结行、traceback 里真正的错误行常在末尾


class ToolResultManager:
    def __init__(self, spill_dir: Path | None = None):
        # 落盘目录；None = 不落盘（测试默认），超限时只给头尾预览并如实说明。
        self._spill_dir = spill_dir

    def process(self, tool_name: str, result: str) -> str:
        """结果不超过 HARD_CAP 原样返回；超过则全文落盘，返回头尾预览 + 文件路径。"""
        if len(result) <= HARD_CAP:
            return result
        path = self._spill(tool_name, result)
        if path:
            recovery = (
                f"完整结果已落盘：{path}。找特定内容（报错、失败用例）用 grep 在该文件里搜关键字，"
                "要通读用 read_file 的 offset 分页读取；不要重跑本工具（结果还是这么大）"
            )
        else:
            recovery = "完整结果未能落盘；需要被省略的部分，请缩小范围重新获取（加过滤条件或分段参数）"
        return _head_tail_preview(result, recovery)

    def _spill(self, tool_name: str, result: str) -> str | None:
        """把完整结果写入落盘目录，返回文件路径；无目录或写盘失败返回 None。

        文件名 = 工具名（清洗成文件名安全字符）+ uuid——主/子代理共用同一项目级目录，
        uuid 保证无碰撞；工具名来自外部 MCP 服务器时不能带路径分隔符。目录懒建，首次落盘才 mkdir。"""
        if self._spill_dir is None:
            return None
        try:
            self._spill_dir.mkdir(parents=True, exist_ok=True)
            prefix = re.sub(r"[^A-Za-z0-9_-]", "_", tool_name)[:40] or "tool"
            path = self._spill_dir / f"{prefix}-{uuid.uuid4().hex[:12]}.txt"
            path.write_text(result, encoding="utf-8")
            return str(path)
        except Exception:
            logger.debug("工具结果落盘失败，回落纯头尾预览", exc_info=True)
            return None


def _head_tail_preview(result: str, recovery: str) -> str:
    """留头 PREVIEW_HEAD + 尾 PREVIEW_TAIL，中段换成「已截断」标记（含省略字数与原文总字数）。

    标记必须显式：模型对无声的截断几乎察觉不到，有明确提示才会去取被省略的部分。"""
    total = len(result)
    omitted = total - PREVIEW_HEAD - PREVIEW_TAIL
    marker = f"\n\n[... 已截断：中段省略 {omitted} 字符（原文共 {total} 字符）。{recovery} ...]\n\n"
    return result[:PREVIEW_HEAD] + marker + result[-PREVIEW_TAIL:]
