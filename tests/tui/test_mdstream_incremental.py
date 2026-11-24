"""流式 Markdown 的增量渲染（mdstream.MarkdownStream）：只重渲染还在写的那一块，写完的块定稿打印。

安全网：逐字喂完一篇文档，滚动区的最终输出必须和整篇一次性渲染完全一致；并且每次更新的渲染量
不随全文长度增长（真机：长回复越写越卡，渲染耗时 × 10 的节流把刷新拖到一秒三次）。"""
from __future__ import annotations

import io

import pytest

import loongcli.tui  # noqa: F401  中日韩断行补丁
from rich.console import Console

from loongcli.tui import mdstream
from loongcli.tui.mdstream import MarkdownStream

DOCS = {
    "标题与段落": "# 一级标题\n\n第一段，中文和 English 混排，`inline code` 也在。\n\n## 二级标题\n\n第二段。\n\n### 三级\n\n收尾。\n",
    "紧凑列表": "要点：\n\n- 甲\n- 乙\n- 丙\n\n后面的段落。\n",
    "松散列表": "- 第一项\n\n- 第二项\n\n- 第三项\n\n结束。\n",
    "有序列表": "步骤：\n\n1. 读文件\n2. 改文件\n\n3. 跑测试\n\n完成。\n",
    "代码块含空行": "看代码：\n\n```python\nimport asyncio\n\n\nasync def main():\n    await asyncio.sleep(0)\n```\n\n说明在后面。\n",
    "表格": "对照：\n\n| API | 用途 |\n|---|---|\n| run | 入口 |\n| sleep | 让出 |\n\n表后段落。\n",
    # 末行很长、把列撑宽：表头和前几行要是写到一半就先打出去，列宽会和后面的行对不上（真机：表头错位）
    "表格末行撑宽列": ("自查：\n\n| 行 | 陈述 | 依据 | 结论 |\n|---|---|---|---|\n| 1 | 短 | 文档 | 通过 |\n"
                    "| 2 | 短 | 文档 | 通过 |\n| 3 | 短 | 文档 | 通过 |\n| 4 | 短 | 文档 | 通过 |\n"
                    "| 5 | gather 按传入顺序返回、默认异常立即上抛且其余不取消、return_exceptions=True 收异常 "
                    "| Python 3.13.7 实测 | 通过 |\n\n结束。\n"),
    "引用与分隔线":"> 引用一行\n> 引用二行\n\n---\n\n正文。\n",
    "长中文段落": ("asyncio 是单线程协作式并发，事件循环只在 `await` 处才可能切换任务；" * 6 + "\n\n") * 3,
    "嵌套列表续行": "- 外层\n  - 内层\n\n  外层的续段\n\n- 下一项\n",
}


def _console(buf):
    return Console(file=buf, width=80, force_terminal=True, color_system="truecolor")


def _reference(text: str) -> str:
    buf = io.StringIO()
    ref = MarkdownStream(_console(buf), live_target=lambda r, status=False: None)
    ref._emit_stable(ref._render_to_lines(text))       # 整篇一次性渲染
    return buf.getvalue()


def _stream(text: str, step: int) -> tuple[str, MarkdownStream]:
    buf = io.StringIO()
    md = MarkdownStream(_console(buf), live_target=lambda r, status=False: None)
    for end in range(step, len(text), step):
        md.when = 0.0                                   # 关掉节流：每个前缀都真的渲染一次
        md.update(text[:end])
    md.update(text, final=True)
    return buf.getvalue(), md


@pytest.mark.parametrize("name", list(DOCS))
@pytest.mark.parametrize("step", [1, 7])
def test_streamed_output_equals_whole_render(name, step):
    text = DOCS[name]
    got, _ = _stream(text, step)
    assert got == _reference(text), name


def test_printed_lines_never_come_back_to_tail():
    """打进滚动区的行不能再出现在动态区：表格开头时曾把同一块里已打印的段落行又放回动态区，
    定稿时动态区多缩一截、输入框跳（e2e metric_jitter_markdown 偶发上跳）。"""
    import re as _re
    para = " ".join(f"P{i:02d}" + "说明文字" * 6 for i in range(30))
    text = para + "\n\n- 要点一\n- 要点二\n\n| 接口 | 用途 |\n|---|---|\n| run | 入口 |\n\n完。\n"
    buf = io.StringIO()
    tails = []

    def target(r, status=False):
        if r is not None:
            c = Console(file=io.StringIO(), width=80, force_terminal=False)
            c.print(r)
            tails.append((buf.getvalue(), c.file.getvalue()))

    md = MarkdownStream(Console(file=buf, width=80, force_terminal=False), live_target=target)
    for end in range(1, len(text)):          # 逐字喂：表格的「|」单独到达、还够不上定稿门槛的那一刻才会出事
        md.when = 0.0
        md.update(text[:end])
    md.update(text, final=True)
    for printed, tail in tails:
        dup = set(_re.findall(r"P\d\d", printed)) & set(_re.findall(r"P\d\d", tail))
        assert not dup, f"已打印的行又出现在动态区：{sorted(dup)[:5]}"


def test_render_cost_does_not_grow_with_reply_length(monkeypatch):
    """每次更新只渲染还没定稿的那一块：喂一篇很长的回复，单次渲染的字数上限 ≈ 一块，而不是全文。"""
    sizes = []
    real = MarkdownStream._render_to_lines
    monkeypatch.setattr(MarkdownStream, "_render_to_lines", lambda self, t: (sizes.append(len(t)), real(self, t))[1])
    text = ("## 小节\n\n" + "段落文字，中英混排 mixed text。" * 8 + "\n\n") * 50
    _stream(text, 40)
    assert len(text) > 8000
    assert max(sizes) < 600, max(sizes)
