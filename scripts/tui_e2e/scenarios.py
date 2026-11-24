"""TUI 回归场景。用法：python scripts/tui_e2e/scenarios.py [场景名 ...]

每个场景 = 新起一个假模型 + 一个隔离的 loongcli，按时间轴送键、断言。失败时把整段滚动区落盘。
场景名前缀：bug_ = 14e 真机发现的问题（修之前应 FAIL）；guard_ = 已经正常的行为（防修坏）；metric_ = 量化指标。
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import (  # noqa: E402
    DOWN, ENTER, ESC, SHIFT_TAB, FakeLLM, Reply, Tui, text, tool,
)

OUT = Path(tempfile.gettempdir()) / "loong-e2e-out"


def slow(t: str, seconds: float, chunk: int = 4, **kw) -> Reply:
    """在约 seconds 秒内流完的正文，制造「模型正在输出」的时间窗。"""
    n = max(1, len(t) // chunk)
    return Reply(text=t, chunk=chunk, delay=seconds / n, **kw)


def plan_create(title: str) -> dict:
    return {"operation": "create", "title": title, "approach": f"{title}：背景与范围、关键决定、风险",
            "steps": [{"description": f"{title}：改 a.txt", "criteria": ["a.txt 里有 done"]}]}


def input_line(t: Tui) -> str:
    """底部输入框当前内容（最后一个 ❯ 行去掉提示符）。"""
    for l in reversed(t.lines()):
        s = l.lstrip().removeprefix("规划 ")
        if s.startswith("❯"):
            return s[1:].strip()
    return ""


def _edge_hits(t: Tui) -> list[str]:
    """滚动区里「最后一列是框线 / 方块字符或带背景色」的行——窗口拉窄时终端会把它们折碎（满宽框的右边框、满宽线、代码块色带）。

    纯文字顶到最后一列不算：终端重排时只是正常折行，照样能读（长路径、填满的中文段落）。
    输入框（❯ 上横线往下）是程序自己重画的，不查。"""
    cols = t.screen.columns
    with t._lock:
        rows = [dict(l) for l in t.screen.history.top]
        display = [dict(t.screen.buffer[y]) for y in range(t.screen.lines)]
    prompt = t._prompt_row()
    rows += display[:prompt - 1] if prompt > 0 else display
    hits = []
    for row in rows:
        ch = row.get(cols - 1)
        if ch is not None and ("─" <= ch.data[:1] <= "▟" or ch.bg != "default"):    # 框线 U+2500–257F、方块 U+2580–259F
            hits.append("".join(row[x].data for x in sorted(row)).rstrip())
    return hits


def _assert_reflow_safe(t: Tui, what: str) -> None:
    hits = _edge_hits(t)
    assert not hits, f"{what}：有 {len(hits)} 行顶到最后一列（拉窄窗口会被终端折碎），如：{hits[0][:60]!r}"


def start(t: Tui) -> None:
    t.wait_for("查看命令", 30)
    time.sleep(0.8)


# ════════════════ bug_：14e 发现的问题 ════════════════

def bug_slash_completion(llm: FakeLLM, t: Tui):
    """输入 / 应弹出命令补全菜单（banner 里没有 /goal、/compact，出现在屏幕上即菜单出来了）。

    先输出一屏内容把输入框推到终端最底部——真机上就是这个状态；刚启动时输入框在顶部、下方空间充足，测不出来。"""
    llm.push(text("\n".join(f"- 第 {i} 行" for i in range(1, 61)) + "\n", delay=0))
    start(t)
    t.type("刷一屏")
    t.send(ENTER)
    t.wait_for("第 60 行", 10)
    t.wait_idle_prompt()
    t.type("/")
    time.sleep(1.5)
    screen = t.text()
    assert "/goal" in screen or "/compact" in screen, "输入 / 后没有出现补全菜单"


def bug_confirm_eats_draft(llm: FakeLLM, t: Tui):
    """模型输出时正在打插话草稿，确认框弹出：选「执行」应确认执行，草稿保留在输入框，不被当成确认的答案。"""
    llm.push(slow("我先看看这个文件，然后把它删掉。" * 6, 3, tools=[("shell", {"command": "rm a.txt"})]),
             text("删完了。"))
    start(t)
    t.type("删掉 a.txt")
    t.send(ENTER)
    t.wait_for("我先看看", 10)
    t.type("顺便加注释")
    t.wait_for(lambda: "要执行这个操作吗" in t.text(), 15, "确认框")
    t.send(ENTER, settle=1.0)              # 默认选中「执行」
    t.wait_for(lambda: "要执行这个操作吗？ →" in t.transcript(), 5, "确认结果")
    assert "要执行这个操作吗？ → 执行" in t.transcript(), "选「执行」没有确认执行（草稿被当成了答案）"
    t.wait_for("删完了", 10)
    time.sleep(0.5)
    assert "顺便加注释" in input_line(t), f"插话草稿丢了，输入框现在是：{input_line(t)!r}"


def bug_ask_leaves_no_trace(llm: FakeLLM, t: Tui):
    """确认框的问题和回答应留在滚动区（事后能看到当时答了什么）。"""
    llm.push(tool("shell", {"command": "rm a.txt"}), text("好了。"))
    start(t)
    t.type("删掉 a.txt")
    t.send(ENTER)
    t.wait_for(lambda: "要执行这个操作吗" in t.text(), 15, "确认框")
    t.send("1", settle=0.5)
    t.wait_for("好了。", 10)
    assert "要执行这个操作吗？ → 执行" in t.transcript(), "滚动区里没有留下「要执行这个操作吗？ → 执行」这一问一答"


def guard_confirm_reject_with_feedback(llm: FakeLLM, t: Tui):
    """确认框选「不执行，告诉模型怎么改」：在这一行直接写说明（不再弹第二个提问），说明随拒绝结果回给模型。"""
    llm.push(tool("shell", {"command": "rm a.txt"}), text("明白，改用回收站。"))
    start(t)
    t.type("删掉 a.txt")
    t.send(ENTER)
    t.wait_for(lambda: "要执行这个操作吗" in t.text(), 15, "确认框")
    t.send(DOWN, settle=0.3)
    t.type("别删，移到回收站")
    t.wait_for(lambda: "2. 不执行，告诉模型怎么改：别删，移到回收站" in t.text(), 3, "说明直接打在选项行上")
    t.send(ENTER)
    t.wait_for("明白，改用回收站", 10)
    sent = json.dumps(llm.main_requests[-1]["messages"], ensure_ascii=False)
    assert "用户说明：别删，移到回收站" in sent, "拒绝说明没有回给模型"
    tr = t.transcript()
    assert "要执行这个操作吗？ → 不执行，告诉模型怎么改：别删，移到回收站" in tr, "拒绝和说明没留在滚动区"
    assert "✓ 已确认" not in tr and "✗ 已拒绝" not in tr, "确认结果又说了一遍（问答留痕已经有了）"


def _goal_with_abandoned_plan(llm: FakeLLM, t: Tui):
    """先跑一个目标、审批时取消 → 留下一个已放弃的「计划A」。"""
    llm.push(tool("plan", plan_create("计划A")), text("计划已建好。"))
    start(t)
    t.type("/goal 目标一")
    t.send(ENTER)
    t.wait_for(lambda: "开始执行这个计划吗" in t.text(), 15, "目标一审批")
    t.wait_for(lambda: "计划A：改 a.txt" in t.transcript(), 5, "goal 计划框")
    _assert_reflow_safe(t, "goal 计划框")
    tr = t.transcript()
    # 输出精简（用户：比 Claude Code / Codex 话多）：目标原文只在回显里出现一次、plan 工具一行进度、不按轮打 tokens
    assert tr.count("目标一") == 1, "目标原文被重复回显"
    assert "● plan(" not in tr and "◆ 计划：已创建「计划A」（1 步）" in tr, "plan 工具没有收成一行进度"
    assert "tokens:" not in tr, "默认模式仍按轮打 tokens 统计行"
    t.send("2")                                      # 1 执行 · 2 取消 · 3 提修改意见（输入项）
    t.wait_for("计划已取消", 5)


def guard_goal_questions_then_plan(llm: FakeLLM, t: Tui):
    """设计类目标先问再规划（todo 23.3）：第一规划回合只提问不建计划 → 弹三选一 → 在输入项打回答 →
    回答进入第二规划回合 → 建计划 → 审批框里方案在步骤上方。"""
    llm.push(text("插入用剪贴板还是逐字输入？我推荐剪贴板：中文输入法下逐字会被拦。"))
    llm.push(tool("plan", plan_create("计划Q")), text("计划已建好。"))
    start(t)
    t.type("/goal 写架构稿")
    t.send(ENTER)
    t.wait_for(lambda: "回答上面的问题" in t.text(), 15, "问答三选一")
    assert "重试" not in t.transcript(), "模型在提问，却被当成没建成计划去重试"
    t.send(DOWN, settle=0.3)
    t.send(DOWN, settle=0.3)
    t.type("用剪贴板，但要恢复原内容")
    t.wait_for(lambda: "用剪贴板，但要恢复原内容" in t.text(), 3, "回答打在输入项上")
    t.send(ENTER)
    t.wait_for(lambda: "开始执行这个计划吗" in t.text(), 15, "第二回合建了计划进入审批")
    sent = json.dumps(llm.main_requests[-1], ensure_ascii=False)
    assert "用剪贴板，但要恢复原内容" in sent, "用户的回答没进入下一个规划回合"
    tr = t.transcript()
    assert tr.index("计划Q：背景与范围") < tr.index("计划Q：改 a.txt"), "方案没显示在步骤上方"
    t.send("2")
    t.wait_for("计划已取消", 5)


def bug_goal_esc_in_planning(llm: FakeLLM, t: Tui):
    """goal 规划阶段按 Esc：应取消本次目标，不能拿上一个已放弃的计划去审批、更不能执行。"""
    _goal_with_abandoned_plan(llm, t)
    seen_a = t.transcript().count("计划A")
    llm.push(slow("我先调研一下现有代码。" * 10, 5))
    t.type("/goal 目标二")
    t.send(ENTER)
    t.wait_for(lambda: t.transcript().count("目标模式 · 规划中") >= 2 and "我先调研" in t.transcript(), 10, "目标二规划中")
    t.send(ESC, settle=2.5)
    tr = t.transcript()
    assert tr.count("计划A") == seen_a, "Esc 之后又拿已放弃的「计划A」出来审批了"
    assert "目标模式 · 执行中" not in tr, "Esc 之后进入了执行阶段"


def bug_goal_esc_at_approval(llm: FakeLLM, t: Tui):
    """goal 计划审批时按 Esc：应等同取消，不能当成「回车=执行」。"""
    llm.push(tool("plan", plan_create("计划B")), text("计划已建好。"))
    start(t)
    t.type("/goal 目标")
    t.send(ENTER)
    t.wait_for(lambda: "开始执行这个计划吗" in t.text(), 15, "审批对话框")
    t.send(ESC, settle=2.0)
    tr = t.transcript()
    assert "目标模式 · 执行中" not in tr, "审批时按 Esc 却开始执行了"
    assert "计划已取消" in tr, "审批时按 Esc 没有取消计划"


def bug_goal_stop_in_planning(llm: FakeLLM, t: Tui):
    """goal 规划阶段发 /goal stop：应结束目标，不应提示「运行中不支持斜杠命令」。"""
    llm.push(slow("我先调研一下现有代码。" * 10, 5), tool("plan", plan_create("计划C")), text("计划已建好。"))
    start(t)
    t.type("/goal 目标")
    t.send(ENTER)
    t.wait_for("我先调研", 10)
    t.type("/goal stop")
    t.send(ENTER, settle=3.0)
    tr = t.transcript()
    assert "运行中不支持斜杠命令" not in tr, "规划阶段的 /goal stop 被拒绝了"
    assert "开始执行这个计划吗" not in t.text(), "/goal stop 之后仍弹出了计划审批"


def bug_unknown_tool_param(llm: FakeLLM, t: Tui):
    """模型给工具传了不存在的参数：应得到「未知参数」的明确报错，而不是 Python 异常原文。"""
    llm.push(tool("plan", {"operation": "list", "description": "多余的参数"}), text("知道了。"))
    start(t)
    t.type("列出计划")
    t.send(ENTER)
    t.wait_for("知道了。", 15)
    tr = t.transcript()
    assert "unexpected keyword argument" not in tr, "工具报错是 Python 异常原文"
    sent = json.dumps(llm.main_requests[-1]["messages"][-2:], ensure_ascii=False)
    assert "未知参数" in sent, "回给模型的工具结果里没有说明是哪个参数不认识"


# ════════════════ metric_：量化 ════════════════

def metric_flicker(llm: FakeLLM, t: Tui):
    """流式输出 60 行 markdown 期间，输入框提示行从屏幕上消失了几次（WT 会把这些中间帧画出来 = 闪烁）。"""
    body = "\n".join(f"- 第 {i} 行：这是一段用来测试流式渲染是否闪烁的正文内容。" for i in range(1, 61)) + "\n\n全文完。\n"
    llm.push(slow(body, 8, chunk=8))
    start(t)
    t_idle = t.now()
    time.sleep(2)
    idle = t.flicker(t_idle, t.now())
    t.type("写 60 行")
    t.send(ENTER)
    t.wait_for("第 1 行", 10)
    t0 = t.now()
    t.wait_for("全文完", 20)
    m = t.flicker(t0, t.now())
    print(f"    空闲 2 秒：{idle}")
    print(f"    流式输出：{m}")
    assert m["drops"] == 0, f"流式输出期间输入框消失 {m['drops']} 次（{m['prompt_missing']}/{m['chunks']} 个输出块里不在屏幕上）"


def _cheatsheet() -> str:
    """贴近真实的长回答：小节标题、会折行的长段落、列表、表格、代码块（真机「文字多了就上下跳」的场景）。"""
    parts = []
    for i in range(1, 6):
        parts.append(f"## 第 {i} 节：asyncio 要点\n\n" + "这一段故意写得很长，用来模拟模型输出的大段说明文字，" * 6 + "\n")
        parts.append("\n".join(f"- 要点 {i}.{j}：`asyncio.gather` 与 `asyncio.wait` 的区别在于返回值和异常传播方式" for j in range(1, 5)) + "\n")
        parts.append("| 接口 | 用途 | 注意事项 |\n|---|---|---|\n" + "\n".join(
            f"| create_task {j} | 并发调度协程 | 要保留引用，否则可能被回收 |" for j in range(1, 4)) + "\n")
        parts.append("```python\nimport asyncio\n\nasync def main():\n    results = await asyncio.gather(*[work(n) for n in range(10)])\n"
                     "    print(results)\n\nasyncio.run(main())\n```\n")
    return "\n".join(parts) + "\n全文完。\n"


def metric_jitter_markdown(llm: FakeLLM, t: Tui):
    """长 markdown 流式输出、屏幕已满之后：输入框不应消失（drops），也不应上下跳（jumps）。"""
    llm.push(slow(_cheatsheet(), 10, chunk=6))
    start(t)
    t.type("写速查表")
    t.send(ENTER)
    t.wait_for(lambda: t._prompt_row() >= t.rows - 3, 15, "屏幕被占满")
    t0 = t.now()
    t.wait_for("全文完", 30)
    m = t.flicker(t0, t.now())
    print(f"    屏幕满后流式输出：{m}")
    assert m["drops"] == 0 and m["jumps"] == 0, f"输入框消失 {m['drops']} 次、上跳 {m['jumps']} 次"


def metric_thinking_spinner(llm: FakeLLM, t: Tui):
    """思考期间只转圈 + 计时，不逐字显示思考内容；输入框不消失、不跳（用户 2026-11-21：思考行多了输入框会闪）。"""
    llm.push(Reply(text="结论：没问题。", think="先看看这个问题，推理片段甲乙丙，" * 30, think_seconds=4))
    start(t)
    t.type("想一想")
    t.send(ENTER)
    t.wait_for(lambda: "思考中… 1s" in t.text(), 5, "思考计时到 1s")
    ls, row = t.lines(), next(i for i, l in enumerate(t.lines()) if "思考中…" in l)
    assert ls[row - 1].strip() == "" and ls[row + 1].strip() == "" and set(ls[row + 2]) == {"─"}, \
        "思考状态行上下没有空行（用户：贴着正文和输入框太密集）：\n" + "\n".join(ls[row - 2:row + 3])
    t0 = t.now()
    t.wait_for(lambda: "思考中… 3s" in t.text(), 5, "思考计时在走")
    m = t.flicker(t0, t.now())
    print(f"    思考期间：{m}")
    t.wait_for("结论：没问题。", 10)
    assert "推理片段" not in t.transcript(), "思考内容被显示出来了"
    assert m["drops"] == 0 and m["jumps"] == 0, f"思考期间输入框消失 {m['drops']} 次、上跳 {m['jumps']} 次"


def metric_tool_output(llm: FakeLLM, t: Tui):
    """工具执行中逐行出输出（⎿ 输出行）：输入框不消失、不跳（用户 2026-11-22：执行工具时 ❯ 闪）。
    命令带换行（heredoc），同真机截图。"""
    cmd = ("python - <<'PY'\nimport time\nfor i in range(16):\n"
           "    print(f'block {i}: ast OK, run rc=0', flush=True); time.sleep(0.25)\nPY")
    llm.push(tool("shell", {"command": cmd}), text("跑完了。"))
    start(t)
    t.type("跑一下")
    t.send(ENTER)
    t.wait_for(lambda: "block 1:" in t.text() or "要执行这个操作吗" in t.text(), 15, "命令输出或确认框")
    if "要执行这个操作吗" in t.text():
        t.send("1")
        t.wait_for(lambda: "block 1:" in t.text(), 10, "命令输出")
    t0 = t.now()
    # 窗口要盖到工具结束：结束那一下（撤输出行 → 打结果块）是真机闪的地方之一
    t.wait_for(lambda: "⎿ block 0:" in t.transcript() and "… +13 行" in t.transcript(), 15, "工具结果块（前 3 行 + 剩余行数）")
    m = t.flicker(t0, t.now())
    print(f"    工具输出期间：{m}")
    t.wait_for("跑完了。", 15)
    assert m["drops"] == 0 and m["jumps"] == 0, f"工具输出期间输入框消失 {m['drops']} 次、上跳 {m['jumps']} 次"
    ls = t.transcript().splitlines()
    i = next(k for k, l in enumerate(ls) if "● shell(" in l)
    assert "⎿" in ls[i + 1], f"工具行被命令里的换行断成多行（执行中转圈行忽高忽低）：{ls[i:i + 3]}"


_PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                     "1f15c4890000000d49444154789c6360000000000100005f5d2a0b0000000049454e44ae426082")


def bug_backspace_placeholder(llm: FakeLLM, t: Tui):
    """拖进两张图得到 [Image #1][Image #2]，退格一次删掉整个 [Image #2]（真机：一个字一个字删，剩「[Image」残片）。"""
    for name in ("a.png", "b.png"):
        (t.cwd / name).write_bytes(_PNG)
    start(t)
    for name in ("a.png", "b.png"):
        t.send(f"\x1b[200~\"{t.cwd / name}\"\x1b[201~", settle=0.4)
    t.wait_for(lambda: input_line(t) == "[Image #1][Image #2]", 3, "两个图片占位符")
    t.send("\x7f", settle=0.4)
    assert input_line(t) == "[Image #1]", f"退格没有整个删掉占位符：{input_line(t)!r}"
    t.send("\x7f", settle=0.4)
    assert input_line(t) == "", f"第二次退格后输入框应为空：{input_line(t)!r}"
    t.type("abc")
    t.send("\x7f", settle=0.3)
    assert input_line(t) == "ab", "普通文字的退格被改坏了"


def bug_drop_many_images(llm: FakeLLM, t: Tui):
    """多张图一起拖：终端一次粘贴送来一串路径（真机：变成两段裸路径）。空格分隔、file:/// 都要折成占位符。"""
    from urllib.parse import quote
    for name in ("a.png", "b.png", "c d.png"):
        (t.cwd / name).write_bytes(_PNG)
    start(t)
    t.send(f"\x1b[200~{t.cwd / 'a.png'} {t.cwd / 'b.png'}\x1b[201~", settle=0.4)
    t.wait_for(lambda: input_line(t) == "[Image #1][Image #2]", 3, "两张图一次拖入 → 两个占位符")
    uri = "file:///" + quote(str(t.cwd / "c d.png").replace("\\", "/"))
    t.send(f"\x1b[200~{uri}\x1b[201~", settle=0.4)
    t.wait_for(lambda: input_line(t) == "[Image #1][Image #2][Image #3]", 3, "file:/// 拖入 → 占位符")


def guard_alt_v_clipboard(llm: FakeLLM, t: Tui):
    """Alt+V（终端里是 Esc + v）贴剪贴板截图：不能被当成 Esc 中断、也不能把 v 打进输入框。
    剪贴板里有什么不确定（读的是这台机器的真剪贴板）：插进占位符，或在上方提示一行，两种都对。"""
    start(t)
    t.send("\x1bv", settle=0.8)
    line = input_line(t)
    assert line in ("",) or line.startswith("[Image #"), f"Alt+V 被当成了别的键：输入框是 {line!r}"
    assert line.startswith("[Image #") or "剪贴板" in t.transcript() or "Pillow" in t.transcript(), \
        "Alt+V 既没贴图也没提示"


def bug_silent_while_writing_tool_args(llm: FakeLLM, t: Tui):
    """模型在流式写工具参数（write_file 一整篇文档）时、以及工具跑完到下一步出字之间，屏幕上要有动静
    （真机：「我来写这份文档。」之后几十秒一片静止，以为终端或 API 卡死了）。"""
    doc = "\n".join(f"## 第 {i} 节\n\n内容内容内容。" for i in range(1, 30))
    llm.push(Reply(text="我来写这份文档。", tools=[("write_file", {"path": "doc.md", "content": doc})], tool_seconds=3),
             Reply(text="写好了。", wait=2.5))
    start(t)
    t.type("写文档")
    t.send(ENTER)
    t.wait_for("我来写这份文档。", 10)
    t.wait_for(lambda: "正在生成 write_file" in t.text() or "要执行这个操作吗" in t.text(), 3,
               "写参数期间的状态行「正在生成 write_file …」")
    if "要执行这个操作吗" in t.text():
        t.send("1")
    t.wait_for(lambda: "● write_file(doc.md)" in t.transcript(), 10, "工具结果")
    t.wait_for(lambda: "思考中" in t.text(), 1.5, "工具跑完、下一步出字前的「思考中…」")
    t.wait_for("写好了。", 10)


def metric_input_pinned(llm: FakeLLM, t: Tui):
    """输入框钉住不动：回车提交、确认框关闭、思考转正文，全程不消失、不上跳（真机 2026-11-22：确认后闪、审批后闪、
    计划更新完到目标完成之间闪，「输入框没有固定」）。"""
    llm.push(tool("shell", {"command": "python -c \"print('hi')\" > out.txt"}),
             Reply(text="写好了 out.txt，内容是 hi。", think="想一想" * 20, think_seconds=1.2))
    start(t)
    t.type("写个文件")
    t0 = t.now()
    t.send(ENTER)
    t.wait_for(lambda: "要执行这个操作吗" in t.text(), 15, "确认框")
    t.send("1")
    t.wait_for(lambda: "思考中" in t.text(), 10, "思考")
    t.wait_for("内容是 hi。", 10)
    t.wait_idle_prompt()
    m = t.flicker(t0, t.now())
    print(f"    提交→确认→思考→正文：{m}")
    assert m["drops"] == 0 and m["jumps"] == 0, f"输入框消失 {m['drops']} 次、上跳 {m['jumps']} 次"


def metric_stream_cadence(llm: FakeLLM, t: Tui):
    """长回复流式输出要顺：后半程（回复已经很长）可见文字的更新间隔。原先每次更新都从头渲染整篇，
    节流按耗时 ×10 放宽，越写越卡（实测后半程一秒 2.5 次、间隔中位 475ms，真机「一顿一顿的」）。"""
    import pyte
    section = ("## 小节标题\n\n" + "asyncio 是单线程协作式并发，事件循环只在 await 处才可能切换任务。" * 3
               + "\n\n- 列表项一\n- 列表项二\n\n```python\nasync def main():\n    await asyncio.sleep(0)\n```\n\n")
    llm.push(slow(section * 28 + "全文结束。", 10))
    start(t)
    t.type("写长文")
    t.send(ENTER)
    t.wait_for(lambda: "小节标题" in t.transcript(), 10, "开始输出")
    t0 = t.now()
    t.wait_for("全文结束。", 30)
    t1 = t.now()
    mid = t0 + (t1 - t0) / 2
    scr = pyte.HistoryScreen(t.screen.columns, t.screen.lines, history=20000)
    st = pyte.Stream(scr)
    with t._lock:
        raw = list(t.raw)
    stamps, last = [], 0
    for at, data in raw:
        st.feed(data)
        if not mid <= at <= t1:
            continue
        n = sum(1 for line in scr.history.top for ch in line.values() if ch.data.strip())
        n += sum(1 for l in scr.display for c in l if c.strip())
        if n > last:
            stamps.append(at)
        last = max(last, n)
    gaps = sorted((b - a) * 1000 for a, b in zip(stamps, stamps[1:]))
    median, worst = gaps[len(gaps) // 2], gaps[-1]
    print(f"    后半程可见文字更新 {len(stamps) / (t1 - mid):.1f} 次/秒，间隔中位 {median:.0f}ms、最大 {worst:.0f}ms")
    assert median < 150 and worst < 500, f"流式输出不顺：间隔中位 {median:.0f}ms、最大 {worst:.0f}ms"


# ════════════════ guard_：防修坏 ════════════════

def guard_interjection_injected(llm: FakeLLM, t: Tui):
    """运行中提交插话：显示已排队，并在下一次请求里注入给模型。"""
    llm.push(slow("先读一下文件。" * 8, 3, tools=[("read_file", {"path": "a.txt"})]), text("收到你的插话了。"))
    start(t)
    t.type("读 a.txt")
    t.send(ENTER)
    t.wait_for("先读一下", 10)
    t.type("插一句话")
    t.send(ENTER)
    t.wait_for("已排队", 5)
    t.wait_for("收到你的插话了", 15)
    assert "插一句话" in json.dumps(llm.main_requests[-1]["messages"], ensure_ascii=False), "插话没注入给模型"


def guard_esc_interrupt_restores_queue(llm: FakeLLM, t: Tui):
    """运行中排队了插话再按 Esc：回合中断，插话放回输入框、不自动发出。"""
    llm.push(slow("这是一段很长的回答。" * 20, 8))
    start(t)
    t.type("讲个长故事")
    t.send(ENTER)
    t.wait_for("很长的回答", 10)
    t.type("还没发出去的话")
    t.send(ENTER)
    t.wait_for("已排队", 5)
    t.send(ESC, settle=1.5)
    assert "回合已中断" in t.transcript()
    assert "还没发出去的话" in input_line(t), f"插话没放回输入框：{input_line(t)!r}"
    assert len(llm.main_requests) == 1, "中断后插话被自动发出了"


def _last_plan_id(body: dict) -> dict:
    """从请求里最近一条「计划已创建：<id>」工具结果取 plan_id（剧本写不死运行时生成的 id）。"""
    ids = re.findall(r"计划已创建[:：]\s*([0-9a-f]{10})", json.dumps(body["messages"], ensure_ascii=False))
    return {"plan_id": ids[-1] if ids else ""}


def guard_plan_mode_suggestion(llm: FakeLLM, t: Tui):
    """规划模式审批菜单：按 4 跳到「提修改建议：」直接打字 → 建议回给模型 → 新计划选 1 批准执行。"""
    llm.push(tool("plan", plan_create("计划P")), tool("exit_plan_mode", _last_plan_id),
             tool("plan", plan_create("计划P2")), tool("exit_plan_mode", _last_plan_id),
             text("开始执行。"))
    start(t)
    t.send(SHIFT_TAB, settle=0.5)
    t.wait_for(lambda: t.lines()[t._prompt_row()].strip() == "规划 ❯", 5, "进入规划模式")
    assert "已进入规划模式" not in t.transcript(), "切换规划模式又往滚动区打了提示（提示符 + 状态行已表明）"
    t.type("加个功能")
    t.send(ENTER)
    t.wait_for(lambda: "批准这个计划吗" in t.text(), 15, "审批对话框")
    t.wait_for(lambda: "执行计划" in t.transcript(), 5, "计划框")
    _assert_reflow_safe(t, "规划审批的计划框")
    tr = t.transcript()
    assert "1. 计划P：改 a.txt" in tr and "0. 计划P" not in tr, "审批框的步骤编号不是从 1 起（模型说「第 1 步」）"
    t.send("4")
    t.type("用参数化")
    t.wait_for(lambda: "4. 提修改建议：用参数化" in t.text(), 3, "建议直接打在选项行上")
    t.send(ENTER)
    t.wait_for("批准这个计划吗？ → 提修改建议：用参数化", 5)
    t.wait_for(lambda: "计划P2" in t.transcript() and "批准这个计划吗" in t.text(), 15, "第二次审批")
    t.send("1")
    t.wait_for("开始执行。", 15)
    assert "用参数化" in json.dumps(llm.main_requests[2]["messages"], ensure_ascii=False), "修改建议没回给模型"


def guard_input_frame(llm: FakeLLM, t: Tui):
    """输入框上下各一条横线、下方状态行；状态行实时反映模型、上下文占用、规划模式、运行中提示。"""
    llm.push(slow("好的，我来看看。" * 6 + "完成。", 2))
    start(t)
    ls = t.lines()
    row = t._prompt_row()
    assert set(ls[row - 1]) == {"─"} and set(ls[row + 1]) == {"─"}, "输入框上下没有横线"
    assert "fake-model" in ls[row + 2] and "Shift+Enter 换行" in ls[row + 2], f"状态行不对：{ls[row + 2]!r}"
    t.send(SHIFT_TAB, settle=0.8)
    t.wait_for(lambda: "规划模式" in t.lines()[t._prompt_row() + 2], 5, "状态行显示规划模式")
    t.wait_for(lambda: t.lines()[t._prompt_row()].strip() == "规划 ❯", 3, "规划模式下提示符变成「规划 ❯」")
    t.send(SHIFT_TAB, settle=0.8)
    t.wait_for(lambda: t.lines()[t._prompt_row()].strip() == "❯", 3, "退出规划模式提示符变回「❯」")
    t.type("你好")
    t.send(ENTER)
    t.wait_for(lambda: "打字即插话" in t.text(), 5, "运行中状态行提示")
    t.wait_for("完成。", 10)
    t.wait_idle_prompt()
    status = lambda: t.lines()[t._prompt_row() + 2]
    t.wait_for(lambda: "上下文 100" in status() and "Shift+Tab" in status(), 3,
               "一轮后状态行显示上下文占用、按键提示回到空闲态")


def _frame_ok(t: Tui) -> bool:
    """输入框框体完整：❯ 上下紧贴横线、横线下面紧跟状态行。"""
    ls, row = t.lines(), t._prompt_row()
    return (row > 0 and row + 2 < len(ls) and set(ls[row - 1]) == {"─"} and set(ls[row + 1]) == {"─"}
            and "Shift+Enter" in ls[row + 2])


def bug_menu_space_restores(llm: FakeLLM, t: Tui):
    """输入 / 弹出补全后，打完完整命令或清掉输入，输入框要缩回原样（真机：撑大后不恢复，回车才恢复）。"""
    llm.push(text("\n".join(f"- 第 {i} 行" for i in range(1, 61)) + "\n", delay=0))
    start(t)
    t.type("刷一屏")
    t.send(ENTER)
    t.wait_for("第 60 行", 10)
    t.wait_idle_prompt()
    t.type("/", per_char=0)
    t.wait_for(lambda: "/help" in t.text(), 3, "补全菜单")
    t.send("\x7f", settle=0.6)                       # 退格清空
    assert _frame_ok(t), "清空输入后输入框没缩回：\n" + "\n".join(t.lines()[-12:])
    t.type("/usage")
    time.sleep(0.6)
    t.send("\x7f" * 6, settle=0.6)
    assert _frame_ok(t), "打完完整命令再删掉后输入框没缩回：\n" + "\n".join(t.lines()[-12:])


def bug_multiline_input_shrinks(llm: FakeLLM, t: Tui):
    """输入折成三行把输入框撑大，清空后再打一行字，输入框要缩回一行（真机：下横线停在撑大后的位置）。"""
    start(t)
    t.type("长" * 130, per_char=0)                   # 120 列下折成三行
    t.wait_for(lambda: t._prompt_row() >= 0 and t.lines()[t._prompt_row() + 2].strip() != "", 3, "长输入上屏")
    t.send("\x7f" * 130, settle=0.6)
    t.type("短的一行", per_char=0)
    time.sleep(0.6)
    row = t._prompt_row()
    assert _frame_ok(t), "清空多行输入后输入框没缩回：\n" + "\n".join(t.lines()[max(0, row - 2):row + 6])


def guard_scrollback_reflow_safe(llm: FakeLLM, t: Tui):
    """滚动区里打出去的东西（欢迎页、/help /usage /config、Markdown 回复里的一级标题 / --- / 代码块、回合分隔）
    都不能顶到最后一列：程序没法重画终端历史，窗口拉窄时终端把超宽行折断，满宽框和线碎成一片（真机）。"""
    reply = ("# 一级标题\n\n段落文字。\n\n---\n\n```python\nprint('hi')\n```\n\n"
             "| 列 A | 列 B |\n|---|---|\n| 1 | 2 |\n\n结束。\n")
    llm.push(text(reply, delay=0))
    start(t)
    assert "█▄▄▄" in t.transcript(), "欢迎页没有 Logo"
    _assert_reflow_safe(t, "欢迎页")
    for cmd, mark in [("/help", "命令列表"), ("/usage", "Token 用量"), ("/config", "配置")]:
        t.type(cmd, per_char=0)
        t.send(ENTER)
        t.wait_for(lambda: mark in t.transcript(), 5, cmd)
        t.wait_idle_prompt()
        _assert_reflow_safe(t, cmd)
    t.type("来段 Markdown", per_char=0)
    t.send(ENTER)
    t.wait_for("结束。", 10)
    t.wait_idle_prompt()
    _assert_reflow_safe(t, "Markdown 回复 / 回合分隔")


def _rule_lines(t: Tui) -> int:
    return sum(1 for l in t.lines() if l and set(l) == {"─"})


def bug_resize_leaves_stale_rules(llm: FakeLLM, t: Tui):
    """反复拉窄、拉宽窗口：屏幕上不应残留旧横线，框体保持完整（真机：第一次正常，之后堆出一摞横线）。"""
    start(t)
    base = _rule_lines(t)
    for rows, cols in [(40, 80), (40, 120), (40, 70), (40, 100), (40, 60)]:
        t.resize(rows, cols)
        assert _frame_ok(t), f"改成 {cols} 列后框体不完整：\n" + "\n".join(t.lines()[-10:])
        assert _rule_lines(t) == base, f"改成 {cols} 列后横线 {_rule_lines(t)} 条（应为 {base}）：\n" + "\n".join(t.lines()[-12:])


def guard_exit_farewell(llm: FakeLLM, t: Tui):
    """/exit：告别语正常打出（打印改走输入框合批后，收尾要把缓冲写掉）、进程退出。"""
    llm.push(text("好的。"))
    start(t)
    t.type("你好")
    t.send(ENTER)
    t.wait_for("好的。", 10)
    t.wait_idle_prompt()
    t.type("/exit")
    t.send(ENTER)
    t.wait_for("再见", 10)
    t.wait_for(lambda: not t.proc.isalive(), 20, "进程退出")


SCENARIOS = {name: fn for name, fn in list(globals().items())
             if name.startswith(("bug_", "guard_", "metric_")) and callable(fn)}


def run(names: list[str]) -> int:
    OUT.mkdir(exist_ok=True)
    failed = 0
    for name in names:
        llm = FakeLLM()
        t = Tui(llm, files={"a.txt": "hello\n"})
        t0 = time.monotonic()
        try:
            SCENARIOS[name](llm, t)
            print(f"PASS  {name}  ({time.monotonic() - t0:.1f}s)")
        except Exception as e:
            failed += 1
            dump = OUT / f"{name}.txt"
            t.dump(dump)
            msg = str(e).split("\n---- 屏幕 ----")[0] if isinstance(e, TimeoutError) else str(e)
            if not isinstance(e, (AssertionError, TimeoutError)):
                msg = traceback.format_exc(limit=3)
            print(f"FAIL  {name}  ({time.monotonic() - t0:.1f}s)：{msg}\n      滚动区：{dump}")
        finally:
            t.close()
            llm.close()
    print(f"\n{len(names) - failed}/{len(names)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    picked = sys.argv[1:] or list(SCENARIOS)
    unknown = [n for n in picked if n not in SCENARIOS]
    if unknown:
        sys.exit(f"未知场景：{unknown}；可选：{list(SCENARIOS)}")
    sys.exit(run(picked))
