from __future__ import annotations
from datetime import datetime, timezone
import logging
import re

from loongcli.core.llm import LLMClient
from loongcli.core.stream_collector import StreamCollector
from loongcli.core.attachments import ATTACHMENT_MARKER, build_attachments
from loongcli.core.messages import message_text, to_content_parts, count_images

logger = logging.getLogger(__name__)

SUMMARY_TOKEN_RESERVE = 13000

# 兜底估算里每张图片的固定 token 估值（Qwen 默认每图 token 预算约 1280，设计文档 §7）。
# message_text 把图片计为 "[图片]" 几个字符，不加这项会严重低估含图历史的体积。
IMAGE_TOKEN_ESTIMATE = 1280


def model_context_window(model: str) -> int:
    """Return the max context window (tokens) for a known model, or a safe default."""
    from loongcli.core.provider import context_window
    return context_window(model)


COMPACT_INSTRUCTION = """\
严禁调用任何工具。不要调用 read_file、shell、grep、glob 或任何其他工具。
工具调用会被拒绝，你只有这一次机会。你的全部回复必须是纯文本。

请总结以上对话历史。

先输出 <analysis> 块，按时间顺序梳理对话中的关键事件。然后输出 <summary> 块，包含以下 9 个固定章节：

1. **主要意图** — 用户的所有明确请求和目标
2. **关键技术概念** — 重要的技术栈、框架、架构决策
3. **文件与代码** — 查看、修改、创建的文件路径，附关键代码片段（路径:行号）和修改原因
4. **错误与修复** — 遇到的每个错误及其解决方式，避免重复踩坑
5. **问题解决** — 已解决的问题及方案
6. **所有用户消息** — 逐条列出所有非工具调用的用户消息原文（纠正、偏好、约束、需求变更一个不落）
7. **待办任务** — 明确请求但尚未完成的任务
8. **当前工作** — 用最细粒度描述摘要前正在进行的工作（精确到文件名、函数名、当前步骤）
9. **下一步** — 根据最近的请求建议下一步，引用对话原文作为依据

用中文输出。<summary> 部分要足够详细，确保仅凭摘要就能继续工作。

再次强调：不要调用任何工具，直接输出纯文本。"""

SUMMARY_MARKER = "[对话历史摘要]"
SUMMARY_ACK = "好的，我已了解之前的对话内容。请继续。"

BOUNDARY_TEMPLATE = (
    "[compact-boundary] mode={mode} | pre_tokens={pre_tokens} | "
    "pre_messages={pre_messages} | timestamp={timestamp}"
)

KEEP_RECENT_TURNS = 3
# 保留区按「段」计（见 _segment_units），最多这么多段。只按用户轮次算的话，一个长回合（goal 一次执行迭代
# 可有上百次工具调用）整个都在「最近 3 轮」里，压了等于没压——2026-11-19 p4-1 实测上下文 236K、57 次超阈值、0 次压缩。
KEEP_RECENT_UNITS = 8
TOOL_RESULT_PLACEHOLDER = "[工具已执行，结果见摘要]"

SNIP_AGE_THRESHOLD = 20
SNIP_MARKER = "[snip] 已删除 {count} 条远古消息，保留最近 {kept} 轮对话"

def snip(messages: list[dict], threshold: int = SNIP_AGE_THRESHOLD) -> tuple[list[dict], int]:
    """Delete ancient messages beyond `threshold` turns, keeping system msg and recent turns.

    Returns (new_messages, count_of_removed_messages).
    Zero API cost — just drops old messages and inserts a marker.

    唯一余生用途：Compactor 摘要请求超窗时扔最老的消息保住请求不 400。
    曾经它是"五层压缩金字塔"的一层（连同 collapse/micro_compact），2026-07-19 金字塔
    已删——bench 实测保前缀吃缓存比删改历史便宜 ~22x（命中 98% vs 12%），只为省钱
    改写中段历史在缓存时代是负资产；且五层机制连作者都讲不清（认知负债）。
    """
    if not messages:
        return messages, 0

    system_msg = messages[0] if messages[0].get("role") == "system" else None
    start = 1 if system_msg else 0

    turns = _segment_turns(messages, start)
    if len(turns) <= threshold:
        return messages, 0

    keep_turns = turns[-threshold:]
    dropped_turns = turns[:-threshold]
    dropped_count = sum(len(t) for t in dropped_turns)

    marker = SNIP_MARKER.format(count=dropped_count, kept=threshold)
    result: list[dict] = []
    if system_msg:
        result.append(system_msg)
    result.append({"role": "user", "content": marker})
    result.append({"role": "assistant", "content": "好的，已了解。"})
    for turn in keep_turns:
        result.extend(turn)

    return _fix_role_alternation(result[:3], result[3:]), dropped_count


def _segment_turns(messages: list[dict], start: int) -> list[list[dict]]:
    turns: list[list[dict]] = []
    current: list[dict] = []
    for msg in messages[start:]:
        if msg["role"] == "user" and current:
            turns.append(current)
            current = []
        current.append(msg)
    if current:
        turns.append(current)
    return turns


def _segment_units(messages: list[dict], start: int) -> list[list[dict]]:
    """按「段」切：每条用户消息开一段，紧跟在工具结果之后的模型回复（回合内的下一次迭代）也开一段。

    一段 = [用户消息] + 模型回复 + 它的工具结果。工具结果永远和发起它的那条模型回复在同一段，
    所以按段截取不会留下孤儿 tool 消息（API 会以 400 拒绝）。"""
    units: list[list[dict]] = []
    current: list[dict] = []
    for msg in messages[start:]:
        starts_unit = msg["role"] == "user" or (
            msg["role"] == "assistant" and current and current[-1]["role"] == "tool"
        )
        if starts_unit and current:
            units.append(current)
            current = []
        current.append(msg)
    if current:
        units.append(current)
    return units


def _kept_unit_count(units: list[list[dict]]) -> int:
    """保留多少段：最近 KEEP_RECENT_TURNS 个用户轮次里的段，但不超过 KEEP_RECENT_UNITS。"""
    user_starts = [i for i, u in enumerate(units) if u[0]["role"] == "user"]
    first = user_starts[-KEEP_RECENT_TURNS] if len(user_starts) >= KEEP_RECENT_TURNS else 0
    return min(len(units) - first, KEEP_RECENT_UNITS)


def _is_compaction_artifact(msg: dict) -> bool:
    """上一次压缩留下的摘要 / 重挂附件消息——不当作「本回合的用户任务」钉住。"""
    text = message_text(msg)
    return text.startswith("[compact-boundary]") or SUMMARY_MARKER in text[:200] or text.startswith(ATTACHMENT_MARKER)


def _replace_tool_results(messages: list[dict]) -> list[dict]:
    result = []
    for msg in messages:
        if msg["role"] == "tool":
            result.append({**msg, "content": TOOL_RESULT_PLACEHOLDER})
        else:
            result.append(msg)
    return result


def _merge_user_content(a, b):
    """合并相邻 user 消息的 content：纯文本走字符串拼接；任一为多模态 list 时，
    归一成内容块列表拼接，保留图片块不丢。"""
    if isinstance(a, str) and isinstance(b, str):
        return a + "\n\n" + b
    return to_content_parts(a) + to_content_parts(b)


def _fix_role_alternation(prefix: list[dict], kept: list[dict]) -> list[dict]:
    result = list(prefix)
    for msg in kept:
        if msg["role"] == "tool":
            result.append(msg)
            continue
        if result and msg["role"] == result[-1]["role"] == "assistant":
            # assistant content 恒为纯文本（模型只吐字），走 message_text 收口即安全
            prev = result[-1]
            merged_content = (message_text(prev) + "\n\n" + message_text(msg)).strip() or None
            result[-1] = {**prev, "content": merged_content}
            if msg.get("tool_calls"):
                result[-1]["tool_calls"] = msg["tool_calls"]
                result[-1]["content"] = result[-1].get("content") or None
        elif result and msg["role"] == result[-1]["role"] == "user":
            # user content 可能含图片（多模态 list），必须保图片地合并
            result[-1] = {**result[-1], "content": _merge_user_content(result[-1].get("content"), msg.get("content"))}
        else:
            result.append(msg)
    return result


class Compactor:
    def __init__(self, llm: LLMClient, threshold: int = 800000, plan_store=None,
                 task_manager=None, skill_registry=None, window_for=None):
        """window_for(model) -> 上下文窗口：给了就按**当前**模型实时算压缩线（会话中 /model 换模型后跟着变）；
        不给则用固定的 threshold（配置里写死 compact_threshold、子代理等）。"""
        self.llm = llm
        self._threshold = threshold
        self.window_for = window_for
        self.plan_store = plan_store
        self.task_manager = task_manager
        self.skill_registry = skill_registry

    @property
    def threshold(self) -> int:
        if self.window_for is not None:
            return max(0, self.window_for(self.llm.model) - SUMMARY_TOKEN_RESERVE)
        return self._threshold

    @threshold.setter
    def threshold(self, value: int) -> None:
        self._threshold = value
        self.window_for = None

    def should_compact(self, prompt_tokens: int, messages: list[dict]) -> bool:
        if prompt_tokens <= 0:
            return False
        if prompt_tokens <= self.threshold:
            return False
        start = 1 if messages and messages[0].get("role") == "system" else 0
        units = _segment_units(messages, start)
        return len(units) > _kept_unit_count(units)

    async def compact(
        self,
        messages: list[dict],
        active_skill: str | None = None,
        mode: str = "auto",
        pre_tokens: int = 0,
    ) -> list[dict]:
        system_msg = messages[0] if messages[0].get("role") == "system" else None
        start = 1 if system_msg else 0

        units = _segment_units(messages, start)
        keep_n = _kept_unit_count(units)
        if len(units) <= keep_n:
            return messages

        kept_units = units[-keep_n:]
        summary = await self._summarize(messages, active_skill, mode=mode, pre_tokens=pre_tokens)
        logger.info("Compacted %d messages into summary (%d chars)", len(messages), len(summary))

        boundary = BOUNDARY_TEMPLATE.format(
            mode=mode,
            pre_tokens=pre_tokens,
            pre_messages=len(messages),
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        # 保留区的工具结果换占位（结论已写进模型自己的回复）——唯一例外是模型还没回应过的那批：
        # 消息停在工具结果上 = 回合中途、马上要再调模型，它下一步要看的正是这些结果，换掉就等于让它白调一次。
        pending = kept_units[-1] if kept_units[-1][-1]["role"] == "tool" else None
        kept_messages = []
        for unit in kept_units:
            kept_messages.extend(unit if unit is pending else _replace_tool_results(unit))
        # 回合中途压缩时保留区从模型回复开始：把本回合的用户消息原样钉在前面，任务原文不靠摘要转述
        if kept_messages[0]["role"] != "user":
            for unit in reversed(units[:-keep_n]):
                if unit[0]["role"] == "user" and not _is_compaction_artifact(unit[0]):
                    kept_messages.insert(0, unit[0])
                    break

        prefix: list[dict] = []
        if system_msg:
            prefix.append(system_msg)
        prefix.append({"role": "user", "content": f"{boundary}\n{SUMMARY_MARKER}\n{summary}"})
        prefix.append({"role": "assistant", "content": SUMMARY_ACK})

        attachments = build_attachments(
            messages, self.plan_store, self.task_manager,
            skill_registry=self.skill_registry, active_skill=active_skill,
        )

        return _fix_role_alternation(prefix + attachments, kept_messages)

    async def _summarize(self, messages: list[dict], active_skill: str | None = None, mode: str = "auto", pre_tokens: int = 0) -> str:
        instruction = COMPACT_INSTRUCTION
        if active_skill:
            instruction += (
                f"\n\n注意：当前正在执行技能 '{active_skill}'。技能指令原文会在压缩后自动重新挂载，"
                "摘要中只需详细记录执行进度（已完成哪几步、当前在哪一步、产出了什么），不要复述技能指令本身。"
            )
        if mode == "auto":
            instruction += "\n\n不要在摘要中提出新问题或建议用户回答任何内容。摘要应纯粹记录事实，不包含后续提问。"

        # 摘要请求的输入：预算内喂完整历史，超预算才 snip 扔最老的保住请求不超窗。
        # 不再按 provider 分流（2026-07-19 金字塔删除）：主流云厂商全有自动前缀缓存
        # （DS 1:120、qwen 实测稳态命中 72%），完整历史吃缓存比先删改历史便宜 ~22x
        # （命中 98% vs 12%）且摘要不丢早期"主要意图"；无缓存 provider（本地模型等）
        # 的摘要是罕见的保险丝事件，一次全价可接受，不为它维护第二条路径。
        window = model_context_window(self.llm.model)
        budget = window - SUMMARY_TOKEN_RESERVE
        est_tokens = pre_tokens if pre_tokens > 0 else _estimate_tokens(messages)
        if est_tokens < budget:
            compact_messages = list(messages) + [{"role": "user", "content": instruction}]
        else:
            snipped, _ = snip(list(messages))
            compact_messages = snipped + [{"role": "user", "content": instruction}]

        collector = StreamCollector()
        async for _ in collector.collect(
            self.llm.chat_stream(messages=compact_messages),
        ):
            pass

        summary = _extract_summary(collector.response.content)
        # 确定性守卫：摘要为空/过短（模型无视"严禁调用工具"的 prompt 禁令硬发
        # tool_calls、输出被截断等）时放弃本次压缩——prompt 禁令赌不得，而拿
        # 空摘要替换全史等于销毁当前工作上下文。抛异常走调用方的失败路径，原历史保留。
        if len(summary.strip()) < 50:
            raise ValueError(f"压缩摘要过短（{len(summary.strip())} 字符），放弃本次压缩以保留原历史")
        return summary


def _estimate_tokens(messages: list[dict]) -> int:
    """无 API 真实值时的 token 兜底估算：文本按字符数 // 2，每张图片加固定估值。"""
    return (
        sum(len(message_text(m)) for m in messages) // 2
        + count_images(messages) * IMAGE_TOKEN_ESTIMATE
    )


def _extract_summary(raw: str) -> str:
    match = re.search(r"<summary>(.*?)</summary>", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    raw = re.sub(r"<analysis>.*?</analysis>", "", raw, flags=re.DOTALL)
    return raw.strip()
