"""resume 时间跳跃标记：告诉模型「以上历史发生在多久以前」。

resume 会重建 messages[0]，那修的是「现在几点」——系统提示是当前世界快照，
里面的日期永远是今天。但恢复回来的历史消息本身没有任何时间戳，模型看到的是
一条无时间的消息流 + 一句「今天是 X」，无从知道第 5 轮其实发生在 6 天前。
结果是把跨周的会话当成一天内跑完，并按今天的日期去理解历史里的
「今天/昨天/刚才/等下」——那些词以当时为准。

修法与 compact 同构：`compact.BOUNDARY_TEMPLATE` 早就在压缩边界写了 timestamp，
「改写 messages 时在边界留元信息」这个模式仓库里已经有，只是 resume 入口漏了。

用 user+assistant 一对而非 role=system 承载标记，两个原因：
1. `main.py` 的 raw resume 会滤掉历史里的 system 消息，system 标记活不过下一次
   resume；user 消息能留下，多次 resume 就在历史上叠出完整的时间脉络。
2. 与 SUMMARY_MARKER / ATTACHMENT_MARKER 的既有形状一致，也不破坏角色交替。
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

# 触发阈值：间隔超过它、或本地日历日已翻篇，就插标记。
# 连续工作的轮次之间插标记纯属噪声，只为「离开又回来」这个稀疏事件付出上下文。
GAP_THRESHOLD = timedelta(hours=4)

RESUME_GAP_MARKER = "[会话恢复]"

RESUME_GAP_TEMPLATE = (
    "{marker} 上次活动 {last_active}，距今 {gap}。\n"
    "以上历史发生在那时，其中的「今天/昨天/刚才/等下/明天」等相对时间词"
    "以当时为准，不要按当前日期理解。"
)

RESUME_GAP_ACK = "好的，我已注意到这段时间跳跃，历史中的相对时间按当时理解。"


def _humanize(delta: timedelta) -> str:
    seconds = int(delta.total_seconds())
    if seconds < 3600:
        return f"{max(seconds // 60, 1)} 分钟"
    if seconds < 86400:
        return f"{seconds // 3600} 小时"
    return f"{seconds // 86400} 天"


def resume_gap_messages(updated_at: str, now: datetime | None = None) -> list[dict]:
    """恢复的历史与现在相隔够远时，返回一对标记消息；否则返回空列表。

    `updated_at` 取自会话 meta（UTC isoformat）。缺失或损坏一律返回空——
    升级前的旧会话文件没有这个字段，不该因此崩掉 resume。
    """
    if not updated_at:
        return []
    try:
        last = datetime.fromisoformat(updated_at)
    except (TypeError, ValueError):
        return []
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    delta = now - last
    if delta <= timedelta(0):
        return []  # 时钟回拨/未来时间戳：没有可信的间隔可报，闭嘴比乱报好

    # 日历日按本地时区判断，与系统提示里的「今天日期」（datetime.now() 本地时间）对齐，
    # 否则会出现「模型看到今天是 8-19，标记却说没跨日」这种自相矛盾。
    last_local = last.astimezone()
    now_local = now.astimezone()
    if delta < GAP_THRESHOLD and last_local.date() == now_local.date():
        return []

    return [
        {
            "role": "user",
            "content": RESUME_GAP_TEMPLATE.format(
                marker=RESUME_GAP_MARKER,
                last_active=last_local.strftime("%Y-%m-%d %H:%M"),
                gap=_humanize(delta),
            ),
        },
        {"role": "assistant", "content": RESUME_GAP_ACK},
    ]


# ---- 进程内跨日：resume 之外的另一半 ----
#
# resume 标记修的是「离开又回来」。但同一个进程不退出、会话开过午夜时，是同一个缺口的
# 另一半：系统提示每会话只构建一次，里面的「今天日期」停在启动那天，而上下文里没有任何
# 东西告诉模型日期已经变了。真机（2026-11-10 14:35 开的会话聊到 11-11 01:20）：模型把
# 11-12 的面试说成「今天下午」，被用户纠正后自述「系统给的日期快照停在 11-10」。
#
# 不去改写系统提示里的日期——那是 messages[0]，一改整段前缀缓存全破。与 resume 标记
# 同构：只在历史末尾追加一对消息，append-only。

DATE_ROLLOVER_MARKER = "[日期变更]"

DATE_ROLLOVER_TEMPLATE = (
    "{marker} 现在是 {now}（星期{weekday}）。"
    "系统提示里的「今天日期」是会话启动时的快照（{stale}），已过期，以此处为准；"
    "此前对话里的「今天/明天/昨天」等相对时间词以当时为准。"
)

DATE_ROLLOVER_ACK = "好的，我已注意到日期变更，后续按新日期推算。"

_WEEKDAYS = "一二三四五六日"


def date_rollover_messages(last_turn_date: date | None, now: datetime | None = None) -> list[dict]:
    """上一回合所在的本地日期与现在不同（会话开过了午夜）时，返回一对标记消息。

    `last_turn_date` 为 None（本进程的第一回合：系统提示刚构建、日期是新鲜的）或日期未变
    时返回空列表。时钟回拨（现在早于上一回合）同样闭嘴——没有可信的「变更」可报。
    """
    if last_turn_date is None:
        return []
    now = now or datetime.now()
    if now.date() <= last_turn_date:
        return []
    return [
        {
            "role": "user",
            "content": DATE_ROLLOVER_TEMPLATE.format(
                marker=DATE_ROLLOVER_MARKER,
                now=now.strftime("%Y-%m-%d %H:%M"),
                weekday=_WEEKDAYS[now.weekday()],
                stale=last_turn_date.strftime("%Y-%m-%d"),
            ),
        },
        {"role": "assistant", "content": DATE_ROLLOVER_ACK},
    ]
