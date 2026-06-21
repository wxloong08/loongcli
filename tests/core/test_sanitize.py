import pytest

from loongcli.core.sanitize import repair_surrogates


# 🎁 (U+1F381)。用 chr() 构造以避免源码字面量歧义：
#   GIFT      = 正确单码点（合法 UTF-8）
#   GIFT_SURR = Windows 控制台粘贴 emoji 收到的损坏形态：UTF-16 代理对 high+low
GIFT = chr(0x1F381)
GIFT_SURR = chr(0xD83C) + chr(0xDF81)
LONE_HIGH = chr(0xD800)
LONE_LOW = chr(0xDC81)


class TestRepairSurrogates:
    def test_clean_text_unchanged(self):
        s = "普通文本 normal text 123"
        assert repair_surrogates(s) is s  # 快路径零拷贝

    def test_recovers_surrogate_pair_into_emoji(self):
        # 核心：成对代理救回真 emoji（忠实保留用户输入）
        assert repair_surrogates(f"奖品{GIFT_SURR}一枚") == f"奖品{GIFT}一枚"

    def test_valid_astral_emoji_untouched(self):
        s = f"已经是合成的 {GIFT} 不动它"
        assert repair_surrogates(s) == s

    def test_lone_surrogate_dropped(self):
        # 无法配对的孤立代理直接丢弃
        assert repair_surrogates("坏" + LONE_HIGH + "字符") == "坏字符"

    def test_mixed_pair_recovered_lone_dropped(self):
        out = repair_surrogates(GIFT_SURR + "text" + LONE_LOW + "更多")
        assert out == f"{GIFT}text更多"

    def test_result_is_utf8_encodable(self):
        # 核心保证：修复后必须能 UTF-8 编码（否则请求/存盘仍会崩）
        for raw in [f"emoji{GIFT_SURR}", "lone" + LONE_LOW + "here", GIFT_SURR + LONE_HIGH + "mix"]:
            repair_surrogates(raw).encode("utf-8")  # 不抛异常即通过

    def test_surrogate_pair_would_crash_utf8(self):
        # 证明问题确实存在：原始代理对无法 UTF-8 编码（正是线上崩溃的根因）
        with pytest.raises(UnicodeEncodeError):
            GIFT_SURR.encode("utf-8")
