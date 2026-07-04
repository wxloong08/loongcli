import pytest
from unittest.mock import MagicMock

from loongcli.core import messages as messages_mod
from loongcli.core.messages import (
    message_text, to_content_parts, image_to_data_url, build_user_content,
    extract_image_paths, is_image_file,
    store_image, store_image_bytes, inline_image_refs, recycle_old_images, count_images,
    detect_image_mime, make_image_sentinel, parse_image_sentinel,
    IMAGE_REF_PREFIX, IMAGE_DROPPED_PLACEHOLDER, IMAGE_LOST_PLACEHOLDER,
)
from loongcli.core.compact import (
    Compactor, _fix_role_alternation, snip, micro_compact,
)
from loongcli.core.context_collapse import collapse
from loongcli.memory.conversation import ConversationStore


# 一条含图的多模态 user 消息（骨架的核心受测对象）
def _img_msg(text="看这个设计稿"):
    return {"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,ABC123"}},
    ]}


def _has_image(content) -> bool:
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "image_url" for b in content
    )


# ── message_text ────────────────────────────────────────────────────

class TestMessageText:
    def test_str_content(self):
        assert message_text({"role": "user", "content": "hello"}) == "hello"

    def test_list_text_and_image(self):
        assert message_text(_img_msg("看这个")) == "看这个[图片]"

    def test_none_content(self):
        assert message_text({"role": "assistant", "content": None}) == ""

    def test_missing_content(self):
        assert message_text({"role": "assistant", "tool_calls": []}) == ""

    def test_pure_image(self):
        msg = {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,X"}},
        ]}
        assert message_text(msg) == "[图片]"

    def test_multiple_text_blocks(self):
        msg = {"role": "user", "content": [
            {"type": "text", "text": "a"}, {"type": "text", "text": "b"},
        ]}
        assert message_text(msg) == "ab"

    def test_non_dict_block(self):
        msg = {"role": "user", "content": ["raw"]}
        assert message_text(msg) == "raw"

    def test_unknown_block_type(self):
        msg = {"role": "user", "content": [{"type": "audio_url"}]}
        assert message_text(msg) == "[audio_url]"


# ── to_content_parts ────────────────────────────────────────────────

class TestToContentParts:
    def test_str_wraps_in_text_block(self):
        assert to_content_parts("hi") == [{"type": "text", "text": "hi"}]

    def test_list_passthrough(self):
        parts = _img_msg()["content"]
        assert to_content_parts(parts) == parts

    def test_none_empty(self):
        assert to_content_parts(None) == []
        assert to_content_parts("") == []


# ── 闸测试：含图 list content 穿过全管线不崩、图片不丢、文本不失真 ──

class TestListContentSurvivesPipeline:
    def test_fix_role_alternation_merges_user_keeping_image(self):
        # 相邻 user 消息合并：第一条含图，第二条纯文本 → 合并后仍保留图片块
        kept = [_img_msg("看图"), {"role": "user", "content": "再补一句"}]
        result = _fix_role_alternation([], kept)
        assert len(result) == 1
        assert _has_image(result[0]["content"])
        merged_text = message_text(result[0])
        assert "看图" in merged_text and "再补一句" in merged_text

    def test_fix_role_alternation_two_images(self):
        kept = [_img_msg("图一"), _img_msg("图二")]
        result = _fix_role_alternation([], kept)
        assert len(result) == 1
        # 两张图都在
        imgs = [b for b in result[0]["content"] if b.get("type") == "image_url"]
        assert len(imgs) == 2

    def test_snip_with_image_no_crash(self):
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(25):
            msgs.append(_img_msg(f"q{i}") if i == 2 else {"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        snipped, _ = snip(msgs)  # 不抛异常
        assert isinstance(snipped, list)

    def test_micro_compact_leaves_image_msg(self):
        msgs = [_img_msg("图"), {"role": "user", "content": "x"}]
        assert micro_compact(msgs) == msgs  # 非工具消息，原样

    def test_collapse_with_image_no_crash(self):
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(10):
            msgs.append(_img_msg(f"q{i}") if i == 1 else {"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": "a" * 600})
        out = collapse(msgs, level=2)  # 不抛异常
        assert any(_has_image(m.get("content")) for m in out)  # 图片消息未被截断破坏

    def test_conversation_save_title_from_image_msg(self, tmp_path):
        store = ConversationStore(base_dir=tmp_path)
        store.save([_img_msg("这是登录页设计稿")])
        # 标题取自 text 部分（不崩、不失真）
        assert store._meta["title"].startswith("这是登录页设计稿")
        # 往返：list content 原样保存
        loaded = store.load(store.session_id)
        assert _has_image(loaded["messages"][0]["content"])

    @pytest.mark.asyncio
    async def test_compact_with_image_no_crash(self):
        llm = _mock_llm()
        compactor = Compactor(llm=llm, threshold=1)
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(5):
            msgs.append(_img_msg(f"q{i} 看图") if i == 3 else {"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        out = await compactor.compact(msgs, mode="auto")  # 不抛异常
        # 保留区里的图片消息应活下来
        assert any(_has_image(m.get("content")) for m in out)


# ── image_to_data_url ───────────────────────────────────────────────

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class TestImageToDataUrl:
    def test_png_by_extension(self, tmp_path):
        p = tmp_path / "x.png"
        p.write_bytes(_PNG_MAGIC + b"data")
        assert image_to_data_url(str(p)).startswith("data:image/png;base64,")

    def test_jpg_by_extension(self, tmp_path):
        p = tmp_path / "x.jpg"
        p.write_bytes(b"\xff\xd8\xff data")
        assert image_to_data_url(str(p)).startswith("data:image/jpeg;base64,")

    def test_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            image_to_data_url(str(tmp_path / "nope.png"))

    def test_oversized_raises(self, tmp_path):
        p = tmp_path / "big.png"
        p.write_bytes(b"x" * (10 * 1024 * 1024 + 1))
        with pytest.raises(ValueError, match="过大"):
            image_to_data_url(str(p))

    def test_unknown_extension_sniffs_magic(self, tmp_path):
        p = tmp_path / "noext"
        p.write_bytes(_PNG_MAGIC + b"data")
        assert image_to_data_url(str(p)).startswith("data:image/png;base64,")

    def test_unrecognized_raises(self, tmp_path):
        p = tmp_path / "x.dat"
        p.write_bytes(b"just random bytes")
        with pytest.raises(ValueError, match="无法识别"):
            image_to_data_url(str(p))


# ── build_user_content ──────────────────────────────────────────────

class TestBuildUserContent:
    def test_no_images_returns_str(self):
        assert build_user_content("hi", None) == "hi"
        assert build_user_content("hi", []) == "hi"

    def test_with_images_returns_ref_blocks(self, tmp_path):
        p = tmp_path / "x.png"
        p.write_bytes(_PNG_MAGIC + b"data")
        content = build_user_content("看图", [str(p)])
        assert content[0] == {"type": "text", "text": "看图"}
        assert content[1]["type"] == "image_url"
        # 消息里只存引用，不内联 base64（发 API 时才由 chat_stream 内联）
        assert content[1]["image_url"]["url"].startswith(IMAGE_REF_PREFIX)

    def test_empty_text_omits_text_block(self, tmp_path):
        p = tmp_path / "x.png"
        p.write_bytes(_PNG_MAGIC + b"data")
        content = build_user_content("", [str(p)])
        assert all(b["type"] == "image_url" for b in content)


# ── store_image / inline_image_refs（存盘持久化 + 发送内联） ─────────

def _ref_msg(ref, text="看图"):
    return {"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": ref}},
    ]}


class TestStoreImage:
    def test_creates_hashed_file(self, tmp_path):
        p = tmp_path / "x.png"
        p.write_bytes(_PNG_MAGIC + b"data")
        ref = store_image(str(p))
        assert ref.startswith(IMAGE_REF_PREFIX) and ref.endswith(".png")
        stored = messages_mod.images_dir() / ref[len(IMAGE_REF_PREFIX):]
        assert stored.read_bytes() == _PNG_MAGIC + b"data"

    def test_dedupe_same_content(self, tmp_path):
        p1 = tmp_path / "a.png"
        p2 = tmp_path / "b.png"
        p1.write_bytes(_PNG_MAGIC + b"same")
        p2.write_bytes(_PNG_MAGIC + b"same")
        assert store_image(str(p1)) == store_image(str(p2))
        assert len(list(messages_mod.images_dir().iterdir())) == 1

    def test_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            store_image(str(tmp_path / "nope.png"))

    def test_unrecognized_raises(self, tmp_path):
        p = tmp_path / "x.dat"
        p.write_bytes(b"just random bytes")
        with pytest.raises(ValueError, match="无法识别"):
            store_image(str(p))


class TestInlineImageRefs:
    def test_ref_inlined_to_data_url(self, tmp_path):
        p = tmp_path / "x.png"
        p.write_bytes(_PNG_MAGIC + b"data")
        msgs = [_ref_msg(store_image(str(p)))]
        out = inline_image_refs(msgs)
        assert out[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_original_not_mutated(self, tmp_path):
        p = tmp_path / "x.png"
        p.write_bytes(_PNG_MAGIC + b"data")
        ref = store_image(str(p))
        msgs = [_ref_msg(ref)]
        inline_image_refs(msgs)
        # 调用方的消息历史里仍是引用
        assert msgs[0]["content"][1]["image_url"]["url"] == ref

    def test_data_url_passthrough(self):
        msgs = [_ref_msg("data:image/png;base64,ABC")]
        out = inline_image_refs(msgs)
        assert out is not None
        assert out[0]["content"][1]["image_url"]["url"] == "data:image/png;base64,ABC"

    def test_str_content_untouched(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert inline_image_refs(msgs)[0] is msgs[0]

    def test_lost_file_degrades_to_placeholder(self):
        # 引用指向的缓存文件不存在 → 降级为占位文本，不抛异常、不卡死会话
        msgs = [_ref_msg(IMAGE_REF_PREFIX + "deadbeef00000000.png")]
        out = inline_image_refs(msgs)
        assert out[0]["content"][1] == {"type": "text", "text": IMAGE_LOST_PLACEHOLDER}

    def test_traversal_ref_degrades(self):
        # 引用名带路径分隔符视为非法 → 同样降级，不读库外文件
        msgs = [_ref_msg(IMAGE_REF_PREFIX + "../secret.png")]
        out = inline_image_refs(msgs)
        assert out[0]["content"][1]["type"] == "text"


# ── recycle_old_images / count_images（旧图回收） ────────────────────

class TestRecycleOldImages:
    def _msgs_with_images(self, n):
        msgs = []
        for i in range(n):
            msgs.append(_ref_msg(f"{IMAGE_REF_PREFIX}{i:016x}.png", text=f"图{i}"))
            msgs.append({"role": "assistant", "content": f"a{i}"})
        return msgs

    def test_under_limit_unchanged(self):
        msgs = self._msgs_with_images(5)
        out, dropped = recycle_old_images(msgs, keep_recent=5)
        assert out is msgs and dropped == 0

    def test_over_limit_drops_oldest(self):
        msgs = self._msgs_with_images(7)
        out, dropped = recycle_old_images(msgs, keep_recent=5)
        assert dropped == 2
        # 最旧 2 张变占位文本，文本块不受影响
        assert out[0]["content"][1] == {"type": "text", "text": IMAGE_DROPPED_PLACEHOLDER}
        assert out[2]["content"][1] == {"type": "text", "text": IMAGE_DROPPED_PLACEHOLDER}
        assert out[0]["content"][0] == {"type": "text", "text": "图0"}
        # 最近 5 张保留
        assert count_images(out) == 5
        assert out[4]["content"][1]["type"] == "image_url"

    def test_original_not_mutated(self):
        msgs = self._msgs_with_images(7)
        recycle_old_images(msgs, keep_recent=5)
        assert count_images(msgs) == 7

    def test_current_turn_batch_protected(self):
        # 一次附 6 张（当前轮=最后一条 assistant 之后）→ 全部保留，不误伤刚上传的图
        msgs = [
            {"role": "user", "content": "早"},
            {"role": "assistant", "content": "好"},
            {"role": "user", "content": [
                {"type": "text", "text": "看这 6 张"},
                *({"type": "image_url", "image_url": {"url": f"{IMAGE_REF_PREFIX}{i:016x}.png"}}
                  for i in range(6)),
            ]},
        ]
        out, dropped = recycle_old_images(msgs, keep_recent=5)
        assert out is msgs and dropped == 0

    def test_current_turn_protected_old_recycled_first(self):
        # 旧图 3 张 + 当前轮 4 张 → 旧图预算 5-4=1，回收最旧 2 张；当前轮 4 张完好
        msgs = self._msgs_with_images(3)
        msgs.append({"role": "user", "content": [
            {"type": "text", "text": "新批次"},
            *({"type": "image_url", "image_url": {"url": f"{IMAGE_REF_PREFIX}{90 + i:016x}.png"}}
              for i in range(4)),
        ]})
        out, dropped = recycle_old_images(msgs, keep_recent=5)
        assert dropped == 2
        assert count_images(out) == 5
        # 当前轮 4 张全部完好
        assert sum(1 for b in out[-1]["content"] if b.get("type") == "image_url") == 4
        # 最旧 2 张变占位
        assert out[0]["content"][1] == {"type": "text", "text": IMAGE_DROPPED_PLACEHOLDER}
        assert out[2]["content"][1] == {"type": "text", "text": IMAGE_DROPPED_PLACEHOLDER}


class TestCountImages:
    def test_str_content(self):
        assert count_images([{"role": "user", "content": "hi"}]) == 0

    def test_mixed(self):
        msgs = [_img_msg(), {"role": "assistant", "content": "ok"}, _img_msg()]
        assert count_images(msgs) == 2


# ── detect_image_mime / 图片 sentinel（read_file 读图，设计 §4） ─────

class TestDetectImageMime:
    def test_by_extension(self, tmp_path):
        p = tmp_path / "x.png"
        p.write_bytes(b"anything")  # 扩展名优先，不看内容
        assert detect_image_mime(str(p)) == "image/png"

    def test_magic_bytes_no_extension(self, tmp_path):
        p = tmp_path / "noext"
        p.write_bytes(_PNG_MAGIC + b"d")
        assert detect_image_mime(str(p)) == "image/png"

    def test_non_image(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_bytes(b"hello world")
        assert detect_image_mime(str(p)) is None

    def test_missing_no_extension(self, tmp_path):
        assert detect_image_mime(str(tmp_path / "nope")) is None


class TestImageSentinel:
    def test_roundtrip_with_text(self):
        s = make_image_sentinel([("loongimg://abc.png", "image/png")], text="页面说明")
        assert parse_image_sentinel(s) == ([("loongimg://abc.png", "image/png")], "页面说明")

    def test_multi_image(self):
        imgs = [("loongimg://a.png", "image/png"), ("loongimg://b.jpg", "image/jpeg")]
        assert parse_image_sentinel(make_image_sentinel(imgs)) == (imgs, "")

    def test_plain_text_none(self):
        assert parse_image_sentinel("1\thello\n2\tworld") is None

    def test_malformed_json_none(self):
        assert parse_image_sentinel('{"__images__" broken') is None

    def test_non_ref_url_rejected(self):
        # 只认 loongimg:// 引用，不给注入任意 URL 留口子
        s = '{"__images__": [{"ref": "https://evil.example/x.png", "mime": "image/png"}], "text": ""}'
        assert parse_image_sentinel(s) is None

    def test_one_bad_ref_rejects_all(self):
        s = ('{"__images__": [{"ref": "loongimg://ok.png", "mime": "image/png"}, '
             '{"ref": "file:///etc/passwd", "mime": "image/png"}], "text": ""}')
        assert parse_image_sentinel(s) is None

    def test_empty_images_none(self):
        assert parse_image_sentinel('{"__images__": [], "text": "x"}') is None


class TestStoreImageBytes:
    def test_stores_and_dedupes(self):
        ref1 = store_image_bytes(_PNG_MAGIC + b"x")
        ref2 = store_image_bytes(_PNG_MAGIC + b"x")
        assert ref1 == ref2 and ref1.endswith(".png")
        assert len(list(messages_mod.images_dir().iterdir())) == 1

    def test_mime_by_magic_bytes(self):
        # 格式一律以 magic bytes 实测为准（外部声明的 mime 不可信）
        assert store_image_bytes(b"\xff\xd8\xff jpeg-bytes").endswith(".jpg")

    def test_unrecognized_raises(self):
        with pytest.raises(ValueError, match="无法识别"):
            store_image_bytes(b"not an image")

    def test_oversized_raises(self):
        with pytest.raises(ValueError, match="过大"):
            store_image_bytes(b"x" * (10 * 1024 * 1024 + 1))


# ── extract_image_paths（TUI 拖图/贴路径） ──────────────────────────

class TestExtractImagePaths:
    def test_bare_existing_path(self, tmp_path):
        p = tmp_path / "shot.png"
        p.write_bytes(_PNG_MAGIC + b"d")
        text, imgs = extract_image_paths(f"看这个 {p}")
        assert imgs == [str(p)]
        assert text == "看这个"

    def test_quoted_path_with_spaces(self, tmp_path):
        d = tmp_path / "my shots"
        d.mkdir()
        p = d / "a.png"
        p.write_bytes(_PNG_MAGIC + b"d")
        text, imgs = extract_image_paths(f'描述 "{p}"')
        assert imgs == [str(p)]
        assert text == "描述"

    def test_only_path_empties_text(self, tmp_path):
        p = tmp_path / "x.png"
        p.write_bytes(_PNG_MAGIC + b"d")
        text, imgs = extract_image_paths(str(p))
        assert imgs == [str(p)]
        assert text == ""

    def test_nonexistent_path_ignored(self):
        text, imgs = extract_image_paths("看 C:/nope/x.png 这个")
        assert imgs == []
        assert text == "看 C:/nope/x.png 这个"

    def test_no_image(self):
        assert extract_image_paths("hello world") == ("hello world", [])

    def test_multiple_images(self, tmp_path):
        p1 = tmp_path / "a.png"
        p1.write_bytes(_PNG_MAGIC + b"a")
        p2 = tmp_path / "b.jpg"
        p2.write_bytes(b"\xff\xd8\xff b")
        text, imgs = extract_image_paths(f"对比 {p1} 和 {p2}")
        assert set(imgs) == {str(p1), str(p2)}


class TestIsImageFile:
    def test_existing_image(self, tmp_path):
        p = tmp_path / "x.png"
        p.write_bytes(_PNG_MAGIC + b"d")
        assert is_image_file(str(p)) is True

    def test_nonexistent(self, tmp_path):
        assert is_image_file(str(tmp_path / "nope.png")) is False

    def test_non_image_ext(self, tmp_path):
        p = tmp_path / "x.txt"
        p.write_bytes(b"hi")
        assert is_image_file(str(p)) is False


def _mock_llm(summary_text: str = "<summary>本轮对话完成了压缩测试所需的全部准备工作，包括消息构造、角色交替修复、附件重建与边界标记，所有细节均已核对无误。</summary>") -> MagicMock:
    llm = MagicMock()
    llm.cache_aware = True
    llm.model = "deepseek-v4-flash"
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = summary_text
    chunk.choices[0].delta.tool_calls = None
    chunk.choices[0].delta.reasoning_content = None
    chunk.choices[0].finish_reason = "stop"
    chunk.usage = None

    async def mock_stream(**kwargs):
        yield chunk

    llm.chat_stream = mock_stream
    return llm
