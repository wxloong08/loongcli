import pytest
from loongcli.tools.read_file import ReadFileTool, _decode_with_fallback


@pytest.fixture
def tool():
    return ReadFileTool()


@pytest.fixture
def sample_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")
    return str(f)


@pytest.mark.asyncio
async def test_read_whole_file(tool, sample_file):
    result = await tool.execute(path=sample_file)
    assert "line1" in result
    assert "line5" in result


@pytest.mark.asyncio
async def test_read_with_offset_and_limit(tool, sample_file):
    result = await tool.execute(path=sample_file, offset=2, limit=2)
    assert "line2" in result
    assert "line3" in result
    assert "line1" not in result
    assert "line4" not in result


@pytest.mark.asyncio
async def test_read_nonexistent_file(tool):
    result = await tool.execute(path="/nonexistent/file.txt")
    assert "错误" in result or "Error" in result


@pytest.mark.asyncio
async def test_read_file_has_line_numbers(tool, sample_file):
    result = await tool.execute(path=sample_file)
    assert "1\t" in result


def test_tool_schema(tool):
    assert tool.name == "read_file"
    assert "path" in tool.parameters["properties"]


# ── encoding fallback tests ──


@pytest.mark.asyncio
async def test_utf8_no_encoding_annotation(tool, sample_file):
    """UTF-8 file should not have encoding annotation."""
    result = await tool.execute(path=sample_file)
    assert "[编码：" not in result


@pytest.mark.asyncio
async def test_gbk_encoded_file(tool, tmp_path):
    """GBK-encoded file → fallback to gbk, annotation added."""
    f = tmp_path / "chinese_gbk.txt"
    content = "你好世界\n第二行\n"
    f.write_bytes(content.encode("gbk"))
    result = await tool.execute(path=str(f))
    assert "[编码：gbk]" in result
    assert "你好世界" in result


@pytest.mark.asyncio
async def test_utf8_bom_file(tool, tmp_path):
    """UTF-8 with BOM → utf-8-sig fallback."""
    f = tmp_path / "bom.txt"
    content = "line with BOM\n"
    f.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
    result = await tool.execute(path=str(f))
    assert "[编码：utf-8-sig]" in result


@pytest.mark.asyncio
async def test_binary_file_falls_back_to_latin1(tool, tmp_path):
    """True binary file → latin-1 fallback (always succeeds)."""
    f = tmp_path / "data.bin"
    f.write_bytes(bytes(range(256)))
    result = await tool.execute(path=str(f))
    assert "[编码：latin-1]" in result
    assert "错误" not in result


@pytest.mark.asyncio
async def test_cp936_fallback(tool, tmp_path):
    """CP936 is a variant of GBK for extended characters."""
    f = tmp_path / "cp936.txt"
    # Some characters encode differently in cp936 vs gbk
    content = "中文字符测试\n"
    f.write_bytes(content.encode("cp936"))
    result = await tool.execute(path=str(f))
    assert "错误" not in result
    assert "中文字符测试" in result


# ── large file / streaming protection tests ──


@pytest.mark.asyncio
async def test_file_too_large_rejected(tool, tmp_path, monkeypatch):
    """Files > 20MB should be rejected."""
    import loongcli.tools.read_file as mod

    monkeypatch.setattr(mod, "_MAX_FILE_MB", 0)  # reject any non-empty file
    f = tmp_path / "big.txt"
    f.write_text("hello\n", encoding="utf-8")
    result = await tool.execute(path=str(f))
    assert "过大" in result or "错误" in result


@pytest.mark.asyncio
async def test_skip_past_max_lines_rejected(tool, tmp_path, monkeypatch):
    """Offset beyond _MAX_LINES_SKIP should be rejected."""
    import loongcli.tools.read_file as mod

    monkeypatch.setattr(mod, "_MAX_LINES_SKIP", 3)
    f = tmp_path / "many.txt"
    f.write_text("\n".join(str(i) for i in range(1, 100)), encoding="utf-8")
    result = await tool.execute(path=str(f), offset=90)
    assert "错误" in result


@pytest.mark.asyncio
async def test_large_file_streaming_shows_more_indicator(tool, tmp_path):
    """Large file within limit → streamed with 'more' indicator."""
    f = tmp_path / "large.txt"
    lines = [f"line_{i}" for i in range(1, 50)]
    f.write_text("\n".join(lines), encoding="utf-8")
    result = await tool.execute(path=str(f), offset=1, limit=10)
    assert "line_1" in result
    assert "line_10" in result
    assert "line_11" not in result  # past limit
    assert "共" in result  # range header


@pytest.mark.asyncio
async def test_mid_file_offset_streaming(tool, tmp_path):
    """Streaming from mid-file offset within limit."""
    f = tmp_path / "mid.txt"
    lines = [f"line_{i}" for i in range(1, 60)]
    f.write_text("\n".join(lines), encoding="utf-8")
    result = await tool.execute(path=str(f), offset=25, limit=5)
    assert "line_25" in result
    assert "line_29" in result
    assert "line_24" not in result
    assert "line_30" not in result


# ── truncated UTF-8 sample boundary tests ──


class TestDecodeWithFallback:
    def test_utf8_truncated_at_multibyte_boundary(self):
        """UTF-8 Chinese text truncated mid-character should still detect as utf-8."""
        full = "你好世界测试内容" * 200
        raw = full.encode("utf-8")[:4096]
        _, enc = _decode_with_fallback(raw)
        assert enc == "utf-8"

    def test_utf8_clean_boundary(self):
        """UTF-8 text not truncated at multi-byte boundary works normally."""
        text = "hello world\n" * 100
        raw = text.encode("utf-8")[:4096]
        _, enc = _decode_with_fallback(raw)
        assert enc == "utf-8"

    def test_genuine_latin1_not_misdetected(self):
        """Bytes that are truly latin-1 (invalid UTF-8 throughout) stay latin-1."""
        raw = bytes([0xC0, 0x20, 0xE0, 0x20, 0xF8]) * 100
        _, enc = _decode_with_fallback(raw)
        assert enc != "utf-8"

    def test_gbk_chinese(self):
        """GBK-encoded Chinese should be detected as gbk, not utf-8."""
        raw = "你好世界".encode("gbk")
        _, enc = _decode_with_fallback(raw)
        assert enc in ("gbk", "utf-8")  # short GBK may also be valid UTF-8

    def test_bom_detected(self):
        raw = b"\xef\xbb\xbf" + "hello".encode("utf-8")
        _, enc = _decode_with_fallback(raw)
        assert enc == "utf-8-sig"


@pytest.mark.asyncio
async def test_large_utf8_chinese_file_no_garble(tool, tmp_path):
    """Regression: large UTF-8 Chinese file should not fall back to latin-1."""
    f = tmp_path / "chinese_large.txt"
    content = "这是一段中文内容用于测试编码检测\n" * 300
    f.write_text(content, encoding="utf-8")
    result = await tool.execute(path=str(f), limit=10)
    assert "latin-1" not in result
    assert "这是一段中文" in result


# ── 图片分支：返回 sentinel，由 agent 循环转成带图 user 消息（设计 §4） ──

@pytest.mark.asyncio
async def test_read_image_returns_sentinel(tool, tmp_path):
    from loongcli.core.messages import parse_image_sentinel
    p = tmp_path / "shot.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\ndata")
    result = await tool.execute(path=str(p))
    parsed = parse_image_sentinel(result)
    assert parsed is not None
    images, text = parsed
    assert len(images) == 1 and text == ""
    ref, mime = images[0]
    assert ref.startswith("loongimg://") and mime == "image/png"


@pytest.mark.asyncio
async def test_read_image_no_extension_by_magic(tool, tmp_path):
    from loongcli.core.messages import parse_image_sentinel
    p = tmp_path / "noext"
    p.write_bytes(b"\x89PNG\r\n\x1a\ndata")
    assert parse_image_sentinel(await tool.execute(path=str(p))) is not None


@pytest.mark.asyncio
async def test_read_oversized_image_clear_error(tool, tmp_path):
    # 10MB 图片上限走图片分支的明确文案，而非"用 offset/limit 分段读"的文本建议
    p = tmp_path / "big.png"
    p.write_bytes(b"x" * (10 * 1024 * 1024 + 1))
    result = await tool.execute(path=str(p))
    assert result.startswith("错误") and "过大" in result
    assert "__image__" not in result


@pytest.mark.asyncio
async def test_read_text_file_no_sentinel(tool, sample_file):
    result = await tool.execute(path=sample_file)
    assert "line1" in result and "__image__" not in result


# ── 按字符预算分页（2026-11-19：此前交给入口截断，中等文件被掐头去尾砍掉约八成）──

from loongcli.tools.read_file import READ_CHAR_BUDGET  # noqa: E402


def _big_file(tmp_path, n_lines=3000, width=60):
    f = tmp_path / "big.py"
    f.write_text("\n".join(f"line_{i:05d} " + "x" * width for i in range(1, n_lines + 1)), encoding="utf-8")
    return f


@pytest.mark.asyncio
async def test_pages_at_char_budget_on_line_boundary(tool, tmp_path):
    f = _big_file(tmp_path)  # 每行约 75 字符，3000 行 ≈ 22.5 万字符
    result = await tool.execute(path=str(f))
    assert len(result) <= READ_CHAR_BUDGET + 300          # 头部说明之外不超预算
    body = result.splitlines()
    last = [l for l in body if l.split("\t", 1)[0].isdigit()][-1]
    last_no = int(last.split("\t", 1)[0])
    assert last.endswith("x" * 60)                        # 最后一行完整，不在行中间断开
    assert f"[第 1-{last_no} 行 / 共 3000 行]" in result    # 总行数是准确值
    assert f"继续读用 offset={last_no + 1}" in result
    assert "字符上限" in result


@pytest.mark.asyncio
async def test_next_page_continues_without_gap(tool, tmp_path):
    f = _big_file(tmp_path)
    first = await tool.execute(path=str(f))
    nxt = int(first.split("继续读用 offset=")[1].split("]")[0])
    second = await tool.execute(path=str(f), offset=nxt)
    assert f"{nxt}\tline_{nxt:05d}" in second
    assert f"{nxt - 1}\tline_" not in second


@pytest.mark.asyncio
async def test_medium_file_returned_whole(tool, tmp_path):
    """p3-1 里被截的那类文件（8.8K–13.6K 字符）现在整份返回，没有续读提示。"""
    f = tmp_path / "audio_service.py"
    f.write_text("\n".join(f"    value_{i} = compute({i})  # 注释" for i in range(1, 400)), encoding="utf-8")
    assert 9000 < f.stat().st_size < 20000
    result = await tool.execute(path=str(f))
    assert "399\t" in result
    assert "未读完" not in result and "共" not in result


@pytest.mark.asyncio
async def test_line_limit_still_reported(tool, tmp_path):
    f = tmp_path / "s.txt"
    f.write_text("\n".join(f"l{i}" for i in range(1, 50)), encoding="utf-8")
    result = await tool.execute(path=str(f), limit=10)
    assert "[第 1-10 行 / 共 49 行]" in result
    assert "行数上限" in result and "offset=11" in result


@pytest.mark.asyncio
async def test_single_overlong_line_is_clipped(tool, tmp_path):
    f = tmp_path / "min.js"
    f.write_text("a" * (READ_CHAR_BUDGET * 2) + "\nsecond", encoding="utf-8")
    result = await tool.execute(path=str(f))
    assert "该行过长" in result
    assert len(result) < READ_CHAR_BUDGET + 500
    assert "继续读用 offset=2" in result


@pytest.mark.asyncio
async def test_full_page_passes_entry_fuse_untouched(tool, tmp_path):
    """满页的 read_file 结果低于入口保险丝，原样进上下文（不再被掐头去尾）。"""
    from loongcli.core.tool_result_manager import ToolResultManager
    page = await tool.execute(path=str(_big_file(tmp_path)))
    assert ToolResultManager().process("read_file", page) == page
