import pytest
from pathlib import Path
from loongcli.tools.edit_file import EditFileTool, _diff_stats


@pytest.fixture
def tool():
    return EditFileTool()


@pytest.fixture
def sample_file(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("def hello():\n    print('hello')\n    return True\n", encoding="utf-8")
    return str(f)


@pytest.mark.asyncio
async def test_edit_replaces_string(tool, sample_file):
    result = await tool.execute(path=sample_file, old_string="print('hello')", new_string="print('world')")
    assert "成功" in result
    content = Path(sample_file).read_text(encoding="utf-8")
    assert "print('world')" in content
    assert "print('hello')" not in content


@pytest.mark.asyncio
async def test_edit_old_string_not_found(tool, sample_file):
    result = await tool.execute(path=sample_file, old_string="nonexistent string", new_string="replacement")
    assert "未找到" in result


@pytest.mark.asyncio
async def test_edit_multiple_matches_fails(tool, tmp_path):
    f = tmp_path / "dup.txt"
    f.write_text("aaa\nbbb\naaa\n", encoding="utf-8")
    result = await tool.execute(path=str(f), old_string="aaa", new_string="ccc")
    assert "多处匹配" in result or "2" in result


@pytest.mark.asyncio
async def test_edit_nonexistent_file(tool):
    result = await tool.execute(path="/nonexistent/file.txt", old_string="a", new_string="b")
    assert "错误" in result


def test_tool_schema(tool):
    assert tool.name == "edit_file"
    assert "old_string" in tool.parameters["properties"]
    assert "new_string" in tool.parameters["properties"]


# ── fuzzy match tests ──


@pytest.mark.asyncio
async def test_strip_match_finds_whitespace_variant(tool, tmp_path):
    """old_string with extra surrounding whitespace should match after strip."""
    f = tmp_path / "code.py"
    f.write_text("def hello():\n    print('hello')\n", encoding="utf-8")
    # original has no leading space, old_string has extra spaces
    result = await tool.execute(
        path=str(f),
        old_string="  print('hello')  ",
        new_string="print('world')",
    )
    assert "成功" in result
    assert "print('world')" in f.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_normalize_whitespace_match(tool, tmp_path):
    """Collapsed whitespace should match spread-out whitespace."""
    f = tmp_path / "code.py"
    f.write_text("def   hello(  ):\n    return   True\n", encoding="utf-8")
    result = await tool.execute(
        path=str(f),
        old_string="def hello( ):",
        new_string="def hello():",
    )
    assert "成功" in result
    content = f.read_text(encoding="utf-8")
    assert "def hello():" in content


@pytest.mark.asyncio
async def test_fuzzy_auto_replace_high_similarity(tool, tmp_path):
    """High similarity (> 0.85) single-line match → auto-replace."""
    f = tmp_path / "code.py"
    f.write_text("import os\nimport systeem\nimport json\n", encoding="utf-8")
    result = await tool.execute(
        path=str(f),
        old_string="import system",
        new_string="import sys",
    )
    assert "成功" in result
    assert "fuzzy" in result.lower()
    content = f.read_text(encoding="utf-8")
    assert "import sys\n" in content
    assert "import systeem" not in content


@pytest.mark.asyncio
async def test_fuzzy_candidate_shown_medium_similarity(tool, tmp_path):
    """Medium similarity (0.6-0.85) → show candidate, don't modify."""
    f = tmp_path / "code.py"
    original = "def calculate_total(items):\n    return sum(items)\n"
    f.write_text(original, encoding="utf-8")
    result = await tool.execute(
        path=str(f),
        old_string="def compute_total(elements):",  # different words
        new_string="def calc_total(items):",
    )
    assert "未找到精确匹配" in result or "疑似" in result
    assert "文件未被修改" in result
    assert f.read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_fuzzy_below_threshold_shows_closest_lines(tool, tmp_path):
    """Very low similarity (< 0.6) → error with closest lines."""
    f = tmp_path / "code.py"
    f.write_text("x = 1\ny = 2\nz = 3\n", encoding="utf-8")
    result = await tool.execute(
        path=str(f),
        old_string="def main():",
        new_string="def run():",
    )
    assert "未找到" in result or "错误" in result
    # Should mention line numbers or content hints
    assert any(word in result for word in ["行", "line", "x", "y"])


@pytest.mark.asyncio
async def test_fuzzy_ambiguous_two_close_matches(tool, tmp_path):
    """Two matches with close scores → ambiguous, show both."""
    f = tmp_path / "code.py"
    f.write_text(
        "def handle_request(req):\n    pass\n\ndef handle_response(res):\n    pass\n",
        encoding="utf-8",
    )
    result = await tool.execute(
        path=str(f),
        old_string="def handle_event(evt):",
        new_string="def handle(evt):",
    )
    # Should show candidates, not auto-replace
    assert "文件未被修改" in result
    assert "候" in result  # 候选


@pytest.mark.asyncio
async def test_fuzzy_multi_line_auto_replace(tool, tmp_path):
    """Multi-line fuzzy match with high similarity → auto-replace."""
    f = tmp_path / "code.py"
    f.write_text(
        "def greet(name):\n"
        "    # say hello\n"
        "    print('hello', name)\n"
        "    return True\n",
        encoding="utf-8",
    )
    result = await tool.execute(
        path=str(f),
        old_string="def greet(name):\n    # say hello\n    print('hello', name)",
        new_string="def greet(name, lang='en'):\n    # greet\n    print('hello', name, lang)",
    )
    assert "成功" in result
    content = f.read_text(encoding="utf-8")
    assert "def greet(name, lang='en')" in content


@pytest.mark.asyncio
async def test_fuzzy_skips_when_old_string_too_long(tool, tmp_path):
    """old_string > 1000 chars → skip fuzzy, direct error."""
    f = tmp_path / "code.py"
    f.write_text("short\n", encoding="utf-8")
    huge_old = "x" * 1001
    result = await tool.execute(path=str(f), old_string=huge_old, new_string="y")
    assert "未找到" in result or "错误" in result
    assert "fuzzy" not in result.lower()


@pytest.mark.asyncio
async def test_fuzzy_empty_file(tool, tmp_path):
    """Empty file → clean error."""
    f = tmp_path / "empty.py"
    f.write_text("", encoding="utf-8")
    result = await tool.execute(path=str(f), old_string="anything", new_string="x")
    assert "空" in result or "错误" in result


class TestDiffStats:
    def test_pure_addition(self):
        result = _diff_stats("a\n", "a\nb\nc\n")
        assert "添加 2 行" in result
        assert "删除" not in result

    def test_pure_deletion(self):
        result = _diff_stats("a\nb\nc\n", "a\n")
        assert "删除 2 行" in result
        assert "添加" not in result

    def test_replacement(self):
        result = _diff_stats("old line\n", "new line\n")
        assert "添加 1 行" in result
        assert "删除 1 行" in result

    def test_mixed_add_and_delete(self):
        result = _diff_stats("a\nb\nc\n", "a\nx\ny\nz\n")
        assert "添加" in result
        assert "删除" in result

    def test_no_change(self):
        result = _diff_stats("same\n", "same\n")
        assert result == ""

    def test_empty_to_content(self):
        result = _diff_stats("", "new\nlines\n")
        assert "添加 2 行" in result

    def test_content_to_empty(self):
        result = _diff_stats("old\nlines\n", "")
        assert "删除 2 行" in result

    def test_both_empty(self):
        assert _diff_stats("", "") == ""


# ── 编码一致性（GBK / UTF-8-BOM 往返）──────────────────────────────

@pytest.mark.asyncio
async def test_edit_gbk_file_preserves_gbk(tool, tmp_path):
    """GBK 文件能改，且写回仍是 GBK（此前硬编码 utf-8 会 UnicodeDecodeError）。"""
    f = tmp_path / "gbk.py"
    f.write_bytes("# 你好\nx = 1\n".encode("gbk"))
    result = await tool.execute(path=str(f), old_string="x = 1", new_string="x = 2")
    assert "成功" in result
    # 仍是合法 GBK 且中文完好、修改生效
    text = f.read_bytes().decode("gbk")
    assert "你好" in text
    assert "x = 2" in text


@pytest.mark.asyncio
async def test_edit_utf8_bom_preserves_bom(tool, tmp_path):
    """UTF-8-BOM 文件编辑后 BOM 不被剥离。"""
    f = tmp_path / "bom.py"
    f.write_bytes(b"\xef\xbb\xbf" + "s = 'a'\n".encode("utf-8"))
    result = await tool.execute(path=str(f), old_string="s = 'a'", new_string="s = 'b'")
    assert "成功" in result
    raw = f.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # BOM 保留
    assert "s = 'b'" in raw.decode("utf-8-sig")
