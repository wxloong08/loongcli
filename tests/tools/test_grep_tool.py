import pytest
from pathlib import Path
from loongcli.tools.grep_tool import GrepTool


@pytest.fixture
def tool():
    return GrepTool()


@pytest.fixture
def project_dir(tmp_path):
    (tmp_path / "a.py").write_text("def hello():\n    print('hello')\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def world():\n    print('world')\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("no functions here\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.py").write_text("def nested():\n    pass\n", encoding="utf-8")
    return str(tmp_path)


@pytest.mark.asyncio
async def test_grep_finds_pattern(tool, project_dir):
    result = await tool.execute(pattern=r"def \w+", path=project_dir)
    assert "hello" in result
    assert "world" in result
    assert "nested" in result


@pytest.mark.asyncio
async def test_grep_with_glob_filter(tool, project_dir):
    result = await tool.execute(pattern=r"def \w+", path=project_dir, glob="*.py")
    assert "hello" in result
    assert "no functions" not in result


@pytest.mark.asyncio
async def test_grep_no_matches(tool, project_dir):
    result = await tool.execute(pattern=r"class \w+", path=project_dir)
    assert "未找到" in result


@pytest.mark.asyncio
async def test_grep_shows_line_numbers(tool, project_dir):
    result = await tool.execute(pattern="print", path=project_dir)
    assert ":2:" in result


@pytest.mark.asyncio
async def test_grep_case_insensitive(tool, project_dir):
    result = await tool.execute(pattern="HELLO", path=project_dir, case_insensitive=True)
    assert "hello" in result


def test_tool_schema(tool):
    assert tool.name == "grep"
    assert "pattern" in tool.parameters["properties"]


@pytest.mark.asyncio
async def test_grep_skips_large_files(tool, tmp_path, monkeypatch):
    """Files exceeding _MAX_FILE_SIZE should be skipped."""
    import loongcli.tools.grep_tool as mod

    monkeypatch.setattr(mod, "_MAX_FILE_SIZE", 10)  # 10 bytes only
    f = tmp_path / "big.py"
    f.write_text("def hello():\n    print('hello')\n", encoding="utf-8")  # > 10 bytes
    result = await tool.execute(pattern="hello", path=str(tmp_path))
    # Should find nothing since all files are "too large"
    assert "未找到" in result


@pytest.mark.asyncio
async def test_grep_reports_skipped_files(tool, tmp_path):
    """Binary/unreadable files should be reported in skip notice."""
    f = tmp_path / "binary.dat"
    f.write_bytes(bytes(range(256)))
    (tmp_path / "ok.py").write_text("hello world\n", encoding="utf-8")
    result = await tool.execute(pattern="hello", path=str(tmp_path))
    # binary file should be skipped and reported
    assert "跳过" in result
    assert "binary.dat" in result


@pytest.mark.asyncio
async def test_grep_single_file(tool, project_dir):
    """Should accept a file path, not just a directory."""
    file_path = str(Path(project_dir) / "a.py")
    result = await tool.execute(pattern="hello", path=file_path)
    assert "hello" in result
    assert "a.py" in result


@pytest.mark.asyncio
async def test_grep_single_file_no_match(tool, project_dir):
    file_path = str(Path(project_dir) / "a.py")
    result = await tool.execute(pattern="zzz_not_here", path=file_path)
    assert "未找到" in result


@pytest.mark.asyncio
async def test_grep_nonexistent_path(tool):
    result = await tool.execute(pattern="hello", path="/does/not/exist")
    assert "不存在" in result


# ── Windows 不可访问文件（WinError 1920 等 OSError）只跳过不炸 ──

@pytest.mark.asyncio
async def test_grep_skips_unreadable_oserror(tmp_path, monkeypatch):
    """单个文件 read 抛 OSError（如 reparse 点）→ 记跳过，搜索继续。"""
    good = tmp_path / "good.py"
    good.write_text("cache_aware = True\n", encoding="utf-8")
    bad = tmp_path / "bad.py"
    bad.write_text("cache_aware = False\n", encoding="utf-8")

    real_read = Path.read_text

    def fake_read(self, *a, **kw):
        if self.name == "bad.py":
            raise OSError(1920, "系统无法访问此文件")
        return real_read(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", fake_read)

    result = await GrepTool().execute(pattern="cache_aware", path=str(tmp_path))
    assert "good.py:1:" in result
    assert "跳过 1 个" in result


@pytest.mark.asyncio
async def test_grep_skips_is_file_oserror(tmp_path, monkeypatch):
    """is_file() 本身抛 OSError（真机截图里的炸点）→ 同样只跳过。"""
    good = tmp_path / "good.py"
    good.write_text("target_here\n", encoding="utf-8")
    bad = tmp_path / "badlink"
    bad.write_text("x", encoding="utf-8")

    real_is_file = Path.is_file

    def fake_is_file(self, *a, **kw):
        if self.name == "badlink":
            raise OSError(1920, "系统无法访问此文件")
        return real_is_file(self, *a, **kw)

    monkeypatch.setattr(Path, "is_file", fake_is_file)

    result = await GrepTool().execute(pattern="target_here", path=str(tmp_path))
    assert "good.py:1:" in result
    assert "跳过 1 个" in result


@pytest.mark.asyncio
async def test_grep_budget_not_burned_by_skip_dirs(tmp_path, monkeypatch):
    """真机回归：.venv 里的海量条目不得消耗扫描预算（此前项目根只搜到 16 个文件就截断）。"""
    import loongcli.tools.grep_tool as mod

    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    for i in range(30):
        (venv / f"junk{i:02d}.py").write_text("cache_aware = 'noise'\n", encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    (src / "target.py").write_text("cache_aware = True\n", encoding="utf-8")

    # 预算 5：旧实现 .venv 的 30 条枚举就触顶、src 永远轮不到
    monkeypatch.setattr(mod, "_MAX_TOTAL_FILES", 5)

    result = await mod.GrepTool().execute(pattern="cache_aware", path=str(tmp_path))
    assert "target.py:1:" in result
    assert "扫描已截断" not in result


@pytest.mark.asyncio
async def test_grep_output_states_search_root(tmp_path):
    """结果必须标明搜索根——模型把 cwd 的结果说成别的项目"里没有"是毒记忆的认知源头。"""
    (tmp_path / "a.py").write_text("needle = 1\n", encoding="utf-8")
    result = await GrepTool().execute(pattern="needle", path=str(tmp_path))
    assert result.startswith("[搜索根: ")
    assert str(tmp_path) in result.splitlines()[0]


@pytest.mark.asyncio
async def test_grep_not_found_disclaims_scope(tmp_path):
    """未找到时明确"只能说明该范围内没搜到"，不给否定断言留空间。"""
    (tmp_path / "a.py").write_text("nothing here\n", encoding="utf-8")
    result = await GrepTool().execute(pattern="ghost_needle", path=str(tmp_path))
    assert "[搜索根: " in result
    assert "不代表其他目录/项目里不存在" in result
