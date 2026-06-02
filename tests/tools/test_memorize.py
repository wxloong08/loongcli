import pytest
from loongcli.tools.memorize import MemorizeTool
from loongcli.tools.base import ToolRegistry
from loongcli.memory.store import MemoryStore


@pytest.fixture
def mem_tool(tmp_path):
    store = MemoryStore(path=tmp_path / "kv.json")
    return MemorizeTool(store), store


class TestMemorizeTool:
    @pytest.mark.asyncio
    async def test_save(self, mem_tool):
        tool, store = mem_tool
        result = await tool.execute(
            operation="save", category="prefs", key="lang",
            value="Python", type="user",
        )
        assert "已保存" in result
        assert store.get("prefs", "lang") == "Python"
        assert store.get_entry("prefs", "lang").type == "user"

    @pytest.mark.asyncio
    async def test_save_requires_value(self, mem_tool):
        tool, _ = mem_tool
        result = await tool.execute(
            operation="save", category="prefs", key="lang",
        )
        assert "错误" in result

    @pytest.mark.asyncio
    async def test_update_existing(self, mem_tool):
        tool, store = mem_tool
        await tool.execute(
            operation="save", category="prefs", key="lang",
            value="Python", type="user",
        )
        await tool.execute(
            operation="save", category="prefs", key="lang",
            value="Rust", type="user",
        )
        assert store.get("prefs", "lang") == "Rust"

    @pytest.mark.asyncio
    async def test_delete(self, mem_tool):
        tool, store = mem_tool
        store.set("prefs", "lang", "Python")
        result = await tool.execute(
            operation="delete", category="prefs", key="lang",
        )
        assert "已删除" in result
        assert store.get("prefs", "lang") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, mem_tool):
        tool, _ = mem_tool
        result = await tool.execute(
            operation="delete", category="nope", key="nope",
        )
        assert "未找到" in result

    @pytest.mark.asyncio
    async def test_unknown_operation(self, mem_tool):
        tool, _ = mem_tool
        result = await tool.execute(
            operation="explode", category="x", key="y",
        )
        assert "未知操作" in result

    def test_tool_schema(self, mem_tool):
        tool, _ = mem_tool
        reg = ToolRegistry()
        reg.register(tool)
        schemas = reg.get_tool_schemas()
        assert len(schemas) == 1
        fn = schemas[0]["function"]
        assert fn["name"] == "memorize"
        props = fn["parameters"]["properties"]
        assert "operation" in props
        assert "category" in props
        assert "key" in props
        assert "value" in props
        assert "type" in props

    @pytest.mark.asyncio
    async def test_default_type_is_project(self, mem_tool):
        tool, store = mem_tool
        await tool.execute(
            operation="save", category="misc", key="k", value="v",
        )
        assert store.get_entry("misc", "k").type == "project"
