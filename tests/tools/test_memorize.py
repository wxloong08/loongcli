import pytest
from loongcli.tools.memorize import MemorizeTool
from loongcli.tools.base import ToolRegistry
from loongcli.memory.markdown_store import MarkdownMemoryStore


@pytest.fixture
def mem_tool(tmp_path):
    store = MarkdownMemoryStore(base_dir=tmp_path)
    return MemorizeTool(store), store


class TestMemorizeTool:
    @pytest.mark.asyncio
    async def test_save(self, mem_tool):
        tool, store = mem_tool
        result = await tool.execute(
            operation="save", name="user-lang",
            description="Prefers Python", type="user",
            content="I mostly write Python.",
        )
        assert "已保存" in result
        mem = store.load("user-lang")
        assert mem is not None
        assert mem["type"] == "user"
        assert "Python" in mem["content"]

    @pytest.mark.asyncio
    async def test_save_requires_content(self, mem_tool):
        tool, _ = mem_tool
        result = await tool.execute(
            operation="save", name="test",
            description="desc", type="user",
        )
        assert "错误" in result

    @pytest.mark.asyncio
    async def test_save_requires_description(self, mem_tool):
        tool, _ = mem_tool
        result = await tool.execute(
            operation="save", name="test", type="user",
            content="body",
        )
        assert "错误" in result

    @pytest.mark.asyncio
    async def test_update_existing(self, mem_tool):
        tool, store = mem_tool
        await tool.execute(
            operation="save", name="k1",
            description="desc1", type="user", content="v1",
        )
        await tool.execute(
            operation="save", name="k1",
            description="desc2", type="user", content="v2",
        )
        mem = store.load("k1")
        assert mem["content"] == "v2"

    @pytest.mark.asyncio
    async def test_delete(self, mem_tool):
        tool, store = mem_tool
        store.save(name="k1", description="d", type="project", content="body")
        result = await tool.execute(operation="delete", name="k1")
        assert "已删除" in result
        assert store.load("k1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, mem_tool):
        tool, _ = mem_tool
        result = await tool.execute(operation="delete", name="nope")
        assert "未找到" in result

    @pytest.mark.asyncio
    async def test_unknown_operation(self, mem_tool):
        tool, _ = mem_tool
        result = await tool.execute(operation="explode", name="x")
        assert "未知操作" in result

    def test_tool_schema(self, mem_tool):
        tool, _ = mem_tool
        reg = ToolRegistry()
        reg.register(tool)
        schemas = reg.get_tool_schemas()
        fn = schemas[0]["function"]
        assert fn["name"] == "memorize"
        props = fn["parameters"]["properties"]
        assert "name" in props
        assert "description" in props
        assert "content" in props
        assert "type" in props
        assert "category" not in props
        assert "key" not in props
