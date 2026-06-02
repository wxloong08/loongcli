import pytest
from unittest.mock import MagicMock

from loongcli.tools.base import ToolRegistry
from loongcli.tools.routing import (
    AgentRole, SUBAGENT_BLACKLIST, BACKGROUND_WHITELIST, filter_tools,
)


def _make_registry(*names: str) -> ToolRegistry:
    registry = ToolRegistry()
    for name in names:
        tool = MagicMock()
        tool.name = name
        tool.description = f"{name} tool"
        tool.parameters = {"type": "object", "properties": {}}
        registry.register(tool)
    return registry


def _schema_names(schemas: list[dict]) -> set[str]:
    return {s["function"]["name"] for s in schemas}


class TestFilterTools:
    def test_main_gets_all(self):
        tools = {"a": 1, "b": 2, "c": 3}
        assert filter_tools(tools, AgentRole.MAIN) == tools

    def test_subagent_blacklist(self):
        tools = {"shell": 1, "delegate": 2, "send_message": 3, "task_status": 4, "glob": 5}
        result = filter_tools(tools, AgentRole.SUBAGENT)
        assert "shell" in result
        assert "glob" in result
        assert "delegate" not in result
        assert "send_message" not in result
        assert "task_status" not in result

    def test_background_whitelist(self):
        tools = {"read_file": 1, "shell": 2, "delegate": 3, "recall": 4, "glob": 5}
        result = filter_tools(tools, AgentRole.BACKGROUND)
        assert "read_file" in result
        assert "shell" in result
        assert "glob" in result
        assert "delegate" not in result
        assert "recall" not in result


class TestRegistryWithRole:
    def test_main_returns_all(self):
        reg = _make_registry("read_file", "shell", "delegate", "send_message")
        schemas = reg.get_tool_schemas(role=AgentRole.MAIN)
        assert _schema_names(schemas) == {"read_file", "shell", "delegate", "send_message"}

    def test_subagent_filters_blacklisted(self):
        reg = _make_registry("read_file", "shell", "delegate", "send_message", "task_status")
        schemas = reg.get_tool_schemas(role=AgentRole.SUBAGENT)
        names = _schema_names(schemas)
        assert "read_file" in names
        assert "shell" in names
        for blocked in SUBAGENT_BLACKLIST:
            assert blocked not in names

    def test_background_only_whitelist(self):
        reg = _make_registry("read_file", "write_file", "edit_file", "shell", "glob", "grep", "delegate", "recall")
        schemas = reg.get_tool_schemas(role=AgentRole.BACKGROUND)
        names = _schema_names(schemas)
        assert names == BACKGROUND_WHITELIST

    def test_default_role_is_main(self):
        reg = _make_registry("delegate", "shell")
        default = reg.get_tool_schemas()
        explicit = reg.get_tool_schemas(role=AgentRole.MAIN)
        assert _schema_names(default) == _schema_names(explicit)

    def test_subagent_recall_allowed(self):
        reg = _make_registry("recall", "read_file", "delegate")
        schemas = reg.get_tool_schemas(role=AgentRole.SUBAGENT)
        names = _schema_names(schemas)
        assert "recall" in names
        assert "delegate" not in names
