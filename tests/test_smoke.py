def test_packages_importable():
    from loongcli.core.llm import LLMClient
    from loongcli.core.agent import AgentLoop
    from loongcli.tools.base import Tool, ToolRegistry
    from loongcli.tui.app import TUI
    from loongcli.memory.store import MemoryStore
    from loongcli.memory.conversation import ConversationStore
    from loongcli.tools.recall import RecallTool
    from loongcli.security.checker import SecurityChecker


def test_tool_registry_empty():
    from loongcli.tools.base import ToolRegistry
    registry = ToolRegistry()
    assert registry.get_tool_schemas() == []


def test_main_entry_exists():
    from loongcli.main import main
    assert callable(main)
