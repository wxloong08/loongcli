from loongcli.core.events import TextDelta, ToolCallStart, ToolCallResult, AgentDone


def test_text_delta():
    e = TextDelta(text="hello")
    assert e.text == "hello"


def test_tool_call_start():
    e = ToolCallStart(tool_name="shell", arguments={"command": "ls"})
    assert e.tool_name == "shell"
    assert e.arguments == {"command": "ls"}


def test_tool_call_result():
    e = ToolCallResult(tool_name="shell", result="file.txt")
    assert e.tool_name == "shell"
    assert e.result == "file.txt"


def test_agent_done():
    e = AgentDone(content="任务完成")
    assert e.content == "任务完成"
