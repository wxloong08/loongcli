"""视觉输入端到端验收 —— 打真实 Qwen 多模态 API。

用真实调用一次性证实：CLI 附图 → 多模态消息 → 模型看到图并正确描述。
这是视觉骨架的验收（文档所述 base64 data URL 格式/字段名/图片数限制，靠真机跑通证实）。

用法：
    # 环境变量给 key（DASHSCOPE_API_KEY 或 QWEN_API_KEY）
    export DASHSCOPE_API_KEY=sk-...
    python tests/e2e_vision.py <图片路径> [提问文本] [模型名]

    # 例：
    python tests/e2e_vision.py ./mockup.png "这是什么界面？描述布局和主要元素"

默认模型 qwen3.7-plus（原生多模态旗舰）；也可传第 3 个参数换 qwen-vl-max 等。
"""
import asyncio
import os
import sys

from loongcli.core.llm import LLMClient
from loongcli.core.agent import AgentLoop
from loongcli.core.events import TextDelta, ThinkingDelta, ToolCallStart, ToolCallResult, AgentDone
from loongcli.tools.base import ToolRegistry
from loongcli.tools.read_file import ReadFileTool
from loongcli.tools.glob_tool import GlobTool
from loongcli.security.permissions import PermissionChecker, PermissionMode

_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "qwen3.7-plus"


async def main():
    if len(sys.argv) < 2:
        print("用法：python tests/e2e_vision.py <图片路径> [提问文本] [模型名]")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "这是什么界面？请描述你看到的内容。"
    model = sys.argv[3] if len(sys.argv) > 3 else _DEFAULT_MODEL

    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not api_key:
        print("ERROR: 未设置 DASHSCOPE_API_KEY / QWEN_API_KEY 环境变量")
        sys.exit(1)
    if not os.path.isfile(image_path):
        print(f"ERROR: 图片不存在：{image_path}")
        sys.exit(1)

    base_url = os.environ.get("QWEN_BASE_URL", _DEFAULT_BASE_URL)
    print(f"Model: {model}  Base: {base_url}")
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    print("=" * 60)

    # vision=True 放行图片输入；thinking=True 走 Qwen 思考模式（dashscope base_url 会被
    # 识别为 qwen 类型，思考经 extra_body.enable_thinking 控制，且强制随流式）。
    llm = LLMClient(api_key=api_key, model=model, base_url=base_url, vision=True, thinking=True)
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(GlobTool())
    checker = PermissionChecker(PermissionMode.SKIP)

    agent = AgentLoop(
        llm=llm,
        tool_registry=registry,
        permission_checker=checker,
        system_prompt="你是一个助手。用户会给你图片，请仔细看图后回答。",
    )

    saw_text = False
    try:
        async for event in agent.run_stream(prompt, images=[image_path]):
            if isinstance(event, ThinkingDelta):
                pass
            elif isinstance(event, TextDelta):
                saw_text = True
                print(event.text, end="", flush=True)
            elif isinstance(event, ToolCallStart):
                print(f"\n  [TOOL] {event.tool_name}({event.arguments})")
            elif isinstance(event, ToolCallResult):
                preview = event.result[:200] if len(event.result) > 200 else event.result
                print(f"  [RESULT] {preview}")
            elif isinstance(event, AgentDone):
                print()
        print("\n" + "=" * 60)
        if saw_text:
            print(">> PASS：模型返回了对图片的描述（请人工确认描述是否贴合图片内容）")
        else:
            print(">> FAIL：没有文本输出")
            sys.exit(1)
    except Exception as e:
        print(f"\n>> FAIL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
