from __future__ import annotations

from loongcli.tools.base import Tool
from loongcli.skills.registry import SkillRegistry


SKILL_TOOL_PREFIX = """\
调用技能来获取特定任务的详细指令和工作流。
技能在被调用时才加载完整内容，平时只占用少量上下文。

可用技能：
"""

NO_SKILLS_DESC = "当前没有可用的技能。"


class SkillTool(Tool):
    name = "skill"
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "要调用的技能名称",
            },
            "args": {
                "type": "string",
                "description": "传给技能的可选参数",
                "default": "",
            },
        },
        "required": ["name"],
    }

    def __init__(self, registry: SkillRegistry):
        self.registry = registry
        self._update_description()

    def _update_description(self):
        listing = self.registry.build_listing(include_model_disabled=False)
        if listing:
            self.description = SKILL_TOOL_PREFIX + listing
        else:
            self.description = NO_SKILLS_DESC

    async def execute(self, name: str, args: str = "") -> str:
        meta = self.registry.get(name)
        if not meta:
            available = ", ".join(s.name for s in self.registry.list_skills())
            return f"未找到技能 '{name}'。可用技能: {available}" if available else f"未找到技能 '{name}'，当前没有可用技能。"

        content = self.registry.load_content(name)
        if not content:
            return f"无法加载技能 '{name}' 的内容。"

        header = f"## 技能: {meta.name}\n{meta.description}\n\n"
        instruction = "请按照以下指令执行：\n\n"
        return header + instruction + content
