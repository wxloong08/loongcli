from __future__ import annotations

from loongcli.tools.base import Tool
from loongcli.skills.registry import SkillRegistry


# 技能清单刻意不嵌进 description：工具 schema 属于请求前缀的静态段，技能一增删
# 就打断最该稳定的前缀缓存。清单由 prompts.py 注入系统提示的易变段（「可用技能」节），
# 这里只保留静态指引；description 仅随「有无技能」这个布尔态变化，远比清单稳定。
SKILL_TOOL_DESC = """\
调用技能来获取特定任务的详细指令和工作流。
用 skill(name) 加载完整内容，可用技能清单见系统提示的「可用技能」一节。"""

NO_SKILLS_DESC = "当前没有可用的技能。"

# 技能内容自管上限（入口 ToolResultManager 对本工具豁免——技能是指令，被一刀切截成
# 2000 预览等于按残缺工作流执行，且重调返回同样的截断）。上限取宽：现存最大技能
# 45.9K，50K 全放行；真超限时截断并指向源文件，绝不建议"重新调用"。
SKILL_CONTENT_CAP = 50000


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
        if self.registry.list_skills(include_model_disabled=False):
            self.description = SKILL_TOOL_DESC
        else:
            self.description = NO_SKILLS_DESC

    async def execute(self, name: str, args: str = "") -> str:
        meta = self.registry.get(name)
        if not meta and ":" in name:
            # Claude Code 插件惯例 namespace:skill（如 superpowers:writing-plans）——
            # 外部技能内容里的交叉引用带命名空间，模型会照抄调用，剥前缀重试
            meta = self.registry.get(name.rsplit(":", 1)[-1].strip())
        if not meta:
            available = ", ".join(s.name for s in self.registry.list_skills())
            return f"未找到技能 '{name}'。可用技能: {available}" if available else f"未找到技能 '{name}'，当前没有可用技能。"

        content = self.registry.load_content(meta.name)
        if not content:
            return f"无法加载技能 '{name}' 的内容。"

        if len(content) > SKILL_CONTENT_CAP:
            kept = content[:SKILL_CONTENT_CAP]
            nl = kept.rfind("\n")
            if nl > SKILL_CONTENT_CAP * 0.7:
                kept = kept[:nl]
            content = kept + (
                f"\n\n[技能原文超长已截断：保留 {len(kept)}/{len(content)} 字符。"
                f"剩余部分用 read_file 读取 {meta.path}（带 offset 分页），不要重新调用 skill 工具]"
            )

        header = f"## 技能: {meta.name}\n{meta.description}\n\n"
        instruction = "请按照以下指令执行：\n\n"
        return header + instruction + content
