from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import os
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import AsyncIterator, Callable

from loongcli.core.llm import LLMClient
from loongcli.core.events import TextDelta, ToolCallStart, ToolCallResult, AgentDone, CompactStart, CompactNotice, TaskNotification, ConfirmRequest, BatchProgress, PlanApproval, QueuedInputInjected
from loongcli.core.stream_collector import StreamCollector
from loongcli.core.compact import Compactor, _estimate_tokens
from loongcli.core.messages import (
    message_text, has_images, build_user_content, recycle_old_images, parse_image_sentinel,
)
from loongcli.core.circuit_breaker import CompactCircuitBreaker
from loongcli.core.telemetry import EventLogger, NULL_TELEMETRY
from loongcli.core.tool_result_manager import ToolResultManager
from loongcli.core.resume_gap import date_rollover_messages
from loongcli.tools.base import ToolRegistry
from loongcli.tools.routing import AgentRole
from loongcli.security.permissions import PermissionChecker, Decision, PermissionMode
from loongcli.security.auto_review import referenced_scripts
from loongcli.memory.conversation import ConversationStore, project_tool_results_dir
from loongcli.hooks.manager import HookManager, HookEvent
from loongcli.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

# 整次 run_stream（一条用户消息的完整执行）的工具调用总额，不是迭代轮级——
# 曾名 PER_TURN，与当时结果截断器的迭代级 reset_turn() 撞名误导，2026-07-21 改名。
MAX_TOOL_CALLS_PER_RUN = 200
# 主代理的迭代兜底，与工具预算对齐。此前是 50：一轮迭代≈一次工具调用，50 轮永远先于
# 200 次工具预算触发，后者那条「收走工具 → 文字总结」的优雅收场通道形同虚设。真机
# dogfooding（2026-11，jobhunter 四个会话）首轮 4/4 撞 50，被截停 4 次之后用户的下一句
# 全是「继续」——100% 误杀。对齐后工具预算是先触发的成本闸，迭代上限退为纯兜底
# （只在「大量迭代不调工具」这种异常形态下才轮得到它）。子代理 30 / 协调者 80 不变。
DEFAULT_MAX_ITERATIONS = MAX_TOOL_CALLS_PER_RUN
LOOP_DETECT_THRESHOLD = 3
# 循环检测的最大周期：1=同参连打（AAA），2/3=交替模式（ABAB…/ABCABC…）。
# 仍是精确签名匹配——整周期原样重复满 THRESHOLD 遍才触发，合法的
# 编辑→测试交替每轮参数都在变，签名不同，不会误伤。
LOOP_DETECT_MAX_PERIOD = 3
_MIN_RECALL_LENGTH = 4

# 两条提示的措辞要点：归因必须落在「loongcli 的配置开关」上。旧措辞「当前模型未开启
# 视觉能力」被模型转述成「当前模型没有视觉能力」（2026-11-09 真机：deepseek-flash 明明
# 支持视觉，只是 roles.main.vision=false，模型却让用户手贴邮件正文）。门控是 harness
# 按配置做的，harness 并不知道模型支不支持——那就别替模型下结论。
NO_VISION_MSG = (
    "⚠ 图片未发送：loongcli 配置里当前角色的 vision 开关是关闭的（roles.<角色>.vision=false）。"
    "这只是配置开关，不代表模型不支持图片——若所用模型支持视觉，"
    "在 ~/.loongcli/config.json 把该角色的 vision 设为 true 后重启即可。"
)

# read_file 读到图片但模型无视觉时喂给模型的工具结果（明确失败，不静默注入垃圾）
NO_VISION_READ_MSG = (
    "该文件是图片，但没有发给你：loongcli 配置里当前角色的 vision 开关是关闭的"
    "（roles.<角色>.vision=false）。这是 harness 的配置开关，不代表模型不支持图片——"
    "向用户转述时请如实说「loongcli 的 vision 配置未开启」，不要说成模型没有视觉能力。"
    "开启方法：在 ~/.loongcli/config.json 把该角色的 vision 设为 true 后重启。"
)

# 图片 sentinel 被拦截后，tool 消息里的占位文本（图片走下一条 user 消息）
IMAGE_READ_PLACEHOLDER = "[图片已读取（{n} 张），内容见下一条消息]"

PLAN_MODE_TOOLS = frozenset({
    "read_file",
    "shell",  # 只读：执行层按 PermissionChecker.is_read_only_command 拦，见 _shell_read_only
    "recall", "search_history",  # 规划期查记忆/翻历史，通用任务的调研通道
    "plan", "exit_plan_mode",
})

# 写文件的工具。一个回合的允许工具集里一个都没有 = 只读回合，shell 也必须只读——
# 否则 shell 就是绕过「不给写工具」的后门（规划阶段在用户批准计划之前）。
WRITE_TOOLS = frozenset({"write_file", "edit_file"})

READ_ONLY_SHELL_REJECTION = (
    "⚠ 规划阶段 shell 只能跑只读命令：git status/log/diff/show/grep/ls-files、ls、cat、head、tail、grep、"
    "pip show/list、python --version 等，不能带重定向（>）或命令替换（$(...)）。这条命令可能会写东西，"
    "等计划批准后再执行；核实环境请换成上面的只读命令。"
)

PLAN_MODE_SYSTEM_INJECTION = """\

## 规划模式

你已进入规划模式，只能使用只读调研工具（read_file / recall / search_history，以及只能跑只读命令的
shell：git log/status/diff/grep/ls-files、ls、cat、grep、pip show、python --version 等）
和已配置的 MCP 工具（每个 MCP 工具首次调用需用户确认）。

工作流程：
1. **调研** — 收集规划所需的事实：代码任务读代码；其他任务查记忆（recall）、
   用 MCP 工具（如搜索）查外部资料、翻已有材料。
   调研深度随任务：改一处已知问题，一两次调用就够；要做设计决定的任务（选技术路线、定架构、划范围），
   要把需求、约束、现有代码和环境查到能为每个决定说出理由为止。
   数据收集、批量统计是执行期的活（审批后一条 shell 解决），不要在规划期硬凑
   （反例：为统计行数逐文件 read——这是把执行搬进调研）
2. **规划** — 用 plan 工具创建计划。approach（方案）写四块：背景与范围（做什么、不做什么）；
   关键决定（每条写选了什么、放弃了什么、为什么）；风险与先验证什么；需要用户拍板的问题（单列，带你的推荐）。
   每一步写 description（改动内容，编码任务具体到文件/函数）和 1-5 条 criteria
   （怎么算这步做完——跑什么/看到什么算过；执行期标完成时每条都要交证据）。取舍和理由写在 approach，步骤只写怎么执行
3. **提交** — 调用 exit_plan_mode 提交计划等待用户审批。界面会把完整计划展示给用户，不要再复述步骤；
   只说需要用户拍板或特别注意的地方，没有就不说

原则：
- 不要猜测，先看事实再规划
- 规划期 shell 只做只读查询：不带重定向（>）和命令替换（$(...)），不跑脚本或 python -c；
  运行时行为要实测的，写进计划的步骤，批准后再跑
- 「运行脚本」「验证正确性」这类没有判据的空步骤不合格——判据是防执行期跑偏的锚
- 判据查内容不查格式：章节数、行数、表格行数不能当判据——格式全对、内容全错也能过。
  产出是文档或设计时，判据写「文档里写明了哪些决定、覆盖了哪些风险、回答了哪些问题」
- 步骤是给执行期的自己看的路标：宁可 5 步具体，不要 2 步空话；控制在 10 步以内
- 领域性任务（求职、调研报告等）若已加载对应 skill，按该 skill 的章法规划
- 有不确定的地方直接问用户\
"""

# 刻意不含计划全文——这段拼在 messages[0] 末尾，而 _rebuild_system_prompt 在计划
# 执行期每轮触发。塞进随进度变化的 format_summary()，就等于每完成一步改写一次
# messages[0]，前缀缓存连同其后的全部对话历史一起作废。计划全貌改由 plan update_step
# 的返回值携带（工具结果天然落在对话尾部、走缓存、零额外代价），compact 后则由
# build_attachments 重新挂载。这里只留「有计划在跑 + 去哪儿看」这两条恒定信息。
ACTIVE_PLAN_INJECTION_TEMPLATE = """\

## 活跃计划

计划 {plan_id} 执行中。按步骤执行；每完成一步用 plan update_step 更新状态，
evidence 里逐条判据交证据（返回值会带上最新的完整计划）；完成所有步骤后用 plan complete 关闭计划，
收尾最后给 1-2 条具体的下一步建议（能直接拿去做的，不是核对表）。
需要当前全貌时用 plan get 查询。\
"""

# /goal 执行期常驻系统提示：目标原话 + 完成审计规则。放系统提示而不是 compact 后重挂——系统消息不参与压缩，
# 目标始终在场。Codex issue #19910 的失败模式正是 compact 摘要丢了「全局目标 + 完成审计要求」，
# 新上下文把局部验证当成整体完成、过早收尾。整个目标期间内容恒定（不含进度），只在目标开始/结束时各改一次。
GOAL_INJECTION_TEMPLATE = """\

## 目标模式

当前目标（用户原话）：
{goal}

计划 {plan_id} 是这个目标的执行计划。规则：
- 每完成一步用 plan update_step 标 completed，evidence 里逐条判据交证据；缺证据会被拒绝
- 局部验证通过不等于目标完成：plan complete 前，对照上面的目标原话和全部判据审计一遍
  （审计结论交在 evidence 里，不要把审计过程写给用户）
- 目标完成后给用户的收尾只说三件事：做了什么、产出在哪、需要用户注意或决定什么，几行即可；
  证据都在计划里，用户没要的话不再列核对表、自查清单、验收清单。
  最后给 1-2 条具体的下一步建议（例：「拿这份架构稿写实施计划」「先确认这 3 个决定」）——能直接拿去做的，不是核对表
- 做不了的步骤标 skipped 并写理由；需要用户操作或决策时说清楚要什么，然后停下等回复
- 需要计划全貌用 plan get\
"""


@dataclass
class AgentServices:
    """AgentLoop 的可选服务依赖集合——把散落在构造函数里的 9 个 service 收拢成一个包，
    构造点只传 services= 一项（此前 18 参构造函数的主要来源）。全部可选（默认 None），
    主装配注入全部、子代理/测试按需注入。内部仍用 self.<name> 引用，不改调用点。"""
    conversation_store: "ConversationStore | None" = None
    compactor: "Compactor | None" = None
    task_manager: object | None = None
    hook_manager: "HookManager | None" = None
    skill_registry: "SkillRegistry | None" = None
    system_prompt_builder: "Callable[[], str] | None" = None
    recall_engine: object | None = None
    auto_extractor: object | None = None
    checkpoint_manager: object | None = None
    telemetry: "EventLogger | None" = None


class AgentLoop:
    def __init__(
        self,
        llm: LLMClient,
        tool_registry: ToolRegistry,
        permission_checker: PermissionChecker,
        system_prompt: str = "",
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_tool_calls: int = MAX_TOOL_CALLS_PER_RUN,
        task=None,
        role: AgentRole = AgentRole.MAIN,
        interactive: bool = True,
        services: AgentServices | None = None,
    ):
        services = services or AgentServices()
        self.llm = llm
        self.tool_registry = tool_registry
        self.permission_checker = permission_checker
        self.role = role
        # 无人工交互通道（子代理 / 后台任务）：CONFIRM 无人应答，故直接当 DENY 处理，
        # 而不是 yield 一个无人消费的 ConfirmRequest（那会 await future 永久挂起）。见 run_stream。
        self.interactive = interactive
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.task = task
        # 服务依赖从 services 包解出，内部引用保持 self.<name> 不变
        self.conversation_store = services.conversation_store
        self.compactor = services.compactor
        self.task_manager = services.task_manager
        self.hook_manager = services.hook_manager
        self.skill_registry = services.skill_registry
        self._system_prompt_builder = services.system_prompt_builder
        self.recall_engine = services.recall_engine
        self.auto_extractor = services.auto_extractor
        self.checkpoint_manager = services.checkpoint_manager
        # 空对象兜底：埋点处直接 emit，无需 None 检查
        self.telemetry = services.telemetry or NULL_TELEMETRY
        self.cost_tracker = None
        self.plan_store = None
        self.lsp_manager = None
        self._plan_mode: bool = False
        self._active_plan_id: str | None = None
        self._goal_description: str = ""  # /goal 执行期的目标原话（常驻系统提示），见 set_goal
        self._last_checkpoint: str | None = None
        self._files_modified_this_turn: list[str] = []
        # 读后写硬闸的状态表：normcase(resolve(path)) → (mtime_ns, size)。
        # 覆盖已存在的文件前必须本会话读过且读后未被外部改动——防「凭（compact 后
        # 退化的）记忆整篇重写」这一事故根因。每个 AgentLoop 实例独立，子代理同样约束。
        self._file_read_state: dict[str, tuple[int, int]] = {}
        self._test_ran_this_turn: bool = False
        # 本回合是否已发过验证提醒：一回合最多一次，提醒之后模型再改代码也不再追问（只提醒、不重试）
        self._verify_nudged: bool = False
        self._turn_tools: list[dict] | None = None
        self._last_prompt_tokens = 0
        self.token_usage = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0,
            "reasoning_tokens": 0,
        }
        self._tool_call_count = 0
        self._recent_tool_sigs: deque[str] = deque(
            maxlen=LOOP_DETECT_THRESHOLD * LOOP_DETECT_MAX_PERIOD
        )
        # read_file 读到的图片先攒着，本轮全部 tool 结果落库后再以 user 消息注入
        self._pending_images: list[dict] = []
        # 自动记忆写入通知（写入可见化：落库即曝光，人眼是第五道闸）
        self.memory_notices: list[dict] = []
        # 本轮允许调用的工具名集合；None = 无限制（派发层硬闸用，区别于 _turn_tools 的 API 参数语义）
        self._turn_tool_names: set[str] | None = None
        self._turn_read_only: bool = False  # 本回合工具集不含写工具（/goal 规划），shell 只读，见 _shell_read_only
        # 撞单轮工具上限后置位：收走工具只留文字总结；模型仍硬发调用则计振出局
        self._tools_exhausted: bool = False
        self._exhausted_strikes: int = 0
        # 任何工具结果超过 HARD_CAP 时全量落盘、上下文留头尾预览 + 指针；主/子代理各自
        # manager 共用项目级目录（uuid 文件名无碰撞）。懒建目录，构造期无副作用。
        self._result_manager = ToolResultManager(spill_dir=project_tool_results_dir())
        self._compact_breaker = CompactCircuitBreaker()
        # 执行中排队的用户插话：TUI key listener 在轮次运行时 queue_user_input()，
        # run_stream 在迭代边界 drain 注入。单事件循环内同步操作，无需锁。
        self._queued_user_inputs: list[str] = []
        # 上一回合开始时的本地日期：会话开过午夜时据此插「日期变更」标记（见 resume_gap）。
        # None = 本进程还没跑过回合，系统提示里的日期是新鲜的，不需要标记。
        self._last_turn_date = None
        self.messages: list[dict] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def _persist(self):
        if self.conversation_store:
            self.conversation_store.save(self.messages)

    def _persist_checkpoint(self):
        """回合内的尽力落盘：失败只记日志，不打断正在进行的回合。

        回合收尾的 _persist 失败会如实抛出（那是最后一次机会）；中途这次只是保险，
        磁盘暂时写不进去不该让一个本来能跑完的回合夭折。"""
        try:
            self._persist()
        except Exception:
            logger.warning("回合内落盘失败（不影响本回合继续）", exc_info=True)

    def _effective_prompt_tokens(self) -> int:
        """自动压缩的触发依据。优先用 API 报的真实 prompt_tokens；provider 不
        回报 usage 时（部分本地 / 代理端点）它恒为 0，回退到基于消息的字符估算——否则
        自动压缩会因"token 数未知"永不触发，上下文无界增长直到 API 400。
        故意不改 should_compact 的"0=无信息"纯语义，只在此处收口。"""
        if self._last_prompt_tokens > 0:
            return self._last_prompt_tokens
        return _estimate_tokens(self.messages)

    def abort_turn(self):
        """用户中断回合（Ctrl+C）后的消息一致性修复。

        取消打在任意 await 点，可能停在「assistant 带 tool_calls 已入列、
        部分工具结果未回」的中间态——不补占位的话，下一次请求会被 API
        以孤儿 tool_call 拒绝（400），会话就废了。修复后落盘。"""
        last_assistant = None
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].get("role") == "assistant":
                last_assistant = i
                break
        if last_assistant is not None:
            answered = {
                m.get("tool_call_id")
                for m in self.messages[last_assistant + 1:]
                if m.get("role") == "tool"
            }
            for tc in self.messages[last_assistant].get("tool_calls") or []:
                if tc.get("id") not in answered:
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "content": "[用户中断，该调用未执行完成]",
                    })
        self.messages.append({"role": "user", "content": "[用户中断了本回合]"})
        self._persist()

    def queue_user_input(self, text: str) -> None:
        """轮次执行中排队一条用户消息，下一迭代边界注入。只应由 TUI 调用——
        非交互模式与子代理不触碰此接口，行为与排队功能出现前逐字节一致。"""
        self._queued_user_inputs.append(text)

    def take_queued_inputs(self) -> list[str]:
        """取走并清空排队消息（轮内 drain 与轮末收尾共用）。"""
        taken = self._queued_user_inputs
        self._queued_user_inputs = []
        return taken

    def enter_plan_mode(self):
        self._plan_mode = True

    def exit_plan_mode(self, plan_id: str | None = None):
        self._plan_mode = False
        self._active_plan_id = plan_id

    @property
    def plan_mode(self) -> bool:
        return self._plan_mode

    @property
    def turn_tool_calls(self) -> int:
        """最近一个回合实际执行的工具调用数（回合开始清零）。/goal 用它判断这轮有没有进展。"""
        return self._tool_call_count

    def set_goal(self, description: str, plan_id: str) -> None:
        """进入 /goal 执行：目标原话与审计规则常驻系统提示，计划设为活跃（草稿覆盖保护也认它）。"""
        self._goal_description = description
        self._active_plan_id = plan_id
        self._rebuild_system_prompt()

    def clear_goal(self) -> None:
        self._goal_description = ""
        self._active_plan_id = None
        self._rebuild_system_prompt()

    def _review_context(self) -> tuple[str, str]:
        """自动审核用的上下文：(目标, 计划当前步骤)。审核器据此判断命令是否超出当前步骤范围。

        目标优先取 /goal 描述；没有（普通对话、子代理）就取第一条用户消息——子代理的第一条就是委派给它的任务。"""
        goal = self._goal_description or ""
        if not goal:
            first = next((m for m in self.messages if m.get("role") == "user"), None)
            if first:
                goal = str(first.get("content") or "")[:500]
        step = ""
        if self._active_plan_id and self.plan_store:
            plan = self.plan_store.load(self._active_plan_id)
            if plan:
                cur = next((s for s in plan.steps if s.status in ("in_progress", "pending")), None)
                if cur:
                    step = f"{cur.number}. {cur.description}"[:500]
        return goal, step

    def _build_plan_injection(self) -> str:
        if self._plan_mode:
            return PLAN_MODE_SYSTEM_INJECTION
        if self._goal_description and self._active_plan_id:
            return GOAL_INJECTION_TEMPLATE.format(goal=self._goal_description, plan_id=self._active_plan_id)
        if self._active_plan_id and self.plan_store:
            plan = self.plan_store.load(self._active_plan_id)
            if plan and plan.status == "active":
                return ACTIVE_PLAN_INJECTION_TEMPLATE.format(
                    plan_id=self._active_plan_id,
                )
        return ""

    def _rebuild_system_prompt(self):
        if not self._system_prompt_builder:
            return
        if not self.messages or self.messages[0].get("role") != "system":
            return
        base = self._system_prompt_builder()
        plan_injection = self._build_plan_injection()
        self.messages[0] = {"role": "system", "content": base + plan_injection}

    @staticmethod
    def _tool_signature(name: str, args: dict) -> str:
        raw = json.dumps({"n": name, "a": args}, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    @staticmethod
    def _extract_file_args(name: str, args: dict) -> list[str]:
        """Extract file paths from tool arguments for checkpointing."""
        files = []
        file_keys = ("file_path", "path", "target_file")
        for key in file_keys:
            if key in args and isinstance(args[key], str):
                files.append(args[key])
        return files

    @staticmethod
    def _file_state_key(path: str) -> str | None:
        """路径归一化为状态表 key。Windows 路径大小写不敏感，normcase 防同文件两个 key。"""
        try:
            return os.path.normcase(str(Path(path).resolve()))
        except (OSError, ValueError):
            return None

    def _record_file_state(self, path: str) -> None:
        key = self._file_state_key(path)
        if not key:
            return
        try:
            st = os.stat(key)
            self._file_read_state[key] = (st.st_mtime_ns, st.st_size)
        except OSError:
            pass

    def _check_write_gate(self, tool_name: str, args: dict) -> str | None:
        """读后写硬闸：修改已存在的文件前，要求本会话读过且读后未被外部改动。

        新文件放行；未读过或已被外部修改则拒绝执行并要求先 read_file——
        确定性代码闸门，不靠 LLM 自觉（prompt 规则在 compact 后会退化）。"""
        from loongcli.core.checkpoint import MODIFY_TOOLS
        if tool_name not in MODIFY_TOOLS:
            return None
        for f in self._extract_file_args(tool_name, args):
            key = self._file_state_key(f)
            if not key or not os.path.isfile(key):
                continue
            recorded = self._file_read_state.get(key)
            if recorded is None:
                return (
                    f"错误：文件已存在且本会话未读取过：{f}\n"
                    f"先用 read_file 查看当前内容，再决定用 edit_file 修改或确认覆盖。"
                )
            try:
                st = os.stat(key)
            except OSError:
                continue
            if (st.st_mtime_ns, st.st_size) != recorded:
                return (
                    f"错误：文件在读取后被外部修改：{f}\n"
                    f"请重新 read_file 获取当前内容后再修改。"
                )
        return None

    def _maybe_checkpoint(self, tool_name: str, args: dict) -> None:
        """Save a checkpoint before file-modifying tool calls."""
        if not self.checkpoint_manager:
            return
        from loongcli.core.checkpoint import MODIFY_TOOLS
        if tool_name not in MODIFY_TOOLS:
            return
        try:
            files = self._extract_file_args(tool_name, args)
            self._last_checkpoint = self.checkpoint_manager.save(files)
            if self._last_checkpoint:
                self._files_modified_this_turn.extend(files)
        except Exception:
            logger.debug("checkpoint save failed", exc_info=True)

    def _release_checkpoint(self) -> None:
        """成功路径：只解除当前引用，快照本身留着，供 /rollback 会话内随时恢复。

        不在成功时 discard，是因为要防的不是「执行中抛异常」（那条路径由
        _restore_checkpoint 兜），而是「几轮之后用户才发现内容被改坏」。留存量由
        CheckpointManager 封顶（MAX_CHECKPOINTS 淘汰最旧），不会无界增长。
        注意快照索引只在内存里、不跨会话——进程退出后旧快照即不可恢复。"""
        self._last_checkpoint = None

    def _restore_checkpoint(self) -> None:
        """修改类工具抛异常后回滚到最近一次 checkpoint。

        恢复修改前的文件内容，使执行中途抛异常（磁盘满、编码崩溃等）
        不会留下半写的文件。
        """
        if self._last_checkpoint and self.checkpoint_manager:
            try:
                self.checkpoint_manager.restore(self._last_checkpoint)
            except Exception:
                logger.debug("checkpoint restore failed", exc_info=True)
            self._last_checkpoint = None

    def _check_loop(self, name: str, args: dict) -> str | None:
        """精确签名循环检测，周期 1~LOOP_DETECT_MAX_PERIOD。

        旧实现只比较相邻调用（周期 1），A-B-A-B 交替每次都与上一次不同、
        计数永远归零，抓不到。改为签名窗口：末尾 period×THRESHOLD 个签名
        恰为同一 period 元组原样平铺时触发。升序扫周期，纯 AAA 由周期 1
        先报，不会重复归因到更大周期。"""
        sig = self._tool_signature(name, args)
        self._recent_tool_sigs.append(sig)
        sigs = list(self._recent_tool_sigs)
        for period in range(1, LOOP_DETECT_MAX_PERIOD + 1):
            need = period * LOOP_DETECT_THRESHOLD
            if len(sigs) < need:
                break  # need 随 period 递增，更大的周期更不够
            window = sigs[-need:]
            cycle = window[:period]
            if any(window[i] != cycle[i % period] for i in range(need)):
                continue
            if period == 1:
                run = 0
                for s in reversed(sigs):
                    if s != sig:
                        break
                    run += 1
                return (
                    f"⚠ 检测到循环：{name} 已用相同参数连续调用 {run} 次。"
                    "请换一种思路或向用户说明无法完成。"
                )
            return (
                f"⚠ 检测到循环：最近 {need} 次工具调用构成周期 {period} 的交替重复"
                f"（同一组调用原样重复 {LOOP_DETECT_THRESHOLD} 遍，没有新信息进入）。"
                "请换一种思路或向用户说明无法完成。"
            )
        return None

    async def _exec_tool_stream(self, tool_name: str, args: dict):
        gate_err = self._check_write_gate(tool_name, args)
        if gate_err:
            yield ("__result__", gate_err)
            return
        self._maybe_checkpoint(tool_name, args)
        tool = self.tool_registry._tools.get(tool_name)
        if not tool or not getattr(tool, 'supports_progress', False):
            async for kind, data in self._exec_with_retry(tool_name, args):
                yield (kind, data)
            return

        queue = asyncio.Queue()
        tool._progress_callback = lambda evt: queue.put_nowait(evt)

        exec_task = asyncio.create_task(
            self._exec_with_retry_single(tool_name, args)
        )
        try:
            while not exec_task.done():
                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=0.3)
                    yield ("__progress__", evt)
                except asyncio.TimeoutError:
                    pass

            while not queue.empty():
                yield ("__progress__", queue.get_nowait())

            if exec_task.exception():
                raise exec_task.exception()
            yield ("__result__", exec_task.result())
        except Exception as e:
            logger.warning("Tool %s failed: %s", tool_name, e)
            yield ("__error__", f"⚠ 工具执行失败: {e}")
        finally:
            # 本协程被取消（stop_task）时 CancelledError 打在上面的 queue.get()，
            # 而 exec_task 是独立 asyncio 任务，不随消费者消亡——不收走的话，
            # 工具协程连同 shell 子进程一起变孤儿。取消后等它跑完自己的清理
            # （shell 的取消路径要 terminate→kill 子进程）再继续传播。
            if not exec_task.done():
                exec_task.cancel()
                try:
                    await exec_task
                except (asyncio.CancelledError, Exception):
                    pass
            tool._progress_callback = None

    async def _exec_with_retry_single(self, tool_name: str, args: dict):
        """Execute a tool once, used by progress-supporting tools (retry not
        supported for streaming tools — too complex to re-stream)."""
        return await self.tool_registry.execute_by_name(tool_name, args)

    async def _exec_with_retry(self, tool_name: str, args: dict):
        """Execute tool with one automatic retry on ToolError."""
        from loongcli.tools.errors import ToolError

        last_error = None
        for attempt in range(2):
            try:
                result = await self.tool_registry.execute_by_name(tool_name, args)
                yield ("__result__", result)
                return
            except ToolError as e:
                last_error = e
                if e.retryable and attempt == 0:
                    await asyncio.sleep(e.retry_after)
                    continue
                yield ("__error__", f"⚠ 工具执行失败: {e.message}")
                return
            except Exception as e:
                logger.warning("Tool %s failed: %s", tool_name, e)
                yield ("__error__", f"⚠ 工具执行失败: {e}")
                return

    def _detect_active_skill(self) -> str | None:
        for msg in reversed(self.messages):
            # content 可能是多模态 list，经 message_text 取纯文本再做正则匹配
            content = message_text(msg)
            match = re.search(r"\[自动加载 skill:\s*(\S+)\]", content)
            if match:
                return match.group(1)
            match = re.search(r"## 技能:\s*(\S+)", content)
            if match:
                return match.group(1)
        return None

    async def _dispatch_tool_call(self, tc: dict) -> AsyncIterator:
        """处理单次工具调用：解析参数 → 各类短路检查 → 权限 → 执行 → 结果落库。

        作为异步生成器，把事件（ToolCallStart/Result、ConfirmRequest、PlanApproval、
        执行进度）透传给 run_stream 再上抛给调用方——ConfirmRequest/PlanApproval 的
        future 由调用方在请求下一个事件前 resolve，委托对 await 时序透明。plan 审批
        对工具集的调整写回 self._turn_tools。从 run_stream 抽出以缩小主循环、隔离这段
        安全攸关的逻辑，便于独立阅读与测试。
        """
        tool_name = tc["function"]["name"]
        # 模型以自由文本流式拼接工具参数；流被截断或调用是幻觉时会产出非法 JSON。
        # 绝不能让它逃出循环（会击穿整个 TUI）——转成一条工具错误喂回模型，
        # 让它下一轮自行改正。
        raw_args = tc["function"].get("arguments") or ""
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except (json.JSONDecodeError, TypeError) as e:
            result = (
                f"⚠ 工具参数不是合法 JSON，无法解析：{e}。"
                "请检查参数格式后重新调用。"
            )
            yield ToolCallResult(tool_name=tool_name, result=result)
            self.messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })
            return
        self._tool_call_count += 1
        exec_t0 = time.monotonic()
        yield ToolCallStart(tool_name=tool_name, arguments=args)

        # 派发层硬闸：schema 过滤只是"告知"，有的模型（真机：qwen 在规划模式下
        # 直接发 shell）会无视工具列表硬调。凡不在本轮允许集里的调用一律拒绝——
        # 闸门必须确定性，不赌模型自觉。
        if self._turn_tool_names is not None:
            if tool_name not in self._turn_tool_names:
                if self._plan_mode:
                    result = (
                        f"⚠ 规划模式下工具 {tool_name} 不可用，只能用只读调研工具"
                        "（read_file/recall/search_history，shell 只能跑只读命令）。请完成调研后用"
                        " plan 工具制定计划，再调用 exit_plan_mode 提交用户审批——审批通过后所有工具自动恢复。"
                    )
                elif self._tools_exhausted:
                    result = "⚠ 已达本次任务工具调用上限，工具已收走，请直接文字总结当前进展。"
                else:
                    result = f"⚠ 工具 {tool_name} 不在本轮允许的工具列表中。"
                yield ToolCallResult(tool_name=tool_name, result=result)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
                return

        # 只读回合的 shell 闸：确定性白名单判定，不赌模型自觉，也不用小模型猜（误放行 = 批准前就执行了写操作）
        if tool_name == "shell" and self._shell_read_only() and not self.permission_checker.is_read_only_command(
            args.get("command", "")
        ):
            yield ToolCallResult(tool_name=tool_name, result=READ_ONLY_SHELL_REJECTION)
            self.messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": READ_ONLY_SHELL_REJECTION,
            })
            return

        if self._tool_call_count > self.max_tool_calls:
            # 标记撞限：run_stream 下一迭代会收走全部工具（不带 tools 参数），只留
            # 文字总结一条路。此前的实现是"无限拒绝不终止"——模型每次重试都膨胀一轮
            # 上下文（真机：单轮烧掉 97 万 prompt tokens），且连 plan/exit_plan_mode
            # 的收场通道也一起拒，模型只能挣扎到 max_iterations 耗尽。
            self._tools_exhausted = True
            result = (
                f"⚠ 已达到本次任务工具调用上限（{self.max_tool_calls}次），"
                "请总结当前进展并回复用户。"
            )
            yield ToolCallResult(tool_name=tool_name, result=result)
            self.messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })
            return

        loop_warning = self._check_loop(tool_name, args)
        if loop_warning:
            yield ToolCallResult(tool_name=tool_name, result=loop_warning)
            self.messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": loop_warning,
            })
            return

        if self.hook_manager:
            pre = await self.hook_manager.run(
                HookEvent.PRE_TOOL_USE,
                {"tool": tool_name, "arguments": args},
                tool_name=tool_name,
            )
            if pre.blocked:
                result = f"⚠ Hook 拦截: {pre.reason}"
                yield ToolCallResult(tool_name=tool_name, result=result)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
                return

        # 外部 MCP 工具能做任意 I/O，且是提示注入的主要目标——必须走确认层。
        # 由工具实例的 is_mcp 标记判定，而非按名字猜（MCP 工具名形如 server__tool）。
        tool_obj = self.tool_registry._tools.get(tool_name)
        is_mcp_tool = bool(getattr(tool_obj, "is_mcp", False))
        decision, reason = self.permission_checker.check_tool(
            tool_name, args, is_mcp=is_mcp_tool,
        )
        tool_errored = False

        reviewer = getattr(self.permission_checker, "auto_reviewer", None)
        if (decision == Decision.CONFIRM and reviewer is not None
                and self.permission_checker.mode == PermissionMode.AUTO):
            # AUTO 档：本该弹确认的调用交给审核器（主代理与子代理同一路径）。放行 → 按 ALLOW 往下走；
            # 拦下（含审核失败）→ 不执行，理由回给模型，让它停下来问人
            goal, step = self._review_context()
            review_t0 = time.monotonic()
            review = await reviewer.review(
                tool_name, args, goal=goal, step=step, cwd=os.getcwd(),
                files=referenced_scripts(tool_name, args, self.permission_checker.project_root),
            )
            self.telemetry.emit(
                "auto_review", role=self.role.value, tool_call_id=tc.get("id", ""), name=tool_name,
                verdict="allow" if review.allowed else "block", category=review.category,
                cached=review.cached, failed=review.failed,
                duration_ms=int((time.monotonic() - review_t0) * 1000),
            )
            if review.allowed:
                decision = Decision.ALLOW
            else:
                decision = Decision.DENY
                reason = (f"自动审核拦下（{review.category}）：{review.reason}。"
                          "需要这么做就停下来向用户说明要做什么、为什么，由用户决定")

        if decision == Decision.DENY:
            result = f"⚠ 操作被禁止: {reason}"
        elif decision == Decision.CONFIRM and not self.interactive:
            result = f"⚠ 需用户确认的操作在子代理/非交互上下文中自动拒绝（{reason}）"
        elif decision == Decision.CONFIRM:
            # 结果：True = 批准；False = 拒绝；字符串 = 拒绝并附用户给模型的说明
            future: asyncio.Future[bool | str] = asyncio.get_running_loop().create_future()
            yield ConfirmRequest(
                tool_name=tool_name,
                arguments=args,
                risk_reason=reason,
                future=future,
            )
            answer = await future
            approved = answer is True
            if approved:
                self.permission_checker.record_approval(tool_name, args, is_mcp=is_mcp_tool)
                result = None
                async for kind, data in self._exec_tool_stream(tool_name, args):
                    if kind == "__progress__":
                        yield data
                    else:
                        result = data
                        tool_errored = kind == "__error__"
            else:
                result = f"⚠ 用户拒绝了此操作（{reason}）"
                if isinstance(answer, str) and answer.strip():
                    result += f"。用户说明：{answer.strip()}"
        else:
            result = None
            async for kind, data in self._exec_tool_stream(tool_name, args):
                if kind == "__progress__":
                    yield data
                else:
                    result = data
                    tool_errored = kind == "__error__"

        if self.hook_manager:
            await self.hook_manager.run(
                HookEvent.POST_TOOL_USE,
                {"tool": tool_name, "arguments": args, "result": result},
                tool_name=tool_name,
            )

        if self.lsp_manager and tool_name in ("edit_file", "write_file"):
            for fp in self._extract_file_args(tool_name, args):
                await self.lsp_manager.invalidate_doc(fp)

        if tool_name in ("edit_file", "write_file"):
            from loongcli.core.verify_loop import filter_code_files
            # 只有改了代码才作废「已测过」：测完把证据写进 todo.md/README 不该再招来一次验证提醒
            # （dogfood exec-2/3：测试通过 → 写 todo.md → 收尾，旧逻辑因此多提醒一轮）
            if filter_code_files(self._extract_file_args(tool_name, args)):
                self._test_ran_this_turn = False

        # 读后写闸门状态更新：读过的文件记录 (mtime_ns, size)；写/改成功视同已知
        # 最新内容，后续连续修改无需重读。工具级错误（"错误："）与拒绝（"⚠"）不记录。
        if isinstance(result, str) and not tool_errored:
            if tool_name == "read_file" and not result.startswith("错误"):
                self._record_file_state(args.get("path", ""))
            elif tool_name in ("write_file", "edit_file") and result.startswith("成功"):
                self._record_file_state(args.get("path", ""))

        if tool_name == "shell" and isinstance(result, str):
            from loongcli.core.verify_loop import detect_test_failure, is_test_command
            # 最后一次改动之后跑过「通过的」测试，收尾时就不用提醒验证；改文件会把它清回 False
            if is_test_command(args.get("command", "")) and not detect_test_failure(result):
                self._test_ran_this_turn = True

        approval_data = None
        if (tool_name == "exit_plan_mode"
                and isinstance(result, str)
                and result.startswith('{"__plan_approval__":')):
            try:
                parsed = json.loads(result)
                # 两个 key 都必需——载荷畸形时落到「按普通工具结果处理」，不崩溃。
                approval_data = (parsed["plan_id"], parsed["plan_summary"])
            except (json.JSONDecodeError, KeyError, TypeError):
                approval_data = None

        if approval_data is not None:
            plan_id, plan_summary = approval_data
            approval_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
            yield PlanApproval(
                plan_id=plan_id,
                plan_summary=plan_summary,
                future=approval_future,
            )
            user_response = await approval_future
            if user_response in ("approve", "approve_auto_edits"):
                self.exit_plan_mode(plan_id=plan_id)
                if user_response == "approve_auto_edits":
                    # 安全语义收在 agent 层：TUI 只回传选择字符串
                    self.permission_checker.auto_accept_edits = True
                    result = (
                        "计划已批准（本会话文件编辑自动接受，敏感路径除外）。"
                        f"现在按计划执行，所有工具已恢复。\n\n{plan_summary}"
                    )
                else:
                    result = f"计划已批准。现在按计划执行，所有工具已恢复。\n\n{plan_summary}"
                self._rebuild_system_prompt()
                self._turn_tools = self.tool_registry.get_tool_schemas(role=self.role)
                self._turn_tool_names = {s["function"]["name"] for s in self._turn_tools}
            elif user_response == "cancel":
                self._plan_mode = False
                result = "用户取消了计划。已退出规划模式。"
                self._rebuild_system_prompt()
                self._turn_tools = self.tool_registry.get_tool_schemas(role=self.role)
                self._turn_tool_names = {s["function"]["name"] for s in self._turn_tools}
            else:
                result = (
                    f"用户要求修改计划：{user_response}\n"
                    "请根据反馈调整计划，然后重新调用 exit_plan_mode 提交。"
                    "注意：计划再次获批之前你仍处于规划模式，只能使用只读工具，不要开始执行。"
                )

        # 修改类工具执行中抛异常时留有备份——回滚该文件，避免半写损坏；
        # 成功则保留快照供 /rollback（会话内随时可恢复）。
        if tool_errored:
            self._restore_checkpoint()
        else:
            self._release_checkpoint()

        # read_file 读到图片：结果是 sentinel。tool 消息只放占位文本，图片攒进
        # _pending_images、由 run_stream 在本轮全部 tool 结果之后以 user 消息注入
        # ——不能插在 tool 结果中间（破坏 assistant(tool_calls)→tool×N 序列），
        # 也不放 tool 消息里（provider 对 tool-role 图片支持不一，设计 §4）。
        # 模型无视觉时给明确失败文本，不静默注入。
        sentinel = parse_image_sentinel(result) if isinstance(result, str) else None
        if sentinel:
            images, extra_text = sentinel
            if getattr(self.llm, "vision", False):
                placeholder = IMAGE_READ_PLACEHOLDER.format(n=len(images))
                # MCP 工具可能图文混排（如截图 + 页面说明），文本部分留在 tool 消息里
                result = f"{extra_text}\n{placeholder}" if extra_text else placeholder
                source = args.get("path") or ""
                label = f"[工具 {tool_name} 返回的图片{'：' + source if source else ''}]"
                self._pending_images.append({"role": "user", "content": [
                    {"type": "text", "text": label},
                    *({"type": "image_url", "image_url": {"url": ref}} for ref, _ in images),
                ]})
            else:
                result = f"{extra_text}\n{NO_VISION_READ_MSG}" if extra_text else NO_VISION_READ_MSG

        yield ToolCallResult(tool_name=tool_name, result=result)

        stored = (
            self._result_manager.process(tool_name, result)
            if isinstance(result, str) else result
        )
        self.messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": stored,
        })
        # 截断判定用长度对比（只有超过 HARD_CAP 落盘时才会变短）。
        # 早退路径（非法 JSON/硬闸/上限/循环/hook 拦截）不经此处，v1 不覆盖。
        # tool_call_id：事件和会话里的工具调用按 id 对上，不靠顺序——早退路径不发本事件、上下文压缩又会删掉前段消息，
        # 按顺序对齐会错位（dogfood 补人工判断时，压缩过的 p4-2 算不出时间线）
        self.telemetry.emit(
            "tool_exec", name=tool_name, role=self.role.value, tool_call_id=tc.get("id", ""),
            duration_ms=int((time.monotonic() - exec_t0) * 1000),
            result_chars=len(result) if isinstance(result, str) else 0,
            truncated=isinstance(result, str) and isinstance(stored, str) and len(stored) < len(result),
            error=tool_errored,
        )

    async def _inject_recall(self, user_input: str) -> None:
        """记忆语义召回并注入。注入到当前 user message 之前——每轮变化的召回内容尽量靠近
        末尾，不打断 system prompt + 工具 schema + 历史对话的前缀缓存（实测 bench/cache_probe.py：
        position 1 注入命中 ~53%，末尾 ~99%）。失败静默跳过，不影响主流程。"""
        if not (self.recall_engine and len(user_input.strip()) >= _MIN_RECALL_LENGTH):
            return
        recall_t0 = time.monotonic()
        try:
            recalled = await self.recall_engine.auto_recall(user_input)
            self.telemetry.emit(
                "recall", count=len(recalled or []),
                duration_ms=int((time.monotonic() - recall_t0) * 1000),
            )
            if not recalled:
                return
            # 先删掉上一轮的召回注入，避免跨轮累积
            self.messages = [
                m for m in self.messages
                if not (m.get("role") == "system" and message_text(m).startswith("# 相关记忆"))
            ]
            injection = self.recall_engine.format_for_injection(recalled)
            if self.messages and self.messages[-1].get("role") == "user":
                insert_idx = len(self.messages) - 1
            else:
                insert_idx = len(self.messages)
            self.messages.insert(insert_idx, {"role": "system", "content": injection})
        except Exception as e:
            logger.warning("Memory recall failed: %s", e)

    async def _maybe_auto_compact(self) -> AsyncIterator:
        """回合边界自动压缩：达阈值且未熔断时，归档原文后 LLM 摘要重建 messages。触发依据用
        _effective_prompt_tokens（provider 不回报 usage 时回退估算）。产出 CompactStart/CompactNotice
        事件透传给调用方（故为异步生成器）。"""
        if not (self.compactor
                and not self._compact_breaker.is_open
                and self.compactor.should_compact(self._effective_prompt_tokens(), self.messages)):
            return
        before = len(self.messages)
        yield CompactStart(message_count=before)
        compact_ok = False
        try:
            active_skill = self._detect_active_skill()
            if self.conversation_store:
                # 压缩会就地替换 messages，先归档原文，否则完整历史在磁盘上也会丢失
                self.conversation_store.archive_segment(self.messages, reason="auto-compact")
            self.messages = await self.compactor.compact(
                self.messages, active_skill=active_skill,
                mode="auto", pre_tokens=self._last_prompt_tokens,
            )
            self._compact_breaker.record_success()
            self._rebuild_system_prompt()
            # 触发依据是「上一次请求的 prompt_tokens」——压完到下一次请求之前它还是旧的大数，不清零的话
            # 下一迭代会拿它再压一次。清零后退回按当前消息估算，下一次请求回报真实值再接管。
            self._last_prompt_tokens = 0
            compact_ok = True
        except Exception as e:
            logger.warning("Compact failed: %s", e)
            self._compact_breaker.record_failure()
        self.telemetry.emit(
            "compact", mode="auto", before=before, after=len(self.messages), ok=compact_ok,
        )
        yield CompactNotice(before=before, after=len(self.messages))

    def _prepare_turn_tools(self, disable_tools: bool, allowed_tools: set[str] | None) -> None:
        """构造本轮工具集。规划模式只留只读白名单 + MCP 工具（MCP 走"用户确认制"：首次调用必过
        确认闸，放行的是人批准过的调用而非 MCP 本身），再套 allowed_tools。_turn_tools 是 API 参数
        （空塌成 None）；_turn_tool_names 是派发层硬闸的允许集，必须能区分「空允许集」和「无限制」，故单独存。"""
        if disable_tools:
            self._turn_tools = None
            self._turn_tool_names = set()
            return
        schemas = self.tool_registry.get_tool_schemas(role=self.role) or []
        if self._plan_mode:
            schemas = [
                s for s in schemas
                if s["function"]["name"] in PLAN_MODE_TOOLS
                or getattr(self.tool_registry._tools.get(s["function"]["name"]), "is_mcp", False)
            ]
        if allowed_tools is not None:
            schemas = [s for s in schemas if s["function"]["name"] in allowed_tools]
        self._turn_tools = schemas or None
        self._turn_tool_names = {s["function"]["name"] for s in schemas}
        # 调用方（/goal 规划）给的工具集里没有写工具 → 本回合只读
        self._turn_read_only = allowed_tools is not None and not (allowed_tools & WRITE_TOOLS)

    def _shell_read_only(self) -> bool:
        """shell 此刻是否只能跑只读命令：规划模式、/goal 规划回合、协调者角色（只读角色，没有写工具）。
        现算而非存标志：规划获批（_plan_mode 清零）后自然恢复。"""
        return self._plan_mode or self._turn_read_only or self.role == AgentRole.COORDINATOR

    async def run_stream(
        self,
        user_input: str,
        disable_tools: bool = False,
        allowed_tools: set[str] | None = None,
        images: list[str] | None = None,
    ) -> AsyncIterator:
        """薄包装：turn 级遥测。end 事件在 finally 里发——正常收尾、异常、
        用户 Esc 取消（CancelledError/GeneratorExit）都有归宿，aborted 标记区分。
        对事件流与 future 协议完全透明，逻辑全部在 _run_stream_impl。"""
        t0 = time.monotonic()
        self.telemetry.emit(
            "turn", phase="start", role=self.role.value,
            input_chars=len(user_input), plan_mode=self._plan_mode,
        )
        aborted = False
        try:
            async for ev in self._run_stream_impl(
                user_input, disable_tools=disable_tools,
                allowed_tools=allowed_tools, images=images,
            ):
                yield ev
        except (asyncio.CancelledError, GeneratorExit):
            aborted = True
            raise
        finally:
            self.telemetry.emit(
                "turn", phase="end", role=self.role.value,
                duration_ms=int((time.monotonic() - t0) * 1000),
                tool_calls=self._tool_call_count, aborted=aborted,
            )

    async def _run_stream_impl(
        self,
        user_input: str,
        disable_tools: bool = False,
        allowed_tools: set[str] | None = None,
        images: list[str] | None = None,
    ) -> AsyncIterator:
        # 带图但模型未开视觉 → 立刻拒绝，连文件都不读（首要配置错误，早失败）。
        if images and not getattr(self.llm, "vision", False):
            yield AgentDone(content=NO_VISION_MSG, failed=True)
            return
        # 有图则把 user 消息构造成多模态 list content；图片加载失败给明确错误。
        if images:
            try:
                content = build_user_content(user_input, images)
            except (FileNotFoundError, ValueError, OSError) as e:
                yield AgentDone(content=f"⚠ 图片加载失败：{e}", failed=True)
                return
        else:
            content = user_input
        # 会话开过了午夜：系统提示里的「今天日期」已过期，在用户消息之前追加一对标记
        # （append-only，不动 messages[0]，不破前缀缓存）。
        now = datetime.now()
        self.messages.extend(date_rollover_messages(self._last_turn_date, now))
        self._last_turn_date = now.date()
        self.messages.append({"role": "user", "content": content})
        # 旧图回收：全局只保留最近 N 张，更旧的替换为占位文本。图片每轮全量重发
        # 且无法被文本摘要，不回收会撑爆 context/成本；原图仍在本地图片库。
        self.messages, recycled = recycle_old_images(self.messages)
        if recycled:
            logger.info("已回收 %d 张旧图片（超出保留窗口），原图仍在本地图片库", recycled)
        self._tool_call_count = 0
        self._recent_tool_sigs.clear()
        self._tools_exhausted = False
        self._exhausted_strikes = 0
        self._pending_images = []  # 防上一轮异常中断的残留
        self._files_modified_this_turn = []
        self._test_ran_this_turn = False
        self._verify_nudged = False

        # 视觉门控兜底：消息里含图片（含未来 read_file 等其他来源）但模型未开视觉
        # → 明确拒绝，不发任何请求（放在 recall/compact 之前，别把图喂给瞎的模型）。
        if has_images(self.messages) and not getattr(self.llm, "vision", False):
            yield AgentDone(content=NO_VISION_MSG, failed=True)
            return

        await self._inject_recall(user_input)

        self._prepare_turn_tools(disable_tools, allowed_tools)

        if self._plan_mode or self._active_plan_id:
            self._rebuild_system_prompt()

        empty_retries = 0  # 空响应（只思考不作答）本轮已重试次数，上限 1 防打摆子烧钱
        for iteration in range(self.max_iterations):
            # 执行中插话在迭代顶部注入：上一迭代的 assistant(tool_calls)→tool×N 序列
            # 已完整落库，此刻插 user 消息 API 合法（与 mailbox 注入同一时间点）；
            # 排在 verify/mailbox 注入之后 = 离模型注意力最近。
            for queued_text in self.take_queued_inputs():
                self.messages.append({"role": "user", "content": queued_text})
                yield QueuedInputInjected(text=queued_text)
            # 压缩检查放在每次调模型之前，而不是只在回合开头：goal 的一次执行迭代就是一个回合，可有上百次工具调用，
            # 只在开头查的话回合内上下文再大也不会压（2026-11-19 p4-1：236K、57 次超阈值、0 次压缩）。
            # 此刻上一迭代的 assistant(tool_calls)→tool×N 已完整落库，按段截取不会留下孤儿 tool 消息。
            async for ev in self._maybe_auto_compact():
                yield ev
            # 撞限后收走全部工具（请求不带 tools），只留文字总结一条路
            if self._tools_exhausted:
                self._turn_tools = None
                self._turn_tool_names = set()
            exhausted_this_iter = self._tools_exhausted
            collector = StreamCollector()

            # 请求永远发完整历史，不做任何中段删改（collapse 金字塔已删，2026-07-19）：
            # 主流云厂商全有自动前缀缓存（DS 1:120、qwen 实测 72% 命中），改写历史省
            # token = 打破前缀缓存 = 负优化（bench 实测 22x）；本地模型 token 免费，
            # 只剩窗口约束——由入口截断 + Compactor 近窗保险丝兜住。
            api_messages = self.messages

            llm_t0 = time.monotonic()
            try:
                async for event in collector.collect(
                    self.llm.chat_stream(messages=api_messages, tools=self._turn_tools),
                ):
                    yield event
            except Exception as e:
                logger.warning("LLM call failed: %s", e)
                self.telemetry.emit(
                    "llm_call", role=self.role.value, model=self.llm.model,
                    iteration=iteration, error=str(e),
                    duration_ms=int((time.monotonic() - llm_t0) * 1000),
                )
                yield AgentDone(content=f"⚠ LLM 调用失败: {e}", failed=True)
                return

            response = collector.response
            self.telemetry.emit(
                "llm_call", role=self.role.value, model=self.llm.model,
                iteration=iteration, finish_reason=response.finish_reason,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                cache_hit=response.prompt_cache_hit_tokens,
                cache_miss=response.prompt_cache_miss_tokens,
                reasoning=response.reasoning_tokens,
                duration_ms=int((time.monotonic() - llm_t0) * 1000),
            )

            if response.prompt_tokens > 0:
                self._last_prompt_tokens = response.prompt_tokens
            self.token_usage["prompt_tokens"] += response.prompt_tokens
            self.token_usage["completion_tokens"] += response.completion_tokens
            self.token_usage["total_tokens"] += response.total_tokens
            self.token_usage["prompt_cache_hit_tokens"] += response.prompt_cache_hit_tokens
            self.token_usage["prompt_cache_miss_tokens"] += response.prompt_cache_miss_tokens
            self.token_usage["reasoning_tokens"] += response.reasoning_tokens

            if self.cost_tracker:
                usage_snap = {
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "prompt_cache_hit_tokens": response.prompt_cache_hit_tokens,
                    "prompt_cache_miss_tokens": response.prompt_cache_miss_tokens,
                    "reasoning_tokens": response.reasoning_tokens,
                }
                self.cost_tracker.record(self.role.value, self.llm.model, usage_snap)

            if not response.tool_calls:
                # 空响应（DeepSeek reasoner 偶发：只吐 reasoning、正文零字节就收流）：
                # 空 assistant 消息不入历史（污染上下文），原地重试一次——消息未变，
                # 重发大概率命中缓存；仍空则显式报错，绝不静默收场。
                if not (response.content or "").strip():
                    logger.warning(
                        "空响应：finish_reason=%s reasoning_tokens=%d（已重试 %d 次）",
                        response.finish_reason or "?", response.reasoning_tokens, empty_retries,
                    )
                    if empty_retries < 1:
                        empty_retries += 1
                        continue
                    yield AgentDone(content=(
                        f"⚠ 模型连续两次只输出思考、未产生正文回复"
                        f"（finish_reason={response.finish_reason or '未知'}，"
                        f"thinking {response.reasoning_tokens} tokens）。请重试或换个问法。"
                    ), failed=True)
                    return

                self.messages.append({"role": "assistant", "content": response.content})

                # 验证提醒：改了代码、最后一次改动之后没有通过的测试 → 提醒一次再收尾。
                # 一回合最多一次，提醒后的那一轮无论结果如何都直接收尾（只提醒、不重试，见 verify_loop 模块说明）。
                if self._files_modified_this_turn and not self._verify_nudged and not self._test_ran_this_turn:
                    from loongcli.core.verify_loop import build_verify_prompt, filter_code_files
                    from loongcli.core.test_discovery import discover_test_command

                    code_files = list(dict.fromkeys(filter_code_files(self._files_modified_this_turn)))
                    if code_files:
                        test_cmd = discover_test_command(changed_files=code_files)
                        self._verify_nudged = True
                        self.telemetry.emit(
                            "verify", phase="start",
                            test_command=test_cmd, changed_files=len(code_files),
                        )
                        prompt = build_verify_prompt(
                            # 只列代码文件（已去重保序：同一文件本回合改 N 次只列一次，真机 setup_env.py 列过 3 遍）
                            changed_files=code_files,
                            test_command=test_cmd,
                        )
                        self.messages.append({"role": "user", "content": prompt})
                        self._files_modified_this_turn = []
                        continue  # 让模型处理验证提醒
                    self._files_modified_this_turn = []

                self._persist()
                yield AgentDone(content=response.content, displayed=True)
                return

            self.messages.append(response.to_message())

            for tc in response.tool_calls:
                async for event in self._dispatch_tool_call(tc):
                    yield event

            # 工具已收走仍硬发调用（qwen 前科）：拒两轮就强制终止本轮，
            # 不陪它耗到 max_iterations——每多一轮都在膨胀上下文烧钱
            if exhausted_this_iter and response.tool_calls:
                self._exhausted_strikes += 1
                if self._exhausted_strikes >= 2:
                    msg = (
                        "⚠ 已达本次任务工具调用上限，且工具收走后模型仍持续尝试调用，"
                        "已强制结束本次执行。请拆分任务或换个说法重试。"
                    )
                    self.messages.append({"role": "assistant", "content": msg})
                    self._persist()
                    yield AgentDone(content=msg)
                    return

            # read_file 读到的图片在全部 tool 结果之后统一注入，随即回收超窗旧图
            # （一次读多图也不让在途数量越过保留窗口太多）。
            if self._pending_images:
                self.messages.extend(self._pending_images)
                self._pending_images = []
                self.messages, recycled = recycle_old_images(self.messages)
                if recycled:
                    logger.info("已回收 %d 张旧图片（超出保留窗口），原图仍在本地图片库", recycled)

            if self.task:
                for mail in self.task.drain_mailbox():
                    self.messages.append({"role": "user", "content": mail})

            if self.task_manager:
                for notif in self.task_manager.drain_notifications():
                    text = self.task_manager.format_notification(notif)
                    self.messages.append({"role": "user", "content": text})
                    yield TaskNotification(
                        task_id=notif["task_id"],
                        result=notif["result"],
                    )

            # 逐迭代落盘：此刻 assistant(tool_calls) → tool×N（及随后的图片/mailbox 注入）
            # 已完整入列，序列对 API 合法，从这里 resume 可以直接续。此前只在回合收尾
            # 落盘——进程中途冻结/崩溃/被杀就丢整个回合，而工具的副作用（写文件、git、
            # 外部写入）已经发生，resume 后模型对它们一无所知（2026-11-13 事故：丢了
            # 含 stash + rebase 的一回合）。实测 1.4MB/878 条消息的会话单次 save 17ms。
            self._persist_checkpoint()

        # 说明写进历史：此前只 yield 给界面、不入 messages——历史停在一条 tool 结果上，
        # 用户说「继续」后模型不知道上一回合是被截停的（会当成自己说完了）。
        msg = (
            f"⚠ 已达到迭代上限（{self.max_iterations}轮），自动停止。"
            "任务可能尚未完成——输入「继续」可接着做。"
        )
        self.messages.append({"role": "assistant", "content": msg})
        self._persist()
        yield AgentDone(content=msg)
