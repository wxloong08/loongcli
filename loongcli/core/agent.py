from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import re
from typing import AsyncIterator, Callable

from loongcli.core.llm import LLMClient
from loongcli.core.events import TextDelta, ToolCallStart, ToolCallResult, AgentDone, CompactStart, CompactNotice, TaskNotification, ConfirmRequest, BatchProgress, PlanApproval
from loongcli.core.stream_collector import StreamCollector
from loongcli.core.compact import Compactor, model_context_window
from loongcli.core.context_collapse import should_collapse, collapse
from loongcli.core.messages import (
    message_text, has_images, build_user_content, recycle_old_images, parse_image_sentinel,
)
from loongcli.core.circuit_breaker import CompactCircuitBreaker
from loongcli.core.tool_result_manager import ToolResultManager
from loongcli.tools.base import ToolRegistry
from loongcli.tools.routing import AgentRole
from loongcli.security.permissions import PermissionChecker, Decision
from loongcli.memory.conversation import ConversationStore
from loongcli.hooks.manager import HookManager, HookEvent
from loongcli.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS_PER_TURN = 200
LOOP_DETECT_THRESHOLD = 3
_MIN_RECALL_LENGTH = 4

NO_VISION_MSG = (
    "⚠ 当前模型未开启视觉能力，无法处理图片。"
    "请在 roles 配置 vision: true 并使用支持图片的模型。"
)

# read_file 读到图片但模型无视觉时喂给模型的工具结果（明确失败，不静默注入垃圾）
NO_VISION_READ_MSG = (
    "该文件是图片，但当前模型未开启视觉能力，无法查看图片内容。"
    "如需看图，请让用户在 roles 配置 vision: true 并使用支持图片的模型。"
)

# 图片 sentinel 被拦截后，tool 消息里的占位文本（图片走下一条 user 消息）
IMAGE_READ_PLACEHOLDER = "[图片已读取（{n} 张），内容见下一条消息]"

PLAN_MODE_TOOLS = frozenset({
    "read_file", "glob", "grep",
    "plan", "exit_plan_mode",
    "lsp_goto_definition", "lsp_find_references",
    "lsp_symbol_search", "lsp_hover", "lsp_diagnostics",
})

PLAN_MODE_SYSTEM_INJECTION = """\

## 规划模式

你已进入规划模式，只能使用只读工具（read_file, glob, grep）调研代码。

工作流程：
1. **调研** — 用只读工具探索代码库，理解任务涉及的文件、接口、依赖
2. **规划** — 用 plan 工具创建结构化计划（标题 + 具体步骤，每步说明改哪个文件、怎么改）
3. **提交** — 调用 exit_plan_mode 提交计划等待用户审批

原则：
- 不要猜测，先读代码再规划
- 计划要具体到文件和函数级别
- 有不确定的地方直接问用户
- 计划控制在 10 步以内\
"""

ACTIVE_PLAN_INJECTION_TEMPLATE = """\

## 活跃计划

{plan_summary}

按步骤执行。每完成一步，用 plan update_step 更新状态。
完成所有步骤后，用 plan complete 关闭计划。\
"""


def _log_task_exception(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.warning("Background task failed: %s", exc)


class AgentLoop:
    def __init__(
        self,
        llm: LLMClient,
        tool_registry: ToolRegistry,
        permission_checker: PermissionChecker,
        system_prompt: str = "",
        max_iterations: int = 50,
        max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
        conversation_store: ConversationStore | None = None,
        compactor: Compactor | None = None,
        task_manager=None,
        task=None,
        role: AgentRole = AgentRole.MAIN,
        hook_manager: HookManager | None = None,
        skill_registry: SkillRegistry | None = None,
        system_prompt_builder: Callable[[], str] | None = None,
        recall_engine=None,
        auto_extractor=None,
        checkpoint_manager=None,
        interactive: bool = True,
    ):
        self.llm = llm
        self.tool_registry = tool_registry
        self.permission_checker = permission_checker
        self.role = role
        # 无人工交互通道（子代理 / 后台任务）：CONFIRM 无人应答，故直接当 DENY 处理，
        # 而不是 yield 一个无人消费的 ConfirmRequest（那会 await future 永久挂起）。见 run_stream。
        self.interactive = interactive
        self.hook_manager = hook_manager
        self.max_iterations = max_iterations
        self.conversation_store = conversation_store
        self.compactor = compactor
        self.task_manager = task_manager
        self.task = task
        self.max_tool_calls = max_tool_calls
        self.skill_registry = skill_registry
        self._system_prompt_builder = system_prompt_builder
        self.recall_engine = recall_engine
        self.auto_extractor = auto_extractor
        self.checkpoint_manager = checkpoint_manager
        self.cost_tracker = None
        self.plan_store = None
        self.lsp_manager = None
        self._plan_mode: bool = False
        self._active_plan_id: str | None = None
        self._last_checkpoint: str | None = None
        self._files_modified_this_turn: list[str] = []
        self._test_ran_this_turn: bool = False
        from loongcli.core.verify_loop import VerifyState, MAX_VERIFY_ROUNDS
        self._verify_state = VerifyState()
        self._max_verify_rounds = MAX_VERIFY_ROUNDS
        self._turn_tools: list[dict] | None = None
        self._last_prompt_tokens = 0
        self.token_usage = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0,
            "reasoning_tokens": 0,
        }
        self._tool_call_count = 0
        self._last_tool_sig: str = ""
        self._repeat_count: int = 0
        # read_file 读到的图片先攒着，本轮全部 tool 结果落库后再以 user 消息注入
        self._pending_images: list[dict] = []
        # 自动记忆写入通知（写入可见化：落库即曝光，人眼是第五道闸）
        self.memory_notices: list[dict] = []
        # 本轮允许调用的工具名集合；None = 无限制（派发层硬闸用，区别于 _turn_tools 的 API 参数语义）
        self._turn_tool_names: set[str] | None = None
        # 撞单轮工具上限后置位：收走工具只留文字总结；模型仍硬发调用则计振出局
        self._tools_exhausted: bool = False
        self._exhausted_strikes: int = 0
        self._result_manager = ToolResultManager()
        self._compact_breaker = CompactCircuitBreaker()
        self.messages: list[dict] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def _schedule_auto_extract(self):
        if self.auto_extractor:
            task = asyncio.create_task(self.auto_extractor.extract(list(self.messages)))

            def _collect(t):
                _log_task_exception(t)
                try:
                    saved = t.result()
                except Exception:
                    return
                if saved:
                    # 写入可见化：攒进 notices，TUI 在下一轮开始时展示
                    #（后台任务完成时用户可能正停在输入行，直接打印会破坏提示符）
                    self.memory_notices.extend(saved)

            task.add_done_callback(_collect)

    def _persist(self):
        if self.conversation_store:
            self.conversation_store.save(self.messages)

    def enter_plan_mode(self):
        self._plan_mode = True

    def exit_plan_mode(self, plan_id: str | None = None):
        self._plan_mode = False
        self._active_plan_id = plan_id

    @property
    def plan_mode(self) -> bool:
        return self._plan_mode

    def _build_plan_injection(self) -> str:
        if self._plan_mode:
            return PLAN_MODE_SYSTEM_INJECTION
        if self._active_plan_id and self.plan_store:
            plan = self.plan_store.load(self._active_plan_id)
            if plan and plan.status == "active":
                return ACTIVE_PLAN_INJECTION_TEMPLATE.format(
                    plan_summary=plan.format_summary(),
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

    def _discard_checkpoint(self) -> None:
        """Discard last checkpoint after successful tool execution."""
        if self._last_checkpoint and self.checkpoint_manager:
            try:
                self.checkpoint_manager.discard(self._last_checkpoint)
            except Exception:
                logger.debug("checkpoint discard failed", exc_info=True)
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
        sig = self._tool_signature(name, args)
        if sig == self._last_tool_sig:
            self._repeat_count += 1
        else:
            self._last_tool_sig = sig
            self._repeat_count = 1
        if self._repeat_count >= LOOP_DETECT_THRESHOLD:
            return (
                f"⚠ 检测到循环：{name} 已用相同参数连续调用 {self._repeat_count} 次。"
                "请换一种思路或向用户说明无法完成。"
            )
        return None

    async def _exec_tool_stream(self, tool_name: str, args: dict):
        self._maybe_checkpoint(tool_name, args)
        tool = self.tool_registry._tools.get(tool_name)
        if not tool or not getattr(tool, 'supports_progress', False):
            async for kind, data in self._exec_with_retry(tool_name, args):
                yield (kind, data)
            return

        queue = asyncio.Queue()
        tool._progress_callback = lambda evt: queue.put_nowait(evt)

        try:
            exec_task = asyncio.create_task(
                self._exec_with_retry_single(tool_name, args)
            )

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
        yield ToolCallStart(tool_name=tool_name, arguments=args)

        # 派发层硬闸：schema 过滤只是"告知"，有的模型（真机：qwen 在规划模式下
        # 直接发 shell）会无视工具列表硬调。凡不在本轮允许集里的调用一律拒绝——
        # 闸门必须确定性，不赌模型自觉。
        if self._turn_tool_names is not None:
            if tool_name not in self._turn_tool_names:
                if self._plan_mode:
                    result = (
                        f"⚠ 规划模式下工具 {tool_name} 不可用，只能用只读调研工具"
                        "（read_file/glob/grep/lsp_*）。请完成调研后用 plan 工具制定计划，"
                        "再调用 exit_plan_mode 提交用户审批——审批通过后所有工具自动恢复。"
                    )
                elif self._tools_exhausted:
                    result = "⚠ 已达单轮工具调用上限，本轮工具已收走，请直接文字总结当前进展。"
                else:
                    result = f"⚠ 工具 {tool_name} 不在本轮允许的工具列表中。"
                yield ToolCallResult(tool_name=tool_name, result=result)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
                return

        if self._tool_call_count > self.max_tool_calls:
            # 标记撞限：run_stream 下一迭代会收走全部工具（不带 tools 参数），只留
            # 文字总结一条路。此前的实现是"无限拒绝不终止"——模型每次重试都膨胀一轮
            # 上下文（真机：单轮烧掉 97 万 prompt tokens），且连 plan/exit_plan_mode
            # 的收场通道也一起拒，模型只能挣扎到 max_iterations 耗尽。
            self._tools_exhausted = True
            result = (
                f"⚠ 已达到单轮工具调用上限（{self.max_tool_calls}次），"
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

        if decision == Decision.DENY:
            result = f"⚠ 操作被禁止: {reason}"
        elif decision == Decision.CONFIRM and not self.interactive:
            result = f"⚠ 需用户确认的操作在子代理/非交互上下文中自动拒绝（{reason}）"
        elif decision == Decision.CONFIRM:
            future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
            yield ConfirmRequest(
                tool_name=tool_name,
                arguments=args,
                risk_reason=reason,
                future=future,
            )
            approved = await future
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
            self._test_ran_this_turn = False

        if tool_name == "shell" and isinstance(result, str):
            from loongcli.core.verify_loop import detect_test_failure, is_test_command
            if self._verify_state.is_active and detect_test_failure(result):
                self._verify_state.test_failed = True
            cmd = args.get("command", "")
            if is_test_command(cmd) and not detect_test_failure(result):
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
                result = f"用户要求修改计划：{user_response}\n请根据反馈调整计划，然后重新调用 exit_plan_mode 提交。"

        # 修改类工具执行中抛异常时留有备份——回滚该文件，避免半写损坏；
        # 否则丢弃备份。
        if tool_errored:
            self._restore_checkpoint()
        else:
            self._discard_checkpoint()

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

        self.messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": self._result_manager.process(tool_name, result) if isinstance(result, str) else result,
        })

    async def run_stream(
        self,
        user_input: str,
        disable_tools: bool = False,
        allowed_tools: set[str] | None = None,
        images: list[str] | None = None,
    ) -> AsyncIterator:
        if self.skill_registry:
            user_input = self.skill_registry.enrich_prompt(user_input)

        # 带图但模型未开视觉 → 立刻拒绝，连文件都不读（首要配置错误，早失败）。
        if images and not getattr(self.llm, "vision", False):
            yield AgentDone(content=NO_VISION_MSG)
            return
        # 有图则把 user 消息构造成多模态 list content；图片加载失败给明确错误。
        if images:
            try:
                content = build_user_content(user_input, images)
            except (FileNotFoundError, ValueError, OSError) as e:
                yield AgentDone(content=f"⚠ 图片加载失败：{e}")
                return
        else:
            content = user_input
        self.messages.append({"role": "user", "content": content})
        # 旧图回收：全局只保留最近 N 张，更旧的替换为占位文本。图片每轮全量重发
        # 且无法被文本摘要，不回收会撑爆 context/成本；原图仍在本地图片库。
        self.messages, recycled = recycle_old_images(self.messages)
        if recycled:
            logger.info("已回收 %d 张旧图片（超出保留窗口），原图仍在本地图片库", recycled)
        self._tool_call_count = 0
        self._last_tool_sig = ""
        self._repeat_count = 0
        self._tools_exhausted = False
        self._exhausted_strikes = 0
        self._pending_images = []  # 防上一轮异常中断的残留
        self._files_modified_this_turn = []
        self._test_ran_this_turn = False
        self._verify_state.reset()

        # 视觉门控兜底：消息里含图片（含未来 read_file 等其他来源）但模型未开视觉
        # → 明确拒绝，不发任何请求（放在 recall/compact 之前，别把图喂给瞎的模型）。
        if has_images(self.messages) and not getattr(self.llm, "vision", False):
            yield AgentDone(content=NO_VISION_MSG)
            return

        if self.recall_engine and len(user_input.strip()) >= _MIN_RECALL_LENGTH:
            try:
                recalled = await self.recall_engine.auto_recall(user_input)
                if recalled:
                    # 先删掉上一轮的召回注入，避免跨轮累积
                    self.messages = [
                        m for m in self.messages
                        if not (m.get("role") == "system" and message_text(m).startswith("# 相关记忆"))
                    ]
                    injection = self.recall_engine.format_for_injection(recalled)
                    # 注入到当前 user message 之前：每轮变化的召回内容尽量靠近末尾，
                    # 不打断 system prompt + 工具 schema + 历史对话的前缀缓存。
                    # 实测（bench/cache_probe.py）：position 1 注入命中率 ~53%，末尾注入 ~99%。
                    if self.messages and self.messages[-1].get("role") == "user":
                        insert_idx = len(self.messages) - 1
                    else:
                        insert_idx = len(self.messages)
                    self.messages.insert(insert_idx, {"role": "system", "content": injection})
            except Exception as e:
                logger.warning("Memory recall failed: %s", e)

        if (self.compactor
                and not self._compact_breaker.is_open
                and self.compactor.should_compact(self._last_prompt_tokens, self.messages)):
            before = len(self.messages)
            yield CompactStart(message_count=before)
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
            except Exception as e:
                logger.warning("Compact failed: %s", e)
                self._compact_breaker.record_failure()
            yield CompactNotice(before=before, after=len(self.messages))

        if disable_tools:
            self._turn_tools = None
            self._turn_tool_names = set()
        else:
            schemas = self.tool_registry.get_tool_schemas(role=self.role) or []
            if self._plan_mode:
                schemas = [s for s in schemas if s["function"]["name"] in PLAN_MODE_TOOLS]
            if allowed_tools is not None:
                schemas = [s for s in schemas if s["function"]["name"] in allowed_tools]
            self._turn_tools = schemas or None
            # 允许集单独存：schemas 为空时 _turn_tools 塌缩成 None（API 参数语义），
            # 但派发层硬闸必须能区分「空允许集」和「无限制」
            self._turn_tool_names = {s["function"]["name"] for s in schemas}

        if self._plan_mode or self._active_plan_id:
            self._rebuild_system_prompt()

        for iteration in range(self.max_iterations):
            self._result_manager.reset_turn()
            # 撞限后收走全部工具（请求不带 tools），只留文字总结一条路
            if self._tools_exhausted:
                self._turn_tools = None
                self._turn_tool_names = set()
            exhausted_this_iter = self._tools_exhausted
            collector = StreamCollector()

            # cache-aware provider（如 DeepSeek）跳过 collapse：保持前缀稳定让缓存命中，
            # 减 token 在自动缓存场景是负优化。其他 provider 走完整压缩金字塔。
            if self.llm.cache_aware:
                api_messages = self.messages
            else:
                level = should_collapse(self._last_prompt_tokens, model_context_window(self.llm.model))
                api_messages = collapse(self.messages, level) if level > 0 else self.messages

            try:
                async for event in collector.collect(
                    self.llm.chat_stream(messages=api_messages, tools=self._turn_tools),
                ):
                    yield event
            except Exception as e:
                logger.warning("LLM call failed: %s", e)
                yield AgentDone(content=f"⚠ LLM 调用失败: {e}")
                return

            response = collector.response

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
                self.messages.append({"role": "assistant", "content": response.content})

                # Verify loop: if code files were modified, kick off verification
                if self._files_modified_this_turn and not self._verify_state.is_active and not self._test_ran_this_turn:
                    from loongcli.core.verify_loop import build_verify_prompt, filter_code_files
                    from loongcli.core.test_discovery import discover_test_command

                    code_files = filter_code_files(self._files_modified_this_turn)
                    if not code_files:
                        self._files_modified_this_turn = []
                        self._persist()
                        self._schedule_auto_extract()
                        yield AgentDone(content=response.content)
                        return

                    test_cmd = discover_test_command(changed_files=code_files)
                    self._verify_state.start(code_files, test_cmd)
                    prompt = build_verify_prompt(
                        changed_files=self._files_modified_this_turn,
                        test_command=test_cmd,
                        round_number=1,
                    )
                    self.messages.append({"role": "user", "content": prompt})
                    self._files_modified_this_turn = []
                    continue  # Let LLM process the verify prompt

                # Verify loop: check if we need to retry or report exhaustion
                if self._verify_state.round > 0:
                    is_failure = self._verify_state.test_failed
                    self._verify_state.test_failed = False

                    if is_failure and self._verify_state.is_active:
                        self._verify_state.last_error = response.content
                        self._verify_state.round += 1
                        prompt = build_verify_prompt(
                            changed_files=self._verify_state.changed_files,
                            test_command=self._verify_state.test_command,
                            round_number=self._verify_state.round,
                            last_error=self._verify_state.last_error,
                        )
                        self.messages.append({"role": "user", "content": prompt})
                        continue  # Retry verification

                    if is_failure and self._verify_state.is_exhausted:
                        response.content = (
                            f"⚠ 验证失败：已重试 {self._max_verify_rounds} 轮仍未通过。\n\n"
                            f"{response.content}"
                        )
                    self._verify_state.reset()

                self._persist()
                self._schedule_auto_extract()
                yield AgentDone(content=response.content)
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
                        "⚠ 已达单轮工具调用上限，且工具收走后模型仍持续尝试调用，"
                        "已强制结束本轮。请拆分任务或换个说法重试。"
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

        msg = f"⚠ 已达到迭代上限（{self.max_iterations}轮），自动停止。"
        self._persist()
        self._schedule_auto_extract()
        yield AgentDone(content=msg)
