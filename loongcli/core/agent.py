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
    ):
        self.llm = llm
        self.tool_registry = tool_registry
        self.permission_checker = permission_checker
        self.role = role
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
        from loongcli.core.verify_loop import VerifyState
        self._verify_state = VerifyState()
        self._last_prompt_tokens = 0
        self.token_usage = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0,
            "reasoning_tokens": 0,
        }
        self._tool_call_count = 0
        self._last_tool_sig: str = ""
        self._repeat_count: int = 0
        self._result_manager = ToolResultManager()
        self._compact_breaker = CompactCircuitBreaker()
        self.messages: list[dict] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def _schedule_auto_extract(self):
        if self.auto_extractor:
            task = asyncio.create_task(self.auto_extractor.extract(list(self.messages)))
            task.add_done_callback(_log_task_exception)

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
        if tool_name == "shell":
            self._last_checkpoint = self.checkpoint_manager.save_workdir()
            return
        files = self._extract_file_args(tool_name, args)
        self._last_checkpoint = self.checkpoint_manager.save(files)
        if self._last_checkpoint:
            self._files_modified_this_turn.extend(files)

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
            content = msg.get("content") or ""
            match = re.search(r"\[自动加载 skill:\s*(\S+)\]", content)
            if match:
                return match.group(1)
            match = re.search(r"## 技能:\s*(\S+)", content)
            if match:
                return match.group(1)
        return None

    async def run_stream(
        self,
        user_input: str,
        disable_tools: bool = False,
        allowed_tools: set[str] | None = None,
    ) -> AsyncIterator:
        if self.skill_registry:
            user_input = self.skill_registry.enrich_prompt(user_input)
        self.messages.append({"role": "user", "content": user_input})
        self._tool_call_count = 0
        self._last_tool_sig = ""
        self._repeat_count = 0
        self._files_modified_this_turn = []
        self._verify_state.reset()

        if self.recall_engine and len(user_input.strip()) >= _MIN_RECALL_LENGTH:
            try:
                recalled = await self.recall_engine.recall(user_input)
                if recalled:
                    injection = self.recall_engine.format_for_injection(recalled)
                    insert_idx = 1 if self.messages and self.messages[0].get("role") == "system" else 0
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
            tools = None
        else:
            schemas = self.tool_registry.get_tool_schemas(role=self.role)
            if self._plan_mode:
                schemas = [s for s in schemas if s["function"]["name"] in PLAN_MODE_TOOLS]
            if allowed_tools is not None:
                schemas = [s for s in schemas if s["function"]["name"] in allowed_tools]
            tools = schemas or None

        if self._plan_mode or self._active_plan_id:
            self._rebuild_system_prompt()

        for iteration in range(self.max_iterations):
            self._result_manager.reset_turn()
            collector = StreamCollector()

            level = should_collapse(self._last_prompt_tokens, model_context_window(self.llm.model))
            api_messages = collapse(self.messages, level) if level > 0 else self.messages

            try:
                async for event in collector.collect(
                    self.llm.chat_stream(messages=api_messages, tools=tools),
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

                # Verify loop: if files were modified, kick off verification
                if self._files_modified_this_turn and not self._verify_state.is_active:
                    from loongcli.core.verify_loop import build_verify_prompt
                    from loongcli.core.test_discovery import discover_test_command

                    test_cmd = discover_test_command(changed_files=self._files_modified_this_turn)
                    self._verify_state.start(self._files_modified_this_turn, test_cmd)
                    prompt = build_verify_prompt(
                        changed_files=self._files_modified_this_turn,
                        test_command=test_cmd,
                        round_number=1,
                    )
                    self.messages.append({"role": "user", "content": prompt})
                    self._files_modified_this_turn = []
                    continue  # Let LLM process the verify prompt

                # Verify loop: already active, check if we need to retry
                if self._verify_state.is_active:
                    is_failure = self._verify_state.test_failed
                    self._verify_state.test_failed = False

                    if is_failure and not self._verify_state.is_exhausted:
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

                    self._verify_state.reset()

                self._persist()
                self._schedule_auto_extract()
                yield AgentDone(content=response.content)
                return

            self.messages.append(response.to_message())

            for tc in response.tool_calls:
                tool_name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                self._tool_call_count += 1
                yield ToolCallStart(tool_name=tool_name, arguments=args)

                if self._tool_call_count > self.max_tool_calls:
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
                    continue

                loop_warning = self._check_loop(tool_name, args)
                if loop_warning:
                    yield ToolCallResult(tool_name=tool_name, result=loop_warning)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": loop_warning,
                    })
                    continue

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
                        continue

                decision, reason = self.permission_checker.check_tool(tool_name, args)

                if decision == Decision.DENY:
                    result = f"⚠ 操作被禁止: {reason}"
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
                        self.permission_checker.record_approval(tool_name, args)
                        result = None
                        async for kind, data in self._exec_tool_stream(tool_name, args):
                            if kind == "__progress__":
                                yield data
                            else:
                                result = data
                    else:
                        result = f"⚠ 用户拒绝了此操作（{reason}）"
                else:
                    result = None
                    async for kind, data in self._exec_tool_stream(tool_name, args):
                        if kind == "__progress__":
                            yield data
                        else:
                            result = data

                if self.hook_manager:
                    await self.hook_manager.run(
                        HookEvent.POST_TOOL_USE,
                        {"tool": tool_name, "arguments": args, "result": result},
                        tool_name=tool_name,
                    )

                if self.lsp_manager and tool_name in ("edit_file", "write_file"):
                    for fp in self._extract_file_args(tool_name, args):
                        await self.lsp_manager.invalidate_doc(fp)

                if (self._verify_state.is_active
                        and tool_name == "shell"
                        and isinstance(result, str)
                        and "[exit code:" in result):
                    self._verify_state.test_failed = True

                if (tool_name == "exit_plan_mode"
                        and isinstance(result, str)
                        and result.startswith('{"__plan_approval__":')):
                    approval_data = json.loads(result)
                    plan_id = approval_data["plan_id"]
                    plan_summary = approval_data["plan_summary"]
                    approval_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
                    yield PlanApproval(
                        plan_id=plan_id,
                        plan_summary=plan_summary,
                        future=approval_future,
                    )
                    user_response = await approval_future
                    if user_response == "approve":
                        self.exit_plan_mode(plan_id=plan_id)
                        result = f"计划已批准。现在按计划执行，所有工具已恢复。\n\n{plan_summary}"
                        self._rebuild_system_prompt()
                        tools = self.tool_registry.get_tool_schemas(role=self.role)
                    elif user_response == "cancel":
                        self._plan_mode = False
                        result = "用户取消了计划。已退出规划模式。"
                        self._rebuild_system_prompt()
                        tools = self.tool_registry.get_tool_schemas(role=self.role)
                    else:
                        result = f"用户要求修改计划：{user_response}\n请根据反馈调整计划，然后重新调用 exit_plan_mode 提交。"

                yield ToolCallResult(tool_name=tool_name, result=result)

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": self._result_manager.process(tool_name, result) if isinstance(result, str) else result,
                })

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
