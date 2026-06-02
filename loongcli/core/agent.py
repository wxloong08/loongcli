from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import re
from typing import AsyncIterator

from loongcli.core.llm import LLMClient
from loongcli.core.events import TextDelta, ToolCallStart, ToolCallResult, AgentDone, CompactStart, CompactNotice, TaskNotification, ConfirmRequest, BatchProgress
from loongcli.core.stream_collector import StreamCollector
from loongcli.core.compact import Compactor
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

    def _persist(self):
        if self.conversation_store:
            self.conversation_store.save(self.messages)

    @staticmethod
    def _tool_signature(name: str, args: dict) -> str:
        raw = json.dumps({"n": name, "a": args}, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

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
        tool = self.tool_registry._tools.get(tool_name)
        if not tool or not getattr(tool, 'supports_progress', False):
            try:
                yield ("__result__", await self.tool_registry.execute_by_name(tool_name, args))
            except Exception as e:
                logger.warning("Tool %s failed: %s", tool_name, e)
                yield ("__error__", f"⚠ 工具执行失败: {e}")
            return

        queue = asyncio.Queue()
        tool._progress_callback = lambda evt: queue.put_nowait(evt)

        try:
            exec_task = asyncio.create_task(
                self.tool_registry.execute_by_name(tool_name, args)
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
            except Exception as e:
                logger.warning("Compact failed: %s", e)
                self._compact_breaker.record_failure()
            yield CompactNotice(before=before, after=len(self.messages))

        if disable_tools:
            tools = None
        else:
            schemas = self.tool_registry.get_tool_schemas(role=self.role)
            if allowed_tools is not None:
                schemas = [s for s in schemas if s["function"]["name"] in allowed_tools]
            tools = schemas or None

        for iteration in range(self.max_iterations):
            self._result_manager.reset_turn()
            collector = StreamCollector()

            try:
                async for event in collector.collect(
                    self.llm.chat_stream(messages=self.messages, tools=tools),
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

            if not response.tool_calls:
                self.messages.append({"role": "assistant", "content": response.content})
                self._persist()
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
        yield AgentDone(content=msg)
