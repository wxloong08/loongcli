from __future__ import annotations
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

EXIT_CODE_BLOCK = 2


class HookEvent(Enum):
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    SESSION_START = "SessionStart"
    SESSION_END = "SessionEnd"


@dataclass
class HookConfig:
    matcher: str
    command: str
    timeout: float = 30.0

    def matches(self, tool_name: str | None) -> bool:
        if not self.matcher or self.matcher == "*":
            return True
        if tool_name is None:
            return self.matcher == "*" or not self.matcher
        for part in self.matcher.split("|"):
            if part.strip() == tool_name:
                return True
        return False


@dataclass
class HookResult:
    blocked: bool = False
    reason: str = ""
    output: str = ""
    exit_code: int = 0


class HookManager:
    def __init__(self):
        self._hooks: dict[HookEvent, list[HookConfig]] = {e: [] for e in HookEvent}

    @classmethod
    def from_config(cls, hooks_dict: dict[str, Any]) -> HookManager:
        manager = cls()
        for event_name, entries in hooks_dict.items():
            try:
                event = HookEvent(event_name)
            except ValueError:
                logger.warning("Unknown hook event: %s", event_name)
                continue
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict) or "command" not in entry:
                    continue
                manager.add(event, HookConfig(
                    matcher=entry.get("matcher", "*"),
                    command=entry["command"],
                    timeout=float(entry.get("timeout", 30)),
                ))
        return manager

    def add(self, event: HookEvent, config: HookConfig):
        self._hooks[event].append(config)

    def has_hooks(self, event: HookEvent) -> bool:
        return len(self._hooks[event]) > 0

    async def run(
        self,
        event: HookEvent,
        context: dict[str, Any],
        tool_name: str | None = None,
    ) -> HookResult:
        matching = [h for h in self._hooks[event] if h.matches(tool_name)]
        if not matching:
            return HookResult()

        combined = HookResult()
        context_json = json.dumps(context, ensure_ascii=False)

        for hook in matching:
            result = await self._exec(hook, context_json)
            if result.output:
                combined.output += (result.output + "\n") if combined.output else result.output
            if result.exit_code == EXIT_CODE_BLOCK:
                combined.blocked = True
                combined.reason = result.output.strip() or f"Hook blocked: {hook.command}"
                combined.exit_code = EXIT_CODE_BLOCK
                break
            if result.exit_code != 0:
                combined.exit_code = result.exit_code
                logger.warning(
                    "Hook '%s' exited with code %d: %s",
                    hook.command, result.exit_code, result.output,
                )

        return combined

    async def _exec(self, hook: HookConfig, stdin_data: str) -> HookResult:
        try:
            proc = await asyncio.create_subprocess_shell(
                hook.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(stdin_data.encode("utf-8")),
                timeout=hook.timeout,
            )
            output = stdout.decode("utf-8", errors="replace").strip()
            if not output and stderr:
                output = stderr.decode("utf-8", errors="replace").strip()

            return HookResult(
                blocked=(proc.returncode == EXIT_CODE_BLOCK),
                reason=output if proc.returncode == EXIT_CODE_BLOCK else "",
                output=output,
                exit_code=proc.returncode or 0,
            )
        except asyncio.TimeoutError:
            logger.warning("Hook '%s' timed out after %.0fs", hook.command, hook.timeout)
            return HookResult(output=f"Hook timed out: {hook.command}", exit_code=1)
        except Exception as e:
            logger.warning("Hook '%s' failed: %s", hook.command, e)
            return HookResult(output=f"Hook error: {e}", exit_code=1)
