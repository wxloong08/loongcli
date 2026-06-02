from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


STEP_STATUSES = ("pending", "in_progress", "completed", "skipped")
PLAN_STATUSES = ("active", "completed", "abandoned")


@dataclass
class PlanStep:
    index: int
    description: str
    status: str = "pending"
    output: str = ""

    def __post_init__(self):
        if self.status not in STEP_STATUSES:
            self.status = "pending"


@dataclass
class Plan:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    title: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    status: str = "active"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def touch(self):
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def progress(self) -> tuple[int, int]:
        done = sum(1 for s in self.steps if s.status in ("completed", "skipped"))
        return done, len(self.steps)

    def format_summary(self) -> str:
        done, total = self.progress()
        lines = [f"**{self.title}** ({done}/{total} 完成)"]
        for s in self.steps:
            icon = {"pending": "○", "in_progress": "◉", "completed": "✓", "skipped": "–"}.get(s.status, "?")
            line = f"  {icon} {s.index}. {s.description}"
            if s.output:
                line += f" → {s.output[:80]}"
            lines.append(line)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Plan:
        steps = [PlanStep(**s) for s in data.pop("steps", [])]
        return cls(steps=steps, **data)


class PlanStore:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path.home() / ".loongcli" / "plans"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, plan_id: str) -> Path:
        return self.base_dir / f"{plan_id}.json"

    def save(self, plan: Plan):
        plan.touch()
        self._path(plan.id).write_text(
            json.dumps(plan.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, plan_id: str) -> Plan | None:
        path = self._path(plan_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return Plan.from_dict(data)

    def delete(self, plan_id: str) -> bool:
        path = self._path(plan_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_plans(self, status: str | None = None) -> list[Plan]:
        plans = []
        for f in sorted(self.base_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                plan = Plan.from_dict(data)
                if status is None or plan.status == status:
                    plans.append(plan)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        return plans

    def get_active_plans(self) -> list[Plan]:
        return self.list_plans(status="active")

    def format_for_prompt(self, max_chars: int = 2000) -> str:
        active = self.get_active_plans()
        if not active:
            return ""
        lines: list[str] = []
        total = 0
        for plan in active:
            summary = plan.format_summary()
            if total + len(summary) > max_chars:
                lines.append(f"... (还有 {len(active) - len(lines)} 个活跃计划)")
                break
            lines.append(summary)
            total += len(summary) + 1
        return "\n\n".join(lines)
