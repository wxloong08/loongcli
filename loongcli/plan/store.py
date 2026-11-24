from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from loongcli.memory.conversation import _path_to_slug, _projects_root


STEP_STATUSES = ("pending", "in_progress", "completed", "skipped")
# 常驻/重挂用的紧凑版里方案只给开头这么多字：重挂整段预算 2000 字，要先保住步骤
APPROACH_BRIEF_CHARS = 600
PLAN_STATUSES = ("active", "completed", "abandoned")


@dataclass
class PlanStep:
    index: int
    description: str
    status: str = "pending"
    output: str = ""
    # 完成判据与逐条证据（evidence[i] 对应 criteria[i]，"" = 还没给）。
    # 旧计划文件没有这两个字段：判据为空的步骤不做证据检查，照常可读可完成。
    criteria: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.status not in STEP_STATUSES:
            self.status = "pending"
        # 证据与判据等长：手改过的文件或旧版本写出的长度不一致时补齐/截断，后面按下标取不越界
        n = len(self.criteria)
        self.evidence = (list(self.evidence) + [""] * n)[:n]

    def missing_evidence(self) -> list[int]:
        """还没有证据的判据下标（0 起，内部用；对外显示 +1）。"""
        return [i for i, e in enumerate(self.evidence) if not e.strip()]

    @property
    def number(self) -> int:
        """对外的步骤编号，从 1 起——给模型、给人看的一律用它。index 是存储用的 0 起下标，计划文件里已有，不动。

        曾经对外也用 0 起：模型时而照工具说「第 0 步」、时而按习惯说「第 1 步」，界面怎么编号都有一边对不上（真机两轮）。"""
        return self.index + 1


@dataclass
class Plan:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    title: str = ""
    # 方案（Markdown）：背景与范围、关键决定（选了什么、放弃了什么、为什么）、风险与先验证什么、待用户拍板的问题。
    # 计划原先只有 title + steps，设计决定无处安放，设计类任务的计划退化成「写某某小节」（todo 第 23 项）。
    # 旧计划文件没有这个字段，读出来是空串。
    approach: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    status: str = "active"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def touch(self):
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def progress(self) -> tuple[int, int]:
        done = sum(1 for s in self.steps if s.status in ("completed", "skipped"))
        return done, len(self.steps)

    def closure_problems(self) -> list[str]:
        """能否关闭（plan complete）：每步要么已完成且证据齐全，要么已跳过且写了理由。返回问题清单，空 = 可关闭。"""
        problems = []
        for s in self.steps:
            if s.status == "skipped":
                if not s.output.strip():
                    problems.append(f"步骤 {s.number} 标了跳过但没写理由")
            elif s.status != "completed":
                problems.append(f"步骤 {s.number} 未完成（{s.status}）：{s.description[:60]}")
            elif s.missing_evidence():
                nums = "、".join(str(i + 1) for i in s.missing_evidence())
                problems.append(f"步骤 {s.number} 判据 {nums} 没有证据")
        return problems

    def format_summary(self, show_evidence: bool = False) -> str:
        """输出规范 markdown：步骤用列表项而非裸换行——Markdown 渲染会把单换行
        折叠成软换行，审批面板里所有步骤会糊成一段（真机反馈）。

        判据逐条列出并标出是否已有证据（✓/○）；证据原文默认不带——update_step 的返回值每次都带全貌，
        刚交的证据再回显一遍是纯 token 浪费，要看原文用 plan get。

        步骤、判据都从 1 编号（同 plan 工具的 step / criterion 入参），模型和界面看到的是同一份编号。"""
        done, total = self.progress()
        lines = [f"**{self.title}**（{done}/{total} 完成）", ""]
        for s in self.steps:
            icon = {"pending": "○", "in_progress": "◉", "completed": "✓", "skipped": "–"}.get(s.status, "?")
            line = f"- {icon} **{s.number}.** {s.description}"
            for i, c in enumerate(s.criteria):
                mark = "✓" if s.evidence[i].strip() else "○"
                line += f"\n  - 判据 {i + 1} [{mark}] {c}"
                if show_evidence and s.evidence[i].strip():
                    line += f"\n    - 证据：{s.evidence[i]}"
            if s.output:
                label = "跳过理由" if s.status == "skipped" else "产出"
                line += f"\n  - {label}：{s.output[:80]}"
            lines.append(line)
        return "\n".join(lines)

    def current_step(self) -> PlanStep | None:
        """第一个还没做完的步骤（in_progress 或 pending）。"""
        for s in self.steps:
            if s.status in ("pending", "in_progress"):
                return s
        return None

    def format_compact(self) -> str:
        """给常驻上下文用的紧凑版：每步一行，只展开当前步骤的判据。

        系统提示与 compact 后重挂都用它——完整版带全部判据，8 步 40 条判据的计划约 6500 字符，
        曾因超出 2000 字符预算被整份丢弃。全貌随 update_step 返回值复述到对话末尾，也可 plan get。"""
        done, total = self.progress()
        icons = {"pending": "○", "in_progress": "◉", "completed": "✓", "skipped": "–"}
        lines = [f"**{self.title}**（{done}/{total} 完成，id {self.id}）"]
        for s in self.steps:
            desc = s.description if len(s.description) <= 100 else s.description[:100] + "…"
            lines.append(f"- {icons.get(s.status, '?')} {s.number}. {desc}")
        cur = self.current_step()
        if cur and cur.criteria:
            lines.append(f"当前步骤 {cur.number} 的判据：")
            for i, c in enumerate(cur.criteria):
                lines.append(f"  - 判据 {i + 1} [{'✓' if cur.evidence[i].strip() else '○'}] {c}")
        if self.approach.strip():
            # 方案放最后、只给摘要：重挂预算超了先截方案，不截步骤（执行期最要紧的是步骤和当前判据）
            brief = self.approach.strip()
            if len(brief) > APPROACH_BRIEF_CHARS:
                brief = brief[:APPROACH_BRIEF_CHARS] + "…"
            lines.append(f"方案摘要（全文 plan get）：\n{brief}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Plan:
        steps = [PlanStep(**s) for s in data.pop("steps", [])]
        return cls(steps=steps, **data)


def _project_plans_dir(project_dir: Path | None = None) -> Path:
    """Plans live alongside sessions under the per-project tree.

    Mirrors ``conversation._project_sessions_dir`` so plans are scoped to the
    cwd they were created in — otherwise a global store leaks unrelated
    projects' plans into every session (and into the system prompt).
    """
    slug = _path_to_slug(project_dir or Path.cwd())
    return _projects_root() / slug / "plans"


class PlanStore:
    def __init__(self, base_dir: Path | None = None, project_dir: Path | None = None):
        self.base_dir = base_dir or _project_plans_dir(project_dir)
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
            summary = plan.format_compact()
            if total + len(summary) > max_chars:
                if not lines:
                    # 首个计划超预算：截断而不是整份丢弃（丢了就只剩一句「还有 N 个」，模型看不到任何计划）
                    lines.append(summary[:max_chars] + "…（已截断，plan get 查看全文）")
                    if len(active) > 1:
                        lines.append(f"... (还有 {len(active) - 1} 个活跃计划)")
                else:
                    lines.append(f"... (还有 {len(active) - len(lines)} 个活跃计划)")
                break
            lines.append(summary)
            total += len(summary) + 1
        return "\n\n".join(lines)
