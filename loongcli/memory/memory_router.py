from __future__ import annotations

from pathlib import Path

from loongcli.memory.markdown_store import MarkdownMemoryStore

_INDEX_FILENAME = "MEMORY.md"

# 三段索引预算占比（基数 = max_bytes - 预留）。绝对值随 max_bytes 缩放，
# 默认 25K 时约为：全局 11.5K / 指针 2.9K / 当前项目 9.6K。
# 预留 1000 覆盖最坏情况：三个段头 ≤200 字节 + store 截断时在预算外追加的
# 两条 ⚠ 截断注 ≤400 字节——保证拼装总量恒 ≤ max_bytes。
_HEADER_RESERVE = 1000
_GLOBAL_SHARE = 12
_POINTER_SHARE = 3
_PROJECT_SHARE = 10
_TOTAL_SHARE = _GLOBAL_SHARE + _POINTER_SHARE + _PROJECT_SHARE


def _strip_index_h1(index_text: str) -> str:
    """去掉 store 索引自带的 '# Memory Index' 标题——三段拼装时各段用自己的小节头。"""
    lines = index_text.split("\n")
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        if lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines).strip()


class MemoryRouter:
    """两层记忆路由：全局库（跨项目 user）+ 当前项目库（project/feedback/reference）。

    鸭子类型兼容 MarkdownMemoryStore 的消费接口（save/load/delete/list_all/
    get_index/index_is_complete），main.py 单点换构造即可，下游零改。

    路由规则：
    - 写：type=user → 全局库；其余 → 当前项目库。真正跨项目的通用原则
      不走 memory，归 LOONG.md（用户既有双层设计）。
    - 读/删：项目优先，未命中回落全局。
    - 索引：三段拼装（全局 + 其他项目指针目录 + 当前项目），总量 ≤ max_bytes。
      其他项目记忆不注入、不召回，靠指针 + read_file 按需拉取（设计如此）。
    """

    def __init__(self, global_dir: Path, project_dir: Path):
        self.global_store = MarkdownMemoryStore(global_dir)
        self.project_store = MarkdownMemoryStore(project_dir)
        # 约定布局 ~/.loongcli/projects/<slug>/memory —— 指针目录扫 projects 根
        self._projects_root = project_dir.parent.parent

    @staticmethod
    def _budgets(max_bytes: int) -> tuple[int, int, int]:
        usable = max(max_bytes - _HEADER_RESERVE, 0)
        return (
            usable * _GLOBAL_SHARE // _TOTAL_SHARE,
            usable * _POINTER_SHARE // _TOTAL_SHARE,
            usable * _PROJECT_SHARE // _TOTAL_SHARE,
        )

    # ---- 写 ----

    def save(self, name: str, description: str, type: str, content: str, source: str = "") -> str:
        if type == "user":
            target, other = self.global_store, self.project_store
        else:
            target, other = self.project_store, self.global_store
        saved = target.save(name, description, type, content, source)
        # name 即身份，一条记忆只活在一个库：改 type 换了作用域（web 编辑场景）时
        # 清掉旧库同名副本——保持单库时代「同名后写覆盖」的语义，否则旧副本会与
        # 新副本互相遮蔽，索引显示与 load 结果分叉。store 去重合并改名的极端情况
        # 只按返回名清理，旧名残留无害（与单库行为一致）。
        other.delete(saved)
        return saved

    # ---- 读 / 删：项目优先，回落全局 ----

    def load(self, name: str) -> dict | None:
        mem = self.project_store.load(name)
        if mem is not None:
            mem["scope"] = "project"
            return mem
        mem = self.global_store.load(name)
        if mem is not None:
            mem["scope"] = "global"
        return mem

    def delete(self, name: str) -> bool:
        return self.project_store.delete(name) or self.global_store.delete(name)

    def list_all(self, type_filter: str | None = None) -> list[dict]:
        """合并当前项目 + 全局条目（不含其他项目——那些只在指针后）。

        同名时项目条目遮蔽全局条目，与 load 的项目优先语义一致。
        每条附加 scope 字段（project/global）供 web UI 区分来源。
        """
        results: list[dict] = []
        seen: set[str] = set()
        for entry in self.project_store.list_all(type_filter):
            entry["scope"] = "project"
            results.append(entry)
            seen.add(entry["name"])
        for entry in self.global_store.list_all(type_filter):
            if entry["name"] in seen:
                continue
            entry["scope"] = "global"
            results.append(entry)
        return results

    # ---- 索引 ----

    def get_index(self, max_bytes: int = 25_000) -> str:
        global_b, pointer_b, project_b = self._budgets(max_bytes)
        sections: list[str] = []

        global_seg = _strip_index_h1(self.global_store.get_index(max_bytes=global_b))
        if global_seg:
            sections.append("### 全局记忆（跨项目）\n" + global_seg)

        pointers = self._project_pointers(pointer_b)
        if pointers:
            sections.append(
                "### 其他项目记忆（未加载；需要时用 read_file 读对应 MEMORY.md 索引，再读具体条目）\n"
                + pointers
            )

        project_seg = _strip_index_h1(self.project_store.get_index(max_bytes=project_b))
        if project_seg:
            sections.append("### 当前项目记忆\n" + project_seg)

        return "\n\n".join(sections)

    def index_is_complete(self, max_bytes: int = 25_000) -> bool:
        """两库索引都完整才算完整（与门），预算与 get_index 同一份分账。

        前提：委派给各库的 index_is_complete 只在「各段远低于自身预算」时与
        get_index 的实际注入一致——拆分后每库都小，当前规模成立。若某个项目库
        逼近自身配额，需要改为直接校验拼装结果。指针段不参与完整性：其他项目
        记忆本就不注入、不在 recall 职责内。
        """
        global_b, _, project_b = self._budgets(max_bytes)
        return (
            self.global_store.index_is_complete(max_bytes=global_b)
            and self.project_store.index_is_complete(max_bytes=project_b)
        )

    def _project_pointers(self, max_bytes: int) -> str:
        """其他项目的指针目录：每项目一行（slug / 条数 / MEMORY.md 路径）。

        扫 ~/.loongcli/projects/*/memory/MEMORY.md；当前项目除外（已全文注入）；
        无索引或零条目的项目跳过。按 slug 排序保证确定性；超预算整行截断。
        """
        if not self._projects_root.is_dir():
            return ""

        current = self.project_store.base_dir.resolve()
        lines: list[str] = []
        for project_dir in sorted(self._projects_root.iterdir()):
            index_path = project_dir / "memory" / _INDEX_FILENAME
            if not index_path.is_file():
                continue
            if index_path.parent.resolve() == current:
                continue
            count = sum(
                1
                for line in index_path.read_text(encoding="utf-8").splitlines()
                if line.startswith("- [")
            )
            if count == 0:
                continue
            lines.append(f"- {project_dir.name}：{count} 条 — {index_path}")

        if not lines:
            return ""

        result = "\n".join(lines)
        while lines and len(result.encode("utf-8")) > max_bytes:
            lines.pop()
            result = "\n".join(lines + ["- …（更多项目未列出）"])
        return result if lines else ""
