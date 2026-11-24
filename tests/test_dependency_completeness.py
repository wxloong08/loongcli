"""守卫：代码里直接 import 的第三方包，必须在 pyproject 里声明。

2026-11-12 线下面试现场演示 harness 时，换到对方机器上报 `ModuleNotFoundError`，
缺的正是 httpx —— `loongcli/core/llm.py` 直接 import 它，pyproject 却没写，
本地一直靠 mcp 的传递依赖活着。换一台干净机器（或 mcp 哪天不再依赖 httpx）就炸。

传递依赖不是依赖。这条测试保证「新增一个 import」和「忘记写进 pyproject」
不能同时发生。
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "loongcli"
PYPROJECT = REPO_ROOT / "pyproject.toml"

# import 名 → 发行包名，只在两者不一致时才需要登记。
IMPORT_TO_DIST = {
    "yaml": "pyyaml",
    "PIL": "pillow",
}


def _normalize(name: str) -> str:
    """PEP 503 归一化：lower + 把 -_. 统一成 -。"""
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_distributions() -> set[str]:
    """pyproject [project].dependencies 里的发行包名（已归一化）。"""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    declared: set[str] = set()
    for spec in data.get("project", {}).get("dependencies", []):
        # 去掉版本约束/环境标记/extras：httpx>=0.27.1,<1.0  ->  httpx
        name = re.split(r"[\s<>=!~\[;]", spec.strip(), maxsplit=1)[0]
        if name:
            declared.add(_normalize(name))
    return declared


def _declared_optional() -> set[str]:
    """pyproject [project.optional-dependencies] 里各 extra 的发行包名（已归一化）。"""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    declared: set[str] = set()
    for specs in data.get("project", {}).get("optional-dependencies", {}).values():
        for spec in specs:
            name = re.split(r"[\s<>=!~\[;]", spec.strip(), maxsplit=1)[0]
            if name:
                declared.add(_normalize(name))
    return declared


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    names = {"ImportError", "ModuleNotFoundError"}
    t = handler.type
    if isinstance(t, ast.Name):
        return t.id in names
    if isinstance(t, ast.Tuple):
        return any(isinstance(e, ast.Name) and e.id in names for e in t.elts)
    return False


def _third_party_imports(optional: bool = False) -> dict[str, set[str]]:
    """包内非标准库 import → 出现位置集合。

    optional=False：必需的 import（没装就崩）；optional=True：包在 try/except ImportError 里的可选 import
    （没装时走降级分支，如 Alt+V 贴图的 Pillow）——它们只需声明在 optional-dependencies。"""
    stdlib = set(sys.stdlib_module_names)
    found: dict[str, set[str]] = {}
    for path in PACKAGE_DIR.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        guarded: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Try) and any(_catches_import_error(h) for h in node.handlers):
                for stmt in node.body:
                    guarded.update(id(n) for n in ast.walk(stmt))
        names: list[str] = []
        for node in ast.walk(tree):
            if (id(node) in guarded) != optional:
                continue
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                # level>0 是包内相对导入；module 为 None 同理
                if node.level or not node.module:
                    continue
                names.append(node.module)
        for full in names:
            top = full.split(".")[0]
            if top in stdlib or top == PACKAGE_DIR.name:
                continue
            found.setdefault(top, set()).add(str(path.relative_to(REPO_ROOT)))
    return found


def test_all_direct_imports_are_declared():
    declared = _declared_distributions()
    missing = {}
    for module, files in _third_party_imports().items():
        dist = _normalize(IMPORT_TO_DIST.get(module, module))
        if dist not in declared:
            missing[module] = files

    assert not missing, (
        "以下第三方包在代码中被直接 import，但没写进 pyproject.toml 的 dependencies：\n"
        + "\n".join(f"  {m}  ({', '.join(sorted(f))})" for m, f in sorted(missing.items()))
        + "\n\n补法：在 [project].dependencies 里加一条（名字不是 import 名时同步更新 IMPORT_TO_DIST）。"
    )


def test_optional_imports_are_declared_as_extras():
    """try/except ImportError 里的可选 import：没装不崩，但也得在 optional-dependencies 里声明，装法有据可查。"""
    declared = _declared_distributions() | _declared_optional()
    missing = {m: f for m, f in _third_party_imports(optional=True).items()
               if _normalize(IMPORT_TO_DIST.get(m, m)) not in declared}
    assert not missing, (
        "以下可选包（try/except ImportError 里 import）没写进 pyproject 的 optional-dependencies：\n"
        + "\n".join(f"  {m}  ({', '.join(sorted(f))})" for m, f in sorted(missing.items()))
    )
    assert "PIL" in _third_party_imports(optional=True), "自检：扫描应能看到 clipboard_image.py 里可选的 PIL"


def test_dependency_check_actually_detects_httpx():
    """自检：确认提取逻辑真的看得见 httpx，否则测试可能因为扫描为空而假绿。"""
    modules = _third_party_imports()
    assert "httpx" in modules, f"扫描结果异常，未发现 httpx：{sorted(modules)}"
    assert _normalize("httpx") in _declared_distributions()


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("httpx>=0.27.1,<1.0.0", "httpx"),
        ("pyyaml>=6.0", "pyyaml"),
        ("prompt_toolkit>=3.0.0", "prompt-toolkit"),
        ("mcp[cli]>=1.0.0", "mcp"),
        ('tomli>=1.1.0; python_version<"3.11"', "tomli"),
    ],
)
def test_spec_parsing(spec, expected):
    """版本约束/extras/环境标记都不能干扰包名提取。"""
    name = re.split(r"[\s<>=!~\[;]", spec.strip(), maxsplit=1)[0]
    assert _normalize(name) == expected
