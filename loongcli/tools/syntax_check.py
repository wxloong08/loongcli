"""写前语法校验：焊进 write_file / edit_file 的写路径，不做独立工具。

独立校验工具靠 LLM 记得调用，和 prompt 规则是同一失效模式；
写盘前确定性校验、失败即拒绝，才是硬闸门。只覆盖能确定性判定的
文本格式，其余扩展名一律放行（不猜语义）。
"""
from __future__ import annotations

import ast
import json
import tomllib
from pathlib import PurePath

import yaml


def check_syntax(path: str, content: str) -> str | None:
    """按扩展名做语法校验。返回 None 表示通过或不适用，str 为带行号的错误描述。"""
    p = PurePath(path)
    suffix = p.suffix.lower()
    if suffix == ".py":
        return _check_python(content)
    if suffix == ".json":
        # JSONC 惯例（VS Code 配置、tsconfig 等允许注释/尾逗号），
        # 严格 json.loads 会误杀合法用法——跳过。.jsonc 扩展名天然不进此分支。
        if _is_jsonc(p):
            return None
        return _check_json(content)
    if suffix == ".toml":
        return _check_toml(content)
    if suffix in (".yaml", ".yml"):
        return _check_yaml(content)
    return None


def _is_jsonc(p: PurePath) -> bool:
    name = p.name.lower()
    if name.startswith("tsconfig") and name.endswith(".json"):
        return True
    return any(part.lower() == ".vscode" for part in p.parts)


def _check_python(content: str) -> str | None:
    try:
        ast.parse(content)
        return None
    except SyntaxError as e:
        loc = f"第 {e.lineno} 行" + (f"第 {e.offset} 列" if e.offset else "")
        return f"Python 语法错误（{loc}）：{e.msg}"


def _check_json(content: str) -> str | None:
    try:
        json.loads(content)
        return None
    except json.JSONDecodeError as e:
        return f"JSON 语法错误（第 {e.lineno} 行第 {e.colno} 列）：{e.msg}"


def _check_toml(content: str) -> str | None:
    try:
        tomllib.loads(content)
        return None
    except tomllib.TOMLDecodeError as e:
        # TOMLDecodeError 的 message 自带 "at line X, column Y"
        return f"TOML 语法错误：{e}"


def _check_yaml(content: str) -> str | None:
    try:
        # safe_load_all 而非 safe_load：多文档 YAML（--- 分隔）是合法用法
        list(yaml.safe_load_all(content))
        return None
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        loc = f"（第 {mark.line + 1} 行第 {mark.column + 1} 列）" if mark else ""
        problem = getattr(e, "problem", None) or str(e)
        return f"YAML 语法错误{loc}：{problem}"
