from __future__ import annotations
import os
import re
import shlex
import subprocess
from enum import Enum
from pathlib import Path

from loongcli.core.subproc import isolated_console_kwargs


class PermissionMode(Enum):
    DEFAULT = "default"
    SKIP = "skip-permissions"
    # 自动审核：确定性判定与 DEFAULT 完全相同；本该弹确认（CONFIRM）的调用交给 utility 模型审核
    # （loongcli/security/auto_review.py），供 /goal 无人值守长跑——两档之间的空档（todo 第 21 项）
    AUTO = "auto-permissions"


class Decision(Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


CATASTROPHIC_PATTERNS = [
    (re.compile(r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+|.*-rf\s+)[/\\](\s|$)"), "递归删除根目录"),
    # rm -rf /*、rm -rf /. —— 通配/点号使根目录内容被清空，旧规则要求 / 后是空白/结尾故漏
    (re.compile(r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+|.*-rf\s+)[/\\][.*]"), "递归删除根目录内容"),
    (re.compile(r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+|.*-rf\s+)/[a-zA-Z]+\s*$"), "递归删除顶级目录"),
    (re.compile(r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+|.*-rf\s+)~(\s|$)"), "递归删除家目录"),
    # rm -rf ~/ 及 ~/子路径 —— 旧规则只认裸 ~，~/ 后跟斜杠时漏
    (re.compile(r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+|.*-rf\s+)~[/\\]"), "递归删除家目录"),
    # --no-preserve-root：显式解除 GNU rm 对根目录的保护，几乎只为删根
    (re.compile(r"\brm\b.*--no-preserve-root"), "解除根目录保护的递归删除"),
    (re.compile(r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+|.*-rf\s+)[A-Z]:\\(\s|$)", re.IGNORECASE), "递归删除磁盘根目录"),
    # find <根/家> ... -delete：绕过 rm 的另一条整树删除路径
    (re.compile(r"\bfind\s+[/~]\S*.*\s-delete\b"), "find 递归删除"),
    (re.compile(r"\bformat\s+[A-Z]:", re.IGNORECASE), "格式化磁盘"),
    (re.compile(r"\bdd\s+.*of\s*=\s*/dev/[a-z]", re.IGNORECASE), "覆写磁盘设备"),
    (re.compile(r"\bmkfs\b", re.IGNORECASE), "格式化文件系统"),
    (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;"), "fork bomb"),
]

# 免确认的只读命令。按「词」匹配，不按字符串前缀——2026-11-19 实测前缀匹配放行了
# `env pip install x`（env 会执行后面的命令）、`setx PATH ...`（命中 "set"）、`git branch -D main`、
# `git tag -d`、`git remote add`、`date -s`，全都无需确认直接执行。
# 本身只读、可以带任意参数的命令（参数里写文件/起外部程序的另由 _UNSAFE_ARG_RE 拦）：
SAFE_COMMANDS_WITH_ARGS = (
    "git status", "git log", "git diff", "git show", "git rev-parse",
    "git ls-files", "git grep", "git blame",
    "ls", "dir", "cd", "cat", "head", "tail", "wc", "grep", "echo",
    "type", "where", "which",
    # 2026-11-23 补（规划期模型常用来核实文件、被拦后白跑一次调用）：这几个没有写文件的参数。
    # xxd、file 只在个别参数下写文件，另有逐命令检查（_PER_COMMAND_SAFE）。
    # 不加：sort（-o 写文件）、uniq（第二个参数是输出文件）、sed（-i）、awk（system()）、tree（-o 写文件）
    "od", "stat", "du", "diff", "cmp", "xxd", "file",
    "pip list", "pip show", "pip freeze", "npm list",
    "python -m pip list", "python -m pip show", "python -m pip freeze",
    "python3 -m pip list", "python3 -m pip show", "python3 -m pip freeze",
    # 2026-11-23 补（understand-1 规划期 6 次拦截里 5 次误拦，模型反复换写法才拿到环境信息）：
    # pip debug 只打印解释器与兼容标签；reg query 只读注册表（写入是 reg add/delete/import 等别的子命令）
    "pip debug", "python -m pip debug", "python3 -m pip debug",
    "reg query",
)
# 只能原样执行：带上参数就可能写（env CMD 执行 CMD、date -s 改时间、git branch -D 删分支）。
# env / set / printenv 不在内：它们打印全部环境变量（含 DEEPSEEK_API_KEY 等密钥），输出进上下文即发给模型服务商——
# 与拦 `cat .env` 同一理由（2026-11-19 审查真实命令语料时发现）。
SAFE_COMMANDS_EXACT = (
    "pwd", "date", "whoami", "hostname",
    "python --version", "python3 --version", "node --version", "npm --version",
    "pip --version", "py -0", "py -0p", "py --list",
    "python -V", "python -VV", "python3 -V", "python3 -VV",
    "pip3 --version", "python -m pip --version", "python3 -m pip --version",
    "git branch", "git branch -a", "git branch -v", "git branch --show-current",
    "git tag", "git tag -l", "git remote", "git remote -v", "git stash list",
)
# 只读命令里会写文件或起外部程序的参数：git diff/log/show --output 写文件，git grep -O 用指定程序打开结果。
_UNSAFE_ARG_RE = re.compile(r"^(--output|--open-files-in-pager|-O)")
# 这几个程序常以全路径调用（`C:/Python313/python.exe --version`，计划里写明要用的解释器）：比对前去掉路径和 .exe。
# 只对它们这样做——对任意程序都去路径的话，项目里一个叫 ls.exe 的脚本也会被当成 ls 放行。
_PATH_NORMALIZED_PROGRAMS = frozenset({"python", "python3", "pip", "pip3", "py", "node", "npm"})
# 不写任何文件的重定向：丢弃输出（2>/dev/null、>nul）与合并输出（2>&1）。判断前先去掉，
# 免得只读命令被当成写文件拦下、或因多出一个 2>&1 对不上「只能原样执行」的条目。
_DEVNULL_REDIRECT_RE = re.compile(r"\s*(?:\d|&)?>\s*(?:/dev/null|nul)\b|\s*\d>&\d", re.IGNORECASE)


def _strip_devnull_redirects(command: str) -> str:
    return _DEVNULL_REDIRECT_RE.sub("", command)


def _xxd_safe(args: list[str]) -> bool:
    """xxd 只读：不带 -r（还原写文件，含 -rp 这类合写）；位置参数至多一个——第二个位置参数是输出文件。"""
    opts = [a for a in args if a.startswith("-")]
    if any(re.match(r"^-[a-z]*r", a) or a.startswith("-revert") for a in opts):
        return False
    return sum(1 for a in args if not a.startswith("-")) <= 1


def _file_safe(args: list[str]) -> bool:
    """file 只读：不带 -C / --compile（编译 magic 文件会写盘）。"""
    return not any(a == "--compile" or re.match(r"^-[A-Za-z]*C", a) for a in args)


# 只在个别参数下会写文件的命令：命中只读表后再过一道逐命令检查
_PER_COMMAND_SAFE = {"xxd": _xxd_safe, "file": _file_safe}


def _builtin_safe(tokens: list[str]) -> bool:
    """按词判断一条（未链式）命令是否属于内置只读命令表。"""
    low = [t.lower() for t in tokens]
    if not low:
        return False
    prog = re.split(r"[/\\]", low[0].strip('"'))[-1]
    prog = prog[:-4] if prog.endswith(".exe") else prog
    if prog in _PATH_NORMALIZED_PROGRAMS:
        low[0] = prog
    # 原样执行的条目：程序名不分大小写（Windows 上 Git/git 同一个），参数区分大小写——
    # python -V 打印版本，python -v 是详细模式会启动解释器，全转小写会把二者混为一谈
    exact = [low[0]] + tokens[1:]
    for entry in SAFE_COMMANDS_EXACT:
        if exact == entry.split():
            return True
    for entry in SAFE_COMMANDS_WITH_ARGS:
        words = entry.split()
        if low[:len(words)] == words:
            rest = tokens[len(words):]
            extra = _PER_COMMAND_SAFE.get(words[0])
            if extra is not None and not extra(rest):
                return False
            return not any(_UNSAFE_ARG_RE.match(t) for t in rest)
    return False

SENSITIVE_PATHS = [
    re.compile(r"(^|[/\\])\.env($|[/\\])"),
    re.compile(r"(^|[/\\])\.ssh[/\\]"),
    re.compile(r"(^|[/\\])\.git[/\\]config$"),
    # .git/hooks/：威胁模型与本表其余项不同——不是密钥外泄，是代码执行。写进去的
    # 脚本会在用户自己 git commit/push 时以用户权限运行，且 .git/ 不被 git 追踪，
    # 所以它是唯一一条「你的动作触发执行、却永远不出现在 git status/diff 里」的路径
    # （项目内其他可执行文件如 conftest.py 至少改动可见）。放这里即得 CONFIRM 而非
    # 项目内直接 ALLOW：正当需求（用户让 agent 装 hook）只多一次确认，且永不进会话
    # 白名单（_make_key 对敏感路径返回 None），注入则必然暴露在人眼前。
    # 副作用：读 hooks 也要确认——接受，agent 翻 hooks 本身就值得看一眼。
    re.compile(r"(^|[/\\])\.git[/\\]hooks[/\\]"),
    re.compile(r"(^|[/\\])credentials", re.IGNORECASE),
    re.compile(r"(^|[/\\])\.aws[/\\]"),
    re.compile(r"(^|[/\\])\.kube[/\\]"),
    # 2026-11-19 审查真实命令语料时补齐的凭据文件——此前 `cat ~/.loongcli/config.json`（就是 loongcli 自己的
    # API key）、`cat ~/.git-credentials` 等全部免确认，read_file 同样直接放行：
    re.compile(r"(^|[/\\])\.loongcli[/\\]config\.json", re.IGNORECASE),     # loongcli 自己的 api_key
    re.compile(r"(^|[/\\])\.git-credentials$", re.IGNORECASE),
    re.compile(r"(^|[/\\])\.config[/\\]gh[/\\]", re.IGNORECASE),             # GitHub CLI token
    re.compile(r"(^|[/\\])(\.npmrc|\.pypirc|\.netrc|_netrc)$", re.IGNORECASE),
    re.compile(r"(^|[/\\])pip[/\\](pip\.ini|pip\.conf)$", re.IGNORECASE),    # 私有源账号密码
    re.compile(r"(^|[/\\])\.docker[/\\]config\.json$", re.IGNORECASE),
]

# 名字像密钥的环境变量（$DEEPSEEK_API_KEY、${GH_TOKEN}、%OPENAI_API_KEY%）：`echo $XXX_KEY` 与 `env` 一样会把密钥
# 打进上下文、发给模型服务商，免确认通道必须拦（`env` 已于同日移出免确认）。
_SECRET_VAR_RE = re.compile(
    r"[$%]\{?[A-Za-z_]*(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)[A-Za-z0-9_]*",
    re.IGNORECASE,
)

UNSAFE_PROJECT_ROOTS = frozenset({
    "/", os.path.expanduser("~"),
    "C:\\", "D:\\", "E:\\",
})

ALWAYS_CONFIRM_COMMANDS = frozenset({
    "rm", "rmdir", "del", "rd",
    "kill", "pkill", "taskkill",
    "chmod", "chown", "icacls",
    "shutdown", "reboot",
    "mkfs", "dd", "format",
})

ALWAYS_CONFIRM_GIT_SUBS = frozenset({
    "push", "reset", "clean", "branch -D", "branch -d",
})

# 能在首 token 之后追加或重定向出「第二条命令」的操作符——它们可绕过对首 token
# 的前缀/白名单匹配。命令替换（$(...)、`...`、<(...)）与文件重定向（>、<）一律不放行。
# 例外：>&N / N>&M 是 fd 重定向（如 2>&1 合并 stderr），不写文件、不藏第二条命令——否则
# 带 2>&1 的日常命令每次必弹确认且会话学习无法生效（短路在前，见 _check_shell）。
_SUBSTITUTION_RE = re.compile(r"\$\(|`|<\(|>(?![&])|<")
# 引号不配对时的退路：不认引号的朴素拆分（会多拆，只会多要确认，不会漏放）。
_CHAIN_SPLIT_RE = re.compile(r"&&|\|\||;|\||&|\n|\r")


def _has_substitution(command: str) -> bool:
    """命令里有没有命令替换（$(...)、反引号、<(...)）或文件重定向（>、<），认引号。

    _SUBSTITUTION_RE 直接全文搜、不认引号：grep -c '^```' 里单引号中的反引号、grep '->' 里的 > 都被当成替换/重定向，
    只读命令也弹确认（真机一个目标 7 次确认里 2 次是这个）。按 bash 规则扫：
    - 单引号里一切都是字面量，跳过；
    - 双引号里 bash 仍展开 $(...) 和反引号，照查（> < 在双引号里是字面量）；
    - 引号外全查；反斜杠转义的下一个字符是字面量。
    引号不配对时退回 _SUBSTITUTION_RE 全文搜——和 _split_commands 同一个原则：宁可多确认，不能漏放。"""
    quote = None
    i, n = 0, len(command)
    while i < n:
        ch = command[i]
        if quote == "'":
            if ch == "'":
                quote = None
            i += 1
            continue
        if ch == "\\" and i + 1 < n:            # 引号外、双引号里：转义的下一个字符是字面量
            i += 2
            continue
        if quote == '"':
            if ch == '"':
                quote = None
            elif ch == "`" or command.startswith("$(", i):
                return True
            i += 1
            continue
        if ch in "'\"":
            quote = ch
        elif ch == "`" or command.startswith(("$(", "<("), i) or ch == "<":
            return True
        elif ch == ">" and not command.startswith(">&", i):
            return True
        i += 1
    if quote:
        return bool(_SUBSTITUTION_RE.search(command))
    return False


def _split_commands(command: str) -> list[str]:
    """把 shell 命令按「能接出第二条命令」的分隔符拆成子命令，逐段独立校验。

    分隔符：&&、||、;、|、单个 &（后台运行，同样会执行后面那条）、换行。2026-11-19 实测旧版只认前四个，
    `ls & pip install x`、`git log -1⏎rm -rf build` 被当成「ls/git log 带参数」免确认执行。
    认引号：引号里的 | ; & 不是分隔符（`grep -E "a|b"` 曾被拆成残片误拦）。2>&1、&> 里的 & 是重定向，不拆。
    引号不配对时退回朴素拆分——宁可多拆多确认，不能少拆漏放。
    """
    parts: list[str] = []
    buf: list[str] = []
    quote = None
    i, n = 0, len(command)
    while i < n:
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < n:
                buf.append(command[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            buf.append(ch)
            i += 1
            continue
        two = command[i:i + 2]
        if two in ("&&", "||"):
            parts.append("".join(buf)); buf = []; i += 2
            continue
        if ch == "&" and (i > 0 and command[i - 1] == ">" or two == "&>"):
            buf.append(ch); i += 1          # >& / 2>&1 / &> 是重定向，不是后台运行
            continue
        if ch in ";|&\n\r":
            parts.append("".join(buf)); buf = []; i += 1
            continue
        buf.append(ch)
        i += 1
    if quote:
        return [p.strip() for p in _CHAIN_SPLIT_RE.split(command) if p.strip()]
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _detect_project_root() -> Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
            # 不继承 stdin、不附着 TUI 控制台（见 core/subproc.py）
            stdin=subprocess.DEVNULL, **isolated_console_kwargs(),
        )
        if result.returncode == 0:
            root = Path(result.stdout.strip()).resolve()
            if str(root) not in UNSAFE_PROJECT_ROOTS:
                return root
    except Exception:
        pass

    cwd = Path.cwd().resolve()
    if str(cwd) not in UNSAFE_PROJECT_ROOTS:
        return cwd
    return None


def _shell_pattern(command: str) -> tuple[str, bool]:
    """Extract a session-allowlist key from a shell command.

    Returns (pattern_key, always_confirm).
    always_confirm=True means the command must always ask, never session-allowed.
    """
    cmd = command.strip()
    try:
        tokens = shlex.split(cmd, posix=False)
    except ValueError:
        tokens = cmd.split()

    if not tokens:
        return "", True

    base = tokens[0].lower()

    # Strip path prefix: /usr/bin/python → python
    base = base.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

    if base in ALWAYS_CONFIRM_COMMANDS:
        return "", True

    if base == "sudo" and len(tokens) > 1:
        actual = tokens[1].lower().rsplit("/", 1)[-1]
        if actual in ALWAYS_CONFIRM_COMMANDS:
            return "", True
        return f"shell:sudo {actual}", False

    if base == "git" and len(tokens) > 1:
        sub = tokens[1].lower()
        rest = " ".join(t.lower() for t in tokens[1:3])
        if sub in ALWAYS_CONFIRM_GIT_SUBS or rest in ALWAYS_CONFIRM_GIT_SUBS:
            return "", True
        return f"shell:git {sub}", False

    return f"shell:{base}", False


def _write_pattern(path: str) -> str:
    """Extract a session-allowlist key from a file write path."""
    try:
        parent = str(Path(path).resolve().parent)
    except (ValueError, OSError):
        parent = str(Path(path).parent)
    return f"write:{parent}"


class PermissionChecker:
    def __init__(
        self,
        mode: PermissionMode = PermissionMode.DEFAULT,
        extra_safe_prefixes: list[str] | None = None,
        trusted_mcp_servers: set[str] | None = None,
    ):
        self.mode = mode
        self.project_root = _detect_project_root()
        self._session_allowed: set[str] = set()
        # 用户在 config.json 的 mcpServers.<name>.trusted 显式标记的可信 MCP server：
        # 其下所有工具免确认（含子代理非交互场景）。信任是 server 级、用户手写的
        # ——自建/审过的 server 才标；没标的仍走"首次调用确认"闸。
        self._trusted_mcp_servers: set[str] = set(trusted_mcp_servers or ())
        # 计划审批选「批准并自动接受编辑」后置位：项目外非敏感写免确认（敏感路径底线不动）。
        # 注意 loongcli 项目内写本就 ALLOW，此标志的增量只是项目外路径。
        self.auto_accept_edits: bool = False
        # 用户在配置里声明的额外安全前缀（如 "cargo build"）：按前缀匹配，用户自己为其负责。
        # 实例私有，避免多实例/测试间追加泄漏（子代理共享同一 checker 时也只在此实例内生效）。
        self._safe_prefixes: list[str] = list(extra_safe_prefixes or ())
        # AUTO 档的审核器（loongcli/security/auto_review.py，main.py 装配）。挂在 checker 上而不是 AgentServices：
        # 子代理、批量委派、协调者克隆都共享同一 checker，自动带上审核器，不会出现漏接线的旁路
        # （子代理 hook 曾因逐处转发漏接而成旁路）。checker 自身判定仍是纯确定性的，不调用它。
        self.auto_reviewer = None

    def record_approval(self, tool_name: str, args: dict, is_mcp: bool = False) -> None:
        """记录一次用户批准，使同类调用本会话内不再重复询问。

        shell 按子命令片段逐个记 key——check 侧要求链式命令每个片段都在白名单，
        只记整条首 token（如 `cd x && python y` 只记 shell:cd）会导致同一条命令
        下次仍要确认。always_confirm 类片段（rm/kill 等）永不入白名单。
        """
        if tool_name == "shell":
            for sc in _split_commands(args.get("command", "")):
                key, always = _shell_pattern(sc)
                if key and not always:
                    self._session_allowed.add(key)
            return
        key = self._make_key(tool_name, args, is_mcp=is_mcp)
        if key:
            self._session_allowed.add(key)

    def check_tool(self, tool_name: str, args: dict, is_mcp: bool = False) -> tuple[Decision, str]:
        if tool_name == "shell":
            return self._check_shell(args.get("command", ""))

        if tool_name in ("write_file", "edit_file"):
            return self._check_file_write(args.get("path", ""))

        if tool_name == "read_file":
            return self._check_file_read(args.get("path", ""))

        if is_mcp:
            if self.mode == PermissionMode.SKIP:
                return Decision.ALLOW, ""
            # 可信 server（工具名形如 server__tool）免确认
            server = tool_name.split("__", 1)[0]
            if server in self._trusted_mcp_servers:
                return Decision.ALLOW, ""
            key = f"mcp:{tool_name}"
            if key in self._session_allowed:
                return Decision.ALLOW, ""
            return Decision.CONFIRM, "MCP 外部工具"

        return Decision.ALLOW, ""

    def _check_shell(self, command: str) -> tuple[Decision, str]:
        for pattern, reason in CATASTROPHIC_PATTERNS:
            if pattern.search(command):
                return Decision.DENY, reason

        if self.mode == PermissionMode.SKIP:
            return Decision.ALLOW, ""

        command = _strip_devnull_redirects(command)
        # 前缀/白名单命中只证明「首 token」安全。命令替换和重定向能把第二条命令
        # 藏进参数里，或把输出写到任意文件，所以它们一律不走快路径。
        if _has_substitution(command):
            return Decision.CONFIRM, "shell 命令含命令替换或重定向，需要确认"

        # 链式（a && b）、管道（a | b）、顺序（a; b）都能追加一条未经校验的第二命令。
        # 仅当「每个」子命令都独立安全时才自动放行。
        subcommands = _split_commands(command)
        if subcommands and all(self._shell_fragment_safe(sc) for sc in subcommands):
            return Decision.ALLOW, ""

        return Decision.CONFIRM, "shell 命令需要确认"

    def is_read_only_command(self, command: str) -> bool:
        """命令是否只读：无命令替换/重定向，每一段都命中内置只读表或配置的安全前缀，且不碰敏感路径。

        规划阶段（用户批准计划之前）放行 shell 的依据。与 DEFAULT 模式免确认的区别：**不认会话学习白名单**
        ——那里是用户批准过的命令（python、pip 等），可以写东西，批准前不能借它执行。"""
        if any(p.search(command) for p, _ in CATASTROPHIC_PATTERNS):
            return False
        command = _strip_devnull_redirects(command)
        if _has_substitution(command):
            return False
        subcommands = _split_commands(command)
        return bool(subcommands) and all(
            self._shell_fragment_safe(sc, use_session=False) for sc in subcommands
        )

    def _shell_fragment_safe(self, command: str, use_session: bool = True) -> bool:
        """单条（未链式）命令命中只读命令表、配置的安全前缀或会话白名单（use_session）则返回 True。"""
        cmd_stripped = command.strip()
        # 敏感路径旁路封堵：cat/head/tail 等安全前缀不能成为读 .ssh/.env/credentials
        # 的免确认通道（read_file 工具会拦，shell 也必须拦，否则子代理可借 `cat ~/.ssh/id_rsa`
        # 绕过确认外泄密钥）。按 token 逐个校验——SENSITIVE_PATHS 锚定在 ^/分隔符，整串
        # search 会漏掉裸文件名（如 `tail .env`，.env 前是空格），拆成 token 后 `.env`
        # 本身即命中 `^\.env$`。任一 token 命中敏感即不走快路径，交回确认层。
        try:
            arg_tokens = shlex.split(cmd_stripped, posix=False)
        except ValueError:
            arg_tokens = cmd_stripped.split()
        for tok in arg_tokens:
            for pattern in SENSITIVE_PATHS:
                if pattern.search(tok):
                    return False
        if _SECRET_VAR_RE.search(cmd_stripped):
            return False
        # 「必须确认」名单最先判：曾经排在白名单之后，`git branch -D` 被 "git branch" 前缀先放行了
        key, always_confirm = _shell_pattern(command)
        if always_confirm:
            return False
        if _builtin_safe(arg_tokens):
            return True
        for prefix in self._safe_prefixes:
            if cmd_stripped == prefix.strip() or cmd_stripped.startswith(prefix):
                return True
        return use_session and key in self._session_allowed

    def _check_file_write(self, path: str) -> tuple[Decision, str]:
        if not path:
            return Decision.CONFIRM, "路径为空"

        for pattern in SENSITIVE_PATHS:
            if pattern.search(path):
                return Decision.CONFIRM, "敏感文件"

        if self.project_root:
            try:
                resolved = Path(path).resolve()
                resolved.relative_to(self.project_root)
                return Decision.ALLOW, ""
            except ValueError:
                pass

        if self.mode == PermissionMode.SKIP:
            return Decision.ALLOW, ""

        # 「批准并自动接受编辑」生效范围：项目外的非敏感写（敏感路径已在上方拦下）
        if self.auto_accept_edits:
            return Decision.ALLOW, ""

        key = _write_pattern(path)
        if key in self._session_allowed:
            return Decision.ALLOW, ""

        return Decision.CONFIRM, "项目目录外写文件"

    def _check_file_read(self, path: str) -> tuple[Decision, str]:
        """敏感路径读取门：.env/.ssh/credentials 等是密钥外泄的主通道，读取一律确认。

        刻意不进会话白名单（_make_key 无 read 分支）——每次读敏感文件都值得人眼
        看一次。普通文件读取放行（与 Claude Code 的读策略一致）。
        """
        if not path or self.mode == PermissionMode.SKIP:
            return Decision.ALLOW, ""
        for pattern in SENSITIVE_PATHS:
            if pattern.search(path):
                return Decision.CONFIRM, "敏感文件读取"
        return Decision.ALLOW, ""

    def _make_key(self, tool_name: str, args: dict, is_mcp: bool = False) -> str | None:
        """构造工具调用的会话白名单 key；始终需确认的调用返回 None。"""
        if tool_name == "shell":
            key, always = _shell_pattern(args.get("command", ""))
            return key if not always else None
        if tool_name in ("write_file", "edit_file"):
            path = args.get("path", "")
            for pattern in SENSITIVE_PATHS:
                if pattern.search(path):
                    return None
            return _write_pattern(path)
        # MCP 由调用方显式告知（工具实例的 is_mcp），不再按名字前缀猜——
        # 真实 MCP 工具名形如 server__tool，永不以 "mcp_" 开头。key 与
        # check_tool 的 is_mcp 分支保持一致，批准才能被正确记住。
        if is_mcp:
            return f"mcp:{tool_name}"
        return None
