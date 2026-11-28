"""loongcli 的 Harbor 适配器（BaseInstalledAgent）：在题目容器里装 loongcli、非交互跑一道题，验收交给题目自带的 test.sh。

用途：同模型换外壳的外部基准——拿 Terminal-Bench 2.1 子集的通过率与 HF Harness Leaderboard 上同模型的其他外壳对照。

运行（在 WSL 里）：
    PYTHONPATH=/mnt/d/HKUDS/loongcli/bench/harbor harbor run -d terminal-bench@2.1 -t <题> \\
        -a loongcli_agent:LoongcliAgent \\
        --ak wheel=/mnt/d/dogfood/tbench/wheels/loongcli-0.1.0-py3-none-any.whl \\
        --ak key_file=$HOME/.config/loongcli-bench/deepseek.key
    大陆本机另加：--ak mirrors=cn --ak uv_bin=~/.local/bin/uv（海外机器不加，直连官方源）

设计取舍：
- 装本地构建的 wheel，不装 GitHub 上的：测的是正在改的代码。
- 权限用 --dangerously-skip-permissions：题目容器是一次性沙盒，正是 skip 档的本意；榜上其他外壳也全自动。
  用自动审核会多一个变量（审核拦下的操作在别的外壳里不会被拦）。
- 不加任何轮数 / 时间上限：与榜单口径一致（各外壳按出厂设置，只受题目超时约束）；loongcli 自带的上限照旧。
- API key 只从 WSL 里的 key 文件读、只作为本次执行的环境变量传入：不写进 Harbor 的任务配置（--ae 会被记进 job 配置）。
- 推理档位 high：与榜单 DeepSeek-V4.1-Flash_high 一致（DeepSeek 用 reasoning_effort 控制）。
- 主代理开 vision：与用户日常配置一致（loongcli 的 vision 只认显式配置、代码默认关）；deepseek-flash 原生支持图片。
  基线首跑漏了这一项，7 次读图被拦，读过图的题已整题重跑。
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

_REMOTE_WHEEL_DIR = "/tmp/loongcli-wheel"
_LOGS = "/logs/agent"
# 偶发断连由 apt 自己重试；只影响装依赖，不改题目
_APT_RETRIES = "echo 'Acquire::Retries \"5\";' > /etc/apt/apt.conf.d/80-retries"
# 国内镜像，默认关，--ak mirrors=cn 打开（本机在大陆时用）：首轮基线 18 次里 5 次装 loongcli 超时、2 次 apt 失败。
# 但在本机走代理时阿里云仍偶发被断开（基线 60 次里 3 次，加重试后冒烟仍失败）；海外机器直连官方源，不该用。
_APT_TO_ALIYUN = (
    "for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list; do "
    '[ -f "$f" ] && sed -i -E '
    "'s#https?://(deb|security)\\.debian\\.org#http://mirrors.aliyun.com#g; "
    "s#https?://(archive|security)\\.ubuntu\\.com#http://mirrors.aliyun.com#g' \"$f\"; "
    "done; true"
)
_MIRROR_ENV = {
    "UV_PYTHON_INSTALL_MIRROR": "https://registry.npmmirror.com/-/binary/python-build-standalone",
    "UV_DEFAULT_INDEX": "https://mirrors.aliyun.com/pypi/simple/",
    "PIP_INDEX_URL": "https://mirrors.aliyun.com/pypi/simple/",
}


class LoongcliAgent(BaseInstalledAgent):
    def __init__(self, *args, wheel: str = "", key_file: str = "", effort: str = "high",
                 skill_dirs: str = "", uv_bin: str = "", mirrors: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self._wheel = Path(wheel) if wheel else None
        # 可选：宿主机上现成的 uv 可执行文件直接上传，免得每个容器再从 GitHub 下（大陆本机用；空 = 容器里装）
        self._uv_bin = Path(uv_bin).expanduser() if uv_bin else None
        if mirrors not in ("", "cn"):
            raise ValueError(f"mirrors 只能是空或 cn，现在是 {mirrors!r}")
        self._cn = mirrors == "cn"
        self._env = _MIRROR_ENV if self._cn else {}
        self._key_file = Path(key_file).expanduser() if key_file else None
        self._effort = effort
        # 逗号分隔的容器内路径；空 = 不带 skill（基线）。24.6 消融时传入
        self._skill_dirs = [s for s in skill_dirs.split(",") if s]

    @staticmethod
    def name() -> str:
        return "loongcli"

    def get_version_command(self) -> str | None:
        return '. "$HOME/.local/bin/env" 2>/dev/null; pip show loongcli 2>/dev/null | head -n 2; loongcli --help >/dev/null && echo ok'

    async def install(self, environment: BaseEnvironment) -> None:
        if not self._wheel or not self._wheel.is_file():
            raise ValueError(f"需要 --ak wheel=<loongcli wheel 路径>，现在是 {self._wheel}")
        await self.exec_as_root(environment, command=f"{_APT_RETRIES}; {_APT_TO_ALIYUN}" if self._cn else _APT_RETRIES)
        # Harbor 只认它登记过的依赖名（ca-certificates 不在内，装 curl 时一般会带上）
        await self.ensure_system_dependencies(environment, ("curl", "git", "coreutils"))
        await self.exec_as_agent(environment, command=f"mkdir -p {_REMOTE_WHEEL_DIR}")
        await environment.upload_file(self._wheel, f"{_REMOTE_WHEEL_DIR}/{self._wheel.name}")
        if self._uv_bin and self._uv_bin.is_file():
            await environment.upload_file(self._uv_bin, f"{_REMOTE_WHEEL_DIR}/uv")
            await self.exec_as_root(environment, command=f"install -m 755 {_REMOTE_WHEEL_DIR}/uv /usr/local/bin/uv")
        # uv 自带独立 Python 3.12：题目镜像的系统 Python 版本五花八门，不能指望它 ≥3.12
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh; "
                '[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"; export PATH="$HOME/.local/bin:$PATH"; '
                f"uv tool install --python 3.12 --force {_REMOTE_WHEEL_DIR}/{shlex.quote(self._wheel.name)}; "
                "loongcli --help >/dev/null && echo loongcli-installed"
            ),
            env=self._env,
        )

    async def configure(self, environment: BaseEnvironment) -> None:
        cfg = {
            "model": "deepseek-flash",
            "base_url": "https://api.deepseek.com",
            "skill_dirs": self._skill_dirs,
            "roles": {
                "main": {"model": "deepseek-flash", "thinking": True, "reasoning_effort": self._effort,
                         "context_window": "1M", "vision": True},
                "sub": {"model": "deepseek-flash", "context_window": "1M"},
                "utility": {"model": "deepseek-flash"},
            },
        }
        await self.exec_as_agent(environment, command='mkdir -p "$HOME/.loongcli"')
        home = (await self.exec_as_agent(environment, command='printf %s "$HOME"')).stdout.strip()
        await self._upload_config_text(environment, content=json.dumps(cfg, ensure_ascii=False, indent=1),
                                       remote_path=f"{home}/.loongcli/config.json", filename="config.json")

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if not self._key_file or not self._key_file.is_file():
            raise ValueError("需要 --ak key_file=<只含 DeepSeek API key 的文件>")
        key = self._key_file.read_text(encoding="utf-8").strip()
        # < /dev/null：非交互模式在非 TTY 下会读 stdin，不闭合会挂死到超时
        # 会话与事件流（轨迹）直接写进 /logs/agent：超时时 Harbor 杀进程，跑完再打包的做法会丢轨迹（基线 2 次超时无轨迹）
        command = (
            '[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"; export PATH="$HOME/.local/bin:$PATH"; '
            f"mkdir -p {_LOGS}/loongcli-projects; "
            f'rm -rf "$HOME/.loongcli/projects"; ln -s {_LOGS}/loongcli-projects "$HOME/.loongcli/projects"; '
            f"loongcli --dangerously-skip-permissions --no-stream {shlex.quote(instruction)} "
            f"< /dev/null > {_LOGS}/loongcli.txt 2>&1; rc=$?; "
            # 留一份配置（不含 key，key 只走环境变量）与目录树，便于事后核对口径
            f'{{ echo "== 工作目录 $PWD"; ls -la; cat "$HOME/.loongcli/config.json"; ls -laR "$HOME/.loongcli"; }} '
            f"> {_LOGS}/loongcli-home-tree.txt 2>&1; "
            "exit $rc"
        )
        await self.exec_as_agent(environment, command=command, env={**self._env, "DEEPSEEK_API_KEY": key})

    def populate_context_post_run(self, context: AgentContext) -> None:
        pass
