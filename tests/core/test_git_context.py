import pytest
from unittest.mock import patch
from pathlib import Path

from loongcli.core.git_context import GitContext, collect_git_context


class TestGitContextToPrompt:
    def test_not_a_repo(self):
        ctx = GitContext(is_repo=False)
        assert ctx.to_prompt() == "- Git 仓库: 否"

    def test_basic_repo(self):
        ctx = GitContext(is_repo=True, branch="main")
        prompt = ctx.to_prompt()
        assert "Git 仓库: 是" in prompt
        assert "当前分支: main" in prompt

    def test_snapshot_label_in_repo(self):
        # 快照标签防模型把过时的 git 状态当现实；非仓库无 git 块、无需标签
        ctx = GitContext(is_repo=True, branch="main")
        assert "快照" in ctx.to_prompt()
        assert "快照" not in GitContext(is_repo=False).to_prompt()

    def test_shows_main_branch_when_different(self):
        ctx = GitContext(is_repo=True, branch="feature", main_branch="main")
        prompt = ctx.to_prompt()
        assert "主分支: main" in prompt

    def test_hides_main_branch_when_same(self):
        ctx = GitContext(is_repo=True, branch="main", main_branch="main")
        prompt = ctx.to_prompt()
        assert "主分支" not in prompt

    def test_shows_ahead_behind(self):
        ctx = GitContext(is_repo=True, branch="dev", ahead_behind="领先 3")
        prompt = ctx.to_prompt()
        assert "领先 3" in prompt

    def test_shows_dirty_files(self):
        ctx = GitContext(is_repo=True, branch="main", dirty_files=["M foo.py", "?? bar.py"])
        prompt = ctx.to_prompt()
        assert "未提交变更" in prompt
        assert "M foo.py" in prompt

    def test_truncates_dirty_files(self):
        files = [f"M file{i}.py" for i in range(20)]
        ctx = GitContext(is_repo=True, branch="main", dirty_files=files)
        prompt = ctx.to_prompt()
        assert "共 20 个文件" in prompt

    def test_shows_recent_commits(self):
        ctx = GitContext(
            is_repo=True, branch="main",
            recent_commits=["abc1234 initial commit", "def5678 add feature"],
        )
        prompt = ctx.to_prompt()
        assert "最近提交" in prompt
        assert "abc1234 initial commit" in prompt

    def test_limits_commits_to_8(self):
        commits = [f"{i:07x} commit {i}" for i in range(12)]
        ctx = GitContext(is_repo=True, branch="main", recent_commits=commits)
        prompt = ctx.to_prompt()
        assert "0000007" in prompt  # 8th commit (index 7)
        assert "0000008" not in prompt  # 9th should be excluded


class TestCollectGitContext:
    def test_not_a_git_repo(self, tmp_path):
        ctx = collect_git_context(tmp_path)
        assert ctx.is_repo is False
        assert ctx.branch == ""

    def _git(self, tmp_path, cmd):
        import subprocess
        subprocess.run(cmd, shell=True, cwd=str(tmp_path), capture_output=True)

    def _init_repo(self, tmp_path):
        self._git(tmp_path, "git init")
        self._git(tmp_path, "git config user.name test")
        self._git(tmp_path, "git config user.email test@test.com")
        self._git(tmp_path, "git commit --allow-empty -m \"init\"")

    def test_collects_from_real_repo(self, tmp_path):
        self._init_repo(tmp_path)
        ctx = collect_git_context(tmp_path)
        assert ctx.is_repo is True
        assert ctx.branch in ("main", "master")

    def test_detects_dirty_files(self, tmp_path):
        self._init_repo(tmp_path)
        (tmp_path / "new.txt").write_text("hello")
        ctx = collect_git_context(tmp_path)
        assert len(ctx.dirty_files) > 0

    def test_detects_recent_commits(self, tmp_path):
        self._init_repo(tmp_path)
        self._git(tmp_path, "git commit --allow-empty -m \"second commit\"")
        ctx = collect_git_context(tmp_path)
        assert len(ctx.recent_commits) >= 2
        assert "second commit" in ctx.recent_commits[0]


class TestPromptIntegration:
    def test_git_context_in_system_prompt(self, monkeypatch):
        monkeypatch.setattr(
            "loongcli.core.prompts.collect_git_context",
            lambda cwd: GitContext(is_repo=True, branch="feat-x", main_branch="main"),
        )
        from loongcli.core.prompts import get_system_prompt
        prompt = get_system_prompt(model="deepseek-v4-flash")
        assert "当前分支: feat-x" in prompt
        assert "主分支: main" in prompt

    def test_no_git_in_system_prompt(self, monkeypatch):
        monkeypatch.setattr(
            "loongcli.core.prompts.collect_git_context",
            lambda cwd: GitContext(is_repo=False),
        )
        from loongcli.core.prompts import get_system_prompt
        prompt = get_system_prompt(model="deepseek-v4-flash")
        assert "Git 仓库: 否" in prompt

    def test_pinned_git_ctx_keeps_prompt_byte_stable(self, monkeypatch):
        """传入 git_ctx 后系统提示词字节稳定——前缀缓存的承重不变量。

        _rebuild_system_prompt 在计划执行期每轮触发。若环境段每次现采 git status，
        agent 每改一个文件 dirty 清单就变 → messages[0] 变 → 前缀缓存连同其后的
        全部对话历史一起作废。固定快照后这条链才断开。
        """
        from loongcli.core.prompts import get_system_prompt

        calls: list[int] = []

        def drifting_collect(cwd=None):
            calls.append(1)
            return GitContext(is_repo=True, branch=f"branch-{len(calls)}", main_branch="main")

        monkeypatch.setattr("loongcli.core.prompts.collect_git_context", drifting_collect)

        pinned = GitContext(is_repo=True, branch="pinned", main_branch="main")
        first = get_system_prompt(model="deepseek-v4-flash", git_ctx=pinned)
        second = get_system_prompt(model="deepseek-v4-flash", git_ctx=pinned)
        assert first == second, "固定 git_ctx 后两次渲染必须字节一致"
        assert calls == [], "传了 git_ctx 就不该再跑 git status"
        assert "当前分支: pinned" in first

        # 不传则保持旧行为（现采）——供一次性构建与测试使用
        assert get_system_prompt(model="deepseek-v4-flash") != get_system_prompt(
            model="deepseek-v4-flash"
        )
        assert len(calls) == 2
