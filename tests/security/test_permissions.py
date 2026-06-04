import pytest
from pathlib import Path
from unittest.mock import patch

from loongcli.security.permissions import (
    PermissionChecker, PermissionMode, Decision,
    CATASTROPHIC_PATTERNS, SAFE_SHELL_PREFIXES,
    ALWAYS_CONFIRM_COMMANDS, ALWAYS_CONFIRM_GIT_SUBS,
    _shell_pattern, _write_pattern,
)


# ── Catastrophic block ──────────────────────────────────────────────


class TestCatastrophicBlock:
    """Safety floor: blocked in ALL modes, no override."""

    def setup_method(self):
        self.default = PermissionChecker(PermissionMode.DEFAULT)
        self.skip = PermissionChecker(PermissionMode.SKIP)

    @pytest.mark.parametrize("cmd", [
        "rm -rf /",
        "rm -rf /home",
        "rm -rf C:\\",
        "rm  -rf  /",
        "format C:",
        "FORMAT D:",
        "dd if=/dev/zero of=/dev/sda",
        "mkfs.ext4 /dev/sda1",
        ":(){ :|:& };:",
    ])
    def test_catastrophic_denied_in_default(self, cmd):
        decision, _ = self.default.check_tool("shell", {"command": cmd})
        assert decision == Decision.DENY

    @pytest.mark.parametrize("cmd", [
        "rm -rf /",
        "format C:",
        "dd if=/dev/zero of=/dev/sda",
    ])
    def test_catastrophic_denied_even_in_skip(self, cmd):
        decision, _ = self.skip.check_tool("shell", {"command": cmd})
        assert decision == Decision.DENY


# ── Shell whitelist ─────────────────────────────────────────────────


class TestShellWhitelist:
    """Safe commands auto-allow in default mode."""

    def setup_method(self):
        self.checker = PermissionChecker(PermissionMode.DEFAULT)

    @pytest.mark.parametrize("cmd", [
        "git status",
        "git log --oneline -10",
        "git diff HEAD",
        "ls -la",
        "pwd",
        "echo hello",
        "cat README.md",
        "python --version",
        "pip list",
        "whoami",
    ])
    def test_safe_commands_allowed(self, cmd):
        decision, _ = self.checker.check_tool("shell", {"command": cmd})
        assert decision == Decision.ALLOW

    @pytest.mark.parametrize("cmd", [
        "pip install requests",
        "npm install express",
        "git push origin main",
        "git commit -m 'test'",
        "curl https://example.com",
        "python script.py",
        "mkdir new_dir",
        "cp file1 file2",
        "mv old new",
        "chmod 777 file",
        "shutdown -h now",
        "sysctl -w net.ipv4.ip_forward=1",
    ])
    def test_unknown_commands_need_confirm(self, cmd):
        decision, _ = self.checker.check_tool("shell", {"command": cmd})
        assert decision == Decision.CONFIRM


# ── Skip mode ───────────────────────────────────────────────────────


class TestSkipMode:
    """Skip mode auto-allows non-catastrophic commands."""

    def setup_method(self):
        self.checker = PermissionChecker(PermissionMode.SKIP)

    def test_normal_command_allowed(self):
        decision, _ = self.checker.check_tool("shell", {"command": "pip install requests"})
        assert decision == Decision.ALLOW

    def test_git_push_allowed(self):
        decision, _ = self.checker.check_tool("shell", {"command": "git push origin main"})
        assert decision == Decision.ALLOW

    def test_catastrophic_still_denied(self):
        decision, _ = self.checker.check_tool("shell", {"command": "rm -rf /"})
        assert decision == Decision.DENY


# ── File write permissions ──────────────────────────────────────────


class TestFileWritePermissions:

    def test_sensitive_path_always_confirm(self):
        checker = PermissionChecker(PermissionMode.DEFAULT)
        decision, _ = checker.check_tool("write_file", {"path": ".env"})
        assert decision == Decision.CONFIRM

    def test_sensitive_path_confirm_even_skip(self):
        checker = PermissionChecker(PermissionMode.SKIP)
        decision, _ = checker.check_tool("write_file", {"path": "/home/user/.ssh/id_rsa"})
        assert decision == Decision.CONFIRM

    def test_project_internal_allowed(self):
        checker = PermissionChecker(PermissionMode.DEFAULT)
        if checker.project_root:
            internal = str(checker.project_root / "src" / "test.py")
            decision, _ = checker.check_tool("write_file", {"path": internal})
            assert decision == Decision.ALLOW

    def test_outside_project_needs_confirm(self):
        checker = PermissionChecker(PermissionMode.DEFAULT)
        decision, _ = checker.check_tool("write_file", {"path": "/tmp/random_file.txt"})
        assert decision == Decision.CONFIRM

    def test_empty_path_confirm(self):
        checker = PermissionChecker(PermissionMode.DEFAULT)
        decision, _ = checker.check_tool("write_file", {"path": ""})
        assert decision == Decision.CONFIRM


# ── Read-only tools ─────────────────────────────────────────────────


class TestReadOnlyTools:

    def setup_method(self):
        self.checker = PermissionChecker(PermissionMode.DEFAULT)

    @pytest.mark.parametrize("tool", ["read_file", "glob", "grep", "recall"])
    def test_readonly_always_allowed(self, tool):
        decision, _ = self.checker.check_tool(tool, {})
        assert decision == Decision.ALLOW


# ── MCP tools ───────────────────────────────────────────────────────


class TestMCPTools:

    def test_mcp_confirm_in_default(self):
        checker = PermissionChecker(PermissionMode.DEFAULT)
        decision, _ = checker.check_tool("searxng__web_search", {}, is_mcp=True)
        assert decision == Decision.CONFIRM

    def test_mcp_allowed_in_skip(self):
        checker = PermissionChecker(PermissionMode.SKIP)
        decision, _ = checker.check_tool("searxng__web_search", {}, is_mcp=True)
        assert decision == Decision.ALLOW


# ── Project root detection ──────────────────────────────────────────


class TestProjectRoot:

    def test_unsafe_root_rejected(self):
        with patch("loongcli.security.permissions._detect_project_root", return_value=None):
            checker = PermissionChecker()
            assert checker.project_root is None


# ── _shell_pattern extraction ───────────────────────────────────────


class TestShellPattern:

    def test_simple_command(self):
        key, always = _shell_pattern("python script.py")
        assert key == "shell:python"
        assert not always

    def test_command_with_path_prefix(self):
        key, always = _shell_pattern("/usr/bin/python script.py")
        assert key == "shell:python"
        assert not always

    def test_windows_path_prefix(self):
        key, always = _shell_pattern("C:\\Python39\\python.exe script.py")
        # shlex.split on Windows may not split backslashes the same way
        assert "python" in key
        assert not always

    def test_git_subcommand(self):
        key, always = _shell_pattern("git commit -m 'test'")
        assert key == "shell:git commit"
        assert not always

    def test_git_push_always_confirm(self):
        key, always = _shell_pattern("git push origin main")
        assert always
        assert key == ""

    def test_git_reset_always_confirm(self):
        key, always = _shell_pattern("git reset --hard HEAD")
        assert always

    def test_git_clean_always_confirm(self):
        key, always = _shell_pattern("git clean -fd")
        assert always

    def test_git_branch_delete_always_confirm(self):
        key, always = _shell_pattern("git branch -D feature")
        assert always

    def test_rm_always_confirm(self):
        key, always = _shell_pattern("rm file.txt")
        assert always

    def test_kill_always_confirm(self):
        key, always = _shell_pattern("kill 1234")
        assert always

    def test_sudo_command(self):
        key, always = _shell_pattern("sudo apt update")
        assert key == "shell:sudo apt"
        assert not always

    def test_sudo_dangerous_always_confirm(self):
        key, always = _shell_pattern("sudo rm file.txt")
        assert always

    def test_empty_command(self):
        key, always = _shell_pattern("")
        assert always

    def test_npm_install(self):
        key, always = _shell_pattern("npm install express")
        assert key == "shell:npm"
        assert not always

    def test_pip_install(self):
        key, always = _shell_pattern("pip install requests")
        assert key == "shell:pip"
        assert not always


# ── _write_pattern extraction ───────────────────────────────────────


class TestWritePattern:

    def test_simple_path(self):
        key = _write_pattern("src/main.py")
        assert key.startswith("write:")
        assert "src" in key

    def test_nested_path(self):
        key = _write_pattern("/home/user/project/src/lib/util.py")
        # Path.resolve() adds drive letter on Windows
        assert key.startswith("write:")
        assert "src" in key and "lib" in key

    def test_root_file(self):
        key = _write_pattern("README.md")
        assert key.startswith("write:")


# ── Session allowlist (record_approval + check_tool) ────────────────


class TestSessionAllowlist:

    def setup_method(self):
        self.checker = PermissionChecker(PermissionMode.DEFAULT)

    def test_shell_approval_remembered(self):
        d1, _ = self.checker.check_tool("shell", {"command": "python script.py"})
        assert d1 == Decision.CONFIRM

        self.checker.record_approval("shell", {"command": "python script.py"})

        d2, _ = self.checker.check_tool("shell", {"command": "python other.py"})
        assert d2 == Decision.ALLOW

    def test_git_commit_approval_remembered(self):
        d1, _ = self.checker.check_tool("shell", {"command": "git commit -m 'init'"})
        assert d1 == Decision.CONFIRM

        self.checker.record_approval("shell", {"command": "git commit -m 'init'"})

        d2, _ = self.checker.check_tool("shell", {"command": "git commit -m 'second'"})
        assert d2 == Decision.ALLOW

    def test_git_push_not_learnable(self):
        """git push is always-confirm, approval should NOT be remembered."""
        self.checker.record_approval("shell", {"command": "git push origin main"})

        decision, _ = self.checker.check_tool("shell", {"command": "git push origin dev"})
        assert decision == Decision.CONFIRM

    def test_rm_not_learnable(self):
        """rm is always-confirm, approval should NOT be remembered."""
        self.checker.record_approval("shell", {"command": "rm file.txt"})

        decision, _ = self.checker.check_tool("shell", {"command": "rm other.txt"})
        assert decision == Decision.CONFIRM

    def test_file_write_approval_remembered(self):
        """Approve write to a directory → subsequent writes in same dir auto-allowed."""
        path1 = "/tmp/project/src/a.py"
        d1, _ = self.checker.check_tool("write_file", {"path": path1})
        assert d1 == Decision.CONFIRM

        self.checker.record_approval("write_file", {"path": path1})

        path2 = "/tmp/project/src/b.py"
        d2, _ = self.checker.check_tool("write_file", {"path": path2})
        assert d2 == Decision.ALLOW

    def test_file_write_different_dir_still_confirms(self):
        """Approval in one dir doesn't cover another."""
        self.checker.record_approval("write_file", {"path": "/tmp/dir_a/f.py"})

        decision, _ = self.checker.check_tool("write_file", {"path": "/tmp/dir_b/f.py"})
        assert decision == Decision.CONFIRM

    def test_sensitive_file_not_learnable(self):
        """Writes to sensitive paths never get session-allowed."""
        self.checker.record_approval("write_file", {"path": ".env"})

        decision, _ = self.checker.check_tool("write_file", {"path": ".env"})
        assert decision == Decision.CONFIRM

    def test_mcp_approval_remembered(self):
        d1, _ = self.checker.check_tool("mcp_searxng", {}, is_mcp=True)
        assert d1 == Decision.CONFIRM

        self.checker.record_approval("mcp_searxng", {})

        d2, _ = self.checker.check_tool("mcp_searxng", {}, is_mcp=True)
        assert d2 == Decision.ALLOW

    def test_different_base_commands_independent(self):
        """Approving python doesn't approve curl."""
        self.checker.record_approval("shell", {"command": "python x.py"})

        decision, _ = self.checker.check_tool("shell", {"command": "curl http://example.com"})
        assert decision == Decision.CONFIRM

    def test_extra_safe_prefixes(self):
        """Constructor param extends SAFE_SHELL_PREFIXES."""
        checker = PermissionChecker(
            PermissionMode.DEFAULT,
            extra_safe_prefixes=["cargo build"],
        )
        decision, _ = checker.check_tool("shell", {"command": "cargo build --release"})
        assert decision == Decision.ALLOW


# ── _make_key ───────────────────────────────────────────────────────


class TestMakeKey:

    def setup_method(self):
        self.checker = PermissionChecker(PermissionMode.DEFAULT)

    def test_shell_returns_key(self):
        key = self.checker._make_key("shell", {"command": "python x.py"})
        assert key == "shell:python"

    def test_shell_dangerous_returns_none(self):
        key = self.checker._make_key("shell", {"command": "rm file.txt"})
        assert key is None

    def test_write_returns_key(self):
        key = self.checker._make_key("write_file", {"path": "src/a.py"})
        assert key is not None
        assert key.startswith("write:")

    def test_write_sensitive_returns_none(self):
        key = self.checker._make_key("write_file", {"path": ".env"})
        assert key is None

    def test_mcp_returns_key(self):
        key = self.checker._make_key("mcp_searxng", {})
        assert key == "mcp:mcp_searxng"

    def test_unknown_tool_returns_none(self):
        key = self.checker._make_key("read_file", {})
        assert key is None
