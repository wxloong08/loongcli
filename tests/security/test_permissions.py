import pytest
from pathlib import Path
from unittest.mock import patch

from loongcli.security.permissions import (
    PermissionChecker, PermissionMode, Decision,
    CATASTROPHIC_PATTERNS, SAFE_SHELL_PREFIXES,
)


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


class TestReadOnlyTools:

    def setup_method(self):
        self.checker = PermissionChecker(PermissionMode.DEFAULT)

    @pytest.mark.parametrize("tool", ["read_file", "glob", "grep", "recall"])
    def test_readonly_always_allowed(self, tool):
        decision, _ = self.checker.check_tool(tool, {})
        assert decision == Decision.ALLOW


class TestMCPTools:

    def test_mcp_confirm_in_default(self):
        checker = PermissionChecker(PermissionMode.DEFAULT)
        decision, _ = checker.check_tool("searxng__web_search", {}, is_mcp=True)
        assert decision == Decision.CONFIRM

    def test_mcp_allowed_in_skip(self):
        checker = PermissionChecker(PermissionMode.SKIP)
        decision, _ = checker.check_tool("searxng__web_search", {}, is_mcp=True)
        assert decision == Decision.ALLOW


class TestProjectRoot:

    def test_unsafe_root_rejected(self):
        with patch("loongcli.security.permissions._detect_project_root", return_value=None):
            checker = PermissionChecker()
            assert checker.project_root is None
