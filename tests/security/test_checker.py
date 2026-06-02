import pytest
from loongcli.security.checker import SecurityChecker


@pytest.fixture
def checker():
    return SecurityChecker()


def test_safe_command_allowed(checker):
    assert checker.is_dangerous_command("ls -la") is False


def test_rm_rf_detected(checker):
    assert checker.is_dangerous_command("rm -rf /tmp/test") is True


def test_git_force_push_detected(checker):
    assert checker.is_dangerous_command("git push --force origin main") is True


def test_git_reset_hard_detected(checker):
    assert checker.is_dangerous_command("git reset --hard HEAD~1") is True


def test_drop_table_detected(checker):
    assert checker.is_dangerous_command("DROP TABLE users") is True


def test_safe_path(checker):
    assert checker.is_sensitive_path("src/main.py") is False


def test_env_file_sensitive(checker):
    assert checker.is_sensitive_path(".env") is True


def test_ssh_dir_sensitive(checker):
    assert checker.is_sensitive_path(".ssh/id_rsa") is True


def test_check_command_returns_reason(checker):
    reason = checker.check_command("rm -rf /tmp")
    assert reason is not None
    assert "删除" in reason


def test_check_command_safe_returns_none(checker):
    assert checker.check_command("ls -la") is None


def test_check_path_returns_reason(checker):
    reason = checker.check_path(".env")
    assert reason is not None
    assert ".env" in reason


def test_check_path_safe_returns_none(checker):
    assert checker.check_path("src/main.py") is None
