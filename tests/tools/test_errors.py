import pytest
from loongcli.tools.errors import ToolError


def test_tool_error_is_exception():
    err = ToolError("disk full", retryable=True)
    assert isinstance(err, Exception)


def test_tool_error_attributes():
    err = ToolError("transient IO error", retryable=True, retry_after=1.0)
    assert err.message == "transient IO error"
    assert err.retryable is True
    assert err.retry_after == 1.0


def test_tool_error_non_retryable():
    err = ToolError("permanent failure", retryable=False)
    assert err.retryable is False


def test_tool_error_defaults():
    err = ToolError("something went wrong")
    assert err.retryable is True
    assert err.retry_after == 0.5


def test_tool_error_str():
    err = ToolError("test message")
    assert str(err) == "test message"
