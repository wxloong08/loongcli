"""_setup_logging：日志进文件不进终端，杜绝 lastResort 裸打 stderr 破坏 TUI。"""
import logging

from loongcli.main import _setup_logging


def test_setup_logging_file_only(tmp_path, monkeypatch):
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    root.handlers = []
    try:
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        _setup_logging()
        assert root.handlers, "应挂上文件 handler"
        assert all(not isinstance(h, logging.StreamHandler) or isinstance(h, logging.FileHandler)
                   for h in root.handlers), "不得有裸 stderr StreamHandler"
        assert (tmp_path / ".loongcli" / "logs" / "loongcli.log").parent.exists()
        # 幂等：再调不重复挂
        n = len(root.handlers)
        _setup_logging()
        assert len(root.handlers) == n
    finally:
        for h in root.handlers[:]:
            h.close()
        root.handlers = old_handlers
