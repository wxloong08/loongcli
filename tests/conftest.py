"""全局测试夹具。"""
import pytest

from loongcli.core import messages


@pytest.fixture(autouse=True)
def _isolate_images_dir(tmp_path, monkeypatch):
    """把本地图片库重定向到临时目录，防止任何测试污染真实的 ~/.loongcli/images/。"""
    monkeypatch.setattr(messages, "images_dir", lambda: tmp_path / "images")
