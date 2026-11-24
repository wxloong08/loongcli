"""从剪贴板取图（Alt+V）：截完图直接复制、不存文件，也能贴进输入框（对齐 Claude Code）。

为什么不是 Ctrl+V：剪贴板里只有图片时，Windows Terminal 自己处理 Ctrl+V、而且什么都不发给程序——loongcli 根本收不到按键。
读剪贴板靠 Pillow 的 ImageGrab（Windows / macOS 原生；Linux 要 xclip 或 wl-paste）。Pillow 是可选依赖
（pip install loongcli[clipboard]），没装就提示，不影响别的功能。
"""
from __future__ import annotations

import time
from pathlib import Path

from loongcli.core.messages import is_image_file

CLIP_DIR = Path.home() / ".loongcli" / "clipboard"
KEEP_SECONDS = 7 * 24 * 3600      # 贴过的截图留 7 天（提交时要按路径读；会话回放也可能用到），再久就清掉


def _cleanup(folder: Path) -> None:
    cutoff = time.time() - KEEP_SECONDS
    for f in folder.glob("clip-*.png"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def grab_clipboard_images(folder: Path | None = None) -> tuple[list[str], str]:
    """读剪贴板里的图：返回 (图片路径列表, 给用户的提示)。拿到图时提示为空。

    剪贴板里是图像数据（截图）→ 存成 png 返回其路径；是复制的文件（资源管理器里 Ctrl+C）→ 挑出其中的图片。"""
    try:
        from PIL import ImageGrab
    except ImportError:
        return [], "贴图需要 Pillow：pip install pillow（或 pip install \"loongcli[clipboard]\"）"
    try:
        data = ImageGrab.grabclipboard()
    except Exception as e:                # Linux 缺 xclip / wl-paste、剪贴板被别的程序占着等
        return [], f"读不到剪贴板：{e}（Linux 需要装 xclip 或 wl-paste）"
    if data is None:
        return [], "剪贴板里没有图片（截图后复制，再按 Alt+V）"
    if isinstance(data, list):            # 复制的是文件
        images = [str(p) for p in data if is_image_file(str(p))]
        return (images, "") if images else ([], "剪贴板里复制的文件不是图片")
    folder = folder or CLIP_DIR
    folder.mkdir(parents=True, exist_ok=True)
    _cleanup(folder)
    path = folder / f"clip-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000:06d}.png"
    try:
        data.save(path, "PNG")
    except Exception as e:
        return [], f"剪贴板图片存盘失败：{e}"
    return [str(path)], ""
