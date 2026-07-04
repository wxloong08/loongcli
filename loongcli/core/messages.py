"""消息内容工具：统一处理 str content 与多模态 list content。

引入视觉输入后，一条消息的 content 可能是字符串，也可能是 OpenAI 内容块列表
（text / image_url 混排）。全管线凡是对 content 做字符串操作（标题、token 估算、
正则匹配、压缩合并的文本侧），都必须经此收口，避免 str 假设被多模态消息打崩。
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# 扩展名 → MIME，识别不了再靠 magic bytes 兜底
_IMAGE_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
}
_MIME_TO_EXT = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif",
    "image/webp": ".webp", "image/bmp": ".bmp",
}
_MAX_IMAGE_FILE_BYTES = 10 * 1024 * 1024  # 单张原图上限 10MB

# 图片引用协议：消息里只存 loongimg://<内容哈希>.<ext>，原图落盘 images_dir()，
# 发 API 时（chat_stream 收口）才内联成 base64 data URL。避免 session JSON 被
# base64 撑爆、resume/归档反复复制大 blob（设计文档 §3 持久化决策）。
IMAGE_REF_PREFIX = "loongimg://"

# 旧图回收：全局只保留最近 N 张，更旧的替换为占位文本。图片原子不可截断、
# 无法被文本摘要，回收是控 context 的唯一手段（设计文档 §7：必需项）。
KEEP_RECENT_IMAGES = 5
IMAGE_DROPPED_PLACEHOLDER = "[图片已省略：超出保留窗口]"
IMAGE_LOST_PLACEHOLDER = "[图片已丢失：本地图片缓存不存在]"

# 识别输入里的图片路径：双/单引号包裹，或无空格的裸路径。终端拖文件通常粘贴成路径。
_IMG_EXT = r"png|jpe?g|gif|webp|bmp"
_IMG_PATH_RE = re.compile(
    rf'"([^"]+\.(?:{_IMG_EXT}))"'
    rf"|'([^']+\.(?:{_IMG_EXT}))'"
    rf"|(\S+\.(?:{_IMG_EXT}))",
    re.IGNORECASE,
)


def message_text(msg: dict) -> str:
    """提取一条消息的纯文本，兼容 str content 和多模态 list content。

    - str：原样返回。
    - list（内容块）：拼接 text 块；image_url 等非文本块计为占位符（不参与文本）。
    - None / 其他：返回 ""。
    """
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "image_url":
                parts.append("[图片]")
            else:
                parts.append(f"[{btype or '未知内容'}]")
        return "".join(parts)
    return ""


def has_images(messages: list[dict]) -> bool:
    """messages 里是否含任何 image_url 内容块。用于视觉能力门控。"""
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                return True
    return False


def to_content_parts(content) -> list:
    """把 content 归一成 OpenAI 内容块列表：str → [text 块]；list 原样拷贝；None/空 → []。

    用于需要保留图片块的场景（如合并相邻消息），区别于 message_text（只取文本）。
    """
    if isinstance(content, list):
        return list(content)
    if content:
        return [{"type": "text", "text": content}]
    return []


def _sniff_mime(raw: bytes) -> str | None:
    """靠 magic bytes 识别常见图片格式（扩展名不可靠时兜底）。"""
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"):
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith(b"BM"):
        return "image/bmp"
    return None


def _read_valid_image(path: str) -> tuple[bytes, str]:
    """读取并校验图片文件，返回 (原始字节, MIME)。

    文件不存在 → FileNotFoundError；过大或格式识别不了 → ValueError。
    先按扩展名定 MIME，不认识再用 magic bytes 兜底。只读一次文件。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"图片不存在：{path}")
    raw = p.read_bytes()
    if len(raw) > _MAX_IMAGE_FILE_BYTES:
        raise ValueError(
            f"图片过大（{len(raw) / 1024 / 1024:.1f}MB，"
            f"上限 {_MAX_IMAGE_FILE_BYTES // 1024 // 1024}MB）：{path}"
        )
    mime = _IMAGE_MIME.get(p.suffix.lower()) or _sniff_mime(raw)
    if mime is None:
        raise ValueError(f"无法识别的图片格式：{path}")
    return raw, mime


def image_to_data_url(path: str) -> str:
    """读取图片文件，返回 base64 data URL。校验规则同 _read_valid_image。"""
    raw, mime = _read_valid_image(path)
    data = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{data}"


def images_dir() -> Path:
    """本地图片库目录（内容哈希命名存储）。独立函数便于测试重定向。"""
    return Path.home() / ".loongcli" / "images"


def store_image_bytes(raw: bytes) -> str:
    """把图片字节按内容哈希存入本地图片库，返回 loongimg:// 引用。

    格式一律以 magic bytes 实测为准——外部来源（MCP server）声明的 mime 可能
    乱标，垃圾字节存成 .png 会在发 API 时才炸。识别不了或超大抛 ValueError。
    同内容图片天然去重（哈希同名，已存在则不重写）。
    """
    if len(raw) > _MAX_IMAGE_FILE_BYTES:
        raise ValueError(
            f"图片过大（{len(raw) / 1024 / 1024:.1f}MB，"
            f"上限 {_MAX_IMAGE_FILE_BYTES // 1024 // 1024}MB）"
        )
    mime = _sniff_mime(raw)
    if mime is None:
        raise ValueError("无法识别的图片格式")
    name = hashlib.sha256(raw).hexdigest()[:16] + _MIME_TO_EXT[mime]
    dest = images_dir() / name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
    return IMAGE_REF_PREFIX + name


def store_image(path: str) -> str:
    """读取并校验图片文件，按内容哈希存入本地图片库，返回 loongimg:// 引用。

    校验失败向上抛，与 image_to_data_url 同规则——错误在入口处暴露，不留到发送时。
    """
    raw, _ = _read_valid_image(path)
    return store_image_bytes(raw)


def _ref_to_path(ref: str) -> Path:
    """loongimg:// 引用 → 图片库内文件路径。引用名含路径分隔符视为非法（防穿越）。"""
    name = ref[len(IMAGE_REF_PREFIX):]
    if not name or "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"非法图片引用：{ref}")
    return images_dir() / name


def inline_image_refs(messages: list[dict]) -> list[dict]:
    """把消息里的 loongimg:// 引用内联成 base64 data URL，返回新列表（不改原消息）。

    发 API 前的唯一收口点（chat_stream 调用）。引用对应文件丢失时降级为
    占位文本块并告警——引用会留在历史里每轮重试，绝不能让一张丢图卡死会话。
    直传的 data URL / 外部 URL 原样透传。
    """
    out: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list) or not any(
            isinstance(b, dict) and b.get("type") == "image_url"
            and str((b.get("image_url") or {}).get("url", "")).startswith(IMAGE_REF_PREFIX)
            for b in content
        ):
            out.append(msg)
            continue
        new_content: list = []
        for block in content:
            url = ""
            if isinstance(block, dict) and block.get("type") == "image_url":
                url = str((block.get("image_url") or {}).get("url", ""))
            if not url.startswith(IMAGE_REF_PREFIX):
                new_content.append(block)
                continue
            try:
                data_url = image_to_data_url(str(_ref_to_path(url)))
                new_content.append({"type": "image_url", "image_url": {"url": data_url}})
            except (FileNotFoundError, ValueError, OSError) as e:
                logger.warning("图片引用无法内联，降级为占位文本：%s", e)
                new_content.append({"type": "text", "text": IMAGE_LOST_PLACEHOLDER})
        out.append({**msg, "content": new_content})
    return out


def count_images(messages: list[dict]) -> int:
    """统计 messages 里 image_url 内容块总数（用于 token 兜底估算等）。"""
    n = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                n += 1
    return n


def recycle_old_images(
    messages: list[dict], keep_recent: int = KEEP_RECENT_IMAGES,
) -> tuple[list[dict], int]:
    """回收旧图片：把超出保留窗口的 image_url 块替换为占位文本块。

    当前轮的图片（最后一条 assistant 之后的消息里的）受保护不回收——用户一次
    附 6 张不能刚上传就丢 1 张，窗口只约束"旧"图；旧图预算 = keep_recent 减去
    当前轮张数。图片每轮全量重发且无法被文本摘要，回收是控 context/成本的唯一
    手段。原图仍在本地图片库，不会真丢。返回 (新消息列表, 回收张数)；无需回收
    时原列表原样返回。不修改传入的消息。
    """
    last_assistant = -1
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            last_assistant = i

    old: list[tuple[int, int]] = []  # (消息下标, 内容块下标)，按出现顺序
    protected_count = 0
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for j, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "image_url":
                if i > last_assistant:
                    protected_count += 1
                else:
                    old.append((i, j))

    budget = max(0, keep_recent - protected_count)
    if len(old) <= budget:
        return messages, 0

    drop = set(old[: len(old) - budget])
    touched = {i for i, _ in drop}
    out: list[dict] = []
    for i, msg in enumerate(messages):
        if i not in touched:
            out.append(msg)
            continue
        new_content = [
            {"type": "text", "text": IMAGE_DROPPED_PLACEHOLDER} if (i, j) in drop else block
            for j, block in enumerate(msg["content"])
        ]
        out.append({**msg, "content": new_content})
    return out, len(drop)


def detect_image_mime(path: str) -> str | None:
    """探测路径是否为图片文件，返回 MIME；非图片/读不了返回 None。

    扩展名优先（零 I/O），无扩展名再读头 16 字节 magic bytes 兜底。
    用于 read_file 的图片分支判定。
    """
    p = Path(path)
    mime = _IMAGE_MIME.get(p.suffix.lower())
    if mime:
        return mime
    try:
        with open(p, "rb") as f:
            head = f.read(16)
    except OSError:
        return None
    return _sniff_mime(head)


# 工具读到图片时的结果 sentinel：工具签名是 -> str，图片放 tool 消息各家兼容性
# 不一，故约定一个 JSON 串，由 agent 循环识别后把图片转成紧随其后的 user 消息
# 注入（设计文档 §4）。read_file 单图、MCP 工具（如 Playwright 截图）可多图带文本。
# sentinel 是瞬态协议：agent 拦截后即替换为占位文本，永不落盘。
_IMAGE_SENTINEL_PREFIX = '{"__images__"'


def make_image_sentinel(images: list[tuple[str, str]], text: str = "") -> str:
    """构造图片工具结果的 sentinel JSON 串。images 为 (ref, mime) 列表。"""
    return json.dumps({
        "__images__": [{"ref": r, "mime": m} for r, m in images],
        "text": text,
    }, ensure_ascii=False)


def parse_image_sentinel(result: str) -> tuple[list[tuple[str, str]], str] | None:
    """工具结果若是图片 sentinel，返回 (图片列表, 附带文本)；否则 None（畸形不抛）。

    只认 loongimg:// 引用，任一引用非法则整体拒绝——sentinel 来自工具输出通道，
    不给注入任意 URL 留口子。
    """
    if not isinstance(result, str) or not result.startswith(_IMAGE_SENTINEL_PREFIX):
        return None
    try:
        data = json.loads(result)
        raw_images = data["__images__"]
        text = data.get("text", "")
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(raw_images, list) or not raw_images or not isinstance(text, str):
        return None
    images: list[tuple[str, str]] = []
    for item in raw_images:
        if not isinstance(item, dict):
            return None
        ref = item.get("ref")
        mime = item.get("mime", "")
        if not (isinstance(ref, str) and ref.startswith(IMAGE_REF_PREFIX) and isinstance(mime, str)):
            return None
        images.append((ref, mime))
    return images, text


def is_image_file(path: str) -> bool:
    """路径是否指向一个真实存在的图片文件（按扩展名判断）。用于 TUI 拖图折叠。"""
    p = Path(path)
    return p.suffix.lower() in _IMAGE_MIME and p.is_file()


def extract_image_paths(text: str) -> tuple[str, list[str]]:
    """从用户输入里识别"真实存在的图片文件路径"，抽出为图片列表。

    返回 (剩余文本, 路径列表)。支持双/单引号包裹或无空格的裸路径（终端拖入文件
    通常粘贴成路径）。**只在文件真实存在时才当图片**，避免把普通的 .png 提及误当附图；
    带空格的路径需加引号。用于 TUI 拖图/贴路径自动附图。
    """
    images: list[str] = []
    spans: list[tuple[int, int]] = []
    for m in _IMG_PATH_RE.finditer(text):
        path = m.group(1) or m.group(2) or m.group(3)
        candidate = path[8:] if path.lower().startswith("file:///") else path
        if Path(candidate).is_file():
            images.append(candidate)
            spans.append((m.start(), m.end()))
    if not images:
        return text, []
    cleaned = text
    for start, end in reversed(spans):
        cleaned = cleaned[:start] + cleaned[end:]
    return cleaned.strip(), images


def build_user_content(text: str, image_paths: list[str] | None):
    """构造 user 消息 content：无图返回原字符串；有图返回内容块列表（text + image_url）。

    图片存入本地图片库，消息里只放 loongimg:// 引用（发 API 时由 chat_stream
    统一内联）。空文本时省略 text 块。图片加载失败会向上抛（由调用方转成明确错误）。
    """
    if not image_paths:
        return text
    blocks: list[dict] = []
    if text:
        blocks.append({"type": "text", "text": text})
    for path in image_paths:
        blocks.append({"type": "image_url", "image_url": {"url": store_image(path)}})
    return blocks
