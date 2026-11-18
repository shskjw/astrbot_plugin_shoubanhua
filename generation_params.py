import io
import math
import re
from typing import Dict, Optional

from PIL import Image, ImageOps


IMAGE_SIZE_MAP = {
    "1K": {
        "1:1": "1024x1024", "16:9": "1280x720", "9:16": "720x1280",
        "5:4": "1040x832", "4:5": "832x1040", "4:3": "1024x768",
        "3:4": "768x1024", "3:2": "1008x672", "2:3": "672x1008",
        "21:9": "1344x576",
    },
    "2K": {
        "1:1": "2048x2048", "16:9": "2048x1152", "9:16": "1152x2048",
        "5:4": "2080x1664", "4:5": "1664x2080", "4:3": "2048x1536",
        "3:4": "1536x2048", "3:2": "2064x1376", "2:3": "1376x2064",
        "21:9": "2016x864",
    },
    "4K": {
        "1:1": "2880x2880", "16:9": "3840x2160", "9:16": "2160x3840",
        "5:4": "3200x2560", "4:5": "2560x3200", "4:3": "3264x2448",
        "3:4": "2448x3264", "3:2": "3504x2336", "2:3": "2336x3504",
        "21:9": "3808x1632",
    },
}

SUPPORTED_ASPECT_RATIOS = tuple(IMAGE_SIZE_MAP["1K"].keys())
_SIZE_TO_PARAMS = {
    size.lower(): (resolution, aspect_ratio)
    for resolution, ratios in IMAGE_SIZE_MAP.items()
    for aspect_ratio, size in ratios.items()
}

# 非标准比例没有预设尺寸，按目标面积等比推算：面积取各档常用预设定面积，
# 保证 9:20、20:9 这类比例的出图体积与同档标准比例接近。
_TARGET_AREA_BY_RESOLUTION = {
    "1K": 1024 * 1024,
    "2K": 2048 * 1536,
    "4K": 3840 * 2160,
}
_MAX_SIZE_SIDE = 4096
_SIZE_ALIGN = 16


def _parse_aspect_ratio_pair(value: Optional[str]) -> Optional[tuple[int, int]]:
    """把 ``9:20`` / ``9/20`` / ``9*20`` / ``9x20`` 这类写法解析成最简整数比。"""
    text = str(value or "").strip()
    if not text:
        return None
    for separator in ("：", "/", "x", "X", "×", "*"):
        text = text.replace(separator, ":")
    text = re.sub(r"\s+", "", text)
    match = re.fullmatch(r"(\d+):(\d+)", text)
    if not match:
        return None
    w, h = int(match.group(1)), int(match.group(2))
    if w <= 0 or h <= 0:
        return None
    ratio = w / h
    if ratio < 0.1 or ratio > 10.0:
        return None
    divisor = math.gcd(w, h)
    return (w // divisor, h // divisor)


def normalize_resolution(value: Optional[str], default: str = "1K") -> str:
    text = str(value or "").strip().upper().replace(" ", "")
    match = re.fullmatch(r"([124])K", text)
    if match:
        return f"{match.group(1)}K"
    normalized_default = str(default or "1K").strip().upper().replace(" ", "")
    return normalized_default if normalized_default in IMAGE_SIZE_MAP else "1K"


def normalize_aspect_ratio(value: Optional[str], default: str = "1:1") -> str:
    """归一化为最简 ``W:H``；保留 9:20 这类非标准比例，非法值回退默认比例。"""
    parsed = _parse_aspect_ratio_pair(value)
    if parsed:
        return f"{parsed[0]}:{parsed[1]}"
    parsed_default = _parse_aspect_ratio_pair(default)
    return f"{parsed_default[0]}:{parsed_default[1]}" if parsed_default else "1:1"


def find_closest_standard_aspect_ratio(aspect_ratio: str) -> str:
    """把任意比例吸附到最接近的预设比例。

    Gemini 的 ``imageConfig.aspectRatio`` 只接受枚举值，发送前需要用它兜底。
    """
    parsed = _parse_aspect_ratio_pair(aspect_ratio)
    if not parsed:
        return "1:1"
    actual_ratio = parsed[0] / parsed[1]
    return min(
        SUPPORTED_ASPECT_RATIOS,
        key=lambda item: abs(actual_ratio - (int(item.split(":")[0]) / int(item.split(":")[1]))),
    )


def compute_size_for_aspect_ratio(resolution: str, aspect_ratio: str) -> str:
    """返回该分辨率下的像素尺寸。

    预设比例直接用映射表；非标准比例按目标面积等比计算，宽高取 16 的整数倍，
    长边不超过 4096，避免 9:20 这类竖图被改成更宽的比例。
    """
    norm_res = normalize_resolution(resolution)
    norm_ratio = normalize_aspect_ratio(aspect_ratio)
    preset_map = IMAGE_SIZE_MAP.get(norm_res, IMAGE_SIZE_MAP["1K"])
    if norm_ratio in preset_map:
        return preset_map[norm_ratio]

    parsed = _parse_aspect_ratio_pair(norm_ratio)
    if not parsed:
        return preset_map["1:1"]
    rw, rh = parsed
    target_area = _TARGET_AREA_BY_RESOLUTION.get(norm_res, 1024 * 1024)

    raw_w = math.sqrt(target_area * (rw / rh))
    raw_h = raw_w * rh / rw

    # 两边使用同一缩放系数，不能单独截断长边，否则竖幅比例会被改掉。
    scale = min(1.0, _MAX_SIZE_SIDE / max(raw_w, raw_h))
    width = max(256, int(round((raw_w * scale) / _SIZE_ALIGN)) * _SIZE_ALIGN)
    height = max(256, int(round((raw_h * scale) / _SIZE_ALIGN)) * _SIZE_ALIGN)
    return f"{width}x{height}"


def detect_aspect_ratio_from_image(image_data: bytes, default: str = "1:1") -> str:
    """读取图片尺寸，返回最简宽高比。

    这里保留原图的真实比例（如手机长截图 9:20），不再吸附到预设比例，
    否则图生图会按错误比例生成，画面被拉伸。
    """
    if not image_data:
        return normalize_aspect_ratio(default)
    try:
        with Image.open(io.BytesIO(image_data)) as image:
            width, height = ImageOps.exif_transpose(image).size
        if width <= 0 or height <= 0:
            return normalize_aspect_ratio(default)

        parsed = _parse_aspect_ratio_pair(f"{width}:{height}")
        return f"{parsed[0]}:{parsed[1]}" if parsed else normalize_aspect_ratio(default)
    except Exception:
        return normalize_aspect_ratio(default)


def resolve_image_generation_params(
    prompt: str,
    default_resolution: str = "1K",
    default_aspect_ratio: str = "1:1",
    resolution: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
) -> Dict[str, str]:
    """解析图片分辨率、比例及接口 size。

    优先级：显式参数 > 提示词 > 配置默认值。显式参数接受任意 ``W:H``；
    提示词解析保持向后兼容，只识别预设比例和映射表中的精确像素尺寸
    （如 3840x2160 → 4K + 16:9），避免把 ``20:30`` 这类文本误判成画幅。
    """
    text = str(prompt or "")
    detected_resolution = None
    detected_ratio = None

    dimension_match = re.search(r"(?<!\d)(\d{3,4})\s*[xX×*]\s*(\d{3,4})(?!\d)", text)
    if dimension_match:
        dimension = f"{dimension_match.group(1)}x{dimension_match.group(2)}".lower()
        if dimension in _SIZE_TO_PARAMS:
            detected_resolution, detected_ratio = _SIZE_TO_PARAMS[dimension]

    ratio_match = re.search(
        r"(?<!\d)(21|16|9|5|4|3|2|1)\s*[:：/xX×*]\s*(16|9|5|4|3|2|1)(?!\d)",
        text,
    )
    if ratio_match:
        candidate = f"{ratio_match.group(1)}:{ratio_match.group(2)}"
        if candidate in SUPPORTED_ASPECT_RATIOS:
            detected_ratio = candidate

    resolution_matches = re.findall(r"(?<![A-Za-z0-9])([124])\s*[kK](?![A-Za-z0-9])", text)
    if resolution_matches:
        detected_resolution = f"{resolution_matches[-1]}K"

    final_resolution = normalize_resolution(resolution, detected_resolution or default_resolution)
    if not resolution:
        final_resolution = normalize_resolution(detected_resolution, default_resolution)

    final_ratio = normalize_aspect_ratio(aspect_ratio, detected_ratio or default_aspect_ratio)
    if not aspect_ratio:
        final_ratio = normalize_aspect_ratio(detected_ratio, default_aspect_ratio)

    return {
        "resolution": final_resolution,
        "aspect_ratio": final_ratio,
        "size": compute_size_for_aspect_ratio(final_resolution, final_ratio),
    }
