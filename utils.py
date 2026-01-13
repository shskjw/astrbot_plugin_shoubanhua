import re
from typing import Any, List


def norm_id(raw_id: Any) -> str:
    """标准化 ID 为字符串"""
    if raw_id is None:
        return ""
    return str(raw_id).strip()


def extract_image_urls_from_text(text: str) -> List[str]:
    """从文本中提取图片链接和本地文件路径"""
    image_urls = []

    # 本地文件路径 (Windows)
    local_patterns = [r'[a-zA-Z]:\\[^\s,，。！？\n]+\.(?:jpg|jpeg|png|gif|bmp|webp)']
    for pattern in local_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if match and match not in image_urls:
                image_urls.append(match)

    # 网络 URL
    url_patterns = [
        r'https?://[^\s<>"\'\)]+\.(?:jpg|jpeg|png|gif|bmp|webp)(?:\?[^\s<>"\'\)]*)?(?=[\s<>"\'\)|$])',
        r'https?://[^\s<>"\'\)]+/(?:s\d+/|upload/|image/|img/|pic/)[^\s<>"\'\)]+\.(?:jpg|jpeg|png|gif|bmp|webp)(?:\?[^\s<>"\'\)]*)?(?=[\s<>"\'\)|$])',
        r'https?://youke\d+\.picui\.cn/[^\s<>"\'\)]+\.(?:jpg|jpeg|png|gif|bmp|webp)(?:\?[^\s<>"\'\)]*)?(?=[\s<>"\'\)|$])'
    ]
    for pattern in url_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if match and match not in image_urls:
                image_urls.append(match)

    return image_urls