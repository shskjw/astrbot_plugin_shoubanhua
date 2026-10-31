import asyncio
import base64
import ipaddress
import json
import re
import socket
from urllib.parse import quote
from urllib.parse import urljoin
from urllib.parse import urlparse
import aiohttp
from typing import List, Dict
from astrbot import logger
from .generation_params import detect_aspect_ratio_from_image, resolve_image_generation_params
from .utils import normalize_api_root


class ApiManager:
    def __init__(self, config: dict):
        self.config = config
        self.key_lock = asyncio.Lock()
        self.generic_idx = 0
        self.gemini_idx = 0
        self.unified_idx = 0
        self._session = None # 保持 Session 持久化，复用 TCP/SSL 连接
        self._last_metrics = {}
        self._last_download_metrics = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # 不在 Session 级别设置 Timeout，改在请求级别设置
            self._session = aiohttp.ClientSession()
        return self._session

    def _reset_metrics(self):
        self._last_metrics = {
            "upstream_duration": 0.0,
            "download_duration": 0.0,
            "total_duration": 0.0,
            "download_route": "",
        }
        self._last_download_metrics = {
            "download_duration": 0.0,
            "download_route": "",
        }

    def get_last_metrics(self) -> Dict:
        return dict(self._last_metrics or {})

    def _normalize_call_api_args(self, legacy_use_power_or_proxy=None, proxy=None, use_text_to_image_api: bool = False):
        """兼容旧版 call_api 调用签名：
        - 新版: call_api(images, prompt, model, proxy, use_text_to_image_api=...)
        - 旧版: call_api(images, prompt, model, False, proxy)
        """
        resolved_proxy = proxy
        resolved_use_text_to_image_api = bool(use_text_to_image_api)

        if isinstance(legacy_use_power_or_proxy, bool):
            # 旧版 use_power 参数，现已废弃，直接忽略
            pass
        else:
            if resolved_proxy is None:
                resolved_proxy = legacy_use_power_or_proxy

        if isinstance(resolved_proxy, bool):
            resolved_proxy = None

        if resolved_proxy is not None:
            resolved_proxy = str(resolved_proxy).strip() or None

        return resolved_proxy, resolved_use_text_to_image_api

    def _get_luxury_request_count(self) -> int:
        """获取奢侈模式并发请求数。"""
        try:
            count = int(self.config.get("luxury_request_count", 3) or 3)
        except Exception:
            count = 3
        return max(1, count)

    async def _call_api_with_luxury_mode(self, images: List[bytes], prompt: str,
                                         model: str, proxy: str = None,
                                         use_text_to_image_api: bool = False,
                                         aspect_ratio: str = None,
                                         resolution: str = None) -> bytes | str:
        """奢侈模式：同一请求并发多次，只取首个成功结果，其余丢弃。"""
        luxury_count = self._get_luxury_request_count()
        if luxury_count <= 1:
            return await self._call_api_once(
                images, prompt, model, proxy, use_text_to_image_api,
                aspect_ratio=aspect_ratio, resolution=resolution
            )

        logger.info(f"奢侈模式已启用：同一请求并发 {luxury_count} 次，仅取其中一张成功图片")

        tasks = [
            asyncio.create_task(self._call_api_once(
                images, prompt, model, proxy, use_text_to_image_api,
                aspect_ratio=aspect_ratio, resolution=resolution
            ))
            for _ in range(luxury_count)
        ]

        first_error = "奢侈模式下所有并发请求均失败"
        success_result = None

        try:
            for completed in asyncio.as_completed(tasks):
                try:
                    result = await completed
                except Exception as e:
                    result = f"系统错误: {e}"

                if isinstance(result, bytes) and result:
                    success_result = result
                    break

                if isinstance(result, str) and result and first_error == "奢侈模式下所有并发请求均失败":
                    first_error = result
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        if success_result is not None:
            return success_result
        return first_error

    def _should_bypass_proxy(self, url: str) -> bool:
        """本地/内网地址不走代理，避免请求本地中转时反而绕远路。"""
        if not url:
            return False

        try:
            host = (urlparse(url).hostname or "").strip().lower()
        except Exception:
            return False

        if not host:
            return False

        if host in {"localhost", "127.0.0.1", "::1"}:
            return True

        if host.endswith(".local") or host.endswith(".lan"):
            return True

        try:
            ip_obj = ipaddress.ip_address(host)
            return any([
                ip_obj.is_private,
                ip_obj.is_loopback,
                ip_obj.is_link_local,
                ip_obj.is_reserved,
            ])
        except ValueError:
            return False

    def _get_request_proxy(self, url: str, proxy: str = None) -> str | None:
        if not proxy:
            return None
        if self._should_bypass_proxy(url):
            logger.info(f"检测到本地/内网地址，自动绕过代理: {url[:120]}")
            return None
        return proxy

    @staticmethod
    def _normalize_keys(keys) -> List[str]:
        """兼容配置面板的多行文本、逗号分隔文本和旧版 list。"""
        if not keys:
            return []
        if isinstance(keys, str):
            return [item.strip() for item in re.split(r"[\r\n,]+", keys) if item.strip()]
        if isinstance(keys, (list, tuple, set)):
            return [str(item).strip() for item in keys if str(item).strip()]
        return [str(keys).strip()] if str(keys).strip() else []

    def _summarize_api_error(self, error, key: str = "", max_length: int = 500) -> str:
        """保留排错所需的错误摘要，隐藏凭证与图片数据。"""
        text = str(error or "")
        text = re.sub(r"data:image/[^,\s]+,[A-Za-z0-9+/=_\-\s]+", "<image data>", text, flags=re.I)
        text = re.sub(r"[A-Za-z0-9+/_-]{160,}={0,2}", "<binary data>", text)
        keys = [key] if key else []
        for field in ("api_keys", "generic_api_keys", "gemini_api_keys", "text_to_image_api_keys"):
            keys.extend(self._normalize_keys(self.config.get(field)))
        for secret in sorted(set(keys), key=len, reverse=True):
            if secret:
                text = text.replace(secret, "<redacted>")
        text = re.sub(r"(?i)(bearer\s+)\S+", r"\1<redacted>", text)
        text = re.sub(
            r'''(?i)(["']?(?:api[_-]?key|access[_-]?token|token|key|authorization)["']?\s*[:=]\s*["']?)[^\s&"',}]+''',
            r"\1<redacted>", text,
        )
        return " ".join(text.split())[:max_length]

    def _get_interface_mode(self) -> str:
        """读取新版接口模式，并兼容旧版 generic 配置。"""
        mode = str(self.config.get("interface_mode", "") or "").strip().lower()
        mode_aliases = {
            "image": "openai_image",
            "images": "openai_image",
            "openai_images": "openai_image",
            "chat": "openai_chat",
            "generic": "openai_chat",
            "gemini": "gemini_official",
            "custom": "custom_endpoint",
        }
        mode = mode_aliases.get(mode, mode)
        valid_modes = {"openai_image", "openai_chat", "gemini_official", "custom_endpoint"}
        if mode in valid_modes:
            return mode

        legacy_mode = str(self.config.get("api_mode", "generic") or "generic").strip().lower()
        if legacy_mode == "gemini_official":
            return "gemini_official"
        if legacy_mode == "openai_image":
            return "openai_image"
        if legacy_mode == "custom_endpoint":
            return "custom_endpoint"
        return "openai_image" if self.config.get("generic_prefer_images_api", False) else "openai_chat"

    def _should_force_images_api(self, interface_mode: str, custom_kind: str = "") -> bool:
        """Whether the selected configuration makes Images API a hard route.

        ``generic_prefer_images_api`` historically behaved as a best-effort hint: it
        silently switched multi-image edits (and unsupported edits endpoints) back
        to chat/completions. That made an explicitly selected image route impossible
        to rely on. Treat both OpenAI Images mode and this compatibility switch as
        routing constraints instead.
        """
        return (
            interface_mode == "openai_image"
            or custom_kind == "image"
            or (
                interface_mode == "openai_chat"
                and bool(self.config.get("generic_prefer_images_api", False))
            )
        )

    def _get_base_url(self, mode: str, use_text_to_image_api: bool = False) -> str:
        unified_base = str(self.config.get("base_url", "") or "").strip()
        if unified_base:
            return unified_base

        if use_text_to_image_api and self.config.get("text_to_image_api_url"):
            return str(self.config.get("text_to_image_api_url") or "").strip()

        if mode == "gemini_official":
            legacy_url = str(self.config.get("gemini_api_url", "") or "").strip()
            return legacy_url or "https://generativelanguage.googleapis.com"
        return str(self.config.get("generic_api_url", "") or "").strip()

    async def get_key(self, mode: str, use_text_to_image_api: bool = False) -> str | None:
        """获取轮询 Key"""
        async with self.key_lock:
            keys = self._normalize_keys(self.config.get("api_keys", []))
            if keys:
                k = keys[self.unified_idx % len(keys)]
                self.unified_idx += 1
                return k

            if use_text_to_image_api:
                keys = self._normalize_keys(self.config.get("text_to_image_api_keys", []))
                if keys:
                    k = keys[self.unified_idx % len(keys)]
                    self.unified_idx += 1
                    return k

            if mode == "gemini_official":
                keys = self._normalize_keys(self.config.get("gemini_api_keys", []))

                if not keys:
                    return None
                k = keys[self.gemini_idx % len(keys)]
                self.gemini_idx += 1
                return k
            else:
                keys = self._normalize_keys(self.config.get("generic_api_keys", []))

                if not keys:
                    return None
                k = keys[self.generic_idx % len(keys)]
                self.generic_idx += 1
                return k

    def extract_image_url(self, data: Dict) -> str | None:
        """解析各种奇怪的 API 返回格式"""
        try:
            # ================== 1. OpenAI DALL-E Standard ==================
            # 格式: {"data": [{"url": "..."}, {"b64_json": "..."}]}
            if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                item = data["data"][0]
                if isinstance(item, dict):
                    encoded_image = item.get("b64_json")
                    if isinstance(encoded_image, str) and encoded_image.strip():
                        return f"data:image/png;base64,{encoded_image}"
                    url_value = item.get("url")
                    if isinstance(url_value, str) and url_value.strip():
                        # 兼容某些接口把纯 base64 误放进 url 字段
                        pure_b64_match = re.fullmatch(r'[A-Za-z0-9+/]{1000,}={0,2}', url_value.strip())
                        if pure_b64_match:
                            return f"data:image/jpeg;base64,{url_value.strip()}"
                        return url_value.strip()

            # ================== 2. OpenAI Chat Completion ==================
            # 格式: {"choices": [{"message": {"content": "..."}}]}
            content = None
            if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                choice = data["choices"][0]
                message = choice.get("message", {})
                
                # 优先 1: 检查非标准 "images" 字段 (某些 OneAPI/中转站实现)
                if "images" in message and isinstance(message["images"], list) and len(message["images"]) > 0:
                    img = message["images"][0]
                    if isinstance(img, str): return img # 可能是 URL
                    if isinstance(img, dict):
                        if "url" in img: return img["url"]
                        # 修复: 兼容 {"type": "image_url", "image_url": {"url": "..."}} 结构
                        if "image_url" in img:
                             if isinstance(img["image_url"], str): return img["image_url"]
                             if isinstance(img["image_url"], dict) and "url" in img["image_url"]:
                                 return img["image_url"]["url"]

                # 优先 2: 检查 tool_calls (Function Calling 格式)
                if "tool_calls" in message and isinstance(message["tool_calls"], list):
                    for tool in message["tool_calls"]:
                        func_args = tool.get("function", {}).get("arguments", "")
                        
                        # 尝试0: 解析 JSON，某些模型会把结构放在深处
                        try:
                            args = json.loads(func_args)
                            # 常见的参数名: url, image_url, images, file_url
                            for k in ["url", "image_url", "file_url", "link", "b64_json", "image", "data", "image_data"]:
                                if k in args:
                                    value = args[k]
                                    if k == "b64_json" and isinstance(value, str):
                                        return f"data:image/png;base64,{value}"
                                    if isinstance(value, str):
                                        pure_b64_match = re.fullmatch(r'[A-Za-z0-9+/]{1000,}={0,2}', value.strip())
                                        if pure_b64_match:
                                            return f"data:image/jpeg;base64,{value.strip()}"
                                    return value
                            
                            # 尝试深度遍历寻找 base64 或 url
                            # 有的API返回 {"response": {"url": "..."}}
                            def find_url(d):
                                if isinstance(d, dict):
                                    for k, v in d.items():
                                        if k in ["url", "image_url", "b64_json", "image", "data", "link"]:
                                            if isinstance(v, str):
                                                if k == "b64_json":
                                                    return f"data:image/png;base64,{v}"
                                                pure_b64_match = re.fullmatch(r'[A-Za-z0-9+/]{1000,}={0,2}', v.strip())
                                                if pure_b64_match:
                                                    return f"data:image/jpeg;base64,{v.strip()}"
                                                return v
                                        res = find_url(v)
                                        if res: return res
                                elif isinstance(d, list):
                                    for item in d:
                                        res = find_url(item)
                                        if res: return res
                                return None
                            
                            deep_url = find_url(args)
                            if deep_url: return deep_url
                        except:
                            pass

                        # 尝试1: 字符串正则查找 base64 (优先找base64，因为它最明确)
                        if "base64" in func_args:
                            b64_match = re.search(r'(data:image\/[\w\-\+\.]+(?:;base64)?,[\w\-\+\/=\s]+)', func_args)
                            if b64_match:
                                found_b64 = b64_match.group(1).replace("\n", "").replace("\r", "").replace(" ", "").replace("\\", "")
                                return found_b64
                                
                        # 尝试2: 纯 base64 无前缀
                        pure_b64_match = re.search(r'"([A-Za-z0-9+/]{1000,}={0,2})"', func_args)
                        if pure_b64_match:
                            return f"data:image/png;base64,{pure_b64_match.group(1)}"

                        # 尝试3: 字符串正则直接解析 url/https
                        if "http" in func_args:
                            urls = re.findall(r"(https?://[^\s<>\"'()\[\]\\]+)", func_args)
                            if urls: return urls[0].strip()

                # 优先 3: 标准 content
                if "content" in message:
                    content = message["content"]
                elif "text" in choice: # Legacy completion
                    content = choice["text"]

            # ================== 3. Google Gemini Official ==================
            # 格式: {"candidates": [{"content": {"parts": [{"inlineData": ...}, {"text": ...}]}}]}
            if "candidates" in data and isinstance(data["candidates"], list) and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if parts and isinstance(parts, list):
                        # 优先找 inlineData
                        for part in parts:
                            if "inlineData" in part:
                                mime = part["inlineData"].get("mimeType", "image/png")
                                d = part["inlineData"].get("data", "")
                                return f"data:{mime};base64,{d}"
                        
                        # 其次找 text 里的链接
                        texts = [p.get("text", "") for p in parts if "text" in p]
                        content = "\n".join(texts)

            # ================== Common Content Extraction ==================
            # 如果从 ChatCompletion 或 Gemini Text 中提取到了文本内容，尝试解析 URL 或 Base64
            if content:
                # 0. 尝试提取其中包含的 data URI (最宽泛的匹配策略)
                # 能够匹配 markdown 内部、纯文本、或者被截断的内容
                # [Fix] 增强正则兼容性: 允许 urlsafe base64 (-_), 允许 mime type 包含特殊字符
                b64_match = re.search(r'(data:image\/[\w\-\+\.]+(?:;base64)?,[\w\-\+\/=\s]+)', content)
                if b64_match:
                    found_b64 = b64_match.group(1).replace("\n", "").replace("\r", "").replace(" ", "")
                    # 简单的有效性检查 (Base64 长度通常较长)
                    if len(found_b64) > 100:
                        return found_b64

                # 0.5 尝试匹配纯 base64 图片内容
                pure_b64_match = re.search(r'([A-Za-z0-9+/]{1000,}={0,2})', content)
                if pure_b64_match:
                    return f"data:image/jpeg;base64,{pure_b64_match.group(1)}"

                # 1. 尝试匹配 Markdown 图片语法 ![...](url) - 这种更精确
                # 匹配 ![description](http...)
                md_match = re.search(r'!\[.*?\]\((.*?)\)', content, re.DOTALL)
                if md_match:
                    url_part = md_match.group(1).strip()
                    # 去除可能存在的 <> 包裹 (e.g. ![img](<url>))
                    url_part = url_part.lstrip("<").rstrip(">")
                    # [Fix] 清理 URL
                    url_part = url_part.strip("'\"").replace("\n", "").replace("\r", "").replace(" ", "")
                    if "data:image" not in url_part and len(url_part) > 5:
                         return url_part

                # 2. (Legacy B64 Match Removed - replaced by step 0)

                # 3. 尝试匹配 HTTP/HTTPS URL
                # 这是一个比较宽泛的匹配
                url_match = re.search(r'(https?://[^\s<>")\]]+)', content)
                if url_match:
                    # 去掉末尾可能的标点
                    return url_match.group(1).rstrip(")>,'\".")

        except Exception as e:
            logger.error(f"Error parsing API response: {e}")
        return None

    def get_mime_type(self, data: bytes) -> str:
        """简单的 MIME 类型检测"""
        if data.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'image/png'
        elif data.startswith(b'\xff\xd8'):
            return 'image/jpeg'
        elif data.startswith(b'GIF87a') or data.startswith(b'GIF89a'):
            return 'image/gif'
        elif data.startswith(b'RIFF') and data[8:12] == b'WEBP':
            return 'image/webp'
        return 'image/png' # 默认

    def _normalize_generic_chat_url(self, base_url: str) -> str:
        """忽略用户填写的版本/接口尾部，按 OpenAI Chat 模式统一补全。"""
        original_url = str(base_url or "").strip().rstrip("/")
        root = normalize_api_root(original_url)
        if not root:
            return ""

        normalized = f"{root}/v1/chat/completions"
        if normalized != original_url:
            logger.info(f"Generic API 地址已规范化为: {normalized}")
        return normalized

    def _convert_to_images_api_url(self, chat_url: str, has_input_image: bool = False) -> str:
        """忽略用户填写的版本/接口尾部，按 OpenAI Images 模式统一补全。"""
        root = normalize_api_root(chat_url)
        if not root:
            return ""
        endpoint = "images/edits" if has_input_image else "images/generations"
        return f"{root}/v1/{endpoint}"

    def _build_gemini_api_url(self, base_url: str, model: str) -> str:
        """忽略用户填写的版本/接口尾部，按 Gemini 官方模式统一补全。"""
        root = normalize_api_root(base_url)
        if not root:
            return ""

        model_id = str(model or "").strip()
        if model_id.lower().startswith("models/"):
            model_id = model_id.split("/", 1)[1]
        return f"{root}/v1beta/models/{quote(model_id, safe='-._~')}:generateContent"

    def _build_candidate_generic_chat_urls(self, base_url: str) -> List[str]:
        """按所选 OpenAI Chat 模式构造唯一请求地址。"""
        normalized = self._normalize_generic_chat_url(base_url)
        return [normalized] if normalized else []

    def _build_candidate_generic_image_urls(self, base_url: str, has_input_image: bool = False) -> List[str]:
        """按所选 OpenAI Images 模式构造唯一请求地址。"""
        normalized = self._convert_to_images_api_url(base_url, has_input_image=has_input_image)
        return [normalized] if normalized else []

    def _should_retry_images_api_with_multipart(self, error_msg: str, has_input_image: bool = False) -> bool:
        """判断 Images API 是否需要回退为 multipart/form-data 方式"""
        if not has_input_image:
            return False
        if self._is_images_edits_unsupported_error(error_msg):
            return False
        error_lower = (error_msg or "").lower()
        keywords = [
            "multipart",
            "form-data",
            "unsupported media type",
            "image must be a file",
            "file upload",
            "use multipart",
            "expected uploadfile",
            "expected file"
        ]
        return any(keyword in error_lower for keyword in keywords)

    def _is_images_edits_unsupported_error(self, error_msg: str) -> bool:
        """Detect providers that do not implement /images/edits at all."""
        error_lower = (error_msg or "").lower()
        keywords = [
            "not support this api",
            "api not supported",
            "unsupported endpoint",
            "unsupported api",
            "暂不支持该接口",
            "不支持该接口",
            "该接口暂不支持",
            "接口不支持",
        ]
        return any(keyword in error_lower for keyword in keywords)

    async def _parse_images_api_success_response(self, resp_text: str, proxy: str = None,
                                                base_url: str = None) -> bytes | str:
        """统一解析 Images API 成功响应"""
        try:
            res_data = json.loads(resp_text)
        except json.JSONDecodeError:
            return f"数据解析失败: 返回内容不是 JSON. 内容: {resp_text[:100]}..."

        if isinstance(res_data, dict) and "error" in res_data:
            return json.dumps(res_data["error"], ensure_ascii=False)

        img_url = self.extract_image_url(res_data)
        if not img_url:
            return f"Images API 返回成功但未找到图片数据。Raw: {str(res_data)[:200]}..."

        if img_url.startswith("data:"):
            try:
                encoded_image = re.sub(r"\s+", "", img_url.split(",", 1)[-1])
                image_data = base64.b64decode(encoded_image, altchars=b"-_", validate=True)
            except ValueError:
                return "Images API 返回了无效的 base64 图片数据"
            return image_data if image_data else "Images API 返回了空的图片数据"

        return await self._download_result_image(img_url, proxy, base_url)

    async def _call_images_api_multipart(self, images: List[bytes], prompt: str,
                                         model: str, key: str, base_url: str, proxy: str = None,
                                         generation_params: Dict = None,
                                         exact_endpoint: bool = False) -> bytes | str:
        """以 multipart/form-data 上传参考图，直接调用 Images 编辑接口。"""
        has_input_image = bool(images)
        candidate_urls = [base_url.rstrip("/")] if exact_endpoint else self._build_candidate_generic_image_urls(
            base_url, has_input_image=has_input_image
        )
        logger.info(f"Using Images API multipart/form-data: {self._summarize_api_error(candidate_urls, key)}")

        headers = {
            "Authorization": f"Bearer {key}"
        }

        generation_params = generation_params or resolve_image_generation_params(
            prompt, self.config.get("image_resolution", "1K")
        )
        res_set = generation_params["resolution"]
        final_prompt = f"(Masterpiece, Best Quality, {res_set} Resolution), {prompt}" if res_set != "1K" else prompt

        timeout_val = self.config.get("timeout", 120)
        timeout = aiohttp.ClientTimeout(total=timeout_val)
        session = await self._get_session()

        try:
            for idx, url in enumerate(candidate_urls):
                form = aiohttp.FormData()
                form.add_field("model", model)
                form.add_field("prompt", final_prompt)
                form.add_field("n", "1")
                form.add_field("size", generation_params["size"])
                if str(model).strip().lower().startswith("dall-e"):
                    form.add_field("response_format", "b64_json")

                if images:
                    # OpenAI-compatible edits accepts one ``image`` field for a
                    # single input and repeated ``image[]`` fields for multiple
                    # references. Do not discard extra references or reroute them
                    # through chat/completions.
                    field_name = "image" if len(images) == 1 else "image[]"
                    for image_index, img in enumerate(images):
                        mime = self.get_mime_type(img)
                        ext = mime.split("/")[-1] if "/" in mime else "png"
                        filename = f"input_{image_index + 1}.{ext}"
                        form.add_field(field_name, img, filename=filename, content_type=mime)

                current_proxy = self._get_request_proxy(url, proxy)
                async with session.post(url, data=form, headers=headers, proxy=current_proxy, timeout=timeout) as resp:
                    resp_text = await resp.text()

                    if "<html" in resp_text.lower() and idx < len(candidate_urls) - 1:
                        logger.warning(f"Images API multipart 返回 HTML 页面，尝试下一个候选地址: {candidate_urls[idx + 1]}")
                        continue

                    if resp.status != 200:
                        err_msg = self._summarize_api_error(resp_text, key)
                        safe_url = self._summarize_api_error(url, key)
                        logger.error(f"Images API multipart HTTP {resp.status}: {err_msg} | URL: {safe_url}")
                        if has_input_image and self._is_images_edits_unsupported_error(err_msg):
                            return f"Images API edits endpoint unsupported {resp.status}: {err_msg} | URL: {safe_url}"
                        return f"Images API Multipart Error {resp.status}: {err_msg} | URL: {safe_url}"

                    if "<html" in resp_text.lower():
                        error = f"HTTP 200: 服务端返回了网页而非图片接口数据 | URL: {self._summarize_api_error(url, key)}"
                        logger.error(error)
                        return error

                    result = await self._parse_images_api_success_response(resp_text, proxy, url)
                    if isinstance(result, str):
                        logger.error(f"Images API multipart 响应未取得图片: {self._summarize_api_error(result, key)}")
                    return result

            return f"Images API Multipart Error: 未找到可用接口地址 | Candidates: {candidate_urls}"
        except asyncio.TimeoutError:
            logger.error(f"Images API multipart 请求超时 ({timeout_val}s)")
            return f"请求超时 ({timeout_val}s)，请稍后再试或检查网络。"
        except Exception as e:
            import traceback
            logger.error(f"Images API Multipart Call Error: {self._summarize_api_error(traceback.format_exc(), key)}")
            err_msg = str(e) or type(e).__name__
            return f"系统错误: {err_msg}"

    async def call_images_api(self, images: List[bytes], prompt: str,
                               model: str, key: str, base_url: str, proxy: str = None,
                               generation_params: Dict = None,
                               exact_endpoint: bool = False) -> bytes | str:
        """Images 编辑接口直接上传文件，文生图和自定义 JSON 接口发送 JSON。"""

        has_input_image = bool(images)
        endpoint_path = urlparse(base_url).path.rstrip("/").lower()
        if has_input_image and (not exact_endpoint or endpoint_path.endswith("/images/edits")):
            return await self._call_images_api_multipart(
                images, prompt, model, key, base_url, proxy,
                generation_params=generation_params,
                exact_endpoint=exact_endpoint,
            )
        if exact_endpoint and endpoint_path.endswith("/images/edits") and not has_input_image:
            return "Images API /images/edits 需要至少一张参考图"

        candidate_urls = [base_url.rstrip("/")] if exact_endpoint else self._build_candidate_generic_image_urls(
            base_url, has_input_image=has_input_image
        )
        logger.info(f"Using Images API JSON: {self._summarize_api_error(candidate_urls, key)}")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }

        # 画质强化 Prompt
        generation_params = generation_params or resolve_image_generation_params(
            prompt, self.config.get("image_resolution", "1K")
        )
        res_set = generation_params["resolution"]
        final_prompt = f"(Masterpiece, Best Quality, {res_set} Resolution), {prompt}" if res_set != "1K" else prompt

        # 构造 Images API 请求
        payload = {
            "model": model,
            "prompt": final_prompt,
            "n": 1,
            "size": generation_params["size"],
        }
        if str(model).strip().lower().startswith("dall-e"):
            payload["response_format"] = "b64_json"

        # 如果有输入图片，兼容不同 Images API 的字段要求
        # 一些服务要求 image_url；另一些只认 image / input_image / images
        if images:
            data_uris = []
            for img in images:
                mime = self.get_mime_type(img)
                b64_img = base64.b64encode(img).decode()
                data_uris.append(f"data:{mime};base64,{b64_img}")

            # Keep singular aliases for compatibility while preserving every
            # reference in the standard plural field.
            payload["image"] = data_uris[0]
            payload["image_url"] = data_uris[0]
            payload["input_image"] = data_uris[0]
            payload["images"] = data_uris

        try:
            timeout_val = self.config.get("timeout", 120)
            timeout = aiohttp.ClientTimeout(total=timeout_val)
            session = await self._get_session()

            for idx, url in enumerate(candidate_urls):
                current_proxy = self._get_request_proxy(url, proxy)
                async with session.post(url, json=payload, headers=headers, proxy=current_proxy, timeout=timeout) as resp:
                    resp_text = await resp.text()

                    if "<html" in resp_text.lower() and idx < len(candidate_urls) - 1:
                        logger.warning(f"Images API 返回 HTML 页面，尝试下一个候选地址: {candidate_urls[idx + 1]}")
                        continue

                    if resp.status != 200:
                        err_msg = resp_text
                        try:
                            err_json = json.loads(resp_text)
                            if "error" in err_json:
                                err_msg = json.dumps(err_json["error"], ensure_ascii=False)
                            else:
                                err_msg = json.dumps(err_json, ensure_ascii=False)
                        except:
                            pass

                        err_msg = self._summarize_api_error(err_msg, key)
                        safe_url = self._summarize_api_error(url, key)
                        logger.error(f"Images API JSON HTTP {resp.status}: {err_msg} | URL: {safe_url}")

                        if has_input_image and self._is_images_edits_unsupported_error(err_msg):
                            if idx < len(candidate_urls) - 1:
                                logger.warning(f"Images API edits endpoint unsupported, trying next candidate: {candidate_urls[idx + 1]}")
                                continue
                            return f"Images API edits endpoint unsupported {resp.status}: {err_msg} | URL: {safe_url}"

                        if self._should_retry_images_api_with_multipart(err_msg, has_input_image):
                            logger.info("Images API JSON 请求失败，检测到服务端更偏好 multipart/form-data，自动重试")
                            return await self._call_images_api_multipart(
                                images, prompt, model, key, base_url, proxy,
                                generation_params=generation_params,
                                exact_endpoint=exact_endpoint,
                            )

                        return f"Images API Error {resp.status}: {err_msg} | URL: {safe_url}"

                    if "<html" in resp_text.lower():
                        error = f"HTTP 200: 服务端返回了网页而非图片接口数据 | URL: {self._summarize_api_error(url, key)}"
                        logger.error(error)
                        return error

                    result = await self._parse_images_api_success_response(resp_text, proxy, url)
                    if isinstance(result, str):
                        logger.error(f"Images API JSON 响应未取得图片: {self._summarize_api_error(result, key)}")
                    return result

            return f"Images API Error: 未找到可用接口地址 | Candidates: {candidate_urls}"

        except asyncio.TimeoutError:
            timeout_val = self.config.get("timeout", 120)
            logger.error(f"Images API JSON 请求超时 ({timeout_val}s)")
            return f"请求超时 ({timeout_val}s)，请稍后再试或检查网络。"
        except Exception as e:
            import traceback
            logger.error(f"Images API Call Error: {self._summarize_api_error(traceback.format_exc(), key)}")
            err_msg = str(e) or type(e).__name__
            return f"系统错误: {err_msg}"

    def _is_chat_not_supported_error(self, error_msg: str) -> bool:
        """检查是否是 chat completions 不支持的错误"""
        error_lower = error_msg.lower()
        return any(keyword in error_lower for keyword in [
            "does not support chat completions",
            "not support chat",
            "chat completions not supported",
            "use images api",
            "images/generations",
            "not a chat model",
            "image generation model"
        ])

    def _should_fallback_to_images_api(self, error_msg: str, has_input_image: bool = False) -> bool:
        """检查是否应切换到 Images API，包括部分图片编辑/生图兼容报错"""
        error_lower = (error_msg or "").lower()
        fallback_keywords = [
            "does not support chat completions",
            "not support chat",
            "chat completions not supported",
            "use images api",
            "images/generations",
            "images/edits",
            "not a chat model",
            "image generation model",
            # 中文兼容层常见报错
            "暂不支持该接口",
            "不支持该接口",
            "当前接口不支持",
            "该接口暂不支持",
            "接口不支持",
            # 有些兼容层即使是文生图，也会错误地回这类 image edits / missing_image 提示
            "image_url is required for image edits",
            "missing_image",
            "image edits",
            "input_image",
            "image_url is required"
        ]

        # 无图时，如果错误点名 messages，也通常意味着 chat/completions 路径不适合当前模型
        if not has_input_image and '"param": "messages"' in error_lower:
            return True

        return any(keyword in error_lower for keyword in fallback_keywords)

    def _resolve_result_image_url(self, img_url: str, base_url: str = None) -> str:
        """将模型返回的结果图地址规范化为可下载的绝对 URL"""
        if not img_url:
            return img_url

        if img_url.startswith(("http://", "https://", "data:")):
            return img_url

        if base_url:
            api_root = normalize_api_root(base_url).rstrip("/") + "/"
            resolved = urljoin(api_root, img_url)
            logger.info(f"检测到相对结果图地址，已自动补全为绝对地址: {resolved}")
            return resolved

        return img_url

    async def _download_result_image(self, img_url: str, proxy: str = None, base_url: str = None) -> bytes | str:
        """下载模型返回的结果图片，增加重试与容错，避免外链偶发重置导致整次任务失败"""
        if not img_url:
            return "结果图片地址为空"

        img_url = self._resolve_result_image_url(img_url, base_url)
        proxy = self._get_request_proxy(img_url, proxy)

        session = await self._get_session()
        request_timeout = max(30, int(self.config.get("timeout", 120)))
        configured_timeout = int(self.config.get("result_image_download_timeout", 0) or 0)
        timeout_val = max(30, configured_timeout or request_timeout, request_timeout)
        retries = max(2, int(self.config.get("result_image_download_retries", 4)))
        timeout = aiohttp.ClientTimeout(
            total=timeout_val,
            sock_read=timeout_val,
        )

        headers = {
            "User-Agent": self.config.get(
                "result_image_user_agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Connection": "keep-alive",
        }

        parsed = urlparse(img_url)
        host = (parsed.netloc or "").lower()
        should_try_dual_route = bool(proxy) and any(
            keyword in host for keyword in [
                "googleapis.com",
                "googleusercontent.com",
                "gstatic.com",
                "storage.googleapis.com",
            ]
        )

        should_try_ipv4_race = (not proxy) and any(
            keyword in host for keyword in [
                "googleapis.com",
                "googleusercontent.com",
                "gstatic.com",
                "storage.googleapis.com",
            ]
        )

        async def _download_once(current_proxy: str, route_name: str, current_session=None):
            active_session = current_session or session
            route_start = asyncio.get_running_loop().time()
            try:
                async with active_session.get(img_url, proxy=current_proxy, timeout=timeout, headers=headers) as img_resp:
                    if img_resp.status != 200:
                        return route_name, None, f"下载结果图失败，HTTP {img_resp.status}"

                    data = await img_resp.read()
                    if not data:
                        return route_name, None, "下载结果图失败，返回内容为空"

                    elapsed = asyncio.get_running_loop().time() - route_start
                    logger.info(f"结果图下载成功 ({elapsed:.2f}s) | 路线: {route_name} | host: {host}")
                    self._last_download_metrics = {
                        "download_duration": elapsed,
                        "download_route": route_name,
                    }
                    return route_name, data, ""
            except asyncio.TimeoutError:
                return route_name, None, f"下载结果图超时 ({timeout_val}s)"
            except Exception as e:
                return route_name, None, str(e) or type(e).__name__

        last_error = ""
        if should_try_dual_route:
            logger.info(f"结果图下载启用直连/代理并发抢跑: {img_url[:120]}")
            for attempt in range(1, retries + 1):
                tasks = [
                    asyncio.create_task(_download_once(None, "direct")),
                    asyncio.create_task(_download_once(proxy, "proxy")),
                ]
                route_errors = {}
                try:
                    for completed in asyncio.as_completed(tasks):
                        route_name, data, error_text = await completed
                        if data:
                            for pending in tasks:
                                if not pending.done():
                                    pending.cancel()
                            if route_name == "direct":
                                logger.info(f"结果图下载通过直连优先成功: {img_url[:120]}")
                            return data
                        route_errors[route_name] = error_text
                finally:
                    for pending in tasks:
                        if not pending.done():
                            pending.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

                last_error = route_errors.get("direct") or route_errors.get("proxy") or "未知下载错误"
                if attempt < retries:
                    logger.warning(f"结果图并发下载失败，准备重试: {img_url[:120]} | {route_errors}")
                    await asyncio.sleep(min(2 * attempt, 5))
        elif should_try_ipv4_race:
            logger.info(f"结果图下载启用默认直连/IPv4直连并发抢跑: {img_url[:120]}")
            for attempt in range(1, retries + 1):
                ipv4_session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(family=socket.AF_INET))
                tasks = [
                    asyncio.create_task(_download_once(None, "direct")),
                    asyncio.create_task(_download_once(None, "direct-ipv4", ipv4_session)),
                ]
                route_errors = {}
                try:
                    for completed in asyncio.as_completed(tasks):
                        route_name, data, error_text = await completed
                        if data:
                            for pending in tasks:
                                if not pending.done():
                                    pending.cancel()
                            if route_name == "direct-ipv4":
                                logger.info(f"结果图下载通过 IPv4 直连优先成功: {img_url[:120]}")
                            return data
                        route_errors[route_name] = error_text
                finally:
                    for pending in tasks:
                        if not pending.done():
                            pending.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    await ipv4_session.close()

                last_error = route_errors.get("direct") or route_errors.get("direct-ipv4") or "未知下载错误"
                if attempt < retries:
                    logger.warning(f"结果图直连/IPv4并发下载失败，准备重试: {img_url[:120]} | {route_errors}")
                    await asyncio.sleep(min(2 * attempt, 5))
        else:
            proxy_candidates = [proxy]
            for proxy_index, current_proxy in enumerate(proxy_candidates, 1):
                route_name = "proxy" if current_proxy else "direct"
                for attempt in range(1, retries + 1):
                    _, data, error_text = await _download_once(current_proxy, route_name)
                    if data:
                        return data
                    last_error = error_text

                    if attempt < retries:
                        await asyncio.sleep(min(2 * attempt, 5))

        logger.error(f"结果图下载失败: {img_url[:120]} | {last_error}")
        return f"结果图片下载失败: {last_error}"

    async def call_api(self, images: List[bytes], prompt: str,
                       model: str, legacy_use_power_or_proxy=None,
                       proxy: str = None,
                       use_text_to_image_api: bool = False,
                       aspect_ratio: str = None,
                       resolution: str = None) -> bytes | str:
        proxy, use_text_to_image_api = self._normalize_call_api_args(
            legacy_use_power_or_proxy, proxy, use_text_to_image_api
        )

        if self.config.get("enable_luxury_mode", False):
            return await self._call_api_with_luxury_mode(
                images, prompt, model, proxy, use_text_to_image_api,
                aspect_ratio=aspect_ratio, resolution=resolution
            )

        return await self._call_api_once(
            images, prompt, model, proxy, use_text_to_image_api,
            aspect_ratio=aspect_ratio, resolution=resolution
        )

    async def _call_api_once(self, images: List[bytes], prompt: str,
                             model: str, proxy: str = None,
                             use_text_to_image_api: bool = False,
                             aspect_ratio: str = None,
                             resolution: str = None) -> bytes | str:
        """核心生成逻辑"""

        self._reset_metrics()
        call_start = asyncio.get_running_loop().time()

        interface_mode = self._get_interface_mode()
        mode = "gemini_official" if interface_mode == "gemini_official" else "generic"

        # 1. 确定 URL
        base = self._get_base_url(interface_mode, use_text_to_image_api=use_text_to_image_api)

        if not base:
            return "API URL 未配置"

        # 2. 获取 Key
        key = await self.get_key(interface_mode, use_text_to_image_api=use_text_to_image_api)
        if not key:
            return "无可用 API Key"

        # 3. 构造请求
        headers = {"Content-Type": "application/json"}
        payload = {}
        url = base.rstrip("/")

        default_aspect_ratio = self.config.get("image_aspect_ratio", "4:3")
        if images:
            default_aspect_ratio = detect_aspect_ratio_from_image(images[0], default_aspect_ratio)

        generation_params = resolve_image_generation_params(
            prompt,
            default_resolution=self.config.get("image_resolution", "1K"),
            default_aspect_ratio=default_aspect_ratio,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
        )
        logger.info(
            f"图片参数已解析: aspect_ratio={generation_params['aspect_ratio']}, "
            f"resolution={generation_params['resolution']}, size={generation_params['size']}"
        )

        custom_kind = ""
        if interface_mode == "custom_endpoint":
            lower_url = url.lower()
            if "generatecontent" in lower_url:
                custom_kind = "gemini"
                mode = "gemini_official"
            elif "chat/completions" in lower_url:
                custom_kind = "chat"
                mode = "generic"
            else:
                custom_kind = "image"

        force_images_api = self._should_force_images_api(interface_mode, custom_kind)
        if force_images_api:
            route_reason = (
                f"interface_mode={interface_mode}"
                if interface_mode != "openai_chat"
                else "generic_prefer_images_api=true"
            )
            target_endpoint = "images/edits" if images else "images/generations"
            logger.info(
                f"API route locked to Images API ({route_reason}); "
                f"endpoint={target_endpoint}, input_images={len(images)}"
            )
            return await self.call_images_api(
                images, prompt, model, key, base, proxy,
                generation_params=generation_params,
                exact_endpoint=(interface_mode == "custom_endpoint"),
            )

        # 画质强化 Prompt
        res_set = generation_params["resolution"]
        final_prompt = f"(Masterpiece, Best Quality, {res_set} Resolution), {prompt}" if res_set != "1K" else prompt

        if mode == "gemini_official":
            if interface_mode != "custom_endpoint":
                # 无论用户填了域名、v1、v1beta 还是完整接口路径，都先还原为基础地址，
                # 再由 gemini_official 模式统一补全官方 v1beta generateContent 路径。
                url = self._build_gemini_api_url(url, model)
                logger.info(f"Gemini API 地址已按接口模式规范化为: {url}")
            headers["x-goog-api-key"] = key

            parts = [{"text": final_prompt}]
            for img in images:
                mime = self.get_mime_type(img)
                parts.append({"inlineData": {"mimeType": mime, "data": base64.b64encode(img).decode()}})

            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "maxOutputTokens": 4096,
                    "responseModalities": ["TEXT", "IMAGE"],
                    "imageConfig": {
                        "aspectRatio": generation_params["aspect_ratio"],
                        "imageSize": generation_params["resolution"],
                    },
                },
                "safetySettings": [{"category": c, "threshold": "BLOCK_NONE"} for c in
                                   ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                                    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT",
                                    "HARM_CATEGORY_CIVIC_INTEGRITY"]] # 参考: 增加 CIVIC_INTEGRITY
            }
        else:
            # OpenAI / Generic 构造：允许用户只填写 Base URL，自动补全 chat/completions
            url = self._normalize_generic_chat_url(url)
            headers["Authorization"] = f"Bearer {key}"
            
            content_list = [{"type": "text", "text": final_prompt}]
            for img in images:
                b64 = base64.b64encode(img).decode()
                mime = self.get_mime_type(img)
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"}
                })
            
            msgs = [{"role": "user", "content": content_list}]
            
            use_stream = self.config.get("use_stream", False)
            
            # [性能优化] 显式设置 max_tokens
            # 如果不设置，某些中转接口可能会等待或者分配过大的 Tokens 空间，增加延迟
            pl = {"model": model, "messages": msgs, "stream": use_stream, "max_tokens": 4096}
            payload.update(pl)

            # 针对 Gemini 系模型的 OpenAI 兼容层特殊处理
            # 参考 bananic_ninjutsu: 如果模型名包含 pro/image/banana，显式添加 modalities
            lower_model = model.lower()
            if "gemini" in lower_model or "pro" in lower_model or "image" in lower_model:
                # 无论何种模式，只要模型名看起来像 Gemini，就尝试注入 modalities
                payload["modalities"] = ["image", "text"]

                # 尝试强制注入 safetySettings (很多中转支持透传此参数)
                # 这能有效防止 finish_reason: content_filter
                payload["safetySettings"] = [{"category": c, "threshold": "BLOCK_NONE"} for c in
                                             ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                                              "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT",
                                              "HARM_CATEGORY_CIVIC_INTEGRITY"]]

                # 兼容通过 OpenAI chat/completions 转发 Gemini 图片模型的中转站。
                payload["image_config"] = {
                    "aspect_ratio": generation_params["aspect_ratio"],
                    "image_size": generation_params["resolution"],
                }
        
        # 4. 发送请求
        try:
            timeout_val = self.config.get("timeout", 120)
            timeout = aiohttp.ClientTimeout(total=timeout_val)
            
            # 使用持久化 Session，避免重复的 TCP/SSL 握手开销
            session = await self._get_session()
            
            candidate_urls = [url]
            if interface_mode == "custom_endpoint":
                candidate_urls = [base.rstrip("/")]
            elif mode == "generic":
                candidate_urls = self._build_candidate_generic_chat_urls(base)

            logger.info(f"Generic API 候选地址: {candidate_urls}")

            resp_text = ""
            last_status = None
            active_url = url

            for idx, candidate_url in enumerate(candidate_urls):
                active_url = candidate_url
                current_proxy = self._get_request_proxy(active_url, proxy)
                async with session.post(active_url, json=payload, headers=headers, proxy=current_proxy, timeout=timeout) as resp:
                    resp_text = await resp.text()
                    last_status = resp.status

                    # 如果返回 HTML，且后面还有候选地址，则继续尝试常见 API 前缀
                    if "<html" in resp_text.lower() and idx < len(candidate_urls) - 1:
                        logger.warning(f"Generic API 地址返回了 HTML 页面，尝试下一个候选地址: {candidate_urls[idx + 1]}")
                        continue

                    if resp.status != 200:
                        try:
                            err_json = json.loads(resp_text)

                            # 兼容标准 OpenAI 错误结构: {"error": {...}}
                            if "error" in err_json:
                                err_msg = json.dumps(err_json["error"], ensure_ascii=False)

                                if interface_mode == "openai_chat" and self._should_fallback_to_images_api(err_msg, bool(images)):
                                    logger.info(f"模型 {model} 当前错误适合切换到 Images API，自动回退处理")
                                    return await self.call_images_api(
                                        images, prompt, model, key, base, proxy,
                                        generation_params=generation_params
                                    )

                                return f"API Error {resp.status}: {err_msg} | URL: {active_url}"

                            # 兼容顶层直接报错结构: {"message": "...", "type": "...", "code": "..."}
                            if any(k in err_json for k in ["message", "type", "code", "param"]):
                                err_msg = json.dumps(err_json, ensure_ascii=False)

                                if interface_mode == "openai_chat" and self._should_fallback_to_images_api(err_msg, bool(images)):
                                    logger.info(f"模型 {model} 返回顶层错误结构，自动回退到 Images API")
                                    return await self.call_images_api(
                                        images, prompt, model, key, base, proxy,
                                        generation_params=generation_params
                                    )

                                return f"API Error {resp.status}: {err_msg} | URL: {active_url}"
                        except:
                            pass

                        if "<html" in resp_text.lower():
                            if mode == "generic":
                                return (
                                    f"HTTP {resp.status}: 服务端返回了网页而非数据。当前尝试地址: {active_url}。\n"
                                    f"请填写 API 基础地址，而不是网站首页。例如应填写接口所在前缀，如 https://域名/api 或 https://域名/openai。"
                                )
                            return f"HTTP {resp.status}: 服务端返回了网页而非数据，请检查URL配置。"

                        if interface_mode == "openai_chat" and self._should_fallback_to_images_api(resp_text, bool(images)):
                            logger.info(f"模型 {model} 当前错误适合切换到 Images API，自动回退处理")
                            return await self.call_images_api(
                                images, prompt, model, key, base, proxy,
                                generation_params=generation_params
                            )

                        return f"HTTP {resp.status}: {resp_text[:200]} | URL: {active_url}"

                    # 命中成功候选地址，跳出循环
                    url = active_url
                    break

            if last_status is not None and last_status != 200:
                return f"HTTP {last_status}: {resp_text[:200]} | URL: {active_url}"

            try:
                res_data = json.loads(resp_text)
            except json.JSONDecodeError:
                # 兼容：处理被强制流式返回的情况 (SSE format)
                if "data: " in resp_text:
                    full_content = ""
                    tool_calls_buffer = {} # {index: "arguments"}

                    lines = resp_text.splitlines()
                    valid_stream = False
                    extracted_images = []
                    extracted_data_arr = []
                    extracted_urls = []

                    for line in lines:
                        line = line.strip()
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                chunk_str = line[6:]
                                if not chunk_str:
                                    continue
                                chunk = json.loads(chunk_str)
                                valid_stream = True
                                if "choices" in chunk and chunk["choices"]:
                                    delta = chunk["choices"][0].get("delta", {})

                                    if "content" in delta and delta["content"]:
                                        full_content += delta["content"]

                                    if "tool_calls" in delta and delta["tool_calls"]:
                                        for tc in delta["tool_calls"]:
                                            idx = tc.get("index", 0)
                                            if idx not in tool_calls_buffer:
                                                tool_calls_buffer[idx] = ""
                                            if "function" in tc and "arguments" in tc["function"]:
                                                tool_calls_buffer[idx] += tc["function"]["arguments"]

                                    if "images" in delta and isinstance(delta["images"], list):
                                        extracted_images.extend(delta["images"])
                                    if "data" in delta and isinstance(delta["data"], list):
                                        extracted_data_arr.extend(delta["data"])
                                    if "image_url" in delta:
                                        image_url = delta["image_url"]
                                        if isinstance(image_url, str):
                                            extracted_urls.append(image_url)
                                        elif isinstance(image_url, dict) and image_url.get("url"):
                                            extracted_urls.append(image_url["url"])
                                    if "url" in delta and isinstance(delta["url"], str):
                                        extracted_urls.append(delta["url"])

                                if "images" in chunk and isinstance(chunk["images"], list):
                                    extracted_images.extend(chunk["images"])
                                if "data" in chunk and isinstance(chunk["data"], list):
                                    extracted_data_arr.extend(chunk["data"])
                                if "image_url" in chunk:
                                    image_url = chunk["image_url"]
                                    if isinstance(image_url, str):
                                        extracted_urls.append(image_url)
                                    elif isinstance(image_url, dict) and image_url.get("url"):
                                        extracted_urls.append(image_url["url"])
                                if "url" in chunk and isinstance(chunk["url"], str):
                                    extracted_urls.append(chunk["url"])
                            except:
                                pass

                    if valid_stream:
                        msg_obj = {"content": full_content, "role": "assistant"}

                        if tool_calls_buffer:
                            msg_obj["tool_calls"] = []
                            for idx in sorted(tool_calls_buffer.keys()):
                                msg_obj["tool_calls"].append({
                                    "function": {"arguments": tool_calls_buffer[idx]}
                                })

                        if extracted_images:
                            msg_obj["images"] = extracted_images
                        if extracted_urls:
                            msg_obj["images"] = msg_obj.get("images", [])
                            msg_obj["images"].extend(extracted_urls)

                        res_data = {"choices": [{"message": msg_obj, "finish_reason": "stop"}]}
                        if extracted_data_arr:
                            res_data["data"] = extracted_data_arr

                        if not full_content and not tool_calls_buffer and not extracted_images and not extracted_data_arr and not extracted_urls:
                            for line in lines:
                                if '"error"' in line:
                                    try:
                                        chunk = json.loads(line.replace("data: ", "").strip())
                                        if "error" in chunk:
                                            return json.dumps(chunk["error"], ensure_ascii=False)
                                    except:
                                        pass
                    else:
                        return f"数据解析失败: 看起来是流式数据但无法解析. 内容: {resp_text[:100]}..."
                else:
                    b64_match = re.search(r'(data:image\/[\w\-\+\.]+(?:;base64)?,[\w\-\+\/=\s]{100,})', resp_text)
                    if b64_match:
                        logger.warning("在非JSON/非标准流解析失败分支中找到了base64，已挽救")
                        img_url = b64_match.group(1).replace("\\n", "").replace("\\r", "").replace(" ", "").replace("\\", "")
                        if img_url.startswith("data:"):
                            return base64.b64decode(img_url.split(",")[-1])

                    pure_b64_match = re.search(r'"([A-Za-z0-9+/]{1000,}={0,2})"', resp_text)
                    if pure_b64_match:
                        logger.warning("在非JSON/非标准流解析失败分支中找到了纯base64，已挽救")
                        img_url = f"data:image/png;base64,{pure_b64_match.group(1)}"
                        return base64.b64decode(img_url.split(",")[-1])

                    return f"数据解析失败: 返回内容不是 JSON. 内容: {resp_text[:100]}... | URL: {active_url}"

            if "error" in res_data:
                return json.dumps(res_data["error"], ensure_ascii=False)

            img_url = self.extract_image_url(res_data)

            # 终极 fallback，检查是否是那种直接放在外层的 tool_calls / images 遗漏
            if not img_url:
                raw_str = str(res_data)
                b64_match = re.search(r'(data:image\/[\w\-\+\.]+(?:;base64)?,[\w\-\+\/=\s]{100,})', raw_str)
                if b64_match:
                    logger.warning("在报错分支的终极fallback中找到了base64，已挽救")
                    img_url = b64_match.group(1).replace("\\n", "").replace("\\r", "").replace(" ", "").replace("\\", "")

            if not img_url:
                raw_str = str(res_data)
                pure_b64_match = re.search(r'"([A-Za-z0-9+/]{1000,}={0,2})"', raw_str)
                if pure_b64_match:
                    logger.warning("在报错分支的终极fallback中找到了纯base64，已挽救")
                    img_url = f"data:image/png;base64,{pure_b64_match.group(1)}"

            if not img_url:
                raw_resp_b64_match = re.search(r'(data:image\/[\w\-\+\.]+(?:;base64)?,[\w\-\+\/=\s]{100,})', resp_text)
                if raw_resp_b64_match:
                    logger.warning("在原始流响应中找到了base64，已挽救")
                    img_url = raw_resp_b64_match.group(1).replace("\\n", "").replace("\\r", "").replace(" ", "").replace("\\", "")

            if not img_url:
                raw_resp_url_match = re.search(r'(https?://[^\s<>")\]]+)', resp_text)
                if raw_resp_url_match:
                    logger.warning("在原始流响应中找到了图片URL，已挽救")
                    img_url = raw_resp_url_match.group(1).rstrip(")>,'\".")

            if not img_url:
                # Gemini 特殊错误诊断 (原生 API)
                if "candidates" in res_data and res_data["candidates"]:
                    cand = res_data["candidates"][0]
                    finish_reason = cand.get("finishReason", "UNKNOWN")

                    if finish_reason not in ["STOP", "MAX_TOKENS"]:
                        return f"生成被终止，原因: {finish_reason} (通常是安全过滤导致)"

                    content_obj = cand.get("content") or {}
                    parts = content_obj.get("parts")
                    if not parts:
                        cand_str = json.dumps(cand, ensure_ascii=False)
                        logger.warning(f"Gemini API returned empty parts: {cand_str}")
                        return f"模型响应为空 (finishReason={finish_reason})。请确认使用的模型 ({model}) 是否支持生图，或者 Prompt 是否触发了隐性过滤。\nRaw: {cand_str[:100]}..."
                    else:
                        texts = [p.get("text", "") for p in parts if "text" in p]
                        if texts:
                            text_msg = "\n".join(texts).strip()
                            if text_msg:
                                return text_msg

                # OpenAI 格式错误诊断 (兼容 API)
                if "choices" in res_data and isinstance(res_data["choices"], list) and len(res_data["choices"]) > 0:
                    choice = res_data["choices"][0]
                    finish_reason = choice.get("finish_reason", "UNKNOWN")
                    msg = choice.get("message", {})
                    content = msg.get("content")
                    has_tools = "tool_calls" in msg and bool(msg["tool_calls"])

                    if content is None and not has_tools:
                        refusal = msg.get("refusal")
                        if refusal:
                            return f"生成请求被拒绝: {refusal}"

                        choice_str = json.dumps(choice, ensure_ascii=False)
                        logger.warning(f"OpenAI API content is None: {choice_str}")
                        return f"API 返回内容为空。finish_reason: {finish_reason}。\nDEBUG: {choice_str[:200]}..."

                    if isinstance(content, str) and not content.strip() and not has_tools:
                        if finish_reason == "content_filter":
                            return "❌ 生成被拦截: 触发了安全过滤 (content_filter)。建议修改 Prompt 或重试。"

                        if "data:image/" in resp_text or '"images"' in resp_text or '"image_url"' in resp_text or '"url"' in resp_text:
                            logger.warning("检测到空文本响应，但原始响应中仍包含疑似图片字段，已跳过空字符串误报")
                        else:
                            return f"API 返回内容为空字符串。finish_reason: {finish_reason}。"

                    if isinstance(content, str) and content.strip():
                        return content.strip()

                    if has_tools:
                        return "API返回了工具调用但无法解析出图片。请重试或检查接口。"

                return f"API请求成功但未找到图片数据。Raw: {str(res_data)[:300]}..."

            # 如果是 Base64 直接返回 Bytes
            if img_url.startswith("data:"):
                self._last_metrics = {
                    "upstream_duration": asyncio.get_running_loop().time() - call_start,
                    "download_duration": 0.0,
                    "total_duration": asyncio.get_running_loop().time() - call_start,
                    "download_route": "inline-base64",
                }
                return base64.b64decode(img_url.split(",")[-1])

            # 如果是 URL，需要再次下载（增加重试与容错，避免外链偶发失败）
            upstream_duration = asyncio.get_running_loop().time() - call_start
            result = await self._download_result_image(img_url, proxy, base)
            total_duration = asyncio.get_running_loop().time() - call_start
            self._last_metrics = {
                "upstream_duration": upstream_duration,
                "download_duration": float(self._last_download_metrics.get("download_duration", 0.0) or 0.0),
                "total_duration": total_duration,
                "download_route": self._last_download_metrics.get("download_route", "url-download") or "url-download",
            }
            return result

        except asyncio.TimeoutError:
            logger.error(f"API Call Timeout after {timeout_val}s")
            self._last_metrics["total_duration"] = asyncio.get_running_loop().time() - call_start
            return f"请求超时 ({timeout_val}s)，请稍后再试或检查网络。"
            
        except Exception as e:
            import traceback
            logger.error(f"API Call Error: {traceback.format_exc()}")
            self._last_metrics["total_duration"] = asyncio.get_running_loop().time() - call_start
            
            err_msg = str(e)
            if not err_msg:
                err_msg = type(e).__name__
            
            return f"系统错误: {err_msg}"
