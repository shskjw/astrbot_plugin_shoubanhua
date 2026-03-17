import asyncio
import base64
import json
import re
from urllib.parse import urljoin
import aiohttp
from typing import List, Dict
from astrbot import logger


class ApiManager:
    def __init__(self, config: dict):
        self.config = config
        self.key_lock = asyncio.Lock()
        self.generic_idx = 0
        self.gemini_idx = 0
        self._session = None # 保持 Session 持久化，复用 TCP/SSL 连接

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # 不在 Session 级别设置 Timeout，改在请求级别设置
            self._session = aiohttp.ClientSession()
        return self._session

    async def get_key(self, mode: str, is_power: bool, use_text_to_image_api: bool = False) -> str | None:
        """获取轮询 Key"""
        async with self.key_lock:
            prefix = "power_" if is_power else ""

            if use_text_to_image_api:
                if mode == "gemini_official":
                    keys = self.config.get("text_to_image_api_keys", [])
                    if not keys:
                        keys = self.config.get(f"{prefix}gemini_api_keys", [])
                    if not keys and is_power:
                        keys = self.config.get("gemini_api_keys", [])
                    if not keys:
                        return None
                    k = keys[self.gemini_idx % len(keys)]
                    self.gemini_idx += 1
                    return k
                else:
                    keys = self.config.get("text_to_image_api_keys", [])
                    if not keys:
                        keys = self.config.get(f"{prefix}generic_api_keys", [])
                    if not keys and is_power:
                        keys = self.config.get("generic_api_keys", [])
                    if not keys:
                        return None
                    k = keys[self.generic_idx % len(keys)]
                    self.generic_idx += 1
                    return k

            if mode == "gemini_official":
                keys = self.config.get(f"{prefix}gemini_api_keys", [])
                if not keys and is_power:
                    keys = self.config.get("gemini_api_keys", [])  # Fallback

                if not keys:
                    return None
                k = keys[self.gemini_idx % len(keys)]
                self.gemini_idx += 1
                return k
            else:
                keys = self.config.get(f"{prefix}generic_api_keys", [])
                if not keys and is_power:
                    keys = self.config.get("generic_api_keys", [])  # Fallback

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
                if "b64_json" in item:
                    return f"data:image/png;base64,{item['b64_json']}"
                if "url" in item:
                    url_value = item["url"]
                    if isinstance(url_value, str):
                        # 兼容某些接口把纯 base64 误放进 url 字段
                        pure_b64_match = re.fullmatch(r'[A-Za-z0-9+/]{1000,}={0,2}', url_value.strip())
                        if pure_b64_match:
                            return f"data:image/jpeg;base64,{url_value.strip()}"
                    return url_value

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
        """将用户填写的 Generic URL 规范化为 chat/completions 地址。
        支持只填写域名或 /v1；如果用户误填了完整接口路径，也会自动忽略尾部路径后重新拼接。
        """
        original_url = (base_url or "").rstrip("/")
        url = original_url
        if not url:
            return url

        # 若用户直接填了完整接口路径，统一裁掉尾部，改按基础 URL 重新拼接
        known_suffixes = [
            "/v1/chat/completions",
            "/chat/completions",
            "/v1/images/generations",
            "/images/generations",
            "/v1/images/edits",
            "/images/edits"
        ]
        lower_url = url.lower()
        for suffix in known_suffixes:
            if lower_url.endswith(suffix):
                url = url[: -len(suffix)].rstrip("/")
                lower_url = url.lower()
                logger.info(f"检测到 Generic API 地址包含完整接口路径，已自动忽略尾部路径并改为基础 URL: {url}")
                break

        if url.endswith("/v1"):
            normalized = f"{url}/chat/completions"
        elif "/v1/" in url:
            idx = lower_url.find("/v1") + 3
            normalized = f"{url[:idx]}/chat/completions"
        else:
            normalized = f"{url}/v1/chat/completions"

        if normalized.rstrip("/") != original_url.rstrip("/"):
            logger.info(f"Generic API 地址已规范化为: {normalized}")

        return normalized

    def _convert_to_images_api_url(self, chat_url: str, has_input_image: bool = False) -> str:
        """将 Generic Base URL / chat URL 转换为图片接口 URL；有输入图时优先使用 edits"""
        url = self._normalize_generic_chat_url(chat_url)
        endpoint = "images/edits" if has_input_image else "images/generations"
        if "chat/completions" in url:
            return url.replace("chat/completions", endpoint)
        elif url.endswith("/v1"):
            return f"{url}/{endpoint}"
        elif "/v1" in url:
            idx = url.find("/v1") + 3
            return f"{url[:idx]}/{endpoint}"
        return f"{url}/v1/{endpoint}"

    def _build_candidate_generic_chat_urls(self, base_url: str) -> List[str]:
        """基于用户填写的基础地址，构造一组常见 Generic API 候选地址"""
        candidates = []
        normalized = self._normalize_generic_chat_url(base_url)
        if normalized:
            candidates.append(normalized)

        raw = (base_url or "").rstrip("/")
        if not raw:
            return candidates

        lower_raw = raw.lower()

        # 仅当用户填的像网站首页/域名，且未包含 v1 或 chat 路径时，才尝试常见前缀
        if "/v1" not in lower_raw and "chat/completions" not in lower_raw:
            extra_prefixes = [
                "/api/v1",
                "/openai/v1",
                "/api/openai/v1",
                "/v1"
            ]
            for prefix in extra_prefixes:
                candidate = f"{raw}{prefix}/chat/completions"
                if candidate not in candidates:
                    candidates.append(candidate)

        return candidates

    def _build_candidate_generic_image_urls(self, base_url: str, has_input_image: bool = False) -> List[str]:
        """基于用户填写的基础地址，构造一组常见 Images API 候选地址"""
        candidates = []
        normalized = self._convert_to_images_api_url(base_url, has_input_image=has_input_image)
        if normalized:
            candidates.append(normalized)

        raw = (base_url or "").rstrip("/")
        if not raw:
            return candidates

        lower_raw = raw.lower()
        endpoint = "images/edits" if has_input_image else "images/generations"

        if "/v1" not in lower_raw and "images/generations" not in lower_raw and "images/edits" not in lower_raw:
            extra_prefixes = [
                "/api/v1",
                "/openai/v1",
                "/api/openai/v1",
                "/v1"
            ]
            for prefix in extra_prefixes:
                candidate = f"{raw}{prefix}/{endpoint}"
                if candidate not in candidates:
                    candidates.append(candidate)

        return candidates

    def _should_retry_images_api_with_multipart(self, error_msg: str, has_input_image: bool = False) -> bool:
        """判断 Images API 是否需要回退为 multipart/form-data 方式"""
        if not has_input_image:
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

    async def _parse_images_api_success_response(self, resp_text: str, proxy: str = None) -> bytes | str:
        """统一解析 Images API 成功响应"""
        try:
            res_data = json.loads(resp_text)
        except json.JSONDecodeError:
            return f"数据解析失败: 返回内容不是 JSON. 内容: {resp_text[:100]}..."

        if "error" in res_data:
            return json.dumps(res_data["error"], ensure_ascii=False)

        img_url = self.extract_image_url(res_data)
        if not img_url:
            return f"Images API 返回成功但未找到图片数据。Raw: {str(res_data)[:200]}..."

        if img_url.startswith("data:"):
            return base64.b64decode(img_url.split(",")[-1])

        session = await self._get_session()
        async with session.get(img_url, proxy=proxy) as img_resp:
            return await img_resp.read()

    async def _call_images_api_multipart(self, images: List[bytes], prompt: str,
                                         model: str, key: str, base_url: str, proxy: str = None) -> bytes | str:
        """以 multipart/form-data 方式调用 Images API，兼容部分仅接受文件上传的编辑接口"""
        has_input_image = bool(images)
        candidate_urls = self._build_candidate_generic_image_urls(base_url, has_input_image=has_input_image)
        logger.info(f"Retry Images API with multipart/form-data, candidate urls: {candidate_urls}")

        headers = {
            "Authorization": f"Bearer {key}"
        }

        res_set = self.config.get("image_resolution", "1K")
        final_prompt = f"(Masterpiece, Best Quality, {res_set} Resolution), {prompt}" if res_set != "1K" else prompt

        timeout_val = self.config.get("timeout", 120)
        timeout = aiohttp.ClientTimeout(total=timeout_val)
        session = await self._get_session()

        form = aiohttp.FormData()
        form.add_field("model", model)
        form.add_field("prompt", final_prompt)
        form.add_field("n", "1")
        form.add_field("response_format", "b64_json")

        if images:
            img = images[0]
            mime = self.get_mime_type(img)
            ext = mime.split("/")[-1] if "/" in mime else "png"
            filename = f"input.{ext}"
            form.add_field("image", img, filename=filename, content_type=mime)

        try:
            for idx, url in enumerate(candidate_urls):
                async with session.post(url, data=form, headers=headers, proxy=proxy, timeout=timeout) as resp:
                    resp_text = await resp.text()

                    if "<html" in resp_text.lower() and idx < len(candidate_urls) - 1:
                        logger.warning(f"Images API multipart 返回 HTML 页面，尝试下一个候选地址: {candidate_urls[idx + 1]}")
                        continue

                    if resp.status != 200:
                        try:
                            err_json = json.loads(resp_text)
                            err_msg = json.dumps(err_json, ensure_ascii=False)
                            return f"Images API Multipart Error {resp.status}: {err_msg} | URL: {url}"
                        except:
                            return f"HTTP {resp.status}: {resp_text[:200]} | URL: {url}"

                    if "<html" in resp_text.lower():
                        return f"HTTP 200: 服务端返回了网页而非图片接口数据 | URL: {url}"

                    return await self._parse_images_api_success_response(resp_text, proxy)

            return f"Images API Multipart Error: 未找到可用接口地址 | Candidates: {candidate_urls}"
        except asyncio.TimeoutError:
            return f"请求超时 ({timeout_val}s)，请稍后再试或检查网络。"
        except Exception as e:
            import traceback
            logger.error(f"Images API Multipart Call Error: {traceback.format_exc()}")
            err_msg = str(e) or type(e).__name__
            return f"系统错误: {err_msg}"

    async def call_images_api(self, images: List[bytes], prompt: str,
                               model: str, key: str, base_url: str, proxy: str = None) -> bytes | str:
        """调用 Images API (DALL-E 风格接口) - 作为 fallback"""

        has_input_image = bool(images)
        candidate_urls = self._build_candidate_generic_image_urls(base_url, has_input_image=has_input_image)
        logger.info(f"Fallback to Images API, candidate urls: {candidate_urls}")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }

        # 画质强化 Prompt
        res_set = self.config.get("image_resolution", "1K")
        final_prompt = f"(Masterpiece, Best Quality, {res_set} Resolution), {prompt}" if res_set != "1K" else prompt

        # 构造 Images API 请求
        payload = {
            "model": model,
            "prompt": final_prompt,
            "n": 1,
            "response_format": "b64_json"
        }

        # 如果有输入图片，兼容不同 Images API 的字段要求
        # 一些服务要求 image_url；另一些只认 image / input_image / images
        if images:
            img = images[0]
            mime = self.get_mime_type(img)
            b64_img = base64.b64encode(img).decode()
            data_uri = f"data:{mime};base64,{b64_img}"
            payload["image"] = data_uri
            payload["image_url"] = data_uri
            payload["input_image"] = data_uri
            payload["images"] = [data_uri]

        try:
            timeout_val = self.config.get("timeout", 120)
            timeout = aiohttp.ClientTimeout(total=timeout_val)
            session = await self._get_session()

            for idx, url in enumerate(candidate_urls):
                async with session.post(url, json=payload, headers=headers, proxy=proxy, timeout=timeout) as resp:
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

                        if self._should_retry_images_api_with_multipart(err_msg, has_input_image):
                            logger.info("Images API JSON 请求失败，检测到服务端更偏好 multipart/form-data，自动重试")
                            return await self._call_images_api_multipart(images, prompt, model, key, base_url, proxy)

                        return f"Images API Error {resp.status}: {err_msg[:300]} | URL: {url}"

                    if "<html" in resp_text.lower():
                        return f"HTTP 200: 服务端返回了网页而非图片接口数据 | URL: {url}"

                    return await self._parse_images_api_success_response(resp_text, proxy)

            return f"Images API Error: 未找到可用接口地址 | Candidates: {candidate_urls}"

        except asyncio.TimeoutError:
            timeout_val = self.config.get("timeout", 120)
            return f"请求超时 ({timeout_val}s)，请稍后再试或检查网络。"
        except Exception as e:
            import traceback
            logger.error(f"Images API Call Error: {traceback.format_exc()}")
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
            normalized_chat_url = self._normalize_generic_chat_url(base_url)
            api_root = normalized_chat_url
            if "/chat/completions" in api_root:
                api_root = api_root.split("/chat/completions", 1)[0] + "/"
            elif not api_root.endswith("/"):
                api_root += "/"

            resolved = urljoin(api_root, img_url.lstrip("/"))
            logger.info(f"检测到相对结果图地址，已自动补全为绝对地址: {resolved}")
            return resolved

        return img_url

    async def _download_result_image(self, img_url: str, proxy: str = None, base_url: str = None) -> bytes | str:
        """下载模型返回的结果图片，增加重试与容错，避免外链偶发重置导致整次任务失败"""
        if not img_url:
            return "结果图片地址为空"

        img_url = self._resolve_result_image_url(img_url, base_url)

        session = await self._get_session()
        retries = max(1, int(self.config.get("result_image_download_retries", 3)))
        timeout_val = max(10, int(self.config.get("result_image_download_timeout", 60)))
        timeout = aiohttp.ClientTimeout(total=timeout_val)

        headers = {
            "User-Agent": self.config.get(
                "result_image_user_agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
            )
        }

        last_error = ""
        for attempt in range(1, retries + 1):
            try:
                async with session.get(img_url, proxy=proxy, timeout=timeout, headers=headers) as img_resp:
                    if img_resp.status != 200:
                        last_error = f"下载结果图失败，HTTP {img_resp.status}"
                    else:
                        data = await img_resp.read()
                        if data:
                            return data
                        last_error = "下载结果图失败，返回内容为空"
            except asyncio.TimeoutError:
                last_error = f"下载结果图超时 ({timeout_val}s)"
            except Exception as e:
                last_error = str(e) or type(e).__name__

            if attempt < retries:
                await asyncio.sleep(min(2 * attempt, 5))

        logger.error(f"结果图下载失败: {img_url[:120]} | {last_error}")
        return f"结果图片下载失败: {last_error}"

    async def call_api(self, images: List[bytes], prompt: str,
                       model: str, use_power: bool, proxy: str = None,
                       use_text_to_image_api: bool = False) -> bytes | str:
        """核心生成逻辑"""

        mode = self.config.get("api_mode", "generic")

        # 1. 确定 URL
        prefix = "power_" if use_power else ""
        if use_text_to_image_api:
            if mode == "gemini_official":
                base = (
                    self.config.get("text_to_image_api_url")
                    or self.config.get(f"{prefix}gemini_api_url")
                    or self.config.get("gemini_api_url")
                )
            else:
                base = (
                    self.config.get("text_to_image_api_url")
                    or self.config.get(f"{prefix}generic_api_url")
                    or self.config.get("generic_api_url")
                )
        else:
            if mode == "gemini_official":
                base = self.config.get(f"{prefix}gemini_api_url") or self.config.get("gemini_api_url")
            else:
                base = self.config.get(f"{prefix}generic_api_url") or self.config.get("generic_api_url")

        if not base:
            return "API URL 未配置"

        # 2. 获取 Key
        key = await self.get_key(mode, use_power, use_text_to_image_api=use_text_to_image_api)
        if not key:
            return "无可用 API Key"

        # 3. 构造请求
        headers = {"Content-Type": "application/json"}
        payload = {}
        url = base.rstrip("/")

        # 对于明确使用 Generic 图片接口的站点，可配置为优先直连 Images API
        if mode == "generic" and self.config.get("generic_prefer_images_api", False):
            logger.info("已启用 generic_prefer_images_api，优先直接走 Images API")
            return await self.call_images_api(images, prompt, model, key, base, proxy)

        # 画质强化 Prompt
        res_set = self.config.get("image_resolution", "1K")
        final_prompt = f"(Masterpiece, Best Quality, {res_set} Resolution), {prompt}" if res_set != "1K" else prompt

        if mode == "gemini_official":
            # --- 修复核心：Gemini URL 智能构造 ---
            # 如果 URL 里没有 'models' 关键字且不是 OneAPI 风格，说明填的是 Base URL

            # 补全 v1/v1beta
            if "/v1" not in url and "/v1beta" not in url:
                url += "/v1beta"

            # 补全 /models
            if "/models" not in url:
                url += "/models"

            # 拼接 Model (防止 url 中已经包含了 model)
            if f"/{model}" not in url:
                url += f"/{model}"

            # 拼接动作
            if ":generateContent" not in url:
                url += ":generateContent"

            # 认证：Query 参数 + Header 双重保险
            if "?" in url:
                url += f"&key={key}"
            else:
                url += f"?key={key}"
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
                    # "aspectRatio": "1:1" # Gemini 可能需要显式比例，暂保持默认
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
        
        # 4. 发送请求
        try:
            timeout_val = self.config.get("timeout", 120)
            timeout = aiohttp.ClientTimeout(total=timeout_val)
            
            # 使用持久化 Session，避免重复的 TCP/SSL 握手开销
            session = await self._get_session()
            
            candidate_urls = [url]
            if mode == "generic":
                candidate_urls = self._build_candidate_generic_chat_urls(base)

            logger.info(f"Generic API 候选地址: {candidate_urls}")

            resp_text = ""
            last_status = None
            active_url = url

            for idx, candidate_url in enumerate(candidate_urls):
                active_url = candidate_url
                async with session.post(active_url, json=payload, headers=headers, proxy=proxy, timeout=timeout) as resp:
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

                                if mode == "generic" and self._should_fallback_to_images_api(err_msg, bool(images)):
                                    logger.info(f"模型 {model} 当前错误适合切换到 Images API，自动回退处理")
                                    return await self.call_images_api(images, prompt, model, key, base, proxy)

                                return f"API Error {resp.status}: {err_msg} | URL: {active_url}"

                            # 兼容顶层直接报错结构: {"message": "...", "type": "...", "code": "..."}
                            if any(k in err_json for k in ["message", "type", "code", "param"]):
                                err_msg = json.dumps(err_json, ensure_ascii=False)

                                if mode == "generic" and self._should_fallback_to_images_api(err_msg, bool(images)):
                                    logger.info(f"模型 {model} 返回顶层错误结构，自动回退到 Images API")
                                    return await self.call_images_api(images, prompt, model, key, base, proxy)

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

                        if mode == "generic" and self._should_fallback_to_images_api(resp_text, bool(images)):
                            logger.info(f"模型 {model} 当前错误适合切换到 Images API，自动回退处理")
                            return await self.call_images_api(images, prompt, model, key, base, proxy)

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
                return base64.b64decode(img_url.split(",")[-1])

            # 如果是 URL，需要再次下载（增加重试与容错，避免外链偶发失败）
            return await self._download_result_image(img_url, proxy, base)

        except asyncio.TimeoutError:
            logger.error(f"API Call Timeout after {timeout_val}s")
            return f"请求超时 ({timeout_val}s)，请稍后再试或检查网络。"
            
        except Exception as e:
            import traceback
            logger.error(f"API Call Error: {traceback.format_exc()}")
            
            err_msg = str(e)
            if not err_msg:
                err_msg = type(e).__name__
            
            return f"系统错误: {err_msg}"
