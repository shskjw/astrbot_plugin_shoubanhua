import asyncio
import base64
import json
import re
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

    async def get_key(self, mode: str, is_power: bool) -> str | None:
        """获取轮询 Key"""
        async with self.key_lock:
            prefix = "power_" if is_power else ""

            if mode == "gemini_official":
                keys = self.config.get(f"{prefix}gemini_api_keys", [])
                if not keys and is_power: keys = self.config.get("gemini_api_keys", [])  # Fallback

                if not keys: return None
                k = keys[self.gemini_idx % len(keys)]
                self.gemini_idx += 1
                return k
            else:
                keys = self.config.get(f"{prefix}generic_api_keys", [])
                if not keys and is_power: keys = self.config.get("generic_api_keys", [])  # Fallback

                if not keys: return None
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
                    return item["url"]

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
                        # 尝试1: 直接解析 arguments 里的 url/https
                        func_args = tool.get("function", {}).get("arguments", "")
                        if "http" in func_args:
                            urls = re.findall(r"(https?://[^\s<>\"'()\[\]]+)", func_args)
                            if urls: return urls[0].strip()

                        # 尝试2: 查找 base64
                        if "base64" in func_args:
                            b64_match = re.search(r'(data:image\/[a-zA-Z]+;base64,[a-zA-Z0-9+/=]+)', func_args)
                            if b64_match:
                                return b64_match.group(1)

                        try:
                            args = json.loads(func_args)
                            # 常见的参数名: url, image_url, images, file_url
                            for k in ["url", "image_url", "file_url", "link", "b64_json", "image", "data", "image_data"]:
                                if k in args: return args[k]
                        except:
                            pass

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

    def _convert_to_images_api_url(self, chat_url: str) -> str:
        """将 chat/completions URL 转换为 images/generations URL"""
        url = chat_url.rstrip("/")
        if "chat/completions" in url:
            return url.replace("chat/completions", "images/generations")
        elif url.endswith("/v1"):
            return f"{url}/images/generations"
        elif "/v1" in url:
            idx = url.find("/v1") + 3
            return f"{url[:idx]}/images/generations"
        return f"{url}/v1/images/generations"

    async def call_images_api(self, images: List[bytes], prompt: str,
                               model: str, key: str, base_url: str, proxy: str = None) -> bytes | str:
        """调用 Images API (DALL-E 风格接口) - 作为 fallback"""
        
        url = self._convert_to_images_api_url(base_url)
        logger.info(f"Fallback to Images API: {url}")
        
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
        
        # 如果有输入图片，尝试添加 image 参数（某些 API 支持）
        if images:
            img = images[0]
            mime = self.get_mime_type(img)
            b64_img = base64.b64encode(img).decode()
            payload["image"] = f"data:{mime};base64,{b64_img}"
        
        try:
            timeout_val = self.config.get("timeout", 120)
            timeout = aiohttp.ClientTimeout(total=timeout_val)
            session = await self._get_session()
            
            async with session.post(url, json=payload, headers=headers, proxy=proxy, timeout=timeout) as resp:
                resp_text = await resp.text()
                
                if resp.status != 200:
                    try:
                        err_json = json.loads(resp_text)
                        if "error" in err_json:
                            err_msg = json.dumps(err_json['error'], ensure_ascii=False)
                            return f"Images API Error {resp.status}: {err_msg}"
                    except:
                        pass
                    return f"HTTP {resp.status}: {resp_text[:200]}"
                
                try:
                    res_data = json.loads(resp_text)
                except json.JSONDecodeError:
                    return f"数据解析失败: 返回内容不是 JSON. 内容: {resp_text[:100]}..."
                
                if "error" in res_data:
                    return json.dumps(res_data["error"], ensure_ascii=False)
                
                # 解析 Images API 响应
                img_url = self.extract_image_url(res_data)
                if not img_url:
                    return f"Images API 返回成功但未找到图片数据。Raw: {str(res_data)[:200]}..."
                
                # 如果是 Base64 直接返回 Bytes
                if img_url.startswith("data:"):
                    return base64.b64decode(img_url.split(",")[-1])
                
                # 如果是 URL，需要再次下载
                async with session.get(img_url, proxy=proxy) as img_resp:
                    return await img_resp.read()
                    
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

    async def call_api(self, images: List[bytes], prompt: str,
                       model: str, use_power: bool, proxy: str = None) -> bytes | str:
        """核心生成逻辑"""

        mode = self.config.get("api_mode", "generic")

        # 1. 确定 URL
        prefix = "power_" if use_power else ""
        if mode == "gemini_official":
            base = self.config.get(f"{prefix}gemini_api_url") or self.config.get("gemini_api_url")
        else:
            base = self.config.get(f"{prefix}generic_api_url") or self.config.get("generic_api_url")

        if not base: return "API URL 未配置"

        # 2. 获取 Key
        key = await self.get_key(mode, use_power)
        if not key: return "无可用 API Key"

        # 3. 构造请求
        headers = {"Content-Type": "application/json"}
        payload = {}
        url = base.rstrip("/")

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
            # OpenAI 构造
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
            
            # [性能优化] 显式设置 max_tokens
            # 如果不设置，某些中转接口可能会等待或者分配过大的 Tokens 空间，增加延迟
            pl = {"model": model, "messages": msgs, "stream": False, "max_tokens": 4096}
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
            
            async with session.post(url, json=payload, headers=headers, proxy=proxy, timeout=timeout) as resp:
                resp_text = await resp.text()

                if resp.status != 200:
                    try:
                        err_json = json.loads(resp_text)
                        if "error" in err_json:
                            err_msg = json.dumps(err_json['error'], ensure_ascii=False)
                            
                            # 检查是否是 chat completions 不支持的错误，自动切换到 Images API
                            if mode == "generic" and self._is_chat_not_supported_error(err_msg):
                                logger.info(f"模型 {model} 不支持 chat completions，自动切换到 Images API")
                                return await self.call_images_api(images, prompt, model, key, base, proxy)
                            
                            return f"API Error {resp.status}: {err_msg}"
                    except:
                        pass
                    if "<html" in resp_text.lower():
                        return f"HTTP {resp.status}: 服务端返回了网页而非数据，请检查URL配置。"
                    
                    # 检查原始响应文本是否包含不支持 chat 的错误
                    if mode == "generic" and self._is_chat_not_supported_error(resp_text):
                        logger.info(f"模型 {model} 不支持 chat completions，自动切换到 Images API")
                        return await self.call_images_api(images, prompt, model, key, base, proxy)
                    
                    # Return error if status != 200
                    return f"HTTP {resp.status}: {resp_text[:200]}"

                try:
                    res_data = json.loads(resp_text)
                except json.JSONDecodeError:
                        # 兼容：处理被强制流式返回的情况 (SSE format)
                        if "data: " in resp_text:
                            full_content = ""
                            tool_calls_buffer = {} # {index: "arguments"}
                            
                            lines = resp_text.splitlines()
                            valid_stream = False
                            
                            for line in lines:
                                line = line.strip()
                                if line.startswith("data: ") and line != "data: [DONE]":
                                    try:
                                        chunk = json.loads(line[6:])
                                        valid_stream = True
                                        if "choices" in chunk and chunk["choices"]:
                                            delta = chunk["choices"][0].get("delta", {})
                                            
                                            # 1. 拼接 content
                                            if "content" in delta and delta["content"]:
                                                full_content += delta["content"]
                                            
                                            # 2. 拼接 tool_calls arguments
                                            if "tool_calls" in delta and delta["tool_calls"]:
                                                for tc in delta["tool_calls"]:
                                                    idx = tc.get("index", 0)
                                                    if idx not in tool_calls_buffer:
                                                        tool_calls_buffer[idx] = ""
                                                    
                                                    if "function" in tc and "arguments" in tc["function"]:
                                                        tool_calls_buffer[idx] += tc["function"]["arguments"]
                                    except:
                                        pass
                            
                            if valid_stream:
                                # 重构为非流式结构以便后续处理
                                msg_obj = {"content": full_content, "role": "assistant"}
                                
                                # 还原 tool_calls
                                if tool_calls_buffer:
                                    msg_obj["tool_calls"] = []
                                    for idx in sorted(tool_calls_buffer.keys()):
                                        msg_obj["tool_calls"].append({
                                            "function": {"arguments": tool_calls_buffer[idx]}
                                        })
                                
                                res_data = {"choices": [{"message": msg_obj, "finish_reason": "stop"}]}
                            else:
                                 return f"数据解析失败: 看起来是流式数据但无法解析. 内容: {resp_text[:100]}..."
                        else:
                            return f"数据解析失败: 返回内容不是 JSON. 内容: {resp_text[:100]}..."

                if "error" in res_data:
                    return json.dumps(res_data["error"], ensure_ascii=False)

                img_url = self.extract_image_url(res_data)
                if not img_url:
                    # Gemini 特殊错误诊断 (原生 API)
                    if "candidates" in res_data and res_data["candidates"]:
                        cand = res_data["candidates"][0]
                        finish_reason = cand.get("finishReason", "UNKNOWN")
                        
                        # 1. 非正常结束
                        if finish_reason not in ["STOP", "MAX_TOKENS"]:
                            return f"生成被终止，原因: {finish_reason} (通常是安全过滤导致)"
                        
                        # 2. 正常结束但无内容
                        content = cand.get("content") or {}
                        parts = content.get("parts")
                        if not parts:
                            # 尝试打印整个 candidate 以便排查
                            cand_str = json.dumps(cand, ensure_ascii=False)
                            logger.warning(f"Gemini API returned empty parts: {cand_str}")
                            return f"模型响应为空 (finishReason={finish_reason})。请确认使用的模型 ({model}) 是否支持生图，或者 Prompt 是否触发了隐性过滤。\nRaw: {cand_str[:100]}..."
                    
                    # OpenAI 格式错误诊断 (兼容 API)
                    if "choices" in res_data and isinstance(res_data["choices"], list) and len(res_data["choices"]) > 0:
                        choice = res_data["choices"][0]
                        finish_reason = choice.get("finish_reason", "UNKNOWN")
                        msg = choice.get("message", {})
                        content = msg.get("content")
                        
                        # 1. content 为 None
                        if content is None:
                            refusal = msg.get("refusal")
                            if refusal: return f"生成请求被拒绝: {refusal}"
                            
                            # 将 choice 打印出来排查
                            choice_str = json.dumps(choice, ensure_ascii=False)
                            logger.warning(f"OpenAI API content is None: {choice_str}")
                            return f"API 返回内容为空。finish_reason: {finish_reason}。\nDEBUG: {choice_str[:200]}..."
                        
                        # 2. content 为空字符串
                        if isinstance(content, str) and not content.strip():
                            if finish_reason == "content_filter":
                                return "❌ 生成被拦截: 触发了安全过滤 (content_filter)。建议修改 Prompt 或重试。"
                            return f"API 返回内容为空字符串。finish_reason: {finish_reason}。"

                    # 尝试提取文本内容作为错误信息提示
                    diag_msg = "未找到图片数据"
                    if "choices" in res_data and res_data["choices"]:
                            c0 = res_data["choices"][0]
                            if c0.get("message", {}).get("content"):
                                diag_msg = f"API返回了文本而非图片: {c0['message']['content'][:200]}"
                    
                    return f"API请求成功但{diag_msg}。Raw: {str(res_data)[:200]}..."

                # 如果是 Base64 直接返回 Bytes
                if img_url.startswith("data:"):
                    return base64.b64decode(img_url.split(",")[-1])

                # 如果是 URL，需要再次下载
                async with session.get(img_url, proxy=proxy) as img_resp:
                    return await img_resp.read()

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
