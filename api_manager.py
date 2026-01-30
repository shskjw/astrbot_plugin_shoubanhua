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
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
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
                # 1. 尝试匹配 Markdown 图片语法 ![...](url) - 这种更精确
                # 匹配 ![description](http...) 或 ![description](data...)
                # group(1) 是 url
                md_match = re.search(r'!\[.*?\]\((.*?)\)', content)
                if md_match:
                    url_part = md_match.group(1)
                    return url_part.strip().strip("'\"")

                # 2. 尝试匹配纯 Base64 标记 (data:image/...)
                # 这种格式比较明显，优先级高
                b64_match = re.search(r'(data:image\/[a-zA-Z]+;base64,[a-zA-Z0-9+/=]+)', content)
                if b64_match:
                    return b64_match.group(1)

                # 3. 尝试匹配 HTTP/HTTPS URL
                # 这是一个比较宽泛的匹配
                url_match = re.search(r'(https?://[^\s<>")\]]+)', content)
                if url_match:
                    # 去掉末尾可能的标点
                    return url_match.group(1).rstrip(")>,'\".")

        except Exception as e:
            logger.error(f"Error parsing API response: {e}")
        return None

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
                # Gemini 官方协议不需要 data:image/png;base64 前缀，只需要纯 base64 字符串
                parts.append({"inlineData": {"mimeType": "image/png", "data": base64.b64encode(img).decode()}})

            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {"maxOutputTokens": 4096},
                "safetySettings": [{"category": c, "threshold": "BLOCK_NONE"} for c in
                                   ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                                    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
            }
        else:
            # OpenAI 构造
            headers["Authorization"] = f"Bearer {key}"
            msgs = [{"role": "system", "content": "You are an AI artist. Output image only."}]

            if images:
                u_content = [{"type": "text", "text": final_prompt}]
                for img in images:
                    b64 = base64.b64encode(img).decode()
                    # OpenAI 格式通常需要 data URL scheme
                    # 增加 detail: high
                    u_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high"
                        }
                    })
                msgs.append({"role": "user", "content": u_content})
            else:
                msgs.append({"role": "user", "content": final_prompt})

            payload = {"model": model, "messages": msgs, "stream": False}

        # 4. 发送请求
        try:
            timeout_val = self.config.get("timeout", 120)
            timeout = aiohttp.ClientTimeout(total=timeout_val)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers, proxy=proxy) as resp:
                    resp_text = await resp.text()

                    if resp.status != 200:
                        try:
                            err_json = json.loads(resp_text)
                            if "error" in err_json:
                                err_msg = json.dumps(err_json['error'], ensure_ascii=False)
                                return f"API Error {resp.status}: {err_msg}"
                        except:
                            pass
                        if "<html" in resp_text.lower():
                            return f"HTTP {resp.status}: 服务端返回了网页而非数据，请检查URL配置。"
                        return f"HTTP {resp.status}: {resp_text[:200]}"

                    try:
                        res_data = json.loads(resp_text)
                    except json.JSONDecodeError:
                        return f"数据解析失败: 返回内容不是 JSON. 内容: {resp_text[:100]}..."

                    if "error" in res_data:
                        return json.dumps(res_data["error"], ensure_ascii=False)

                    img_url = self.extract_image_url(res_data)
                    if not img_url:
                        # Gemini 特殊错误诊断
                        if "candidates" in res_data and res_data["candidates"]:
                            cand = res_data["candidates"][0]
                            finish_reason = cand.get("finishReason", "UNKNOWN")
                            
                            # 1. 非正常结束
                            if finish_reason not in ["STOP", "MAX_TOKENS"]:
                                return f"生成被终止，原因: {finish_reason} (通常是安全过滤导致)"
                            
                            # 2. 正常结束但无内容
                            content = cand.get("content") or {}
                            if not content.get("parts"):
                                return f"模型未生成任何内容 (finishReason={finish_reason})。可能是由于Prompt被拒绝响应。"
                                
                        return f"API返回成功但未找到图片数据: {str(res_data)[:200]}..."

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