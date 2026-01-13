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
            # 1. Generic OpenAI Image
            if "data" in data and isinstance(data["data"], list):
                if "b64_json" in data["data"][0]: return f"data:image/png;base64,{data['data'][0]['b64_json']}"
                if "url" in data["data"][0]: return data["data"][0]["url"]

            # 2. Chat Completion content
            if "choices" in data:
                content = data["choices"][0]["message"]["content"]
                # Markdown Image
                if match := re.search(r'(data:image\/[a-zA-Z]+;base64,[a-zA-Z0-9+/=]+)', content):
                    return match.group(1)
                # HTTP URL
                if match := re.search(r'https?://[^\s<>")\]]+', content):
                    return match.group(0).rstrip(")>,'\"")

            # 3. Gemini Official
            if "candidates" in data:
                part = data["candidates"][0]["content"]["parts"][0]
                if "inlineData" in part:
                    return f"data:{part['inlineData']['mimeType']};base64,{part['inlineData']['data']}"
        except:
            pass
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
        url = base

        # 画质强化 Prompt
        res_set = self.config.get("image_resolution", "1K")
        final_prompt = f"(Masterpiece, Best Quality, {res_set} Resolution), {prompt}" if res_set != "1K" else prompt

        if mode == "gemini_official":
            # Gemini 构造
            if "models/" not in url and not url.endswith(":generateContent"):
                url = url.rstrip("/") + f"/models/{model}:generateContent"
            headers["x-goog-api-key"] = key

            parts = [{"text": final_prompt}]
            for img in images:
                parts.append({"inlineData": {"mimeType": "image/png", "data": base64.b64encode(img).decode()}})

            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {"maxOutputTokens": 2048},
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
                    u_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
                msgs.append({"role": "user", "content": u_content})
            else:
                msgs.append({"role": "user", "content": final_prompt})

            payload = {"model": model, "messages": msgs, "stream": False}

        # 4. 发送请求
        try:
            timeout = aiohttp.ClientTimeout(total=self.config.get("timeout", 120))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers, proxy=proxy) as resp:
                    if resp.status != 200:
                        return f"HTTP {resp.status}: {await resp.text()}"

                    res_data = await resp.json()

                    if "error" in res_data:
                        return json.dumps(res_data["error"], ensure_ascii=False)

                    img_url = self.extract_image_url(res_data)
                    if not img_url:
                        return f"未找到图片数据: {str(res_data)[:100]}..."

                    # 如果是 Base64 直接返回 Bytes
                    if img_url.startswith("data:"):
                        return base64.b64decode(img_url.split(",")[-1])

                    # 如果是 URL，需要再次下载
                    async with session.get(img_url, proxy=proxy) as img_resp:
                        return await img_resp.read()

        except Exception as e:
            logger.error(f"API Call Error: {e}")
            return f"系统错误: {e}"