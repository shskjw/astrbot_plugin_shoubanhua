import asyncio
import base64
import functools
import io
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import aiohttp
from PIL import Image as PILImage

from astrbot import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import At, Image, Reply, Plain
from astrbot.core.platform.astr_message_event import AstrMessageEvent

PRESET_MODELS = [
    "nano-banana",
    "nano-banana-2-4k",
    "nano-banana-2-2k",
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image",
    "nano-banana-hd",
    "gemini-2.5-flash-image-preview"
]


@register(
    "astrbot_plugin_shoubanhua",
    "shskjw",
    "Google Gemini 手办化/图生图插件",
    "1.5.6",
    "https://github.com/shkjw/astrbot_plugin_shoubanhua",
)
class FigurineProPlugin(Star):
    class ImageWorkflow:
        def __init__(self, proxy_url: str | None = None, max_retries: int = 3, timeout: int = 60):
            if proxy_url:
                logger.info(f"ImageWorkflow 使用代理: {proxy_url}")
            self.proxy = proxy_url
            self.max_retries = max_retries
            self.timeout = timeout

        async def _download_image(self, url: str) -> bytes | None:
            logger.info(f"正在下载图片: {url}")

            for i in range(self.max_retries + 1):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, proxy=self.proxy, timeout=self.timeout) as resp:
                            resp.raise_for_status()
                            return await resp.read()
                except Exception as e:
                    if i < self.max_retries:
                        logger.warning(f"下载失败 ({i + 1}/{self.max_retries}): {e}, 1秒后重试...")
                        await asyncio.sleep(1)
                    else:
                        logger.error(f"下载最终失败: {url}, 错误: {e}")
                        return None
            return None

        async def _get_avatar(self, user_id: str) -> bytes | None:
            if not user_id.isdigit():
                return None

            avatar_url = f"https://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640"
            return await self._download_image(avatar_url)

        def _extract_first_frame_sync(self, raw: bytes) -> bytes:
            img_io = io.BytesIO(raw)
            try:
                with PILImage.open(img_io) as img:
                    if getattr(img, "is_animated", False):
                        img.seek(0)

                    img_converted = img.convert("RGBA")
                    out_io = io.BytesIO()
                    img_converted.save(out_io, format="PNG")
                    return out_io.getvalue()
            except Exception:
                pass
            return raw

        async def _load_bytes(self, src: str) -> bytes | None:
            raw: bytes | None = None
            loop = asyncio.get_running_loop()

            if Path(src).is_file():
                raw = await loop.run_in_executor(None, Path(src).read_bytes)
            elif src.startswith("http"):
                raw = await self._download_image(src)
            elif src.startswith("base64://"):
                raw = await loop.run_in_executor(None, base64.b64decode, src[9:])

            if not raw:
                return None

            return await loop.run_in_executor(None, self._extract_first_frame_sync, raw)

        async def get_images(self, event: AstrMessageEvent) -> List[bytes]:
            img_bytes_list: List[bytes] = []
            at_user_ids: List[str] = []

            for seg in event.message_obj.message:
                if isinstance(seg, Reply) and seg.chain:
                    for s_chain in seg.chain:
                        if isinstance(s_chain, Image):
                            if s_chain.url and (img := await self._load_bytes(s_chain.url)):
                                img_bytes_list.append(img)
                            elif s_chain.file and (img := await self._load_bytes(s_chain.file)):
                                img_bytes_list.append(img)

            for seg in event.message_obj.message:
                if isinstance(seg, Image):
                    if seg.url and (img := await self._load_bytes(seg.url)):
                        img_bytes_list.append(img)
                    elif seg.file and (img := await self._load_bytes(seg.file)):
                        img_bytes_list.append(img)
                elif isinstance(seg, At):
                    at_user_ids.append(str(seg.qq))

            if img_bytes_list:
                return img_bytes_list

            if at_user_ids:
                for user_id in at_user_ids:
                    if avatar := await self._get_avatar(user_id):
                        img_bytes_list.append(avatar)
                return img_bytes_list

            if avatar := await self._get_avatar(event.get_sender_id()):
                img_bytes_list.append(avatar)

            return img_bytes_list

        async def terminate(self):
            pass

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.plugin_data_dir = StarTools.get_data_dir()

        self.user_counts_file = self.plugin_data_dir / "user_counts.json"
        self.group_counts_file = self.plugin_data_dir / "group_counts.json"
        self.user_checkin_file = self.plugin_data_dir / "user_checkin.json"

        self.user_counts: Dict[str, int] = {}
        self.group_counts: Dict[str, int] = {}
        self.user_checkin_data: Dict[str, str] = {}
        self.prompt_map: Dict[str, str] = {}

        self.key_index = 0
        self.key_lock = asyncio.Lock()

        self.iwf: Optional[FigurineProPlugin.ImageWorkflow] = None

    async def initialize(self):
        use_proxy = self.conf.get("use_proxy", False)
        proxy_url = self.conf.get("proxy_url") if use_proxy else None
        
        retries = self.conf.get("download_retries", 3)
        timeout = self.conf.get("timeout", 120)

        self.iwf = self.ImageWorkflow(proxy_url, max_retries=retries, timeout=timeout)

        await self._load_user_counts()
        await self._load_group_counts()
        await self._load_user_checkin_data()
        await self._load_prompt_map()

        logger.info("FigurinePro 插件已加载")

        if not self.conf.get("api_keys") and not self.conf.get("custom_model_1_key") and not self.conf.get(
                "custom_model_2_key"):
            logger.warning("FigurinePro: 未配置任何 API 密钥")

    async def _load_prompt_map(self):
        self.prompt_map.clear()

        prompts_cfg = self.conf.get("prompts", {})
        if isinstance(prompts_cfg, dict):
            for k, v in prompts_cfg.items():
                if isinstance(v, dict) and "default" in v:
                    self.prompt_map[k] = v["default"]
                elif isinstance(v, str):
                    self.prompt_map[k] = v

        prompt_list = self.conf.get("prompt_list", [])
        if isinstance(prompt_list, list):
            for item in prompt_list:
                if ":" in item:
                    k, v = item.split(":", 1)
                    self.prompt_map[k.strip()] = v.strip()

    def _get_all_models(self) -> List[str]:
        models = list(PRESET_MODELS)

        c1 = self.conf.get("custom_model_1", "").strip()
        c2 = self.conf.get("custom_model_2", "").strip()

        if c1:
            models.append(c1)
        if c2:
            models.append(c2)

        return models

    def is_global_admin(self, event: AstrMessageEvent) -> bool:
        return event.get_sender_id() in self.context.get_config().get("admins_id", [])

    @filter.command("切换API模式", aliases={"SwitchApi"}, prefix_optional=True)
    async def on_switch_api_mode(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            yield event.plain_result("❌ 只有管理员可以执行此操作。")
            return

        current_mode = self.conf.get("api_mode", "generic")
        raw = event.message_str.strip()
        parts = raw.split()
        target_mode = parts[1].lower() if len(parts) > 1 else ""

        if not target_mode:
            msg = f"ℹ️ 当前 API 模式: **{current_mode}**\n"
            msg += "可选项:\n"
            msg += "1. `generic` (通用格式)\n"
            msg += "2. `gemini_official` (Gemini原生格式)\n"
            msg += "用法: `#切换API模式 <模式名>`"
            yield event.plain_result(msg)
            return

        if target_mode not in ["generic", "gemini_official"]:
            yield event.plain_result("❌ 模式无效。")
            return

        self.conf["api_mode"] = target_mode
        try:
            if hasattr(self.conf, "save"):
                self.conf.save()
        except:
            pass

        yield event.plain_result(f"✅ API 模式已切换为: **{target_mode}**")

    @filter.command("切换模型", aliases={"SwitchModel", "模型列表"}, prefix_optional=True)
    async def on_switch_model(self, event: AstrMessageEvent):
        all_models = self._get_all_models()
        raw_msg = event.message_str.strip()
        parts = raw_msg.split()

        if len(parts) == 1:
            current_model = self.conf.get("model", "nano-banana")
            current_api_mode = self.conf.get("api_mode", "generic")

            msg = "📋 **可用模型列表**:\n"
            msg += "------------------\n"

            for idx, model_name in enumerate(all_models):
                seq_num = idx + 1
                status = "✅ (当前)" if model_name == current_model else ""
                is_custom = idx >= len(PRESET_MODELS)
                type_mark = " [自]" if is_custom else ""
                msg += f"{seq_num}. {model_name}{type_mark} {status}\n"

            msg += "------------------\n"
            msg += f"📡 **当前API模式**: {current_api_mode}\n"
            msg += "------------------\n"
            msg += "📝 **指令**:\n"
            msg += "1. `#切换模型 <序号>`\n"
            msg += "2. `#切换API模式 <模式名>`\n"
            msg += "3. `#手办化(序号) [图片]`"

            yield event.plain_result(msg)
            return

        arg = parts[1]
        if not self.is_global_admin(event):
            yield event.plain_result("❌ 只有管理员可以更改全局默认模型。")
            return

        if not arg.isdigit():
            yield event.plain_result("❌ 格式错误。请输入数字序号。")
            return

        target_idx = int(arg) - 1

        if 0 <= target_idx < len(all_models):
            new_model = all_models[target_idx]
            self.conf["model"] = new_model
            try:
                if hasattr(self.conf, "save"):
                    self.conf.save()
            except:
                pass
            yield event.plain_result(f"✅ 切换成功！\n当前默认模型: **{new_model}**")
        else:
            yield event.plain_result(f"❌ 序号无效。")

    async def _get_pool_api_key(self) -> str | None:
        keys = self.conf.get("api_keys", [])
        if not keys:
            return None

        async with self.key_lock:
            key = keys[self.key_index]
            self.key_index = (self.key_index + 1) % len(keys)
            return key

    def _extract_image_url_from_response(self, data: Dict[str, Any]) -> str | None:
        try:
            if "candidates" in data:
                parts = data["candidates"][0]["content"]["parts"]
                for p in parts:
                    if "inlineData" in p:
                        return f"data:{p['inlineData']['mimeType']};base64,{p['inlineData']['data']}"
                    if "text" in p:
                        match = re.search(r'https?://[^\s<>")\]]+', p["text"])
                        if match:
                            return match.group(0).rstrip(")>,'\"")
        except:
            pass

        try:
            return data["choices"][0]["message"]["images"][0]["image_url"]["url"]
        except:
            pass

        try:
            if "choices" in data:
                content = data["choices"][0]["message"]["content"]
                match = re.search(r'https?://[^\s<>")\]]+', content)
                if match:
                    return match.group(0).rstrip(")>,'\"")
        except:
            pass

        return None

    async def _call_api(self, image_bytes_list: List[bytes], prompt: str,
                        override_model: str | None = None) -> bytes | str:
        api_url = self.conf.get("api_url")
        if not api_url:
            return "API URL 未配置"

        model_name = override_model or self.conf.get("model", "nano-banana")

        api_key = None
        c1 = self.conf.get("custom_model_1", "").strip()
        c2 = self.conf.get("custom_model_2", "").strip()

        if c1 and model_name == c1:
            api_key = self.conf.get("custom_model_1_key") or await self._get_pool_api_key()
        elif c2 and model_name == c2:
            api_key = self.conf.get("custom_model_2_key") or await self._get_pool_api_key()
        else:
            api_key = await self._get_pool_api_key()

        if not api_key:
            return "无可用 API Key (请检查通用池或自定义Key配置)"

        headers = {
            "Content-Type": "application/json",
            "Connection": "close"
        }

        api_mode = self.conf.get("api_mode", "generic")
        payload = {}
        final_url = api_url

        if api_mode == "gemini_official":
            if "models/" in api_url:
                base = api_url.split("models/")[0]
                final_url = f"{base}models/{model_name}:generateContent"
            else:
                base = api_url.rstrip("/")
                if not base.endswith("v1beta"):
                    base += "/v1beta"
                final_url = f"{base}/models/{model_name}:generateContent"

            headers["x-goog-api-key"] = api_key

            parts = [{"text": prompt}]
            for img in image_bytes_list:
                b64 = base64.b64encode(img).decode("utf-8")
                parts.append({
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": b64
                    }
                })

            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {"maxOutputTokens": 2048},
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            }

        else:
            headers["Authorization"] = f"Bearer {api_key}"

            content = [{"type": "text", "text": prompt}]
            for img in image_bytes_list:
                b64 = base64.b64encode(img).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })

            use_stream = self.conf.get("use_stream", True)
            payload = {
                "model": model_name,
                "max_tokens": 1500,
                "stream": use_stream,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": content}
                ]
            }
        
        timeout = self.conf.get("timeout", 120)

        try:
            if not self.iwf:
                return "工作流未初始化"

            async with aiohttp.ClientSession() as session:
                async with session.post(final_url, json=payload, headers=headers, proxy=self.iwf.proxy,
                                        timeout=timeout) as resp:

                    if resp.status == 404 and api_mode == "gemini_official":
                        return f"API 404错误: 模型 '{model_name}' 不存在或路径错误。\nURL: {final_url}"

                    if resp.status != 200:
                        text = await resp.text()
                        return f"API 请求失败 (HTTP {resp.status}): {text[:300]}"

                    # 处理流式响应
                    if api_mode == "generic" and payload.get("stream"):
                        full_content = ""
                        try:
                            async for line in resp.content:
                                line_str = line.decode('utf-8').strip()
                                if not line_str or line_str.startswith(":"):
                                    continue
                                
                                if line_str == "data: [DONE]":
                                    break
                                
                                if line_str.startswith("data: "):
                                    json_str = line_str[6:]
                                    try:
                                        chunk = json.loads(json_str)
                                        if "choices" in chunk and len(chunk["choices"]) > 0:
                                            delta = chunk["choices"][0].get("delta", {})
                                            if "content" in delta:
                                                full_content += delta["content"]
                                    except json.JSONDecodeError:
                                        continue
                            
                            # 构造模拟响应数据以复用现有提取逻辑
                            data = {
                                "choices": [{
                                    "message": {
                                        "content": full_content
                                    }
                                }]
                            }
                        except Exception as e:
                            logger.error(f"流式响应解析失败: {e}", exc_info=True)
                            return f"流式响应解析错误: {e}"
                    else:
                        # 非流式响应
                        data = await resp.json()

                    if "error" in data:
                        return json.dumps(data["error"], ensure_ascii=False)

                    if "promptFeedback" in data:
                        pf = data["promptFeedback"]
                        if pf.get("blockReason"):
                            return f"Gemini 安全拦截: {pf['blockReason']}"

                    url_or_b64 = self._extract_image_url_from_response(data)

                    if not url_or_b64:
                        return f"生成失败，无图片数据。响应: {json.dumps(data)[:200]}..."

                    if url_or_b64.startswith("data:"):
                        b64 = url_or_b64.split(",")[-1]
                        return base64.b64decode(b64)
                    else:
                        return await self.iwf._download_image(url_or_b64) or "下载生成图片失败"

        except asyncio.TimeoutError:
            return "请求超时"
        except Exception as e:
            logger.error(f"API 调用异常: {e}", exc_info=True)
            return f"系统错误: {e}"

    @filter.event_message_type(filter.EventMessageType.ALL, priority=5)
    async def on_figurine_request(self, event: AstrMessageEvent):
        if self.conf.get("prefix", True) and not event.is_at_or_wake_command:
            return

        text = event.message_str.strip()
        if not text:
            return

        full_cmd_match = text.split()[0].strip()
        suffix_match = re.search(r"[\(（](\d+)[\)）]$", full_cmd_match)

        temp_model_idx = None
        cmd = full_cmd_match

        if suffix_match:
            temp_model_idx = int(suffix_match.group(1))
            cmd = full_cmd_match[:suffix_match.start()]

        bnn_command = self.conf.get("extra_prefix", "bnn")
        user_prompt = ""
        is_bnn = False

        if cmd == bnn_command:
            user_prompt = text.removeprefix(full_cmd_match).strip()
            is_bnn = True
            if not user_prompt:
                return

        elif cmd in self.prompt_map:
            user_prompt = self.prompt_map.get(cmd)

        else:
            cmd_map = {
                "手办化": "figurine_1", "手办化2": "figurine_2", "手办化3": "figurine_3",
                "手办化4": "figurine_4", "手办化5": "figurine_5", "手办化6": "figurine_6",
                "Q版化": "q_version",
                "痛屋化": "pain_room_1", "痛屋化2": "pain_room_2",
                "痛车化": "pain_car",
                "cos化": "cos", "cos自拍": "cos_selfie",
                "孤独的我": "clown",
                "第三视角": "view_3", "鬼图": "ghost", "第一视角": "view_1",
                "手办化帮助": "help"
            }
            if cmd in cmd_map:
                key = cmd_map[cmd]
                if key == "help":
                    yield event.plain_result(self.conf.get("help_text", "帮助未配置"))
                    return
                user_prompt = self.prompt_map.get(key)

            if not user_prompt:
                return

        if not user_prompt:
            yield event.plain_result(f"❌ 指令 '{cmd}' 未配置提示词。")
            return

        sender_id = event.get_sender_id()
        group_id = event.get_group_id()
        is_master = self.is_global_admin(event)

        if not is_master:
            if sender_id in self.conf.get("user_blacklist", []): return
            if group_id and group_id in self.conf.get("group_blacklist", []): return
            if self.conf.get("user_whitelist", []) and sender_id not in self.conf.get("user_whitelist", []): return
            if group_id and self.conf.get("group_whitelist", []) and group_id not in self.conf.get("group_whitelist",
                                                                                                   []): return

            user_count = self._get_user_count(sender_id)
            has_user_cnt = self.conf.get("enable_user_limit", True) and user_count > 0

            if self.conf.get("enable_user_limit", True) and not has_user_cnt:
                if group_id and self.conf.get("enable_group_limit", False):
                    if self._get_group_count(group_id) <= 0:
                        yield event.plain_result("❌ 本群和您的次数均已用尽。")
                        return
                else:
                    yield event.plain_result("❌ 您的使用次数已用完。")
                    return

        if not self.iwf or not (img_bytes_list := await self.iwf.get_images(event)):
            if not is_bnn:
                yield event.plain_result("请发送或引用一张图片。")
                return

        images_to_process = []
        display_cmd = cmd
        if is_bnn:
            MAX_IMAGES = 5
            if len(img_bytes_list) > MAX_IMAGES:
                images_to_process = img_bytes_list[:MAX_IMAGES]
                yield event.plain_result(f"🎨 检测到 {len(img_bytes_list)} 张图片，已选取前 {MAX_IMAGES} 张…")
            else:
                images_to_process = img_bytes_list
            display_cmd = user_prompt[:10] + '...' if len(user_prompt) > 10 else user_prompt
        else:
            images_to_process = [img_bytes_list[0]]

        override_model_name = None
        all_models = self._get_all_models()
        if temp_model_idx is not None:
            if 1 <= temp_model_idx <= len(all_models):
                override_model_name = all_models[temp_model_idx - 1]
                display_cmd += f" (模型: {override_model_name})"
            else:
                yield event.plain_result(f"⚠️ 指定的模型序号 {temp_model_idx} 无效，将使用默认模型。")

        yield event.plain_result(f"🎨 收到请求，正在生成 [{display_cmd}]...")

        if not is_master:
            if self.conf.get("enable_group_limit", False) and group_id and self._get_group_count(group_id) > 0:
                await self._decrease_group_count(group_id)
            elif self.conf.get("enable_user_limit", True) and self._get_user_count(sender_id) > 0:
                await self._decrease_user_count(sender_id)

        start_time = datetime.now()
        res = await self._call_api(images_to_process, user_prompt, override_model=override_model_name)
        elapsed = (datetime.now() - start_time).total_seconds()

        if isinstance(res, bytes):
            caption_parts = [f"✅ 生成成功 ({elapsed:.2f}s)", f"预设: {display_cmd}"]
            if is_master:
                caption_parts.append("剩余: ∞")
            else:
                if self.conf.get("enable_user_limit", True):
                    caption_parts.append(f"个人: {self._get_user_count(sender_id)}")

            yield event.chain_result([Image.fromBytes(res), Plain(" | ".join(caption_parts))])
        else:
            msg = f"❌ 生成失败 ({elapsed:.2f}s)\n原因: {res}"
            if not is_master:
                msg += "\n(注: 触发即扣次)"
            yield event.plain_result(msg)

        event.stop_event()

    @filter.command("文生图", prefix_optional=True)
    async def on_text_to_image(self, event: AstrMessageEvent):
        raw_cmd = event.message_str.strip()
        prompt = raw_cmd
        override_model_name = None

        match = re.match(r"^[\(（](\d+)[\)）]\s*(.*)", prompt)
        if match:
            idx = int(match.group(1))
            prompt = match.group(2)
            all_models = self._get_all_models()
            if 1 <= idx <= len(all_models):
                override_model_name = all_models[idx - 1]
            else:
                yield event.plain_result(f"⚠️ 指定的模型序号 {idx} 无效。")
                return

        if not prompt:
            yield event.plain_result("请提供描述。用法: #文生图 [可选:(序号)] <描述>")
            return

        sender_id = event.get_sender_id()
        if not self.is_global_admin(event):
            if self.conf.get("enable_user_limit", True) and self._get_user_count(sender_id) <= 0:
                yield event.plain_result("❌ 您的使用次数已用完。")
                return

        info_str = f"🎨 收到文生图请求，正在生成 [{prompt[:10]}...]"
        if override_model_name:
            info_str += f" (模型: {override_model_name})"
        yield event.plain_result(info_str)

        if not self.is_global_admin(event):
            if self.conf.get("enable_user_limit", True) and self._get_user_count(sender_id) > 0:
                await self._decrease_user_count(sender_id)

        start_time = datetime.now()
        res = await self._call_api([], prompt, override_model=override_model_name)
        elapsed = (datetime.now() - start_time).total_seconds()

        if isinstance(res, bytes):
            yield event.chain_result([Image.fromBytes(res), Plain(f"✅ 生成成功 ({elapsed:.2f}s)")])
        else:
            yield event.plain_result(f"❌ 生成失败: {res}")

        event.stop_event()

    @filter.command("设置自定义key", aliases={"setk"}, prefix_optional=True)
    async def set_custom_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return

        parts = event.message_str.strip().split()
        if len(parts) < 3:
            yield event.plain_result("格式错误。用法: #设置自定义key <1或2> <key>")
            return

        idx = parts[1]
        key_val = parts[2]
        if idx == "1":
            self.conf["custom_model_1_key"] = key_val
            msg = "✅ 自定义模型1 的 Key 已更新。"
        elif idx == "2":
            self.conf["custom_model_2_key"] = key_val
            msg = "✅ 自定义模型2 的 Key 已更新。"
        else:
            yield event.plain_result("❌ 仅支持设置 1 或 2。")
            return

        try:
            if hasattr(self.conf, "save"):
                self.conf.save()
        except:
            pass

        yield event.plain_result(msg)

    @filter.command("lm添加", aliases={"lma"}, prefix_optional=True)
    async def add_lm_prompt(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return

        raw = re.sub(r'^[#\/]?(lm添加|lma)\s*', '', event.message_str.strip(), flags=re.IGNORECASE).strip()
        if ":" not in raw:
            yield event.plain_result('格式错误, 示例: 触发词:提示词')
            return

        key, new_value = map(str.strip, raw.split(":", 1))
        prompt_list = self.conf.get("prompt_list", [])
        found = False

        for idx, item in enumerate(prompt_list):
            if item.strip().startswith(key + ":"):
                prompt_list[idx] = f"{key}:{new_value}"
                found = True
                break

        if not found:
            prompt_list.append(f"{key}:{new_value}")

        self.conf["prompt_list"] = prompt_list
        if hasattr(self.conf, "save"):
            self.conf.save()

        await self._load_prompt_map()
        yield event.plain_result(f"✅ 已保存预设:\n{key}:{new_value}")

    @filter.command("lm帮助", aliases={"lmh", "手办化帮助"}, prefix_optional=True)
    async def on_prompt_help(self, event: AstrMessageEvent):
        parts = event.message_str.strip().split()
        keyword = parts[1] if len(parts) > 1 else ""

        if not keyword:
            if help_text := self.conf.get("help_text"):
                yield event.plain_result(help_text)
                return
            keys = sorted(list(self.prompt_map.keys()))
            yield event.plain_result(f"🎨 预设列表: {', '.join(keys) or '(无)'}")
            return

        prompt = self.prompt_map.get(keyword)
        yield event.plain_result(f"📄 预设 [{keyword}] 内容:\n{prompt}" if prompt else f"❌ 未找到 [{keyword}]")

    async def _load_user_counts(self):
        if not self.user_counts_file.exists():
            self.user_counts = {}
            return
        try:
            content = await asyncio.to_thread(self.user_counts_file.read_text, "utf-8")
            self.user_counts = json.loads(content)
        except:
            self.user_counts = {}

    async def _save_user_counts(self):
        try:
            data = json.dumps(self.user_counts, indent=4)
            await asyncio.to_thread(self.user_counts_file.write_text, data, "utf-8")
        except:
            pass

    def _get_user_count(self, uid: str) -> int:
        return self.user_counts.get(str(uid), 0)

    async def _decrease_user_count(self, uid: str):
        count = self._get_user_count(uid)
        if count > 0:
            self.user_counts[str(uid)] = count - 1
            await self._save_user_counts()

    async def _load_group_counts(self):
        if not self.group_counts_file.exists():
            self.group_counts = {}
            return
        try:
            content = await asyncio.to_thread(self.group_counts_file.read_text, "utf-8")
            self.group_counts = json.loads(content)
        except:
            self.group_counts = {}

    async def _save_group_counts(self):
        try:
            data = json.dumps(self.group_counts, indent=4)
            await asyncio.to_thread(self.group_counts_file.write_text, data, "utf-8")
        except:
            pass

    def _get_group_count(self, group_id: str) -> int:
        return self.group_counts.get(str(group_id), 0)

    async def _decrease_group_count(self, group_id: str):
        count = self._get_group_count(group_id)
        if count > 0:
            self.group_counts[str(group_id)] = count - 1
            await self._save_group_counts()

    async def _load_user_checkin_data(self):
        if not self.user_checkin_file.exists():
            self.user_checkin_data = {}
            return
        try:
            content = await asyncio.to_thread(self.user_checkin_file.read_text, "utf-8")
            self.user_checkin_data = json.loads(content)
        except:
            self.user_checkin_data = {}

    async def _save_user_checkin_data(self):
        try:
            data = json.dumps(self.user_checkin_data, indent=4)
            await asyncio.to_thread(self.user_checkin_file.write_text, data, "utf-8")
        except:
            pass

    @filter.command("手办化签到", prefix_optional=True)
    async def on_checkin(self, event: AstrMessageEvent):
        if not self.conf.get("enable_checkin", False):
            yield event.plain_result("📅 签到未开启。")
            return

        uid = event.get_sender_id()
        today = datetime.now().strftime("%Y-%m-%d")

        if self.user_checkin_data.get(uid) == today:
            yield event.plain_result(f"已签到。剩余: {self._get_user_count(uid)}")
            return

        reward = int(self.conf.get("checkin_fixed_reward", 3))
        if self.conf.get("enable_random_checkin", False):
            max_r = int(self.conf.get("checkin_random_reward_max", 5))
            reward = random.randint(1, max(1, max_r))

        self.user_counts[uid] = self._get_user_count(uid) + reward
        await self._save_user_counts()
        self.user_checkin_data[uid] = today
        await self._save_user_checkin_data()

        yield event.plain_result(f"🎉 签到成功 +{reward}次。")

    @filter.command("手办化增加用户次数", prefix_optional=True)
    async def on_add_user_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return

        text = event.message_str.strip()
        at_seg = next((s for s in event.message_obj.message if isinstance(s, At)), None)
        target, count = None, 0

        if at_seg:
            target = str(at_seg.qq)
            match = re.search(r"(\d+)\s*$", text)
            if match:
                count = int(match.group(1))
        else:
            match = re.search(r"(\d+)\s+(\d+)", text)
            if match:
                target, count = match.group(1), int(match.group(2))

        if target:
            self.user_counts[str(target)] = self._get_user_count(target) + count
            await self._save_user_counts()
            yield event.plain_result(f"✅ 已为 {target} 增加 {count} 次。")

    @filter.command("手办化增加群组次数", prefix_optional=True)
    async def on_add_group_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return

        match = re.search(r"(\d+)\s+(\d+)", event.message_str.strip())
        if match:
            gid, count = match.group(1), int(match.group(2))
            self.group_counts[str(gid)] = self._get_group_count(gid) + count
            await self._save_group_counts()
            yield event.plain_result(f"✅ 已为群 {gid} 增加 {count} 次。")

    @filter.command("手办化查询次数", prefix_optional=True)
    async def on_query_counts(self, event: AstrMessageEvent):
        uid = event.get_sender_id()
        if self.is_global_admin(event):
            at_seg = next((s for s in event.message_obj.message if isinstance(s, At)), None)
            if at_seg:
                uid = str(at_seg.qq)
            else:
                match = re.search(r"(\d+)", event.message_str)
                if match:
                    uid = match.group(1)

        msg = f"👤 用户 {uid} 剩余: {self._get_user_count(uid)}"
        if gid := event.get_group_id():
            msg += f"\n👥 本群剩余: {self._get_group_count(gid)}"

        yield event.plain_result(msg)

    @filter.command("手办化添加key", prefix_optional=True)
    async def on_add_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return

        new_keys = event.message_str.strip().split()[1:]
        if not new_keys:
            yield event.plain_result("格式错误。")
            return

        keys = self.conf.get("api_keys", [])
        added = [k for k in new_keys if k not in keys]
        keys.extend(added)
        self.conf["api_keys"] = keys
        if hasattr(self.conf, "save"):
            self.conf.save()

        yield event.plain_result(f"✅ 添加 {len(added)} 个Key。")

    @filter.command("手办化key列表", prefix_optional=True)
    async def on_list_keys(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return

        keys = self.conf.get("api_keys", [])
        msg = "\n".join([f"{i + 1}. {k[:6]}..." for i, k in enumerate(keys)])
        yield event.plain_result(f"🔑 通用 Key 池:\n{msg}")

    @filter.command("手办化删除key", prefix_optional=True)
    async def on_delete_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return

        parts = event.message_str.strip().split()
        if len(parts) < 2:
            yield event.plain_result("格式: #手办化删除key <序号|all>")
            return

        param = parts[1]
        keys = self.conf.get("api_keys", [])

        if param == "all":
            self.conf["api_keys"] = []
        elif param.isdigit():
            idx = int(param) - 1
            if 0 <= idx < len(keys):
                keys.pop(idx)
                self.conf["api_keys"] = keys

        if hasattr(self.conf, "save"):
            self.conf.save()

        yield event.plain_result("✅ 操作完成。")

    async def terminate(self):
        if self.iwf:
            await self.iwf.terminate()
        logger.info("[FigurinePro] 插件已终止")
