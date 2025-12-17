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
from astrbot.core.message.components import At, Image, Reply, Plain, Node, Nodes
from astrbot.core.platform.astr_message_event import AstrMessageEvent


@register(
    "astrbot_plugin_shoubanhua",
    "shskjw",
    "支持第三方所有OpenAI绘图格式和原生Google Gemini 终极缝合怪，文生图/图生图插件",
    "1.7.6",
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

        async def terminate(self):
            """清理资源"""
            pass

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
            """增强的图片获取方法，支持多@用户和混合@与图片"""
            img_bytes_list: List[bytes] = []
            at_user_ids: List[str] = []

            logger.info("=== 开始获取图片资源 ===")
            logger.info(f"消息平台: {event.platform}")
            logger.info(f"消息内容: {event.message_str}")

            # 1. 处理回复链中的图片
            for seg in event.message_obj.message:
                if isinstance(seg, Reply) and seg.chain:
                    logger.info(f"发现回复链，长度: {len(seg.chain)}")
                    for s_chain in seg.chain:
                        if isinstance(s_chain, Image):
                            logger.info("在回复链中发现图片")
                            if s_chain.url and (img := await self._load_bytes(s_chain.url)):
                                img_bytes_list.append(img)
                                logger.info("成功从回复链URL加载图片")
                            elif s_chain.file and (img := await self._load_bytes(s_chain.file)):
                                img_bytes_list.append(img)
                                logger.info("成功从回复链文件加载图片")

            # 2. 处理当前消息中的图片
            for seg in event.message_obj.message:
                if isinstance(seg, Image):
                    logger.info("在当前消息中发现图片")
                    if seg.url and (img := await self._load_bytes(seg.url)):
                        img_bytes_list.append(img)
                        logger.info("成功从当前消息URL加载图片")
                    elif seg.file and (img := await self._load_bytes(seg.file)):
                        img_bytes_list.append(img)
                        logger.info("成功从当前消息文件加载图片")

            # 3. 处理@用户（支持多@）
            for seg in event.message_obj.message:
                if isinstance(seg, At):
                    at_user_ids.append(str(seg.qq))
                    logger.info(f"发现@用户: {seg.qq}")

            # 4. 处理命令文本中的@用户（从文本提取QQ号）
            import re
            text_at_matches = re.findall(r'@(\d+)', event.message_str)
            for qq in text_at_matches:
                if qq not in at_user_ids:
                    at_user_ids.append(qq)
                    logger.info(f"从文本提取到@用户: {qq}")

            logger.info(f"总共发现 {len(at_user_ids)} 个@用户")
            if at_user_ids:
                logger.info(f"@用户详情: {at_user_ids}")

            # 5. 获取@用户的头像
            if at_user_ids:
                for user_id in at_user_ids:
                    logger.info(f"尝试获取用户 [{user_id}] 的头像...")
                    if avatar := await self._get_avatar(user_id):
                        img_bytes_list.append(avatar)
                        logger.info(f"成功获取用户 [{user_id}] 的头像")
                    else:
                        logger.warning(f"无法获取用户 [{user_id}] 的头像")

            logger.info(f"成功获取 {len(img_bytes_list)} 个@用户头像")

            logger.info(f"最终获取到 {len(img_bytes_list)} 张图片")
            return img_bytes_list

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.plugin_data_dir = StarTools.get_data_dir()

        self.user_counts_file = self.plugin_data_dir / "user_counts.json"
        self.group_counts_file = self.plugin_data_dir / "group_counts.json"
        self.user_checkin_file = self.plugin_data_dir / "user_checkin.json"
        self.daily_stats_file = self.plugin_data_dir / "daily_stats.json"
        self.preset_images_file = self.plugin_data_dir / "preset_images.json"
        self.preset_images_dir = self.plugin_data_dir / "preset_images"

        self.user_counts: Dict[str, int] = {}
        self.group_counts: Dict[str, int] = {}
        self.user_checkin_data: Dict[str, str] = {}
        self.daily_stats: Dict[str, Any] = {}
        self.prompt_map: Dict[str, str] = {}
        self.preset_images: Dict[str, str] = {}  # 预设词 -> 图片文件名映射

        self.generic_key_index = 0
        self.gemini_key_index = 0
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
        await self._load_daily_stats()
        await self._load_prompt_map()
        await self._load_preset_images()

        # 创建预设图片目录
        if not self.preset_images_dir.exists():
            self.preset_images_dir.mkdir(parents=True, exist_ok=True)

        logger.info("FigurinePro 插件已加载")

        g_keys = self.conf.get("generic_api_keys", [])
        o_keys = self.conf.get("gemini_api_keys", [])

        if not g_keys and not o_keys:
            logger.warning("FigurinePro: 未配置任何 API Key")

    def _extract_image_urls_from_text(self, text: str) -> List[str]:
        """从文本中提取图片链接和本地文件路径"""
        image_urls = []

        # 1. 匹配本地文件路径 (仅Windows绝对路径)
        # 匹配 C:\path\to\image.jpg 格式
        local_file_patterns = [
            r'[a-zA-Z]:\\[^\s,，。！？\n]+\.(?:jpg|jpeg|png|gif|bmp|webp)',  # Windows绝对路径
        ]

        for pattern in local_file_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match and match not in image_urls:
                    # 检查文件是否存在
                    if Path(match).exists():
                        image_urls.append(match)

        # 2. 匹配常见的图片链接格式
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

    async def _download_preset_image(self, image_url: str) -> bytes | None:
        """下载预设内容中的图片（支持本地文件和网络图片）"""
        import ssl
        from pathlib import Path

        # 清理URL，移除可能的尾随标点符号
        clean_url = image_url.strip().rstrip('.,;:!?')

        # 检查是否是本地文件路径
        if Path(clean_url).is_file():
            logger.info(f"检测到本地文件路径: {clean_url}")
            try:
                # 使用现有的 _load_bytes 方法处理本地文件
                return await self.iwf._load_bytes(clean_url)
            except Exception as e:
                logger.error(f"加载本地文件失败: {clean_url}, 错误: {e}")
                return None

        # 网络图片处理（原有的下载逻辑）
        for attempt in range(3):  # 最多重试3次
            try:
                logger.info(f"正在下载预设内容中的网络图片: {clean_url} (尝试 {attempt + 1}/3)")

                # 创建SSL上下文，允许更多SSL配置
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

                # 创建不使用代理的下载器，使用自定义SSL上下文
                timeout = aiohttp.ClientTimeout(total=60)
                connector = aiohttp.TCPConnector(ssl=ssl_context, limit=10)

                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    async with session.get(clean_url, headers=headers) as resp:
                        resp.raise_for_status()
                        return await resp.read()

            except Exception as e:
                logger.warning(f"下载预设图片失败 (尝试 {attempt + 1}/3): {clean_url}, 错误: {e}")
                if attempt < 2:  # 如果不是最后一次尝试，等待1秒
                    await asyncio.sleep(1)
                else:
                    logger.error(f"下载预设图片最终失败: {clean_url}, 错误: {e}")
                    return None
        return None

    async def _load_prompt_map(self):
        self.prompt_map.clear()

        # 1. 内置基础映射 (硬编码的指令)
        base_cmd_map = {
            "手办化": "figurine_1", "手办化2": "figurine_2", "手办化3": "figurine_3",
            "手办化4": "figurine_4", "手办化5": "figurine_5", "手办化6": "figurine_6",
            "Q版化": "q_version",
            "痛屋化": "pain_room_1", "痛屋化2": "pain_room_2",
            "痛车化": "pain_car",
            "cos化": "cos", "cos自拍": "cos_selfie",
            "孤独的我": "clown",
            "第三视角": "view_3", "鬼图": "ghost", "第一视角": "view_1"
        }
        for k in base_cmd_map.keys():
            self.prompt_map[k] = "[内置预设]"

        # 2. 从配置的 prompts 加载
        prompts_cfg = self.conf.get("prompts", {})
        if isinstance(prompts_cfg, dict):
            for k, v in prompts_cfg.items():
                if isinstance(v, dict) and "default" in v:
                    self.prompt_map[k] = v["default"]
                elif isinstance(v, str):
                    self.prompt_map[k] = v

        # 3. 从 prompt_list 加载
        prompt_list = self.conf.get("prompt_list", [])
        if isinstance(prompt_list, list):
            for item in prompt_list:
                if ":" in item:
                    k, v = item.split(":", 1)
                    self.prompt_map[k.strip()] = v.strip()

    def _get_all_models(self) -> List[str]:
        """从配置的 model_list 中获取所有 model ID (纯字符串列表)"""
        raw_list = self.conf.get("model_list", [])
        models = []
        # 兼容处理：确保返回的是字符串列表
        for item in raw_list:
            if isinstance(item, str):
                models.append(item.strip())
            elif isinstance(item, dict) and "id" in item:
                # 兼容旧配置
                models.append(item["id"])
        return models

    def is_global_admin(self, event: AstrMessageEvent) -> bool:
        return event.get_sender_id() in self.context.get_config().get("admins_id", [])

    def _norm_id(self, raw_id: Any) -> str:
        if raw_id is None:
            return ""
        return str(raw_id).strip()

    @filter.command("切换API模式", aliases={"SwitchApi"}, prefix_optional=True)
    async def on_switch_api_mode(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            yield event.plain_result("❌ 只有管理员可以执行此操作。")
            return

        current_mode = self.conf.get("api_mode", "generic")
        raw = event.message_str.strip()
        parts = raw.split()
        
        # 别名映射
        alias_map = {
            "gemini": "gemini_official",
            "google": "gemini_official",
            "official": "gemini_official",
            "openai": "generic",
            "gpt": "generic",
            "3rd": "generic",
            "generic": "generic",
            "gemini_official": "gemini_official"
        }

        if len(parts) <= 1:
            msg = f"ℹ️ 当前 API 模式: **{current_mode}**\n"
            msg += "可选项:\n"
            msg += "1. `generic` (通用OpenAI格式)\n"
            msg += "2. `gemini_official` (Gemini官方格式)\n"
            msg += "用法: `#切换API模式 <模式名>`"
            yield event.plain_result(msg)
            return

        input_mode = parts[1].lower().strip()
        target_mode = alias_map.get(input_mode)

        if not target_mode:
            yield event.plain_result("❌ 模式无效。支持: generic, gemini_official (或 gemini, openai)")
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
            resolution = self.conf.get("image_resolution", "1K")

            msg = "📋 **可用模型列表**:\n"
            msg += "------------------\n"

            for idx, model_name in enumerate(all_models):
                seq_num = idx + 1
                status = "✅ (当前)" if model_name == current_model else ""
                msg += f"{seq_num}. {model_name} {status}\n"

            msg += "------------------\n"
            msg += f"📡 **当前API模式**: {current_api_mode}\n"
            msg += f"🖥️ **画质设置**: {resolution}\n"
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

    async def _get_pool_api_key(self, mode: str, use_power_mode: bool = False) -> str | None:
        keys = []
        async with self.key_lock:
            if use_power_mode:
                # 强力模式优先使用独立的Key池
                if mode == "gemini_official":
                    power_keys = self.conf.get("power_gemini_api_keys", [])
                    # 如果强力模式Key池为空，使用普通模式的Key池
                    if not power_keys:
                        keys = self.conf.get("gemini_api_keys", [])
                    else:
                        keys = power_keys
                else:
                    power_keys = self.conf.get("power_generic_api_keys", [])
                    # 如果强力模式Key池为空，使用普通模式的Key池
                    if not power_keys:
                        keys = self.conf.get("generic_api_keys", [])
                    else:
                        keys = power_keys
            else:
                # 普通模式使用常规Key池
                if mode == "gemini_official":
                    keys = self.conf.get("gemini_api_keys", [])
                else:
                    keys = self.conf.get("generic_api_keys", [])

            if not keys: return None

            if mode == "gemini_official":
                key = keys[self.gemini_key_index]
                self.gemini_key_index = (self.gemini_key_index + 1) % len(keys)
                return key
            else:
                key = keys[self.generic_key_index]
                self.generic_key_index = (self.generic_key_index + 1) % len(keys)
                return key

    def _extract_image_url_from_response(self, data: Dict[str, Any]) -> str | None:
        # 1. 优先检查 content 文本中是否包含 Markdown 格式的 Base64 图片
        # 常见于 nano-banana 等逆向模型，格式: ![image](data:image/png;base64,...)
        try:
            if "choices" in data:
                content = data["choices"][0]["message"]["content"]
                # 匹配 data:image/...;base64,......
                # 使用非贪婪匹配或精确字符集以避免匹配过长
                match = re.search(r'(data:image\/[a-zA-Z]+;base64,[a-zA-Z0-9+/=]+)', content)
                if match:
                    return match.group(1)
        except:
            pass

        # 2. Google Gemini Official Structure
        try:
            if "candidates" in data:
                parts = data["candidates"][0]["content"]["parts"]
                for p in parts:
                    if "inlineData" in p:
                        return f"data:{p['inlineData']['mimeType']};base64,{p['inlineData']['data']}"
                    if "text" in p:
                        # 尝试从文本中提取 http 链接
                        match = re.search(r'https?://[^\s<>")\]]+', p["text"])
                        if match:
                            return match.group(0).rstrip(")>,'\"")
        except:
            pass

        # 3. OpenAI-style Image Generation Structure (DALL-E format)
        try:
            if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                item = data["data"][0]
                if "b64_json" in item:
                    return f"data:image/png;base64,{item['b64_json']}"
                if "url" in item:
                    return item["url"]
        except:
            pass

        # 4. OpenAI-style Chat Completion Structure (Custom providers image_url)
        try:
            return data["choices"][0]["message"]["images"][0]["image_url"]["url"]
        except:
            pass

        # 5. OpenAI-style Chat Completion (Raw HTTP URL in content text)
        try:
            if "choices" in data:
                content = data["choices"][0]["message"]["content"]
                match = re.search(r'https?://[^\s<>")\]]+', content)
                if match:
                    return match.group(0).rstrip(")>,'\"")
        except:
            pass

        return None

    def _build_limit_exhausted_message(
            self,
            group_id: Optional[str],
            use_power_mode: bool = False,
            required_cost: int = 1,
    ) -> str:
        if use_power_mode:
            # 强力模式只提示个人次数不足
            msg = f"❌ 个人次数不足。需要 {required_cost} 次。"
        elif group_id and self.conf.get("enable_group_limit", False):
            msg = "❌ 本群或您的使用次数已用尽 (优先扣除群次数)。"
        else:
            msg = "❌ 您的使用次数已用完。"

        extra_cost = max(0, required_cost - 1)
        if use_power_mode and extra_cost > 0:
            msg += f"\n⚙️ 强力模式每次额外扣除 {extra_cost} 次。"

        if self.conf.get("enable_checkin", False) and self.conf.get("enable_user_limit", True):
            msg += "\n📅 发送 \"手办化签到\" 指令（请按当前命令前缀或唤醒方式触发）可补充个人次数。"

        return msg

    def _get_required_invocation_cost(self, use_power_mode: bool) -> int:
        base_cost = 1
        if use_power_mode and self.conf.get("enable_power_model", False):
            extra = self.conf.get("power_model_extra_cost", 1)
            try:
                extra = int(extra)
            except (TypeError, ValueError):
                extra = 1
            base_cost += max(0, extra)
        return max(1, base_cost)

    def _get_power_mode_hint(self, command_hint: str) -> Optional[str]:
        if not self.conf.get("power_model_tip_enabled", False):
            return None
        if not self.conf.get("enable_power_model", False):
            return None

        keyword = (self.conf.get("power_model_keyword") or "").strip()
        if not keyword:
            return None

        total_cost = self._get_required_invocation_cost(True)
        return f"💡 输入 \"{command_hint} {keyword} ...\" 可消耗 {total_cost} 次个人次数调用强力模型。"

    def _format_error_message(self, status_text: str, elapsed: float, detail: Any) -> str:
        """构造错误消息：默认只发overview，调试模式下在终端输出完整错误"""
        summary = f"❌ {status_text} ({elapsed:.2f}s)"

        # 如果detail包含图片下载失败的信息，返回概述+详细信息给用户
        if isinstance(detail, str) and (
                "图片下载失败" in detail or "图片获取未完成" in detail) and "请手动访问链接查看" in detail:
            # 移除"失败"等敏感词，避免被插件拦截
            safe_detail = detail.replace("图片下载失败", "图片获取未完成").replace("失败", "未完成")
            return f"{summary}\n{safe_detail}"

        if self.conf.get("debug_mode", False):
            logger.error(f"调试模式错误详情: {detail}")
        return summary

    async def _call_api(self, image_bytes_list: List[bytes], prompt: str,
                        override_model: str | None = None, use_power_mode: bool = False) -> bytes | str:

        api_mode = self.conf.get("api_mode", "generic")

        # 根据是否强力模式选择对应的API配置
        if use_power_mode:
            if api_mode == "gemini_official":
                base_url = self.conf.get("power_gemini_api_url", "")
                # 如果强力模式URL为空，使用普通模式的URL
                if not base_url:
                    base_url = self.conf.get("gemini_api_url", "https://generativelanguage.googleapis.com")
            else:
                base_url = self.conf.get("power_generic_api_url", "")
                # 如果强力模式URL为空，使用普通模式的URL
                if not base_url:
                    base_url = self.conf.get("generic_api_url", "https://api.bltcy.ai/v1/chat/completions")
        else:
            if api_mode == "gemini_official":
                base_url = self.conf.get("gemini_api_url", "https://generativelanguage.googleapis.com")
            else:
                base_url = self.conf.get("generic_api_url", "https://api.bltcy.ai/v1/chat/completions")

        if not base_url:
            return "API URL 未配置"

        model_name = override_model or self.conf.get("model", "nano-banana")

        # 根据是否强力模式选择对应的API密钥
        api_key = await self._get_pool_api_key(api_mode, use_power_mode)
        if not api_key:
            return f"无可用 API Key (请在 {api_mode} 池中添加Key)"

        # --- 构造最终 Prompt (注入指令以强制画图) ---
        if len(image_bytes_list) > 0:
            # 图生图
            final_prompt = f"Re-imagine the attached image with the following style/description: {prompt}. Draw it directly. Do not analyze."
        else:
            # 文生图
            final_prompt = f"Generate a high quality image based on this description: {prompt}"
        
        # --- 应用分辨率设置 ---
        resolution_setting = self.conf.get("image_resolution", "1K")
        if resolution_setting and resolution_setting != "1K":
            # 修复：将分辨率提示词移到最前面，并加强权重，确保 Gemini 等模型能生效
            final_prompt = f"(Masterpiece, Best Quality, {resolution_setting} Resolution), {final_prompt}"

        headers = {
            "Content-Type": "application/json",
            "Connection": "keep-alive"
        }

        payload = {}
        final_url = base_url

        if api_mode == "gemini_official":
            base = base_url.rstrip("/")
            if "models/" in base:
                base = base.split("models/")[0].rstrip("/")

            if not base.endswith("v1beta"):
                base += "/v1beta"

            final_url = f"{base}/models/{model_name}:generateContent"

            headers["x-goog-api-key"] = api_key

            parts = [{"text": final_prompt}]
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
                ],
                "toolConfig": {
                    "functionCallingConfig": {
                        "mode": "NONE"
                    }
                }
            }

        else:
            headers["Authorization"] = f"Bearer {api_key}"

            messages = []
            # 优化 System Prompt，极度严格地禁止聊天，强制画图模式
            system_instruction = (
                "You are an expert AI artist tool. Your ONLY job is to generate images based on user inputs. "
                "Do NOT describe the image. Do NOT ask questions. Do NOT start a conversation. "
                "Directly output the generated image url or data."
            )
            messages.append({"role": "system", "content": system_instruction})

            if len(image_bytes_list) > 0:
                # 包含图片的 Vision 请求结构
                user_content_list = [{"type": "text", "text": final_prompt}]
                for img in image_bytes_list:
                    b64 = base64.b64encode(img).decode("utf-8")
                    user_content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    })
                messages.append({"role": "user", "content": user_content_list})
            else:
                # 纯文本请求结构
                messages.append({"role": "user", "content": final_prompt})

            use_stream = self.conf.get("use_stream", True)
            payload = {
                "model": model_name,
                "max_tokens": 4000,
                "stream": use_stream,
                "messages": messages
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

                    if api_mode == "generic" and payload.get("stream"):
                        full_content = ""
                        buffer = b""
                        try:
                            # 修复流式 Chunk too big 问题：
                            # 使用 iter_chunked 绕过 aiohttp 默认的单行长度限制
                            async for chunk in resp.content.iter_chunked(4096):
                                buffer += chunk
                                while b'\n' in buffer:
                                    try:
                                        line_data, buffer = buffer.split(b'\n', 1)
                                        line_str = line_data.decode('utf-8').strip()

                                        if not line_str or line_str.startswith(":"):
                                            continue
                                        if line_str == "data: [DONE]":
                                            break
                                        if line_str.startswith("data: "):
                                            json_str = line_str[6:]
                                            try:
                                                chunk_json = json.loads(json_str)
                                                if "choices" in chunk_json and len(chunk_json["choices"]) > 0:
                                                    delta = chunk_json["choices"][0].get("delta", {})
                                                    if "content" in delta:
                                                        full_content += delta["content"]
                                            except json.JSONDecodeError:
                                                continue
                                    except ValueError:
                                        # 解码失败等情况，跳过当前行
                                        break

                            # 构造完整的响应对象，供后续提取图片使用
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
                        # 尝试下载图片，如果下载失败则返回图片链接
                        downloaded_image = await self.iwf._download_image(url_or_b64)
                        if downloaded_image:
                            return downloaded_image
                        else:
                            logger.warning(f"图片获取未完成，返回图片链接: {url_or_b64}")
                            return f"图片获取未完成，请手动访问链接查看: {url_or_b64}"

        except asyncio.TimeoutError:
            return "请求超时"
        except Exception as e:
            logger.error(f"API 调用异常: {e}", exc_info=True)
            return f"系统错误: {e}"

    # 修复：使用 ctx=None 替代 *args 以避免 _empty() 错误，同时兼容框架传递的额外参数
    @filter.event_message_type(filter.EventMessageType.ALL, priority=5)
    async def on_figurine_request(self, event: AstrMessageEvent, ctx=None):
        if self.conf.get("prefix", True) and not event.is_at_or_wake_command:
            return

        text = event.message_str.strip()
        if not text:
            return

        tokens = text.split()
        if not tokens:
            return

        def _normalize_token(token: str) -> Tuple[str, Optional[int]]:
            token = token.strip()
            match = re.search(r"[\(（](\d+)[\)）]$", token)
            if match:
                idx = int(match.group(1))
                return token[:match.start()].strip(), idx
            return token, None

        raw_cmd_token = tokens[0].strip()
        command_token, temp_model_idx = _normalize_token(raw_cmd_token)
        consumed_tokens = 1

        cmd = command_token
        if not cmd:
            return

        # 强力模式参数解析
        raw_power_keyword = (self.conf.get("power_model_keyword") or "").strip()
        keyword_lower = raw_power_keyword.lower()
        power_mode_requested = False

        if keyword_lower and keyword_lower in cmd.lower():
            cmd = cmd.lower().replace(keyword_lower, "").strip()
            power_mode_requested = True
            logger.info(f"在命令中检测到强力模式触发词'{keyword_lower}'，移除后命令='{cmd}'")
        elif keyword_lower and len(tokens) > consumed_tokens:
            next_token = tokens[consumed_tokens].strip().lower()
            if next_token == keyword_lower:
                power_mode_requested = True
                consumed_tokens += 1
                logger.info(f"检测到强力模式触发词作为独立token: '{keyword_lower}'")

        power_model_name = (self.conf.get("power_model_id") or "").strip()
        use_power_model = False  # [FIX] 确保变量名正确初始化
        if power_mode_requested:
            if not power_model_name:
                yield event.plain_result("⚠️ 强力模式触发失败：请先在管理面板配置强力模型ID。")
                return
            use_power_model = True  # [FIX] 使用 use_power_model

        # 指令解析
        bnn_command = self.conf.get("extra_prefix", "bnn")
        user_prompt = ""
        is_bnn = False

        base_cmd = cmd
        append_text = ""

        if "%" in cmd:
            parts = cmd.split("%", 1)
            if len(parts) == 2:
                base_cmd = parts[0].strip()
                append_text = parts[1].strip()
                logger.info(f"检测到%符号分割: 基础命令='{base_cmd}', 追加内容='{append_text}'")

        if base_cmd == bnn_command:
            remaining_tokens = tokens[consumed_tokens:]
            user_prompt = " ".join(remaining_tokens).strip()
            is_bnn = True

        elif base_cmd in self.prompt_map:
            val = self.prompt_map.get(base_cmd)
            if val and val != "[内置预设]":
                user_prompt = val
                if append_text:
                    user_prompt = user_prompt + append_text
                    logger.info(f"将追加内容'{append_text}'添加到预设prompt后面")

        if not user_prompt and not is_bnn:
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
            if base_cmd in cmd_map:
                key = cmd_map[base_cmd]
                if key == "help":
                    yield self._get_help_result(event)
                    return
                user_prompt = self.prompt_map.get(key) or self.prompt_map.get(base_cmd)
                if append_text:
                    user_prompt = user_prompt + append_text
                    logger.info(f"将追加内容'{append_text}'添加到映射命令prompt后面")

        if power_mode_requested:
            logger.info(f"🚀 强力模式已激活！触发词: '{raw_power_keyword}', 使用模型: '{power_model_name}'")

        if not user_prompt:
            if is_bnn:
                if not user_prompt and not power_mode_requested:
                    pass
            else:
                return

        # --- 权限与次数逻辑 ---
        sender_id = self._norm_id(event.get_sender_id())
        group_id = self._norm_id(event.get_group_id()) if event.get_group_id() else None

        user_blacklist = [self._norm_id(x) for x in (self.conf.get("user_blacklist") or [])]
        if sender_id in user_blacklist: return

        if group_id:
            group_blacklist = [self._norm_id(x) for x in (self.conf.get("group_blacklist") or [])]
            if group_id in group_blacklist: return

        raw_g_whitelist = self.conf.get("group_whitelist") or []
        group_whitelist = [self._norm_id(x) for x in raw_g_whitelist]

        raw_u_whitelist = self.conf.get("user_whitelist") or []
        user_whitelist = [self._norm_id(x) for x in raw_u_whitelist]

        is_master = self.is_global_admin(event)
        deduction_source = None
        required_cost = self._get_required_invocation_cost(use_power_model)

        if is_master:
            deduction_source = 'free'
        elif group_id and group_id in group_whitelist:
            deduction_source = 'free'
        elif group_id and len(group_whitelist) > 0:
            yield event.plain_result("❌ 本群未授权使用此功能。")
            return
        elif len(user_whitelist) > 0 and sender_id not in user_whitelist:
            return

        if deduction_source is None:
            if use_power_model:
                allow_group_fallback = bool(self.conf.get("power_mode_fallback_to_group", False))
                if self.conf.get("enable_user_limit", True):
                    u_cnt = self._get_user_count(sender_id)
                    if u_cnt >= required_cost:
                        deduction_source = 'user'
                    else:
                        if allow_group_fallback and group_id and self.conf.get("enable_group_limit", False):
                            g_cnt = self._get_group_count(group_id)
                            if g_cnt >= required_cost:
                                deduction_source = 'group'
                            else:
                                yield event.plain_result(
                                    f"❌ 次数不足。需要 {required_cost} 次。\n👤 用户剩余: {u_cnt}\n👥 本群剩余: {g_cnt}"
                                )
                                return
                        else:
                            yield event.plain_result(f"❌ 个人次数不足。需要 {required_cost} 次，当前剩余 {u_cnt} 次。")
                            return
                else:
                    deduction_source = 'free'
            else:
                if group_id and self.conf.get("enable_group_limit", False):
                    g_cnt = self._get_group_count(group_id)
                    if g_cnt >= required_cost:
                        deduction_source = 'group'

                if deduction_source is None and self.conf.get("enable_user_limit", True):
                    u_cnt = self._get_user_count(sender_id)
                    if u_cnt >= required_cost:
                        deduction_source = 'user'

                if deduction_source is None:
                    if not self.conf.get("enable_group_limit", False) and not self.conf.get("enable_user_limit", True):
                        deduction_source = 'free'
                    else:
                        yield event.plain_result(
                            self._build_limit_exhausted_message(group_id, use_power_model, required_cost)
                        )
                        return

        # --- 图片获取 (融合逻辑) ---
        images_to_process = []
        is_text_to_image = False

        if self.iwf:
            # [修改] ImageWorkflow.get_images 现在不会自动获取头像
            img_bytes_list = await self.iwf.get_images(event)

            if not img_bytes_list:
                # [修改] 智能判断 BNN 模式
                if is_bnn:
                    # bnn 模式 + 无图 = 纯文生图
                    if not user_prompt:
                        yield event.plain_result(f"请在指令后添加描述。例如: #{bnn_command} 一个可爱的女孩")
                        return
                    is_text_to_image = True
                    images_to_process = []
                    logger.info("BNN模式下未检测到图片，自动切换为纯文生图模式")
                else:
                    # 手办化等预设模式 + 无图 = 尝试取发送者头像 (兼容旧习惯)
                    logger.info(f"预设模式下未检测到图片，尝试获取发送者 [{sender_id}] 的头像...")
                    if avatar := await self.iwf._get_avatar(sender_id):
                        img_bytes_list = [avatar]
                        logger.info("成功获取发送者头像作为图生图源")
                    else:
                        yield event.plain_result("请发送或引用一张图片。")
            else:
                # 检测到图片，走图生图
                is_text_to_image = False
                logger.info("检测到明确的图片输入，模式确定为图生图")

            if not is_text_to_image and img_bytes_list:
                images_to_process = img_bytes_list

        if not is_bnn and user_prompt and not is_text_to_image:
            image_urls = self._extract_image_urls_from_text(user_prompt)
            if image_urls:
                logger.info(f"在预设内容中发现 {len(image_urls)} 个图片链接: {image_urls}")
                for image_url in image_urls:
                    if downloaded_image := await self._download_preset_image(image_url):
                        images_to_process.append(downloaded_image)
                        logger.info(f"成功下载预设内容中的图片: {image_url}")
                    else:
                        logger.warning(f"无法下载预设内容中的图片: {image_url}")

        display_cmd = cmd
        if is_bnn:
            if not is_text_to_image:
                MAX_IMAGES = 5
                if len(images_to_process) > MAX_IMAGES:
                    images_to_process = images_to_process[:MAX_IMAGES]
                    yield event.plain_result(f"🎨 检测到 {len(img_bytes_list)} 张图片，已选取前 {MAX_IMAGES} 张…")

            display_cmd = user_prompt[:10] + '...' if len(user_prompt) > 10 else user_prompt
        elif len(images_to_process) > 0:
            MAX_FIGURINE_IMAGES = 10
            if len(images_to_process) > MAX_FIGURINE_IMAGES:
                images_to_process = images_to_process[:MAX_FIGURINE_IMAGES]
                yield event.plain_result(
                    f"🎨 检测到 {len(img_bytes_list)} 张图片（含@用户头像），已选取前 {MAX_FIGURINE_IMAGES} 张…")

        if append_text:
            display_cmd = f"{base_cmd}%{append_text[:5]}..."

        override_model_name = None
        all_models = self._get_all_models()
        if temp_model_idx is not None:
            if 1 <= temp_model_idx <= len(all_models):
                override_model_name = all_models[temp_model_idx - 1]
            else:
                yield event.plain_result(f"⚠️ 指定的模型序号 {temp_model_idx} 无效，将使用默认模型。")

        if use_power_model:
            override_model_name = power_model_name

        display_label = display_cmd
        base_model_name = (self.conf.get("model", "nano-banana") or "nano-banana").strip() or "nano-banana"
        model_in_use = (override_model_name or base_model_name).strip() or base_model_name
        show_model_info = self.conf.get("show_model_info", False)

        mode_prefix = "增强" if use_power_model else ""
        action_type = "文生图" if is_text_to_image else "图生图"

        info_msg = f"🎨 收到{mode_prefix}{action_type}请求，正在生成 [{display_label}]..."
        yield event.plain_result(info_msg)

        if deduction_source == 'group' and group_id:
            await self._decrease_group_count(group_id, required_cost)
        elif deduction_source == 'user':
            await self._decrease_user_count(sender_id, required_cost)

        start_time = datetime.now()
        # [FIX] 使用 use_power_model (布尔值)
        res = await self._call_api(images_to_process, user_prompt, override_model=override_model_name,
                                   use_power_mode=use_power_model)
        elapsed = (datetime.now() - start_time).total_seconds()

        if isinstance(res, bytes):
            await self._record_daily_usage(sender_id, group_id)

            if base_cmd in self.prompt_map and not is_bnn:
                await self._save_preset_image(base_cmd, res)

            status_text = "增强生成成功" if use_power_model else "生成成功"
            caption_parts = [f"✅ {status_text} ({elapsed:.2f}s)", f"预设: {display_label}"]

            if deduction_source == 'free':
                caption_parts.append("剩余: ∞")
            else:
                if group_id and self.conf.get("enable_group_limit", False):
                    caption_parts.append(f"本群剩余: {self._get_group_count(group_id)}")
                if self.conf.get("enable_user_limit", True):
                    caption_parts.append(f"用户剩余: {self._get_user_count(sender_id)}")

            if show_model_info:
                caption_parts.append(f"模型: {model_in_use}")

            message_text = " | ".join(caption_parts)
            if not use_power_model:
                if hint := self._get_power_mode_hint(cmd):
                    message_text += f"\n{hint}"

            yield event.chain_result([Image.fromBytes(res), Plain(message_text)])
        else:
            status_text = "增强生成失败" if use_power_model else "生成失败"
            msg = self._format_error_message(status_text, elapsed, res)
            if deduction_source in ['group', 'user']:
                msg += "\n(注: 触发即扣次)"
            if show_model_info:
                msg += f"\n模型: {model_in_use}"
            if not use_power_model:
                if hint := self._get_power_mode_hint(cmd):
                    msg += f"\n{hint}"
            yield event.plain_result(msg)

        event.stop_event()

    def _get_help_result(self, event: AstrMessageEvent):
        """生成合并转发帮助消息对象"""
        help_text = self.conf.get("help_text", "帮助文档未配置")

        bot_uin = "2854196310"
        try:
            if hasattr(event, "robot") and event.robot:
                bot_uin = str(event.robot.id)
            elif hasattr(event, "bot") and hasattr(event.bot, "self_id"):
                bot_uin = str(event.bot.self_id)
        except:
            pass

        node = Node(
            name="手办化助手",
            uin=str(bot_uin),
            content=[Plain(help_text)]
        )
        return event.chain_result([Nodes(nodes=[node])])

    # 修复：使用 ctx=None 替代 *args
    @filter.command("文生图", prefix_optional=True)
    async def on_text_to_image(self, event: AstrMessageEvent, ctx=None):
        raw_cmd = event.message_str.strip()
        cmd_name = "文生图"
        override_model_name = None

        cmd_pos = raw_cmd.find(cmd_name)
        prompt = raw_cmd[cmd_pos + len(cmd_name):].strip() if cmd_pos != -1 else raw_cmd

        power_model_name = (self.conf.get("power_model_id") or "").strip()
        keyword = (self.conf.get("power_model_keyword") or "").strip()
        keyword_lower = keyword.lower()
        power_mode_requested = False

        if self.conf.get("enable_power_model", False) and keyword_lower:
            prompt_tokens = prompt.split()
            if prompt_tokens and prompt_tokens[0].lower() == keyword_lower:
                power_mode_requested = True
                prompt = " ".join(prompt_tokens[1:]).strip()

        if power_mode_requested and not power_model_name:
            yield event.plain_result("⚠️ 强力模式触发失败：请先在管理面板配置强力模型ID。")
            return

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

        if power_mode_requested:
            override_model_name = power_model_name

        prompt = prompt.strip()
        if not prompt:
            yield event.plain_result("请提供描述。用法: #文生图 [可选:(序号)] <描述>")
            return

        sender_id = self._norm_id(event.get_sender_id())
        group_id = self._norm_id(event.get_group_id()) if event.get_group_id() else None

        use_power_model = power_mode_requested
        required_cost = self._get_required_invocation_cost(use_power_model)

        deduction_source = None
        if self.is_global_admin(event):
            deduction_source = 'free'
        else:
            if use_power_model:
                allow_group_fallback = bool(self.conf.get("power_mode_fallback_to_group", False))
                if self.conf.get("enable_user_limit", True):
                    u_cnt = self._get_user_count(sender_id)
                    if u_cnt >= required_cost:
                        deduction_source = 'user'
                    else:
                        if allow_group_fallback and group_id and self.conf.get("enable_group_limit", False):
                            g_cnt = self._get_group_count(group_id)
                            if g_cnt >= required_cost:
                                deduction_source = 'group'
                            else:
                                yield event.plain_result(
                                    f"❌ 次数不足。需要 {required_cost} 次。\n👤 用户剩余: {u_cnt}\n👥 本群剩余: {g_cnt}"
                                )
                                return
                        else:
                            yield event.plain_result(f"❌ 个人次数不足。需要 {required_cost} 次，当前剩余 {u_cnt} 次。")
                            return
                else:
                    deduction_source = 'free'
            else:
                if group_id and self.conf.get("enable_group_limit", False):
                    if self._get_group_count(group_id) >= required_cost:
                        deduction_source = 'group'

                if deduction_source is None and self.conf.get("enable_user_limit", True):
                    if self._get_user_count(sender_id) >= required_cost:
                        deduction_source = 'user'

                if deduction_source is None:
                    if not self.conf.get("enable_group_limit", False) and not self.conf.get("enable_user_limit", True):
                        deduction_source = 'free'
                    else:
                        yield event.plain_result(
                            self._build_limit_exhausted_message(group_id, use_power_model, required_cost)
                        )
                        return

        display_prompt = prompt[:10] + "..." if len(prompt) > 10 else prompt
        mode_prefix = "增强" if power_mode_requested else ""
        info_str = f"🎨 收到{mode_prefix}文生图请求，正在生成 [{display_prompt}]"
        yield event.plain_result(info_str)

        base_model_name = (self.conf.get("model", "nano-banana") or "nano-banana").strip() or "nano-banana"
        model_in_use = (override_model_name or base_model_name).strip() or base_model_name
        show_model_info = self.conf.get("show_model_info", False)

        if deduction_source == 'group' and group_id:
            await self._decrease_group_count(group_id, required_cost)
        elif deduction_source == 'user':
            await self._decrease_user_count(sender_id, required_cost)

        start_time = datetime.now()
        res = await self._call_api([], prompt, override_model=override_model_name, use_power_mode=use_power_model)
        elapsed = (datetime.now() - start_time).total_seconds()

        if isinstance(res, bytes):
            await self._record_daily_usage(sender_id, group_id)

            status_text = "增强生成成功" if power_mode_requested else "生成成功"
            caption_parts = [f"✅ {status_text} ({elapsed:.2f}s)"]
            if deduction_source == 'free':
                caption_parts.append("剩余: ∞")
            else:
                if group_id and self.conf.get("enable_group_limit", False):
                    caption_parts.append(f"本群剩余: {self._get_group_count(group_id)}")
                if self.conf.get("enable_user_limit", True):
                    caption_parts.append(f"用户剩余: {self._get_user_count(sender_id)}")
            if show_model_info:
                caption_parts.append(f"模型: {model_in_use}")

            message_text = " | ".join(caption_parts)
            if not power_mode_requested:
                if hint := self._get_power_mode_hint(cmd_name):
                    message_text += f"\n{hint}"

            yield event.chain_result([Image.fromBytes(res), Plain(message_text)])
        else:
            status_text = "增强生成失败" if power_mode_requested else "生成失败"
            msg = self._format_error_message(status_text, elapsed, res)
            if show_model_info:
                msg += f"\n模型: {model_in_use}"
            if not power_mode_requested:
                if hint := self._get_power_mode_hint(cmd_name):
                    msg += f"\n{hint}"
            yield event.plain_result(msg)

        event.stop_event()

    @filter.command("lm添加", aliases={"lma"}, prefix_optional=True)
    async def add_lm_prompt(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return

        full_msg = event.message_str or ""
        clean_msg = full_msg.strip()

        cmd_prefix = "lm添加"
        if "lma" in clean_msg.lower() and not clean_msg.startswith(cmd_prefix):
            cmd_prefix = "lma"

        if clean_msg.lower().startswith(cmd_prefix.lower()):
            clean_msg = clean_msg[len(cmd_prefix):].strip()

        clean_msg = clean_msg.lstrip("#/ ")

        if ":" not in clean_msg:
            yield event.plain_result('格式错误, 示例: #lm添加 触发词:提示词')
            return

        key, new_value = map(str.strip, clean_msg.split(":", 1))

        prompt_list = self.conf.get("prompt_list", [])
        if not isinstance(prompt_list, list):
            prompt_list = []

        found = False
        for idx, item in enumerate(prompt_list):
            if isinstance(item, str) and item.strip().startswith(key + ":"):
                prompt_list[idx] = f"{key}:{new_value}"
                found = True
                break

        if not found:
            prompt_list.append(f"{key}:{new_value}")

        self.conf["prompt_list"] = prompt_list
        try:
            if hasattr(self.conf, "save"):
                self.conf.save()
        except Exception as e:
            logger.error(f"保存配置失败: {e}")

        await self._load_prompt_map()
        yield event.plain_result(f"✅ 已保存预设:\n{key}:{new_value}")

    @filter.command("lm查看", aliases={"lmv", "lm预览"}, prefix_optional=True)
    async def lm_preview_prompt(self, event: AstrMessageEvent):
        raw = event.message_str.strip()
        parts = raw.split()
        if len(parts) < 2:
            yield event.plain_result("用法: #lm查看 <关键词>")
            return

        keyword = parts[1].strip()
        prompt_content = self.prompt_map.get(keyword)

        if prompt_content:
            yield event.plain_result(f"🔍 关键词【{keyword}】的提示词：\n\n{prompt_content}")
        else:
            yield event.plain_result(f"❌ 未找到关键词【{keyword}】的预设。")

    @filter.command("lm列表", aliases={"lmlist", "预设列表"}, prefix_optional=True)
    async def on_get_preset_list(self, event: AstrMessageEvent):
        """输出所有可用预设列表，5xN表格格式，上面是图片，下面是预设名称"""
        if not self.prompt_map:
            yield event.plain_result("⚠️ 当前没有可用的预设。")
            return

        # 整理预设
        built_in = []
        custom = []

        for key, val in self.prompt_map.items():
            if val == "[内置预设]":
                built_in.append(key)
            else:
                custom.append(key)

        built_in.sort()
        custom.sort()

        # 合并所有预设并按名称排序
        all_presets = []
        for preset in built_in:
            all_presets.append((preset, True))  # True表示内置预设
        for preset in custom:
            all_presets.append((preset, False))  # False表示自定义预设

        # 按预设名称排序
        all_presets.sort(key=lambda x: x[0])

        if not all_presets:
            yield event.plain_result("⚠️ 当前没有可用的预设。")
            return

        try:
            # 创建表格图片
            table_image = await self._create_preset_table_image(all_presets)

            # 发送图片和标题
            yield event.chain_result([
                Image.fromBytes(table_image)
            ])

        except Exception as e:
            logger.error(f"创建预设表格图片失败: {e}")
            # 如果图片创建失败，回退到文本模式
            plain_msg = "📜 **可用预设列表**\n"
            plain_msg += "==================\n"

            if built_in:
                plain_msg += "📌 **内置预设**:\n"
                for preset in built_in:
                    plain_msg += f"  • {preset}\n"
                plain_msg += "\n"

            if custom:
                plain_msg += "✨ **自定义预设**:\n"
                for preset in custom:
                    plain_msg += f"  • {preset}\n"
            else:
                plain_msg += "✨ **自定义预设**: (无)\n\n"

            plain_msg += "==================\n"
            plain_msg += "使用方法: #预设名 [图片]"

            yield event.plain_result(plain_msg)

    async def _create_preset_table_image(self, presets: List[Tuple[str, bool]]) -> bytes:
        """创建5xN表格图片，上面是图片，下面是预设名称"""
        # 根据配置选择表格质量
        quality = self.conf.get("preset_table_quality", "高清")

        # 表格参数 - 根据质量设置尺寸
        cols = self.conf.get("preset_table_columns", 5)  # 从配置获取列数，默认5列
        if quality == "标准":
            cell_width = 200  # 标准单元格宽度
            cell_height = 250  # 标准单元格高度
            image_area_height = 200  # 标准图片区域
            text_area_height = 50  # 标准文字区域
            padding = 10  # 标准内边距
            font_size = 16
            title_font_size = 20
        elif quality == "高清":
            cell_width = 300  # 增大单元格宽度
            cell_height = 380  # 增大单元格高度
            image_area_height = 320  # 增大图片区域
            text_area_height = 60  # 增大文字区域
            padding = 15  # 增大内边距
            font_size = 24
            title_font_size = 32
        else:  # 超清
            cell_width = 400  # 超大单元格宽度
            cell_height = 500  # 超大单元格高度
            image_area_height = 420  # 超大图片区域
            text_area_height = 80  # 超大文字区域
            padding = 20  # 超大内边距
            font_size = 30
            title_font_size = 40

        # 计算行数
        rows = (len(presets) + cols - 1) // cols

        # 计算图片尺寸
        table_width = cols * cell_width + (cols + 1) * padding
        table_height = rows * cell_height + (rows + 1) * padding

        # 创建白色背景图片
        table_img = PILImage.new('RGB', (table_width, table_height), 'white')

        # 准备字体（尝试使用支持中文的字体）
        try:
            from PIL import ImageFont
            # 尝试使用支持中文的字体
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",  # 黑体
                "C:/Windows/Fonts/simsun.ttc",  # 宋体
                "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
                "C:/Windows/Fonts/msyhbd.ttc",  # 微软雅黑粗体
                "arial.ttf"  # 英文字体作为最后备选
            ]

            font = None
            title_font = None

            for font_path in font_paths:
                try:
                    if Path(font_path).exists():
                        font = ImageFont.truetype(font_path, font_size)  # 根据质量设置字体大小
                        title_font = ImageFont.truetype(font_path, title_font_size)  # 根据质量设置标题字体
                        break
                except:
                    continue

            # 如果都找不到，使用默认字体
            if not font:
                font = ImageFont.load_default()
                title_font = ImageFont.load_default()

        except:
            font = None
            title_font = None

        # 创建绘图对象
        from PIL import ImageDraw
        draw = ImageDraw.Draw(table_img)

        # 启用抗锯齿（如果可用）
        try:
            from PIL import ImageDraw
            # 使用更平滑的绘图方法
            if hasattr(draw, 'text'):  # 确保draw对象有text方法
                pass  # PIL版本支持
        except ImportError:
            pass

        # 绘制每个单元格
        for i, (preset_name, is_built_in) in enumerate(presets):
            row = i // cols
            col = i % cols

            # 计算单元格位置
            x = padding + col * (cell_width + padding)
            y = padding + row * (cell_height + padding)

            # 获取预设图片
            image_path = self._get_preset_image_path(preset_name)

            # 绘制图片区域
            if image_path:
                try:
                    # 加载并调整图片大小
                    preset_img = PILImage.open(image_path)
                    # 转换为RGB模式以确保兼容性
                    if preset_img.mode != 'RGB':
                        preset_img = preset_img.convert('RGB')
                    # 保持纵横比，填充到更大尺寸，使用最高质量的LANCZOS重采样
                    preset_img.thumbnail((cell_width - 2 * padding, image_area_height - 2 * padding),
                                         PILImage.Resampling.LANCZOS)

                    # 计算居中位置
                    img_width, img_height = preset_img.size
                    img_x = x + (cell_width - img_width) // 2
                    img_y = y + (image_area_height - img_height) // 2

                    # 粘贴图片
                    table_img.paste(preset_img, (img_x, img_y))

                except Exception as e:
                    logger.error(f"加载预设图片失败 {preset_name}: {e}")
                    # 绘制占位符
                    draw.rectangle(
                        [x + padding, y + padding, x + cell_width - padding, y + image_area_height - padding],
                        outline='lightgray', width=2)
                    placeholder_text = "无图片"
                    if font:
                        bbox = draw.textbbox((0, 0), placeholder_text, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                    else:
                        text_width = len(placeholder_text) * (font_size // 2)  # 根据字体大小调整字符宽度
                        text_height = font_size
                    text_x = x + (cell_width - text_width) // 2
                    text_y = y + (image_area_height - text_height) // 2
                    draw.text((text_x, text_y), placeholder_text, fill='gray', font=font)
            else:
                # 没有图片，绘制占位符
                draw.rectangle([x + padding, y + padding, x + cell_width - padding, y + image_area_height - padding],
                               outline='lightgray', width=2)
                placeholder_text = "无图片"
                if font:
                    bbox = draw.textbbox((0, 0), placeholder_text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                else:
                    text_width = len(placeholder_text) * (font_size // 2)  # 根据字体大小调整字符宽度
                    text_height = font_size
                text_x = x + (cell_width - text_width) // 2
                text_y = y + (image_area_height - text_height) // 2
                draw.text((text_x, text_y), placeholder_text, fill='gray', font=font)

            # 绘制文字区域背景
            text_y_pos = y + image_area_height
            draw.rectangle([x, text_y_pos, x + cell_width, text_y_pos + text_area_height], fill='lightgray')

            # 绘制预设名称
            # 根据字体大小调整截断长度
            if font_size <= 16:
                max_length = 10  # 小字体可以显示更多字符
            elif font_size <= 24:
                max_length = 8  # 中等字体
            else:
                max_length = 6  # 大字体显示更少字符
            display_name = preset_name[:max_length] + '...' if len(preset_name) > max_length else preset_name
            if is_built_in:
                display_name = f"📌{display_name}"
            else:
                display_name = f"✨{display_name}"

            if font:
                bbox = draw.textbbox((0, 0), display_name, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width = len(display_name) * (font_size // 2)  # 根据字体大小调整字符宽度
                text_height = font_size

            text_x = x + (cell_width - text_width) // 2
            text_y = text_y_pos + (text_area_height - text_height) // 2
            draw.text((text_x, text_y), display_name, fill='black', font=font)

            # 绘制单元格边框
            draw.rectangle([x, y, x + cell_width, y + cell_height], outline='black', width=1)

        # 保存为字节 - 使用更高质量设置
        img_byte_arr = io.BytesIO()
        # 使用PNG格式，质量设置为最高
        table_img.save(img_byte_arr, format='PNG', optimize=True, compress_level=1)
        return img_byte_arr.getvalue()

    @filter.command("lm帮助", aliases={"lmh", "手办化帮助"}, prefix_optional=True)
    async def on_prompt_help(self, event: AstrMessageEvent):
        parts = event.message_str.strip().split()
        keyword = parts[1] if len(parts) > 1 else ""

        if not keyword:
            yield self._get_help_result(event)
            return

        prompt = self.prompt_map.get(keyword)
        content = f"📄 预设 [{keyword}] 内容:\n{prompt}" if prompt else f"❌ 未找到 [{keyword}]"

        bot_uin = "2854196310"
        try:
            if hasattr(event, "robot") and event.robot:
                bot_uin = str(event.robot.id)
            elif hasattr(event, "bot") and hasattr(event.bot, "self_id"):
                bot_uin = str(event.bot.self_id)
        except:
            pass

        node = Node(
            name="手办化助手",
            uin=str(bot_uin),
            content=[Plain(content)]
        )
        yield event.chain_result([Nodes(nodes=[node])])

    # ---------------- 统计与存储 ----------------

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
        return self.user_counts.get(self._norm_id(uid), 0)

    async def _decrease_user_count(self, uid: str, amount: int = 1):
        uid = self._norm_id(uid)
        count = self._get_user_count(uid)
        if amount <= 0 or count <= 0:
            return
        deduction = min(amount, count)
        self.user_counts[uid] = count - deduction
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
        return self.group_counts.get(self._norm_id(group_id), 0)

    async def _decrease_group_count(self, group_id: str, amount: int = 1):
        gid = self._norm_id(group_id)
        count = self._get_group_count(gid)
        if amount <= 0 or count <= 0:
            return
        deduction = min(amount, count)
        self.group_counts[gid] = count - deduction
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

    async def _load_daily_stats(self):
        if not self.daily_stats_file.exists():
            self.daily_stats = {"date": "", "users": {}, "groups": {}}
            return
        try:
            content = await asyncio.to_thread(self.daily_stats_file.read_text, "utf-8")
            self.daily_stats = json.loads(content)
        except:
            self.daily_stats = {"date": "", "users": {}, "groups": {}}

    async def _save_daily_stats(self):
        try:
            data = json.dumps(self.daily_stats, indent=4)
            await asyncio.to_thread(self.daily_stats_file.write_text, data, "utf-8")
        except:
            pass

    async def _record_daily_usage(self, user_id: str, group_id: str | None):
        today = datetime.now().strftime("%Y-%m-%d")

        if self.daily_stats.get("date") != today:
            self.daily_stats = {
                "date": today,
                "users": {},
                "groups": {}
            }

        uid = self._norm_id(user_id)
        current_u = self.daily_stats["users"].get(uid, 0)
        self.daily_stats["users"][uid] = current_u + 1

        if group_id:
            gid = self._norm_id(group_id)
            current_g = self.daily_stats["groups"].get(gid, 0)
            self.daily_stats["groups"][gid] = current_g + 1

        await self._save_daily_stats()

    async def _load_preset_images(self):
        if not self.preset_images_file.exists():
            self.preset_images = {}
            return
        try:
            content = await asyncio.to_thread(self.preset_images_file.read_text, "utf-8")
            self.preset_images = json.loads(content)
        except:
            self.preset_images = {}

    async def _save_preset_images(self):
        try:
            data = json.dumps(self.preset_images, indent=4)
            await asyncio.to_thread(self.preset_images_file.write_text, data, "utf-8")
        except:
            pass

    async def _save_preset_image(self, preset_key: str, image_bytes: bytes):
        """保存预设图片到文件和记录中"""
        try:
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{preset_key}_{timestamp}.png"
            filepath = self.preset_images_dir / filename

            # 保存图片文件
            await asyncio.to_thread(filepath.write_bytes, image_bytes)

            # 删除旧的图片文件（如果存在）
            if preset_key in self.preset_images:
                old_filename = self.preset_images[preset_key]
                old_filepath = self.preset_images_dir / old_filename
                if old_filepath.exists():
                    await asyncio.to_thread(old_filepath.unlink)

            # 更新记录
            self.preset_images[preset_key] = filename
            await self._save_preset_images()

            logger.info(f"已保存预设图片: {preset_key} -> {filename}")
            return True
        except Exception as e:
            logger.error(f"保存预设图片失败: {preset_key}, 错误: {e}")
            return False

    def _get_preset_image_path(self, preset_key: str) -> Optional[str]:
        """获取预设图片的文件路径"""
        if preset_key not in self.preset_images:
            return None

        filename = self.preset_images[preset_key]
        filepath = self.preset_images_dir / filename

        if filepath.exists():
            return str(filepath)
        else:
            # 文件不存在，清理记录
            del self.preset_images[preset_key]
            asyncio.create_task(self._save_preset_images())
            return None

    async def _cleanup_preset_images(self, max_age_days: int = 30):
        """清理超过指定天数的预设图片"""
        try:
            current_time = datetime.now()
            cleaned_count = 0

            for preset_key, filename in list(self.preset_images.items()):
                filepath = self.preset_images_dir / filename
                if filepath.exists():
                    # 获取文件创建时间
                    file_time = datetime.fromtimestamp(filepath.stat().st_mtime)
                    age_days = (current_time - file_time).days

                    if age_days > max_age_days:
                        # 删除文件和记录
                        await asyncio.to_thread(filepath.unlink)
                        del self.preset_images[preset_key]
                        cleaned_count += 1
                        logger.info(f"清理过期预设图片: {preset_key} ({filename})")

            if cleaned_count > 0:
                await self._save_preset_images()
                logger.info(f"预设图片清理完成，共清理 {cleaned_count} 个文件")

            return cleaned_count
        except Exception as e:
            logger.error(f"清理预设图片失败: {e}")
            return 0

    @filter.command("预设图片清理", prefix_optional=True)
    async def on_cleanup_preset_images(self, event: AstrMessageEvent):
        """清理过期的预设图片"""
        if not self.is_global_admin(event):
            yield event.plain_result("❌ 只有管理员可以执行此操作。")
            return

        # 默认清理30天前的图片
        max_age_days = 30
        args = event.message_str.strip().split()
        if len(args) > 1 and args[1].isdigit():
            max_age_days = int(args[1])

        cleaned_count = await self._cleanup_preset_images(max_age_days)

        total_images = len(self.preset_images)
        msg = f"✅ 预设图片清理完成！\n"
        msg += f"📊 清理了 {cleaned_count} 个过期图片\n"
        msg += f"📁 当前剩余 {total_images} 个预设图片\n"
        msg += f"⏰ 清理条件: 超过 {max_age_days} 天的图片"

        yield event.plain_result(msg)

    @filter.command("预设图片统计", prefix_optional=True)
    async def on_preset_images_stats(self, event: AstrMessageEvent):
        """显示预设图片统计信息"""
        if not self.is_global_admin(event):
            yield event.plain_result("❌ 只有管理员可以执行此操作。")
            return

        total_images = len(self.preset_images)

        # 统计文件大小
        total_size = 0
        for filename in self.preset_images.values():
            filepath = self.preset_images_dir / filename
            if filepath.exists():
                total_size += filepath.stat().st_size

        # 转换为MB
        total_size_mb = total_size / (1024 * 1024)

        # 显示每个预设的图片信息
        msg = f"📊 **预设图片统计**\n"
        msg += f"==================\n"
        msg += f"📁 总预设数: {total_images}\n"
        msg += f"💾 总大小: {total_size_mb:.2f} MB\n"
        msg += f"📂 存储目录: {self.preset_images_dir}\n\n"

        if total_images > 0:
            msg += "📸 **详细列表**:\n"
            for preset, filename in sorted(self.preset_images.items()):
                filepath = self.preset_images_dir / filename
                if filepath.exists():
                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    msg += f"  • {preset}: {size_mb:.2f} MB\n"

        yield event.plain_result(msg)

    @filter.command("手办化今日统计", prefix_optional=True)
    async def get_daily_stats_report(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            yield event.plain_result("❌ 权限不足")
            return

        today = datetime.now().strftime("%Y-%m-%d")
        if self.daily_stats.get("date") != today:
            yield event.plain_result(f"📊 {today} 今日暂无统计数据。")
            return

        users_sorted = sorted(self.daily_stats["users"].items(), key=lambda x: x[1], reverse=True)[:10]
        groups_sorted = sorted(self.daily_stats["groups"].items(), key=lambda x: x[1], reverse=True)[:10]

        msg = f"📊 **手办化今日统计 ({today})**\n"
        msg += "--------------------\n"
        msg += "👥 **群组消耗排行**:\n"
        if groups_sorted:
            for i, (gid, count) in enumerate(groups_sorted):
                msg += f"{i + 1}. 群{gid}: {count}次\n"
        else:
            msg += "(无数据)\n"

        msg += "\n👤 **用户消耗排行**:\n"
        if users_sorted:
            for i, (uid, count) in enumerate(users_sorted):
                msg += f"{i + 1}. {uid}: {count}次\n"
        else:
            msg += "(无数据)\n"

        yield event.plain_result(msg)

    @filter.command("手办化签到", prefix_optional=True)
    async def on_checkin(self, event: AstrMessageEvent):
        if not self.conf.get("enable_checkin", False):
            yield event.plain_result("📅 签到未开启。")
            return

        uid = self._norm_id(event.get_sender_id())
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
            target = self._norm_id(at_seg.qq)
            match = re.search(r"(\d+)\s*$", text)
            if match:
                count = int(match.group(1))
        else:
            match = re.search(r"(\d+)\s+(\d+)", text)
            if match:
                target, count = self._norm_id(match.group(1)), int(match.group(2))

        if target:
            old_cnt = self._get_user_count(target)
            new_cnt = old_cnt + count
            self.user_counts[target] = new_cnt
            await self._save_user_counts()

            msg = f"✅ 已为用户 {target} 增加 {count} 次。\n"
            msg += f"📊 变动: {old_cnt} + {count} = {new_cnt}\n"
            msg += f"👤 用户剩余: {new_cnt}"
            if gid := event.get_group_id():
                msg += f"\n👥 本群剩余: {self._get_group_count(self._norm_id(gid))}"

            yield event.plain_result(msg)

    @filter.command("手办化增加群组次数", prefix_optional=True)
    async def on_add_group_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return

        match = re.search(r"(\d+)\s+(\d+)", event.message_str.strip())
        if match:
            gid, count = self._norm_id(match.group(1)), int(match.group(2))

            old_cnt = self._get_group_count(gid)
            new_cnt = old_cnt + count
            self.group_counts[gid] = new_cnt
            await self._save_group_counts()

            msg = f"✅ 已为群 {gid} 增加 {count} 次。\n"
            msg += f"📊 变动: {old_cnt} + {count} = {new_cnt}\n"
            msg += f"👥 本群剩余: {new_cnt}"

            yield event.plain_result(msg)

    @filter.command("手办化查询次数", prefix_optional=True)
    async def on_query_counts(self, event: AstrMessageEvent):
        uid = self._norm_id(event.get_sender_id())

        if self.is_global_admin(event):
            at_seg = next((s for s in event.message_obj.message if isinstance(s, At)), None)
            if at_seg:
                uid = self._norm_id(at_seg.qq)
            else:
                parts = event.message_str.strip().split()
                if len(parts) > 1 and parts[1].isdigit():
                    uid = self._norm_id(parts[1])

        msg = f"👤 用户 {uid} 剩余: {self._get_user_count(uid)}"
        if gid := event.get_group_id():
            msg += f"\n👥 本群剩余: {self._get_group_count(self._norm_id(gid))}"

        yield event.plain_result(msg)

    @filter.command("手办化添加key", prefix_optional=True)
    async def on_add_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return

        new_keys = event.message_str.strip().split()[1:]
        if not new_keys:
            yield event.plain_result("格式错误。用法: #手办化添加key <key1> ...")
            return

        current_mode = self.conf.get("api_mode", "generic")

        # 检查是否有强力模式参数
        use_power_mode = False
        if new_keys and new_keys[0].lower() in ["power", "强力", "p"]:
            use_power_mode = True
            new_keys = new_keys[1:]  # 移除参数

        if not new_keys:
            yield event.plain_result("格式错误。用法: #手办化添加key [power/强力/p] <key1> ...")
            return

        # 根据模式和是否强力模式选择目标字段
        if use_power_mode:
            target_field = "power_gemini_api_keys" if current_mode == "gemini_official" else "power_generic_api_keys"
            mode_desc = f"【强力模式-{current_mode}】"
        else:
            target_field = "gemini_api_keys" if current_mode == "gemini_official" else "generic_api_keys"
            mode_desc = f"【{current_mode}】"

        keys = self.conf.get(target_field, [])
        added = [k for k in new_keys if k not in keys]
        keys.extend(added)
        self.conf[target_field] = keys

        if hasattr(self.conf, "save"):
            self.conf.save()

        yield event.plain_result(f"✅ 已向 {mode_desc} 模式添加 {len(added)} 个Key。")

    @filter.command("手办化key列表", prefix_optional=True)
    async def on_list_keys(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return

        current_mode = self.conf.get("api_mode", "generic")

        # 获取普通模式Key池
        normal_target_field = "gemini_api_keys" if current_mode == "gemini_official" else "generic_api_keys"
        normal_keys = self.conf.get(normal_target_field, [])

        # 获取强力模式Key池
        power_target_field = "power_gemini_api_keys" if current_mode == "gemini_official" else "power_generic_api_keys"
        power_keys = self.conf.get(power_target_field, [])

        msg = f"🔑 API模式: 【{current_mode}】\n\n"

        # 普通模式Key列表
        msg += f"📌 普通模式Key池 ({len(normal_keys)}个):\n"
        if normal_keys:
            msg += "\n".join([f"{i + 1}. {k[:6]}..." for i, k in enumerate(normal_keys)]) + "\n"
        else:
            msg += "(空)\n"

        # 强力模式Key列表
        msg += f"\n⚡ 强力模式Key池 ({len(power_keys)}个):\n"
        if power_keys:
            msg += "\n".join([f"{i + 1}. {k[:6]}..." for i, k in enumerate(power_keys)]) + "\n"
        else:
            msg += "(空)\n"

        # 如果强力模式Key池为空，显示提示
        if not power_keys:
            msg += "\n💡 提示: 强力模式Key池为空时将使用普通模式Key池"

        yield event.plain_result(msg)

    @filter.command("手办化删除key", prefix_optional=True)
    async def on_delete_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return

        parts = event.message_str.strip().split()
        if len(parts) < 2:
            yield event.plain_result("格式: #手办化删除key [power/强力/p] <序号|all>")
            return

        # 检查是否有强力模式参数
        use_power_mode = False
        param_idx = 1

        if parts[1].lower() in ["power", "强力", "p"]:
            use_power_mode = True
            param_idx = 2
            if len(parts) < 3:
                yield event.plain_result("格式: #手办化删除key [power/强力/p] <序号|all>")
                return

        param = parts[param_idx]

        current_mode = self.conf.get("api_mode", "generic")

        # 根据是否强力模式选择目标字段
        if use_power_mode:
            target_field = "power_gemini_api_keys" if current_mode == "gemini_official" else "power_generic_api_keys"
            mode_desc = f"【强力模式-{current_mode}】"
        else:
            target_field = "gemini_api_keys" if current_mode == "gemini_official" else "generic_api_keys"
            mode_desc = f"【{current_mode}】"

        keys = self.conf.get(target_field, [])

        if param == "all":
            self.conf[target_field] = []
        elif param.isdigit():
            idx = int(param) - 1
            if 0 <= idx < len(keys):
                keys.pop(idx)
                self.conf[target_field] = keys

        if hasattr(self.conf, "save"):
            self.conf.save()

        yield event.plain_result(f"✅ 已从 {mode_desc} 模式删除Key。")

    async def terminate(self):
        if self.iwf:
            await self.iwf.terminate()
        logger.info("[FigurinePro] 插件已终止")
