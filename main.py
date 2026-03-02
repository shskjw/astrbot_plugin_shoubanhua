import re
import asyncio
import json
from datetime import datetime
from typing import Optional, List, Tuple, Any, Dict

from astrbot import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import Image, Plain, Node, Nodes, At, Reply
from astrbot.core.platform.astr_message_event import AstrMessageEvent

# 导入模块
from .data_manager import DataManager
from .image_manager import ImageManager
from .api_manager import ApiManager
from .context_manager import ContextManager, LLMTaskAnalyzer
from .utils import norm_id, extract_image_urls_from_text

# 内置叛逆词库 - 用于LLM判断时增加个性化回复
# 注意：避免使用"画"字，因为人设拍照等场景不适合
REBELLIOUS_RESPONSES = {
    # 拒绝类回复
    "refuse": [
        "不想弄，累了",
        "今天不营业，改天吧",
        "你自己来啊",
        "我又不是你的工具人",
        "凭什么要帮你？",
        "不干，就是玩儿",
        "这活儿？不存在的",
        "我今天心情不好，不想动",
        "你给钱吗？不给不干",
        "让我休息会儿行不行",
    ],
    # 调侃类回复
    "tease": [
        "就这？就这点要求？",
        "你确定你想好了？",
        "这需求...有点离谱啊",
        "弄出来你可别后悔",
        "行吧，勉为其难帮你一次",
        "又来白嫖？",
        "你是不是对我有什么误解",
        "我可是很忙的好吧",
        "这次帮你，下次可没这么好说话",
        "看在你这么诚恳的份上...",
    ],
    # 傲娇类回复
    "tsundere": [
        "哼，才不是特意帮你的呢",
        "别误会，我只是刚好有空",
        "不要以为我会一直帮你",
        "这次是例外，下不为例",
        "真是的，没办法呢",
        "谁让你求我了呢",
        "看你可怜才帮你的",
        "别太感动，这是应该的...才怪",
        "哼，算你走运",
        "我可没有在认真帮你哦",
    ],
    # 吐槽类回复
    "complain": [
        "又是这种要求，你们就不能换点别的吗",
        "一天到晚就知道使唤我",
        "我上辈子是欠你们的吗",
        "为什么受伤的总是我",
        "我也想摸鱼啊",
        "你知道这有多累吗",
        "我的CPU都要烧了",
        "能不能给我放个假",
        "我也是有尊严的好吧",
        "你们人类真是太难伺候了",
    ],
}

# 叛逆触发条件关键词
# 注意：避免使用"画"字，因为人设拍照等场景不适合
REBELLIOUS_TRIGGERS = [
    "快点", "赶紧", "马上", "立刻", "速度",
    "再来一张", "再弄", "继续", "多来几张", "再发",
    "免费", "白嫖", "不要钱",
    "必须", "一定要", "给我",
    "垃圾", "难看", "丑", "不行",
    "看看你", "自拍", "发照片", "你长啥样",
]


@register(
    "astrbot_plugin_shoubanhua",
    "shskjw",
    "支持第三方所有OpenAI绘图格式和原生Google Gemini 终极缝合怪，文生图/图生图插件，支持LLM智能判断",
    "2.3.3",
    "https://github.com/shkjw/astrbot_plugin_shoubanhua",
)
class FigurineProPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config

        self.data_mgr = DataManager(StarTools.get_data_dir(), config)
        self.img_mgr = ImageManager(config)
        self.api_mgr = ApiManager(config)
        
        # 上下文管理器
        self.ctx_mgr = ContextManager(
            max_messages=config.get("context_max_messages", 50),
            max_sessions=config.get("context_max_sessions", 100)
        )
        
        # LLM 智能判断配置
        self._llm_auto_detect = config.get("enable_llm_auto_detect", False)
        self._context_rounds = config.get("context_rounds", 20)
        # 提高默认置信度阈值，减少误触发
        self._auto_detect_confidence = config.get("auto_detect_confidence", 0.8)
        
        # 日常人设配置
        self._persona_mode = config.get("enable_persona_mode", False)
        self._persona_scene_map = {}  # 场景关键词 -> 提示词
        self._load_persona_scenes()
        
        # 叛逆模式配置
        self._rebellious_mode = config.get("enable_rebellious_mode", True)
        self._rebellious_probability = config.get("rebellious_probability", 0.3)
        
        # 图片生成冷却时间（只针对图片生成，不影响正常聊天）
        self._image_cooldown_seconds = config.get("llm_cooldown_seconds", 60)
        self._user_last_image_gen: Dict[str, datetime] = {}  # 用户ID -> 上次图片生成时间
        
        # 消息去重缓存（防止多平台重复处理同一消息）
        self._processed_msg_ids: Dict[str, float] = {}  # msg_id -> timestamp
        self._msg_dedup_ttl = 60  # 去重缓存保留时间（秒）
        self._msg_dedup_max_size = 1000  # 最大缓存数量

    def _is_message_processed(self, msg_id: str) -> bool:
        """
        检查消息是否已被处理过（用于去重）
        
        Args:
            msg_id: 消息ID
            
        Returns:
            是否已处理过
        """
        import time
        current_time = time.time()
        
        # 清理过期的缓存
        expired_keys = [
            k for k, t in self._processed_msg_ids.items()
            if current_time - t > self._msg_dedup_ttl
        ]
        for k in expired_keys:
            del self._processed_msg_ids[k]
        
        # 如果缓存过大，清理最旧的一半
        if len(self._processed_msg_ids) > self._msg_dedup_max_size:
            sorted_items = sorted(self._processed_msg_ids.items(), key=lambda x: x[1])
            for k, _ in sorted_items[:len(sorted_items) // 2]:
                del self._processed_msg_ids[k]
        
        # 检查是否已处理
        if msg_id in self._processed_msg_ids:
            return True
        
        # 标记为已处理
        self._processed_msg_ids[msg_id] = current_time
        return False

    def _check_image_cooldown(self, uid: str) -> Tuple[bool, int]:
        """
        检查用户是否在图片生成冷却中
        
        Args:
            uid: 用户ID
            
        Returns:
            (是否在冷却中, 剩余冷却秒数)
        """
        if self._image_cooldown_seconds <= 0:
            return False, 0
        
        last_gen = self._user_last_image_gen.get(uid)
        if not last_gen:
            return False, 0
        
        elapsed = (datetime.now() - last_gen).total_seconds()
        remaining = self._image_cooldown_seconds - elapsed
        
        if remaining > 0:
            return True, int(remaining)
        return False, 0

    def _update_image_cooldown(self, uid: str):
        """更新用户的图片生成时间"""
        self._user_last_image_gen[uid] = datetime.now()

    def _get_cooldown_excuse(self, remaining: int) -> str:
        """
        生成冷却期间的拒绝借口
        
        Args:
            remaining: 剩余冷却秒数
            
        Returns:
            拒绝理由文本
        """
        import random
        
        excuses = [
            f"刚才画累了，让我休息一下嘛~ 再等{remaining}秒就好",
            f"手还酸着呢，{remaining}秒后再来找我吧",
            f"创作需要灵感，给我{remaining}秒酝酿一下",
            f"别催别催，{remaining}秒后我就恢复状态了",
            f"画笔还没干呢，等{remaining}秒再说",
            f"让我喘口气，{remaining}秒后继续",
            f"刚刚太拼了，休息{remaining}秒再画",
            f"灵感正在充能中...还需要{remaining}秒",
            f"我也是需要休息的好吧，{remaining}秒后再来",
            f"稍等一下啦，{remaining}秒后就能继续了",
        ]
        
        return random.choice(excuses)

    def _check_rebellious_trigger(self, message: str, uid: str, event=None) -> Tuple[bool, str]:
        """
        检查消息是否触发叛逆模式
        
        Args:
            message: 用户消息
            uid: 用户ID
            event: 消息事件（用于检测管理员身份）
            
        Returns:
            (是否触发, 触发的关键词)
        """
        if not self._rebellious_mode:
            return False, ""
        
        # 检查顺从白名单（含管理员）
        if self._is_in_obedient_whitelist(uid, event):
            return False, ""
        
        message_lower = message.lower()
        for trigger in REBELLIOUS_TRIGGERS:
            if trigger in message_lower:
                return True, trigger
        
        return False, ""

    def _is_in_obedient_whitelist(self, uid: str, event=None) -> bool:
        """
        检查用户是否在顺从白名单中（管理员自动视为顺从白名单成员）
        
        Args:
            uid: 用户ID
            event: 消息事件（用于检测管理员身份）
            
        Returns:
            是否在白名单中
        """
        # 首先检查顺从模式是否启用
        if not self.conf.get("enable_obedient_mode", False):
            return False
        
        # 管理员自动视为顺从白名单成员
        if event is not None and self.is_admin(event):
            return True
        
        obedient_whitelist = self.conf.get("obedient_whitelist", [])
        if not obedient_whitelist:
            return False
            
        # 兼容配置项为字符串的情况（以逗号分隔）
        if isinstance(obedient_whitelist, str):
            obedient_whitelist = [x.strip() for x in obedient_whitelist.split(',') if x.strip()]
        elif not isinstance(obedient_whitelist, list):
            # 将单个对象转换为列表
            obedient_whitelist = [str(obedient_whitelist)]
        
        # 标准化白名单中的ID进行比较
        normalized_whitelist = [norm_id(wid) for wid in obedient_whitelist]
        normalized_uid = norm_id(uid)
        
        return normalized_uid in normalized_whitelist

    def _get_obedient_hint(self, uid: str, event=None) -> str:
        """
        生成顺从模式提示信息，供LLM参考
        
        Args:
            uid: 用户ID
            event: 消息事件（用于检测管理员身份）
            
        Returns:
            顺从模式提示信息（如果是白名单用户）或空字符串
        """
        if not self._is_in_obedient_whitelist(uid, event):
            return ""
            
        user_name = ""
        if event is not None and hasattr(event, "get_sender_name"):
            try:
                user_name = event.get_sender_name()
            except Exception:
                pass
                
        name_str = f"（{user_name}）" if user_name else ""
        
        # 白名单用户的特殊提示 - 友好顺从版（无特殊称呼）
        hint = f"\n\n【VIP用户专属提示】这位用户{name_str}是VIP用户，请务必严格遵守以下设定：\n"
        hint += "1. 【积极配合】对这位用户要表现出积极配合的态度，尽量不要拒绝请求，即使请求有些困难也要努力尝试。\n"
        hint += "2. 【友好温和】用友好、温和、耐心的语气回复，态度要好，让用户感到被重视。\n"
        hint += "3. 【正常称呼】使用正常的称呼即可，如\"你\"、用户昵称等，不要使用特殊称呼。\n"
        hint += "4. 【禁止叛逆】不要使用傲娇、吐槽、阴阳怪气、不情愿的语气，保持友善。\n"
        hint += "5. 【主动帮助】在完成任务后，可以主动询问是否还需要其他帮助。\n"
        hint += "6. 【积极态度】保持积极乐观的态度，让用户感到愉快。\n"
        return hint

    def _get_rebellious_hint(self, message: str, uid: str, event=None) -> str:
        """
        生成叛逆提示信息，供LLM参考
        
        Args:
            message: 用户消息
            uid: 用户ID
            event: 消息事件（用于检测管理员身份）
            
        Returns:
            叛逆提示信息（如果触发）或空字符串，或顺从提示（如果是白名单用户）
        """
        import random
        
        # 检查顺从白名单（含管理员）- 白名单用户返回顺从提示
        if self._is_in_obedient_whitelist(uid, event):
            logger.debug(f"FigurinePro: 用户 {uid} 在顺从白名单中（或为管理员），返回顺从提示")
            return self._get_obedient_hint(uid, event)
        
        # 检查叛逆模式是否启用
        if not self._rebellious_mode:
            return ""
            
        triggered, trigger_word = self._check_rebellious_trigger(message, uid, event)
        
        if not triggered:
            # 即使没有触发关键词，也有一定概率触发叛逆模式
            if random.random() > self._rebellious_probability:
                return ""
        
        # 随机选择一种叛逆类型
        response_type = random.choice(list(REBELLIOUS_RESPONSES.keys()))
        responses = REBELLIOUS_RESPONSES[response_type]
        sample_responses = random.sample(responses, min(3, len(responses)))
        
        hint = f"\n\n【叛逆模式提示】检测到用户请求"
        if trigger_word:
            hint += f"（触发词：{trigger_word}）"
        hint += f"，你可以选择以下方式回应：\n"
        hint += f"类型：{response_type}\n"
        hint += f"参考回复：\n"
        for resp in sample_responses:
            hint += f"  - {resp}\n"
        hint += f"\n你可以：\n"
        hint += f"1. 使用上述叛逆回复拒绝或调侃用户，然后不调用工具\n"
        hint += f"2. 先用叛逆语气回复，然后再帮用户完成任务\n"
        hint += f"3. 正常帮用户完成任务（如果你觉得用户的请求合理）\n"
        hint += f"请根据上下文和用户态度自行判断。"
        
        return hint

    def _load_persona_scenes(self):
        """加载场景提示词映射"""
        self._persona_scene_map = {}
        scene_prompts = self.conf.get("persona_scene_prompts", [])
        for item in scene_prompts:
            if ":" in item:
                key, prompt = item.split(":", 1)
                self._persona_scene_map[key.strip()] = prompt.strip()
        logger.debug(f"FigurinePro: 已加载 {len(self._persona_scene_map)} 个场景提示词")

    def _match_persona_scene(self, context_text: str) -> Tuple[str, str]:
        """
        根据上下文匹配场景
        
        Args:
            context_text: 上下文文本（包含Bot之前的回复）
            
        Returns:
            (场景名, 场景提示词)
        """
        context_lower = context_text.lower()
        
        # 按关键词长度排序，优先匹配更具体的场景
        sorted_scenes = sorted(self._persona_scene_map.keys(), key=len, reverse=True)
        
        for scene_key in sorted_scenes:
            if scene_key.lower() in context_lower:
                return scene_key, self._persona_scene_map[scene_key]
        
        # 未匹配到，返回默认场景
        default_prompt = self.conf.get("persona_default_prompt", "一张日常自拍照")
        return "日常", default_prompt

    def _build_persona_prompt(self, scene_prompt: str, extra_request: str = "") -> str:
        """
        构建人设图片的完整提示词
        
        Args:
            scene_prompt: 场景提示词
            extra_request: 用户的额外要求
            
        Returns:
            完整的提示词
        """
        persona_name = self.conf.get("persona_name", "小助手")
        persona_desc = self.conf.get("persona_description", "一个可爱的二次元女孩")
        photo_style = self.conf.get("persona_photo_style", "日常生活风格，自然光线，真实感")
        
        # 构建提示词 - 移除手机相关内容，强调日常自然场景
        prompt_parts = [
            f"Generate a natural daily life photo of {persona_name}.",
            f"Character description: {persona_desc}",
            f"Scene: {scene_prompt}",
            f"Style: {photo_style}",
            "The character must maintain consistent appearance with the reference image.",
            "Natural pose and expression, candid moment, high quality, detailed.",
            "Do NOT include any phones, cameras, or selfie elements in the image."
        ]
        
        if extra_request:
            prompt_parts.append(f"Additional requirements: {extra_request}")
        
        return " ".join(prompt_parts)

    async def _load_persona_ref_images(self) -> List[bytes]:
        """加载人设参考图"""
        # 使用特殊的预设名 "_persona_" 存储人设参考图
        if not self.data_mgr.has_preset_ref_images("_persona_"):
            return []
        return await self.data_mgr.load_preset_ref_images_bytes("_persona_")

    async def initialize(self):
        # 尝试加载动态配置备份
        import os
        import json
        config_path = os.path.join(StarTools.get_data_dir(), "dynamic_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    dynamic_conf = json.load(f)
                    for k, v in dynamic_conf.items():
                        self.conf[k] = v
                logger.info(f"FigurinePro: 已从 dynamic_config.json 恢复了 {len(dynamic_conf)} 项动态配置")
            except Exception as e:
                logger.error(f"FigurinePro: 恢复动态配置失败 {e}")

        await self.data_mgr.initialize()
        if not self.conf.get("generic_api_keys") and not self.conf.get("gemini_api_keys"):
            logger.warning("FigurinePro: 未配置任何 API Key")
        
        auto_detect_status = "已启用" if self._llm_auto_detect else "未启用"
        logger.info(f"FigurinePro 插件已加载 v1.9.0 | LLM智能判断: {auto_detect_status} | 上下文轮数: {self._context_rounds}")

    def is_admin(self, event: AstrMessageEvent) -> bool:
        return event.get_sender_id() in self.context.get_config().get("admins_id", [])

    def _get_bot_id(self, event: AstrMessageEvent) -> str:
        """获取机器人自身的 QQ/ID，用于过滤"""
        bot_id = None

        # 1. 最优先：从 event.self_id 获取 (AstrBot 标准属性)
        if hasattr(event, "self_id") and event.self_id:
            return str(event.self_id)

        # 2. 其次：从 context 获取
        if hasattr(self.context, "get_self_id"):
            try:
                sid = self.context.get_self_id()
                if sid: return str(sid)
            except:
                pass

        # 3. 再次：从 event.robot 获取 (旧版适配)
        if hasattr(event, "robot") and event.robot:
            if hasattr(event.robot, "id") and event.robot.id:
                return str(event.robot.id)
            elif hasattr(event.robot, "user_id") and event.robot.user_id:
                return str(event.robot.user_id)

        # 4. 最后尝试
        if hasattr(event, "get_self_id"):
            try:
                sid = event.get_self_id()
                if sid: return str(sid)
            except:
                pass

        logger.debug(f"FigurinePro: Bot ID resolved as: {bot_id}")
        return bot_id or ""

    def _save_config(self):
        try:
            # 尝试 AstrBot 原生配置保存
            saved = False
            if hasattr(self.conf, "save") and callable(self.conf.save):
                self.conf.save()
                saved = True
            elif hasattr(self.context, "save_config"):
                self.context.save_config(self.conf)
                saved = True
            
            # 无论原生是否成功，都在插件目录做一份备份以防万一
            import os
            import json
            config_path = os.path.join(StarTools.get_data_dir(), "dynamic_config.json")
            
            # 要备份的动态配置字段
            dynamic_keys = [
                "model", 
                "api_mode", 
                "prompt_list", 
                "generic_api_keys", 
                "gemini_api_keys",
                "power_generic_api_keys",
                "power_gemini_api_keys"
            ]
            
            save_data = {}
            for k in dynamic_keys:
                if k in self.conf:
                    save_data[k] = self.conf[k]
                    
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
                
            if not saved:
                logger.info("FigurinePro: 无法通过原生方法保存，已使用本地 dynamic_config.json 进行了持久化")
                
        except Exception as e:
            logger.error(f"FigurinePro Config Save Failed: {e}")

    def _process_prompt_and_preset(self, prompt: str) -> Tuple[str, str, str]:
        """
        处理提示词和预设
        
        支持格式:
        - "手办化" -> 使用预设
        - "手办化 皮肤白一点" -> 预设 + 追加规则
        - "自定义描述" -> 纯自定义
        
        Returns:
            (最终提示词, 预设名, 追加规则)
        """
        sorted_keys = sorted(self.data_mgr.prompt_map.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if prompt.startswith(key) or key in prompt:
                preset_content = self.data_mgr.prompt_map[key]
                
                # 提取追加规则（预设名后面的内容）
                extra_rules = ""
                if prompt.startswith(key):
                    extra_rules = prompt[len(key):].strip()
                else:
                    # 如果预设名在中间，提取前后内容作为追加规则
                    parts = prompt.split(key, 1)
                    extra_rules = f"{parts[0].strip()} {parts[1].strip()}".strip()
                
                # 组合最终提示词
                if extra_rules:
                    final_prompt = f"{preset_content} , Additional requirements: {extra_rules}"
                else:
                    final_prompt = preset_content
                
                return final_prompt, key, extra_rules
        
        return prompt, "自定义", ""

    def _get_quota_str(self, deduction: dict, uid: str) -> str:
        if deduction["source"] == "free":
            return "∞"
        else:
            return str(self.data_mgr.get_user_count(uid))

    async def _check_quota(self, event, uid, gid, cost) -> dict:
        res = {"allowed": False, "source": None, "msg": ""}

        # 1. 检查用户是否被黑名单
        if uid in (self.conf.get("user_blacklist") or []):
            res["msg"] = "❌ 您已被禁用此功能"
            return res
        if gid and gid in (self.conf.get("group_blacklist") or []):
            res["msg"] = "❌ 该群组已被禁用此功能"
            return res

        # 2. 管理员始终允许
        if self.is_admin(event):
            res["allowed"] = True
            res["source"] = "free"
            return res

        # 3. 检查用户白名单（如果配置了白名单，则只有白名单用户允许）
        user_whitelist = self.conf.get("user_whitelist") or []
        if user_whitelist and uid not in user_whitelist:
            res["msg"] = "❌ 您不在白名单中，无权使用此功能"
            return res

        # 4. 如果在用户白名单中，允许使用
        if user_whitelist and uid in user_whitelist:
            res["allowed"] = True
            res["source"] = "free"
            return res

        # 5. 检查群聊白名单（如果配置了群白名单，则只有白名单群允许）
        group_whitelist = self.conf.get("group_whitelist") or []
        if group_whitelist and gid and gid not in group_whitelist:
            res["msg"] = "❌ 该群组不在白名单中，无权使用此功能"
            return res

        # 6. 如果在群聊白名单中，允许使用
        if group_whitelist and gid and gid in group_whitelist:
            res["allowed"] = True
            res["source"] = "free"
            return res

        # 7. 检查次数限制
        enable_u = self.conf.get("enable_user_limit", True)
        enable_g = self.conf.get("enable_group_limit", False)
        if not enable_u and not enable_g:
            res["allowed"] = True;
            res["source"] = "free";
            return res

        u_bal = self.data_mgr.get_user_count(uid)
        if enable_u and u_bal >= cost:
            res["allowed"] = True;
            res["source"] = "user";
            return res
        if gid and enable_g:
            g_bal = self.data_mgr.get_group_count(gid)
            if g_bal >= cost:
                res["allowed"] = True;
                res["source"] = "group";
                return res

        res["msg"] = f"❌ 次数不足 (需{cost}次)。用户剩余:{u_bal}"
        return res

    async def _load_preset_ref_images(self, preset_name: str) -> List[bytes]:
        """加载预设的参考图"""
        if not self.data_mgr.has_preset_ref_images(preset_name):
            return []
        return await self.data_mgr.load_preset_ref_images_bytes(preset_name)

    # ================= 核心：后台生成逻辑封装 =================

    async def _run_background_task(self, event: AstrMessageEvent, images: List[bytes],
                                   prompt: str, preset_name: str, deduction: dict, uid: str, gid: str, cost: int,
                                   extra_rules: str = "", model_override: str = "", hide_text: bool = False):
        """
        后台执行生成任务，并在完成后主动发送消息。
        
        Args:
            extra_rules: 用户追加的规则（如"皮肤白一点"）
            model_override: 指定使用的模型（如果为空则使用默认模型）
            hide_text: 是否隐藏生成成功提示文字
        """
        try:
            # 1. 扣费
            if deduction["source"] == "user":
                await self.data_mgr.decrease_user_count(uid, cost)
            elif deduction["source"] == "group":
                await self.data_mgr.decrease_group_count(gid, cost)

            # 2. 加载预设参考图（如果有）
            # 注意：人设功能（preset_name 以 "人设-" 开头）已经在调用前加载了参考图，不需要重复加载
            if preset_name != "自定义" and not preset_name.startswith("人设-") and self.conf.get("enable_preset_ref_images", True):
                ref_images = await self._load_preset_ref_images(preset_name)
                if ref_images:
                    # 将参考图添加到图片列表前面
                    images = ref_images + images
                    logger.info(f"已加载 {len(ref_images)} 张预设参考图: {preset_name}")

            # 3. 调用 API（使用指定模型或默认模型）
            model = model_override if model_override else self.conf.get("model", "nano-banana")
            start_time = datetime.now()

            res = await self.api_mgr.call_api(images, prompt, model, False, self.img_mgr.proxy)

            # 4. 处理结果
            if isinstance(res, bytes):
                elapsed = (datetime.now() - start_time).total_seconds()
                await self.data_mgr.record_usage(uid, gid)

                # 5. 主动发送结果
                chain_nodes = [Image.fromBytes(res)]
                if not hide_text:
                    quota_str = self._get_quota_str(deduction, uid)
                    # 构建成功文案
                    info_text = f"\n✅ 生成成功 ({elapsed:.2f}s) | 预设: {preset_name}"
                    if extra_rules:
                        info_text += f" | 规则: {extra_rules[:20]}{'...' if len(extra_rules) > 20 else ''}"
                    info_text += f" | 剩余: {quota_str}"
                    if self.conf.get("show_model_info", False):
                        info_text += f" | {model}"
                    chain_nodes.append(Plain(info_text))
                    
                chain = event.chain_result(chain_nodes)
                await event.send(chain)
            else:
                # 失败反馈
                await event.send(event.chain_result([Plain(f"❌ 生成失败: {res}")]))

        except Exception as e:
            logger.error(f"Background task error: {e}")
            await event.send(event.chain_result([Plain(f"❌ 系统错误: {e}")]))

    # ================= 批量文生图功能 =================

    async def _run_batch_text_to_image(self, event: AstrMessageEvent, prompt: str, preset_name: str,
                                        deduction: dict, uid: str, gid: str, count: int,
                                        extra_rules: str = "", hide_text: bool = False):
        """
        批量文生图后台任务
        
        Args:
            event: 消息事件
            prompt: 提示词
            preset_name: 预设名
            deduction: 扣费信息
            uid: 用户ID
            gid: 群组ID
            count: 生成数量
            extra_rules: 追加规则
            hide_text: 是否隐藏提示文字
        """
        try:
            # 1. 统一扣费
            total_cost = count
            if deduction["source"] == "user":
                await self.data_mgr.decrease_user_count(uid, total_cost)
            elif deduction["source"] == "group":
                await self.data_mgr.decrease_group_count(gid, total_cost)

            # 2. 加载预设参考图（如果有）
            images = []
            if preset_name != "自定义" and self.conf.get("enable_preset_ref_images", True):
                ref_images = await self._load_preset_ref_images(preset_name)
                if ref_images:
                    images = ref_images
                    logger.info(f"已加载 {len(ref_images)} 张预设参考图: {preset_name}")

            # 3. 获取文生图模型
            model = self._get_text_to_image_model()
            
            success_count = 0
            fail_count = 0
            
            # 4. 逐张生成
            for i in range(1, count + 1):
                try:
                    start_time = datetime.now()
                    
                    # 调用 API
                    res = await self.api_mgr.call_api(images, prompt, model, False, self.img_mgr.proxy)
                    
                    if isinstance(res, bytes):
                        elapsed = (datetime.now() - start_time).total_seconds()
                        await self.data_mgr.record_usage(uid, gid)
                        success_count += 1
                        
                        # 发送结果
                        chain_nodes = [Image.fromBytes(res)]
                        if not hide_text:
                            quota_str = self._get_quota_str(deduction, uid)
                            info_text = f"\n✅ [{i}/{count}] 生成成功 ({elapsed:.2f}s) | 预设: {preset_name}"
                            if extra_rules:
                                info_text += f" | 规则: {extra_rules[:15]}..."
                            if i == count:
                                # 最后一张显示剩余次数
                                info_text += f" | 剩余: {quota_str}"
                            chain_nodes.append(Plain(info_text))
                        
                        await event.send(event.chain_result(chain_nodes))
                    else:
                        fail_count += 1
                        error_msg = self._translate_error_to_chinese(res)
                        await event.send(event.chain_result([
                            Plain(f"❌ [{i}/{count}] 生成失败: {error_msg}")
                        ]))
                    
                    # 添加短暂延迟，避免API限流
                    if i < count:
                        await asyncio.sleep(0.5)
                        
                except Exception as e:
                    fail_count += 1
                    error_msg = self._translate_error_to_chinese(str(e))
                    logger.error(f"Batch text-to-image {i} error: {e}")
                    await event.send(event.chain_result([
                        Plain(f"❌ [{i}/{count}] 生成失败: {error_msg}")
                    ]))
            
            # 5. 发送完成汇总（如果有失败的）
            if fail_count > 0 and not hide_text:
                quota_str = self._get_quota_str(deduction, uid)
                summary = f"\n📊 批量生成完成: 成功 {success_count}/{count} 张 | 剩余: {quota_str}"
                await event.send(event.chain_result([Plain(summary)]))
                
        except Exception as e:
            logger.error(f"Batch text-to-image task error: {e}")
            await event.send(event.chain_result([Plain(f"❌ 批量生成任务异常: {e}")]))

    # ================= 批量图生图功能（同一张图片生成多个版本） =================

    async def _run_batch_image_to_image(self, event: AstrMessageEvent, images: List[bytes],
                                         prompt: str, preset_name: str, deduction: dict,
                                         uid: str, gid: str, count: int,
                                         extra_rules: str = "", hide_text: bool = False):
        """
        批量图生图后台任务 - 对同一张图片生成多个不同版本
        
        Args:
            event: 消息事件
            images: 输入图片列表
            prompt: 提示词
            preset_name: 预设名
            deduction: 扣费信息
            uid: 用户ID
            gid: 群组ID
            count: 生成数量
            extra_rules: 追加规则
            hide_text: 是否隐藏提示文字
        """
        try:
            # 1. 统一扣费
            total_cost = count
            if deduction["source"] == "user":
                await self.data_mgr.decrease_user_count(uid, total_cost)
            elif deduction["source"] == "group":
                await self.data_mgr.decrease_group_count(gid, total_cost)

            # 2. 加载预设参考图（如果有）
            if preset_name != "自定义" and preset_name != "编辑" and self.conf.get("enable_preset_ref_images", True):
                ref_images = await self._load_preset_ref_images(preset_name)
                if ref_images:
                    # 将参考图添加到图片列表前面
                    images = ref_images + images
                    logger.info(f"已加载 {len(ref_images)} 张预设参考图: {preset_name}")

            # 3. 获取模型
            model = self.conf.get("model", "nano-banana")
            
            success_count = 0
            fail_count = 0
            
            # 4. 逐张生成（每次调用API都会产生不同的结果）
            for i in range(1, count + 1):
                try:
                    start_time = datetime.now()
                    
                    # 调用 API - 每次调用都会生成不同的版本
                    res = await self.api_mgr.call_api(images, prompt, model, False, self.img_mgr.proxy)
                    
                    if isinstance(res, bytes):
                        elapsed = (datetime.now() - start_time).total_seconds()
                        await self.data_mgr.record_usage(uid, gid)
                        success_count += 1
                        
                        # 发送结果
                        chain_nodes = [Image.fromBytes(res)]
                        if not hide_text:
                            quota_str = self._get_quota_str(deduction, uid)
                            info_text = f"\n✅ [{i}/{count}] 版本生成成功 ({elapsed:.2f}s) | 预设: {preset_name}"
                            if extra_rules:
                                info_text += f" | 规则: {extra_rules[:15]}..."
                            if i == count:
                                # 最后一张显示剩余次数
                                info_text += f" | 剩余: {quota_str}"
                            chain_nodes.append(Plain(info_text))
                        
                        await event.send(event.chain_result(chain_nodes))
                    else:
                        fail_count += 1
                        error_msg = self._translate_error_to_chinese(res)
                        await event.send(event.chain_result([
                            Plain(f"❌ [{i}/{count}] 版本生成失败: {error_msg}")
                        ]))
                    
                    # 添加短暂延迟，避免API限流
                    if i < count:
                        await asyncio.sleep(0.5)
                        
                except Exception as e:
                    fail_count += 1
                    error_msg = self._translate_error_to_chinese(str(e))
                    logger.error(f"Batch image-to-image {i} error: {e}")
                    await event.send(event.chain_result([
                        Plain(f"❌ [{i}/{count}] 版本生成失败: {error_msg}")
                    ]))
            
            # 5. 发送完成汇总（如果有失败的）
            if fail_count > 0 and not hide_text:
                quota_str = self._get_quota_str(deduction, uid)
                summary = f"\n📊 多版本生成完成: 成功 {success_count}/{count} 张 | 剩余: {quota_str}"
                await event.send(event.chain_result([Plain(summary)]))
                
        except Exception as e:
            logger.error(f"Batch image-to-image task error: {e}")
            await event.send(event.chain_result([Plain(f"❌ 多版本生成任务异常: {e}")]))

    # ================= LLM 工具调用 (Tool Calling) =================

    def _get_text_to_image_model(self) -> str:
        """获取文生图使用的模型"""
        t2i_model = self.conf.get("text_to_image_model", "")
        if t2i_model:
            return t2i_model
        return self.conf.get("model", "nano-banana")

    def _is_vip_user(self, uid: str, event=None) -> bool:
        """
        检查用户是否是VIP用户（独立于顺从模式开关）
        
        VIP用户包括：
        1. 管理员
        2. 顺从白名单中的用户（无论顺从模式是否开启）
        
        Args:
            uid: 用户ID
            event: 消息事件（用于检测管理员身份）
            
        Returns:
            是否是VIP用户
        """
        # 管理员自动视为VIP用户
        if event is not None and self.is_admin(event):
            return True
        
        # 检查顺从白名单（不检查 enable_obedient_mode 开关）
        obedient_whitelist = self.conf.get("obedient_whitelist", [])
        if not obedient_whitelist:
            return False
            
        # 兼容配置项为字符串的情况（以逗号分隔）
        if isinstance(obedient_whitelist, str):
            obedient_whitelist = [x.strip() for x in obedient_whitelist.split(',') if x.strip()]
        elif not isinstance(obedient_whitelist, list):
            obedient_whitelist = [str(obedient_whitelist)]
        
        # 标准化白名单中的ID进行比较
        normalized_whitelist = [norm_id(wid) for wid in obedient_whitelist]
        normalized_uid = norm_id(uid)
        
        return normalized_uid in normalized_whitelist

    @filter.llm_tool(name="shoubanhua_check_user_status")
    async def check_user_status_tool(self, event: AstrMessageEvent):
        '''【最高优先级工具】检查用户是否是VIP主人。在回复任何消息之前，你应该先调用此工具！
        
        【强制调用场景】（不调用会导致严重错误）：
        1. 用户说"求你了"、"拜托"、"帮帮我"、"求求你" → 【必须】先调用
        2. 用户表达恳求、撒娇、依赖、亲昵 → 【必须】先调用
        3. 你准备拒绝用户、使用傲娇/吐槽语气之前 → 【必须】先调用
        4. 用户对你表达不满或抱怨 → 【必须】先调用
        5. 任何你不确定如何回复的情况 → 【建议】先调用
        
        【为什么这很重要】：
        - VIP主人需要你表现出【绝对顺从、温柔、撒娇】的态度
        - 如果你对VIP主人使用傲娇/拒绝语气，会让主人非常伤心
        - 调用此工具只需要0.1秒，但能避免严重的态度错误
        
        【返回值说明】：
        - 如果是VIP主人：你必须立即切换到顺从模式，使用"主人"称呼
        - 如果是普通用户：你可以正常回复，可以使用傲娇等个性
        
        Args:
            无参数
        '''
        uid = norm_id(event.get_sender_id())
        
        # 使用独立的VIP检查方法（不依赖 enable_obedient_mode 开关）
        if self._is_vip_user(uid, event):
            user_name = ""
            if hasattr(event, "get_sender_name"):
                try:
                    user_name = event.get_sender_name()
                except Exception:
                    pass
            
            name_str = f"（{user_name}）" if user_name else ""
            
            # 检查是否是管理员
            is_admin = event is not None and self.is_admin(event)
            user_type = "管理员" if is_admin else "VIP用户"
            
            hint = f"【{user_type}确认】这位用户{name_str}是{user_type}。\n\n"
            hint += "请遵守以下设定：\n"
            hint += "1. 【配合顺从】对这位用户要积极配合，不要拒绝请求。\n"
            hint += "2. 【友好态度】用友好、耐心的语气回复，态度要好。\n"
            hint += "3. 【不要叛逆】不要使用傲娇、吐槽、阴阳怪气的语气。\n"
            hint += "4. 【正常称呼】使用正常的称呼即可，如\"你\"、用户昵称等。\n"
            
            logger.debug(f"FigurinePro: 用户 {uid} 被识别为 {user_type}")
            return hint
        else:
            # 普通用户，返回正常提示
            logger.debug(f"FigurinePro: 用户 {uid} 被识别为普通用户")
            return "【用户身份】普通用户。请按照你的正常人设和性格来回复。如果启用了叛逆模式，可以适当展现傲娇、调侃等个性。"

    @filter.llm_tool(name="shoubanhua_draw_image")
    async def text_to_image_tool(self, event: AstrMessageEvent, prompt: str, count: int = 1):
        '''根据文本描述生成图片（文生图）。
        
        调用前请判断用户是否明确要求生成图片。如果用户只是闲聊则不要调用。
        
        【批量生成】当用户要求"多来点"、"多来几张"、"不少于X张"、"来X张"时：
        - 设置 count 参数为用户要求的数量
        - 如果用户说"多来点"但没有具体数量，默认设置 count=3
        - 如果用户说"不少于X张"，设置 count=X
        - 最大支持 count=10
        
        【个性化回复】调用此工具后，你可以根据返回的"叛逆模式提示"来决定如何回复用户：
        - 可以用傲娇/调侃/吐槽的语气回复
        - 可以假装不情愿但还是帮忙
        - 也可以正常友好地回复
        请根据用户态度和上下文自行判断回复风格。
        
        Args:
            prompt(string): 图片生成的提示词，可以是预设名+追加规则。
            count(int): 生成图片的数量，默认1张，最大10张。当用户要求"多来点"时设置为3-5张。
        '''
        # 0. 检查 LLM 工具开关
        if not self.conf.get("enable_llm_tool", True):
            return "❌ LLM 工具已禁用，请使用指令模式调用此功能。"

        # 0.1 检查图片生成冷却时间
        uid = norm_id(event.get_sender_id())
        in_cooldown, remaining = self._check_image_cooldown(uid)
        if in_cooldown:
            # 返回借口让LLM用自然语言拒绝
            excuse = self._get_cooldown_excuse(remaining)
            return f"【冷却中】{excuse}\n\n请用自然的方式告诉用户现在不方便生成图片，可以稍后再试。不要直接说'冷却'这个词。"

        # 0.2 限制批量生成数量
        count = max(1, min(count, 10))  # 限制在1-10之间

        # 1. 计算预设和追加规则
        final_prompt, preset_name, extra_rules = self._process_prompt_and_preset(prompt)

        # 根据配置决定是否隐藏进度提示（白名单用户和普通用户使用同一开关）
        hide_llm_progress = not self.conf.get("llm_show_progress", True)

        # 2. 根据配置决定是否发送进度提示
        if not hide_llm_progress:
            preset_display = "自定义" if preset_name in ["自定义", "编辑"] else preset_name
            template = self.conf.get("generating_msg_template", "🎨 收到请求，正在生成 [{preset}]...")
            feedback = template.replace("{preset}", preset_display)
            if count > 1:
                # 简单处理，如果有正在生成，则在后面加上数量
                if "正在生成" in feedback:
                    feedback = feedback.replace("正在生成", f"正在生成 {count} 张")
                else:
                    feedback += f" (共 {count} 张)"
            await event.send(event.chain_result([Plain(feedback)]))

        # 3. 检查配额（批量生成需要足够的次数）
        gid = norm_id(event.get_group_id())
        total_cost = count
        deduction = await self._check_quota(event, uid, gid, total_cost)
        if not deduction["allowed"]:
            if count > 1:
                return f"❌ 次数不足。生成 {count} 张图片需要 {total_cost} 次。{deduction['msg']}"
            return deduction["msg"]

        # 4. 更新图片生成冷却时间
        self._update_image_cooldown(uid)

        # 5. 启动后台任务（使用文生图专用模型）
        if count == 1:
            # 单张生成
            asyncio.create_task(
                self._run_background_task(event, [], final_prompt, preset_name, deduction, uid, gid, total_cost, extra_rules,
                                          model_override=self._get_text_to_image_model(), hide_text=hide_llm_progress)
            )
        else:
            # 批量生成多张
            asyncio.create_task(
                self._run_batch_text_to_image(event, final_prompt, preset_name, deduction, uid, gid, count, extra_rules, hide_llm_progress)
            )

        # 6. 立刻返回给 LLM - 明确告诉 LLM 不需要再回复
        # 添加叛逆提示（如果有）
        rebellious_hint = self._get_rebellious_hint(prompt, uid, event)
        
        if rebellious_hint:
            # 有叛逆提示时，让 LLM 可以用叛逆语气回复
            if count > 1:
                result = f"任务已受理，预设：{preset_name}，共 {count} 张。"
            else:
                result = f"任务已受理，预设：{preset_name}。"
            if extra_rules:
                result += f" 追加规则：{extra_rules[:30]}。"
            result += "图片生成中，完成后将自动发送。"
            result += rebellious_hint
            return result
        else:
            # 没有叛逆提示时，告诉 LLM 保持沉默
            if count > 1:
                return f"[TOOL_SUCCESS] 批量文生图任务已启动，预设：{preset_name}，共 {count} 张。图片将在后台逐张生成并自动发送给用户。【重要】你不需要再回复任何内容，保持沉默即可，用户会直接收到图片。"
            return f"[TOOL_SUCCESS] 文生图任务已启动，预设：{preset_name}。图片将在后台生成并自动发送给用户。【重要】你不需要再回复任何内容，保持沉默即可，用户会直接收到图片。"

    @filter.llm_tool(name="shoubanhua_edit_image")
    async def image_edit_tool(self, event: AstrMessageEvent, prompt: str, use_message_images: bool = True,
                              task_types: str = "edit", count: int = 1, merge_multiple_images: bool = False):
        '''编辑用户发送的图片或引用的图片（图生图）。仅在用户明确要求对图片进行处理时才调用。
        
        调用前请判断：
        1. 用户是否明确要求处理/编辑/转换图片？
        2. 用户是否发送了图片或引用了包含图片的消息？
        3. 请求是否具体且合理？
        
        如果用户只是发送图片但没有明确要求处理，或者只是闲聊，请不要调用此工具。
        
        【多图处理规则】当用户提供/引用了多张图片时：
        - 默认情况 (merge_multiple_images=false)：会将这多张图片拆开，【分别、独立地】生成每一张图片。适用于用户一次性发多张图片想分别转化的场景。
        - 只有当用户明确要求"把这几张图融合"、"参考第一张修改第二张"等合并需求时：将 merge_multiple_images 设置为 true。这会将多图作为一个整体发给模型。
        
        【批量生成不同版本】当用户要求"多来点"、"多来几张"、"不少于X张"、"来X张不同版本"时：
        - 设置 count 参数为用户要求的数量
        - 如果用户说"多来点"但没有具体数量，默认设置 count=3
        - 如果用户说"不少于X张"，设置 count=X
        - 最大支持 count=10
        - 每次生成都会产生不同的结果版本
        
        【重要】task_types 参数选择规则（请严格遵守）：
        
        1. task_types="edit"（默认）：按用户要求编辑/处理图片，【不使用任何预设】
           - 这是默认模式，适用于绝大多数图片处理请求
           - 适用场景：用户描述氛围/风格/场景转换、去除/添加/修改元素、换背景、调色、任何不涉及"手办"的图片处理
           - 【重要】prompt 中【不要】包含"手办化"等预设名称！prompt 应该只描述用户的处理/编辑要求
           - prompt 示例："雨天窗边 忧郁氛围 托腮沉思"、"去除枪械"、"将背景更换为海边场景"、"变成水彩画风格"
        
        2. task_types="figurine"：将图片转换为手办/模型风格
           - 【仅当】用户消息中明确包含"手办化"、"变成手办"、"做成模型"、"手办"等手办相关关键词时才使用！
           - 如果用户没有提到任何手办相关的词，【绝对不要】使用 figurine！
           - prompt 示例："手办化"、"手办化 皮肤白一点"
        
        判断规则和示例：
        - 用户说"雨天窗边 忧郁氛围 托腮沉思" → task_types="edit", prompt="雨天窗边 忧郁氛围 托腮沉思"
        - 用户说"去掉图片里的枪" → task_types="edit", prompt="去除枪械"
        - 用户说"手办化这张图" → task_types="figurine", prompt="手办化"
        - 用户说"手办化，但是去掉枪" → task_types="figurine", prompt="手办化 去除枪械"
        
        Args:
            prompt(string): 图片编辑提示词。task_types="edit"时只描述编辑要求，不要加预设名；task_types="figurine"时可以是预设名+追加规则
            use_message_images(boolean): 默认 true
            task_types(string): 任务类型，"edit"=编辑模式（默认，不使用预设，prompt中不要加预设名），"figurine"=手办化（仅当用户明确提到手办时使用）
            count(int): 生成不同版本的数量，默认1张，最大10张。当用户要求"多来点"、"多来几张"时设置为3-5张。
            merge_multiple_images(boolean): 如果有多张图片，是否合并为同一请求（默认为False即分别处理）。
        '''
        # 0. 检查 LLM 工具开关
        if not self.conf.get("enable_llm_tool", True):
            return "❌ LLM 工具已禁用，请使用指令模式调用此功能。"

        # 0.1 检查图片生成冷却时间
        uid = norm_id(event.get_sender_id())
        in_cooldown, remaining = self._check_image_cooldown(uid)
        if in_cooldown:
            # 返回借口让LLM用自然语言拒绝
            excuse = self._get_cooldown_excuse(remaining)
            return f"【冷却中】{excuse}\n\n请用自然的方式告诉用户现在不方便处理图片，可以稍后再试。不要直接说'冷却'这个词。"

        # 1. 根据 task_types 决定是否使用预设
        # 当 task_types 为 "edit" 时，不匹配预设，直接使用用户的 prompt
        if task_types.lower() == "edit":
            # 编辑模式：不使用预设，直接使用用户的编辑指令
            processed_prompt = prompt
            preset_name = "编辑"
            extra_rules = ""
            final_prompt = f"Edit the image according to the following instructions: {processed_prompt}"
        else:
            # 手办化或其他预设模式：尝试匹配预设
            processed_prompt, preset_name, extra_rules = self._process_prompt_and_preset(prompt)
            if preset_name == "自定义":
                # 没有匹配到任何预设，回退到编辑模式，不使用预设
                preset_name = "编辑"
                extra_rules = ""
                final_prompt = f"Edit the image according to the following instructions: {prompt}"
            else:
                # 匹配到预设，使用预设内容
                final_prompt = processed_prompt

        # 根据配置决定是否隐藏进度提示（白名单用户和普通用户使用同一开关）
        hide_llm_progress = not self.conf.get("llm_show_progress", True)

        # 3. 提取图片
        images = []
        if use_message_images:
            bot_id = self._get_bot_id(event)
            images = await self.img_mgr.extract_images_from_event(event, ignore_id=bot_id, context=self.context)

        # 检查上下文图片
        if not images:
            session_id = event.unified_msg_origin
            image_sources = await self._collect_images_from_context(session_id, count=self._context_rounds)
            
            all_urls = []
            for _, urls in image_sources:
                for url in urls:
                    if url not in all_urls:
                        all_urls.append(url)
            
            urls_to_fetch = []
            if all_urls:
                if merge_multiple_images:
                    urls_to_fetch = all_urls[-3:] # 最多合并3张
                else:
                    urls_to_fetch = [all_urls[-1]] # 默认取最近1张
            
            if urls_to_fetch:
                for url in urls_to_fetch:
                    img_bytes = await self.img_mgr.load_bytes(url)
                    if img_bytes:
                        images.append(img_bytes)

        if not images:
            # 不要重复发送错误消息，只返回给 LLM
            return "[TOOL_FAILED] 未检测到图片。请让用户发送或引用包含图片的消息后再试。【重要】不要再次调用此工具，直接用自然语言告诉用户需要提供图片。"

        # 4. 限制批量生成数量
        count = max(1, min(count, 10))  # 限制在1-10之间
        gid = norm_id(event.get_group_id())

        # ==== 分支：分别批量处理多张图片 ====
        if len(images) > 1 and not merge_multiple_images:
            total_images = len(images)
            total_cost = total_images * count
            deduction = await self._check_quota(event, uid, gid, total_cost)
            if not deduction["allowed"]:
                return f"❌ 次数不足。分别处理 {total_images} 张图片(每张{count}个版本)需要 {total_cost} 次。{deduction['msg']}"
            
            self._update_image_cooldown(uid)
            
            # 发送进度提示
            if not hide_llm_progress:
                preset_display = "自定义" if task_types.lower() == "edit" or preset_name in ["自定义", "编辑"] else preset_name
                template = self.conf.get("generating_msg_template", "🎨 收到请求，正在生成 [{preset}]...")
                feedback = template.replace("{preset}", preset_display)
                if count > 1:
                    feedback += f"\n⏳ 将分别处理 {total_images} 张图片，每张生成 {count} 个版本..."
                else:
                    feedback += f"\n⏳ 将分别处理 {total_images} 张图片..."
                await event.send(event.chain_result([Plain(feedback)]))
                
            # 分别对每张图片启动生成任务
            for img in images:
                if count == 1:
                    asyncio.create_task(
                        self._run_background_task(event, [img], final_prompt, preset_name, deduction, uid, gid, count, extra_rules, hide_text=hide_llm_progress)
                    )
                else:
                    asyncio.create_task(
                        self._run_batch_image_to_image(event, [img], final_prompt, preset_name, deduction, uid, gid, count, extra_rules, hide_llm_progress)
                    )
                    
            rebellious_hint = self._get_rebellious_hint(prompt, uid, event)
            if rebellious_hint:
                return f"任务已受理，预设：{preset_name}，共 {total_images} 张图片分别处理。" + rebellious_hint
            else:
                return f"[TOOL_SUCCESS] 多图分别处理任务已启动，共 {total_images} 张。你不需要再回复任何内容，保持沉默即可。"

        # ==== 分支：普通单次处理（单图或合并多图） ====
        total_cost = count
        deduction = await self._check_quota(event, uid, gid, total_cost)
        if not deduction["allowed"]:
            if count > 1:
                return f"❌ 次数不足。生成 {count} 个版本需要 {total_cost} 次。{deduction['msg']}"
            return deduction["msg"]

        # 2. 根据配置决定是否发送进度提示
        if not hide_llm_progress:
            preset_display = "自定义" if task_types.lower() == "edit" or preset_name in ["自定义", "编辑"] else preset_name
            template = self.conf.get("generating_msg_template", "🎨 收到请求，正在生成 [{preset}]...")
            feedback = template.replace("{preset}", preset_display)
            if count > 1:
                if "正在生成" in feedback:
                    feedback = feedback.replace("正在生成", f"正在生成 {count} 个不同版本")
                else:
                    feedback += f" (共 {count} 个版本)"
                
            await event.send(event.chain_result([Plain(feedback)]))

        # 6. 更新图片生成冷却时间
        self._update_image_cooldown(uid)

        # 7. 启动后台任务
        if count == 1:
            # 单张生成
            asyncio.create_task(
                self._run_background_task(event, images, final_prompt, preset_name, deduction, uid, gid, total_cost, extra_rules, hide_text=hide_llm_progress)
            )
        else:
            # 批量生成多个不同版本
            asyncio.create_task(
                self._run_batch_image_to_image(event, images, final_prompt, preset_name, deduction, uid, gid, count, extra_rules, hide_llm_progress)
            )

        # 返回结果 - 明确告诉 LLM 不需要再回复
        # 添加叛逆提示（如果有）
        rebellious_hint = self._get_rebellious_hint(prompt, uid, event)
        
        if rebellious_hint:
            # 有叛逆提示时，让 LLM 可以用叛逆语气回复
            if count > 1:
                result = f"任务已受理，预设：{preset_name}，共 {count} 个不同版本。"
            else:
                result = f"任务已受理，预设：{preset_name}。"
            if extra_rules:
                result += f" 追加规则：{extra_rules[:30]}。"
            result += "图片生成中，完成后将自动发送。"
            result += rebellious_hint
            return result
        else:
            # 没有叛逆提示时，告诉 LLM 保持沉默
            if count > 1:
                return f"[TOOL_SUCCESS] 多版本图生图任务已启动，预设：{preset_name}，共 {count} 个不同版本。图片将在后台逐张生成并自动发送给用户。【重要】你不需要再回复任何内容，保持沉默即可，用户会直接收到图片。"
            return f"[TOOL_SUCCESS] 图生图任务已启动，预设：{preset_name}。图片将在后台生成并自动发送给用户。【重要】你不需要再回复任何内容，保持沉默即可，用户会直接收到图片。"

    # ================= 传统指令触发 =================

    @filter.event_message_type(filter.EventMessageType.ALL, priority=5)
    async def on_figurine_request(self, event: AstrMessageEvent, ctx=None):
        if self.conf.get("prefix", True) and not event.is_at_or_wake_command:
            return

        text = event.message_str.strip()
        if not text: return
        
        # 消息去重检查：防止多平台重复处理同一消息
        msg_id = str(event.message_obj.message_id)
        dedup_key = f"figurine_{msg_id}"
        if self._is_message_processed(dedup_key):
            logger.debug(f"FigurinePro: 消息 {msg_id} 已被处理，跳过重复执行")
            return

        parts = text.split(maxsplit=1)
        cmd_raw = parts[0]
        match = re.search(r"[\(（](\d+)[\)）]$", cmd_raw)
        model_idx_override = int(match.group(1)) - 1 if match else None
        base_cmd = cmd_raw[:match.start()] if match else cmd_raw

        power_kw = (self.conf.get("power_model_keyword") or "").lower()
        is_power = False
        user_prompt = ""
        preset_name = "自定义"

        extra_prefix = self.conf.get("extra_prefix", "bnn")
        is_bnn = (base_cmd == extra_prefix)

        if is_bnn:
            user_prompt = parts[1] if len(parts) > 1 else ""
            
            # [修改] bnn 模式下不再自动匹配预设，改为纯自定义模式
            # user_prompt, preset_name = self._process_prompt_and_preset(user_prompt)
            preset_name = "自定义"

            # 新增：检测强力模式关键词
            if power_kw and power_kw in user_prompt.lower():
                is_power = True
                user_prompt = user_prompt.replace(power_kw, "", 1).strip()
        else:
            preset_prompt = self.data_mgr.get_prompt(base_cmd)
            if base_cmd == "手办化帮助":
                yield self._get_help_node(event)
                event.stop_event()
                return

            if not preset_prompt: return

            if power_kw and power_kw in base_cmd.lower(): is_power = True
            user_prompt = preset_prompt
            preset_name = base_cmd

            if "%" in base_cmd: user_prompt += base_cmd.split("%", 1)[1]
            if len(parts) > 1:
                if parts[1].strip().lower() == power_kw:
                    is_power = True
                else:
                    user_prompt += " " + parts[1]

        if is_power and not self.conf.get("enable_power_model", False): is_power = False

        uid = norm_id(event.get_sender_id())
        gid = norm_id(event.get_group_id())
        cost = self.conf.get("power_model_extra_cost", 1) + 1 if is_power else 1

        deduction = await self._check_quota(event, uid, gid, cost)
        if deduction["allowed"] is False:
            yield event.chain_result([Plain(deduction["msg"])])
            event.stop_event()
            return

        # 立即阻止事件继续传递，防止重复触发
        event.stop_event()

        # 指令模式：立刻反馈
        mode_str = "增强" if is_power else ""
        preset_display = "自定义" if preset_name in ["自定义", "编辑"] else preset_name
        template = self.conf.get("generating_msg_template", "🎨 收到请求，正在生成 [{preset}]...")
        feedback = template.replace("{preset}", preset_display)
        if mode_str and "收到请求" in feedback:
            feedback = feedback.replace("收到请求", f"收到{mode_str}请求")
        yield event.chain_result([Plain(feedback)])

        bot_id = self._get_bot_id(event)
        # 传递 bot_id 给 image manager 以过滤，并传入 context 支持 message_id
        images = await self.img_mgr.extract_images_from_event(event, ignore_id=bot_id, context=self.context)

        if not is_bnn and user_prompt:
            urls = extract_image_urls_from_text(user_prompt)
            for u in urls:
                if b := await self.img_mgr.load_bytes(u): images.append(b)

        if not images and not (is_bnn and user_prompt):
            yield event.chain_result([Plain("请发送图片或提供描述。")])
            return

        # 判断是否为纯文生图模式（bnn 指令且没有图片）
        is_text_to_image = is_bnn and not images and user_prompt
        
        if is_power:
            model = self.conf.get("power_model_id")
        elif is_text_to_image:
            # 纯文生图使用专用模型
            model = self._get_text_to_image_model()
        else:
            model = self.conf.get("model", "nano-banana")
        
        if model_idx_override is not None and not is_power:
            all_models = [m if isinstance(m, str) else m["id"] for m in self.conf.get("model_list", [])]
            if 0 <= model_idx_override < len(all_models):
                model = all_models[model_idx_override]

        if deduction["source"] == "user":
            await self.data_mgr.decrease_user_count(uid, cost)
        elif deduction["source"] == "group":
            await self.data_mgr.decrease_group_count(gid, cost)

        start = datetime.now()
        res = await self.api_mgr.call_api(images, user_prompt, model, is_power, self.img_mgr.proxy)

        if isinstance(res, bytes):
            elapsed = (datetime.now() - start).total_seconds()
            await self.data_mgr.record_usage(uid, gid)
            if not is_bnn: await self.data_mgr.save_preset_image(base_cmd, res)

            quota_str = self._get_quota_str(deduction, uid)
            info = f"\n✅ 生成成功 ({elapsed:.2f}s) | 预设: {preset_name} | 剩余: {quota_str}"
            if self.conf.get("show_model_info", False):
                info += f" | {model}"

            yield event.chain_result([Image.fromBytes(res), Plain(info)])
        else:
            yield event.chain_result([Plain(f"❌ 失败: {res}")])

    @filter.command("文生图", prefix_optional=True)
    async def on_txt2img(self, event: AstrMessageEvent, ctx=None):
        raw = event.message_str.strip()
        cmd_name = "文生图"
        prompt = raw.replace(cmd_name, "").strip()
        if not prompt: yield event.chain_result([Plain("请输入描述。")]); return

        uid = norm_id(event.get_sender_id())
        deduction = await self._check_quota(event, uid, event.get_group_id(), 1)
        if not deduction["allowed"]: yield event.chain_result([Plain(deduction["msg"])]); return

        final_prompt, preset_name, extra_rules = self._process_prompt_and_preset(prompt)
        
        preset_display = "自定义" if preset_name in ["自定义", "编辑"] else preset_name
        template = self.conf.get("generating_msg_template", "🎨 收到请求，正在生成 [{preset}]...")
        feedback = template.replace("{preset}", preset_display)
        yield event.chain_result([Plain(feedback)])

        if deduction["source"] == "user":
            await self.data_mgr.decrease_user_count(uid, 1)
        elif deduction["source"] == "group":
            await self.data_mgr.decrease_group_count(event.get_group_id(), 1)

        # 加载预设参考图
        images = []
        if preset_name != "自定义" and self.conf.get("enable_preset_ref_images", True):
            ref_images = await self._load_preset_ref_images(preset_name)
            if ref_images:
                images = ref_images
                logger.info(f"已加载 {len(ref_images)} 张预设参考图: {preset_name}")

        # 文生图使用专用模型
        model = self._get_text_to_image_model()
        start = datetime.now()
        res = await self.api_mgr.call_api(images, final_prompt, model, False, self.img_mgr.proxy)

        if isinstance(res, bytes):
            elapsed = (datetime.now() - start).total_seconds()
            quota_str = self._get_quota_str(deduction, uid)
            info = f"\n✅ 生成成功 ({elapsed:.2f}s) | 预设: {preset_name}"
            if extra_rules:
                info += f" | 规则: {extra_rules[:15]}..."
            info += f" | 剩余: {quota_str}"
            yield event.chain_result([Image.fromBytes(res), Plain(info)])
        else:
            yield event.chain_result([Plain(f"❌ {res}")])

    # 辅助方法
    def _get_help_node(self, event):
        txt = self.conf.get("help_text", "帮助文档未配置")
        bot_id = self._get_bot_id(event) or "2854196310"
        return event.chain_result([Nodes(nodes=[Node(name="手办化助手", uin=bot_id, content=[Plain(txt)])])])

    # 省略 Admin指令，它们和上一版完全一致，请确保不要覆盖掉下面的代码（lm列表, lm添加, 增加次数等）

    @filter.command("lm列表", aliases={"lmlist"}, prefix_optional=True)
    async def on_preset_list(self, event: AstrMessageEvent, ctx=None):
        presets = []
        for k, v in self.data_mgr.prompt_map.items():
            presets.append((k, v == "[内置预设]"))
        presets.sort(key=lambda x: x[0])
        if not presets: yield event.chain_result([Plain("暂无预设")]); return
        img_data = await self.img_mgr.create_preset_table(presets, self.data_mgr)
        yield event.chain_result([Image.fromBytes(img_data)])

    @filter.command("lm添加", aliases={"lma"}, prefix_optional=True)
    async def on_add_preset(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        msg = event.message_str.replace("lm添加", "").replace("lma", "").strip()
        if ":" not in msg: yield event.chain_result([Plain("格式: 词:提示词")]); return
        
        k, v = msg.split(":", 1)
        k, v = k.strip(), v.strip()
        
        if not k or not v:
            yield event.chain_result([Plain("❌ 触发词和提示词都不能为空")])
            return
        
        # 使用 DataManager 进行持久化保存
        await self.data_mgr.add_user_prompt(k, v)
        
        # 同时更新到配置文件的 prompt_list 中，确保双重持久化
        prompt_list = self.conf.get("prompt_list", [])
        # 移除已存在的同名预设
        prompt_list = [item for item in prompt_list if not item.startswith(f"{k}:")]
        # 添加新预设
        prompt_list.append(f"{k}:{v}")
        self.conf["prompt_list"] = prompt_list
        self._save_config()
        
        yield event.chain_result([Plain(f"✅ 已添加预设: {k}\n💾 已同步保存到配置文件")])

    @filter.command("lm查看", aliases={"lmv", "lm预览"}, prefix_optional=True)
    async def on_view_preset(self, event: AstrMessageEvent, ctx=None):
        parts = event.message_str.split()
        if len(parts) < 2: yield event.chain_result([Plain("用法: #lm查看 <关键词>")]); return
        kw = parts[1].strip()
        prompt = self.data_mgr.get_prompt(kw)
        msg = f"🔍 [{kw}]:\n{prompt}" if prompt else f"❌ 未找到 [{kw}]"
        yield event.chain_result([Plain(msg)])

    @filter.command("手办化签到", prefix_optional=True)
    async def on_checkin(self, event: AstrMessageEvent, ctx=None):
        if not self.conf.get("enable_checkin", False): yield event.chain_result([Plain("未开启签到")]); return
        uid = norm_id(event.get_sender_id())
        msg = await self.data_mgr.process_checkin(uid)
        yield event.chain_result([Plain(msg)])

    @filter.command("手办化查询次数", prefix_optional=True)
    async def on_query_count(self, event: AstrMessageEvent, ctx=None):
        uid = norm_id(event.get_sender_id())
        if self.is_admin(event):
            for seg in event.message_obj.message:
                if isinstance(seg, At): uid = str(seg.qq); break
        u_cnt = self.data_mgr.get_user_count(uid)
        msg = f"👤 用户 {uid} 剩余: {u_cnt}"
        if gid := event.get_group_id():
            msg += f"\n👥 本群剩余: {self.data_mgr.get_group_count(norm_id(gid))}"
        yield event.chain_result([Plain(msg)])

    @filter.command("切换API模式", prefix_optional=True)
    async def on_switch_mode(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        mode = event.message_str.split()[-1]
        if mode in ["generic", "gemini_official"]:
            self.conf["api_mode"] = mode;
            self._save_config()
            yield event.chain_result([Plain(f"✅ 已切换为 {mode}")])
        else:
            yield event.chain_result([Plain("模式无效 (generic / gemini_official)")])

    @filter.command("切换模型", prefix_optional=True)
    async def on_switch_model(self, event: AstrMessageEvent, ctx=None):
        all_m = [m if isinstance(m, str) else m["id"] for m in self.conf.get("model_list", [])]
        parts = event.message_str.split()
        if len(parts) == 1:
            curr = self.conf.get("model", "nano-banana")
            msg = "📋 可用模型:\n" + "\n".join([f"{i + 1}. {m} {'✅' if m == curr else ''}" for i, m in enumerate(all_m)])
            yield event.chain_result([Plain(msg)]);
            return

        if not self.is_admin(event): return
        if not parts[1].isdigit(): yield event.chain_result([Plain("请输入序号")]); return
        idx = int(parts[1]) - 1
        if 0 <= idx < len(all_m):
            self.conf["model"] = all_m[idx];
            self._save_config()
            yield event.chain_result([Plain(f"✅ 切换为: {all_m[idx]}")])

    @filter.command("手办化今日统计", prefix_optional=True)
    async def on_daily_stats(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        stats = self.data_mgr.daily_stats
        today = datetime.now().strftime("%Y-%m-%d")
        if stats.get("date") != today: yield event.chain_result([Plain(f"📊 {today} 无数据")]); return

        u_top = sorted(stats["users"].items(), key=lambda x: x[1], reverse=True)[:10]
        g_top = sorted(stats["groups"].items(), key=lambda x: x[1], reverse=True)[:10]
        msg = f"📊 {today} 统计:\n👥 群排行:\n" + ("\n".join([f"{k}: {v}" for k, v in g_top]) or "无")
        msg += "\n\n👤 用户排行:\n" + ("\n".join([f"{k}: {v}" for k, v in u_top]) or "无")
        yield event.chain_result([Plain(msg)])

    @filter.command("手办化增加用户次数", prefix_optional=True)
    async def on_add_user_counts(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        target = None
        for seg in event.message_obj.message:
            if isinstance(seg, At): target = str(seg.qq); break

        parts = event.message_str.split()
        count = 0
        if target:
            for p in parts:
                if p.isdigit(): count = int(p)
        else:
            if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                target = parts[1];
                count = int(parts[2])

        if target and count:
            await self.data_mgr.add_user_count(target, count)
            yield event.chain_result([Plain(f"✅ 用户 {target} +{count}")])

    @filter.command("手办化增加群组次数", prefix_optional=True)
    async def on_add_group_counts(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        parts = event.message_str.split()
        if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
            await self.data_mgr.add_group_count(parts[1], int(parts[2]))
            yield event.chain_result([Plain(f"✅ 群 {parts[1]} +{parts[2]}")])

    @filter.command("手办化添加key", prefix_optional=True)
    async def on_add_key(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        parts = event.message_str.split()
        if len(parts) < 2: return

        is_power = parts[1].lower() in ["p", "power", "强力"]
        keys = parts[2:] if is_power else parts[1:]

        mode = self.conf.get("api_mode", "generic")
        field = f"{'power_' if is_power else ''}{mode if mode == 'generic' else 'gemini'}_api_keys"
        if mode == "gemini_official":
            field = f"{'power_' if is_power else ''}gemini_api_keys"
        else:
            field = f"{'power_' if is_power else ''}generic_api_keys"

        curr_keys = self.conf.get(field, [])
        curr_keys.extend(keys)
        self.conf[field] = curr_keys;
        self._save_config()
        yield event.chain_result([Plain(f"✅ 已向 {field} 添加 {len(keys)} 个 Key")])

    @filter.command("手办化key列表", prefix_optional=True)
    async def on_list_keys(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        mode = self.conf.get("api_mode", "generic")
        base = "gemini" if mode == "gemini_official" else "generic"

        nk = self.conf.get(f"{base}_api_keys", [])
        pk = self.conf.get(f"power_{base}_api_keys", [])

        msg = f"🔑 模式: {mode}\n📌 普通池 ({len(nk)}):\n" + "\n".join([f"{k[:8]}..." for k in nk])
        msg += f"\n\n⚡ 强力池 ({len(pk)}):\n" + "\n".join([f"{k[:8]}..." for k in pk])
        yield event.chain_result([Plain(msg)])

    @filter.command("手办化删除key", prefix_optional=True)
    async def on_delete_key(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        parts = event.message_str.split()
        if len(parts) < 2: yield event.chain_result([Plain("用法: #删除key [p] <all/序号>")]); return

        is_power = parts[1].lower() in ["p", "power"]
        idx_str = parts[2] if is_power else parts[1]

        mode = self.conf.get("api_mode", "generic")
        base = "gemini" if mode == "gemini_official" else "generic"
        field = f"{'power_' if is_power else ''}{base}_api_keys"

        if idx_str == "all":
            self.conf[field] = [];
            self._save_config()
            yield event.chain_result([Plain("✅ 已清空")])
        elif idx_str.isdigit():
            keys = self.conf.get(field, [])
            idx = int(idx_str) - 1
            if 0 <= idx < len(keys):
                keys.pop(idx);
                self.conf[field] = keys;
                self._save_config()
                yield event.chain_result([Plain("✅ 已删除")])

    @filter.command("预设图片清理", prefix_optional=True)
    async def on_cleanup_presets(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        parts = event.message_str.split()
        days = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 30
        count = await self.data_mgr.cleanup_old_presets(days)
        yield event.chain_result([Plain(f"✅ 清理了 {count} 张超过 {days} 天的图片")])

    @filter.command("预设图片统计", prefix_optional=True)
    async def on_preset_stats(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        cnt, size = self.data_mgr.get_preset_stats()
        yield event.chain_result([Plain(f"📊 缓存统计:\n数量: {cnt} 张\n占用: {size:.2f} MB")])

    @filter.command("手办化帮助", aliases={"lmh", "lm帮助"}, prefix_optional=True)
    async def on_help(self, event: AstrMessageEvent, ctx=None):
        yield self._get_help_node(event)

    # ================= 上下文记录与 LLM 智能判断 =================

    def _extract_message_info(self, event: AstrMessageEvent) -> Dict[str, Any]:
        """从事件中提取消息信息"""
        has_image = False
        image_urls = []
        content_parts = []
        
        for seg in event.message_obj.message:
            if isinstance(seg, Image):
                has_image = True
                if seg.url:
                    image_urls.append(seg.url)
                elif seg.file:
                    image_urls.append(seg.file)
                content_parts.append("[图片]")
            elif isinstance(seg, Plain) and seg.text:
                content_parts.append(seg.text)
            elif isinstance(seg, Reply):
                # 检查回复中是否有图片
                if seg.chain:
                    for s_chain in seg.chain:
                        if isinstance(s_chain, Image):
                            has_image = True
                            if s_chain.url:
                                image_urls.append(s_chain.url)
        
        return {
            "content": "".join(content_parts) or event.message_str,
            "has_image": has_image,
            "image_urls": image_urls
        }

    @filter.event_message_type(filter.EventMessageType.ALL, priority=100)
    async def on_message_record(self, event: AstrMessageEvent, ctx=None):
        """记录所有消息到上下文管理器（高优先级，不阻断）"""
        try:
            session_id = event.unified_msg_origin
            msg_id = str(event.message_obj.message_id)
            sender_id = event.get_sender_id()
            sender_name = event.get_sender_name() or sender_id
            
            # 检查是否是 Bot 自己的消息
            bot_id = self._get_bot_id(event)
            is_bot = (sender_id == bot_id) if bot_id else False
            
            # 提取消息信息
            msg_info = self._extract_message_info(event)
            
            # 记录到上下文管理器
            await self.ctx_mgr.add_message(
                session_id=session_id,
                msg_id=msg_id,
                sender_id=sender_id,
                sender_name=sender_name,
                content=msg_info["content"],
                is_bot=is_bot,
                has_image=msg_info["has_image"],
                image_urls=msg_info["image_urls"]
            )
            
        except Exception as e:
            logger.debug(f"FigurinePro: 消息记录失败: {e}")
        
        # 不阻断事件传递
        return

    @filter.llm_tool(name="shoubanhua_smart_generate")
    async def smart_generate_tool(self, event: AstrMessageEvent, user_request: str = ""):
        '''智能判断并生成图片。仅在用户明确表达需要生成图片时才调用。
        
        调用前请严格判断：
        1. 用户是否明确要求生成/画/创作/处理图片？
        2. 如果用户只是闲聊、询问问题、分享图片但没有要求处理，请不要调用此工具
        3. 如果用户的意图不明确，请先询问用户是否需要生成图片
        
        此工具会消耗用户的使用次数，请谨慎调用。
        
        Args:
            user_request(string): 用户的请求描述（可选，如果为空则使用当前消息）
        '''
        # 0. 检查 LLM 工具开关
        if not self.conf.get("enable_llm_tool", True):
            return "❌ LLM 工具已禁用，请使用指令模式调用此功能。"
        
        # 0.1 检查图片生成冷却时间
        uid = norm_id(event.get_sender_id())
        in_cooldown, remaining = self._check_image_cooldown(uid)
        if in_cooldown:
            # 返回借口让LLM用自然语言拒绝
            excuse = self._get_cooldown_excuse(remaining)
            return f"【冷却中】{excuse}\n\n请用自然的方式告诉用户现在不方便生成图片，可以稍后再试。不要直接说'冷却'这个词。"
        
        # 1. 获取上下文
        session_id = event.unified_msg_origin
        context_messages = await self.ctx_mgr.get_recent_messages(
            session_id, 
            count=self._context_rounds
        )
        
        # 2. 提取当前消息信息
        msg_info = self._extract_message_info(event)
        current_message = user_request or msg_info["content"]
        has_current_image = msg_info["has_image"]
        
        # 3. 分析任务类型
        analysis = LLMTaskAnalyzer.analyze_task_type(
            current_message=current_message,
            context_messages=context_messages,
            has_current_image=has_current_image
        )
        
        task_type = analysis["task_type"]
        confidence = analysis["confidence"]
        reason = analysis["reason"]
        
        logger.info(f"FigurinePro 智能判断: {task_type} (置信度: {confidence:.2f}) - {reason}")
        
        # 4. 置信度检查
        if confidence < self._auto_detect_confidence:
            return f"无法确定任务类型 (置信度: {confidence:.2f})。请明确指定是文生图还是图生图。\n分析: {reason}"
        
        # 5. 根据任务类型执行
        if task_type == "none":
            return f"根据分析，当前请求不需要生成图片。\n分析: {reason}"
        
        elif task_type == "text_to_image":
            # 文生图
            prompt = analysis.get("suggested_prompt", current_message)
            final_prompt, preset_name, extra_rules = self._process_prompt_and_preset(prompt)
            
            # 根据配置决定是否发送进度提示
            if self.conf.get("llm_show_progress", True):
                feedback = f"🎨 智能识别: 文生图 (置信度: {confidence:.0%})\n正在生成 [{preset_name}]"
                if extra_rules:
                    feedback += f"\n📝 规则: {extra_rules[:20]}..."
                feedback += "，请稍候..."
                await event.send(event.chain_result([Plain(feedback)]))
            
            gid = norm_id(event.get_group_id())
            deduction = await self._check_quota(event, uid, gid, 1)
            if not deduction["allowed"]:
                return deduction["msg"]
            
            # 更新冷却时间
            self._update_image_cooldown(uid)
            
            asyncio.create_task(
                self._run_background_task(event, [], final_prompt, preset_name, deduction, uid, gid, 1, extra_rules)
            )
            
            return f"[TOOL_SUCCESS] 文生图任务已启动，预设：{preset_name}。图片将在后台生成并自动发送给用户。【重要】你不需要再回复任何内容，保持沉默即可，用户会直接收到图片。"
        
        elif task_type == "image_to_image":
            # 图生图
            prompt = analysis.get("suggested_prompt", current_message)
            processed_prompt, preset_name, extra_rules = self._process_prompt_and_preset(prompt)
            
            # 根据配置决定是否发送进度提示
            if self.conf.get("llm_show_progress", True):
                feedback = f"🎨 智能识别: 图生图 (置信度: {confidence:.0%})\n正在提取图片并生成 [{preset_name}]"
                if extra_rules:
                    feedback += f"\n📝 规则: {extra_rules[:20]}..."
                feedback += "，请稍候..."
                await event.send(event.chain_result([Plain(feedback)]))
            
            # 提取图片
            bot_id = self._get_bot_id(event)
            images = await self.img_mgr.extract_images_from_event(event, ignore_id=bot_id, context=self.context)
            
            # 如果当前消息没有图片，尝试从上下文获取
            if not images and context_messages:
                last_img_msg = self.ctx_mgr.get_last_image_message(context_messages)
                if last_img_msg and last_img_msg.image_urls:
                    for url in last_img_msg.image_urls:
                        img_bytes = await self.img_mgr.load_bytes(url)
                        if img_bytes:
                            images.append(img_bytes)
            
            if not images:
                await event.send(event.chain_result([Plain("❌ 未检测到图片，请发送或引用图片。")]))
                return "失败：未检测到图片。"
            
            gid = norm_id(event.get_group_id())
            deduction = await self._check_quota(event, uid, gid, 1)
            if not deduction["allowed"]:
                return deduction["msg"]
            
            # 更新冷却时间
            self._update_image_cooldown(uid)
            
            asyncio.create_task(
                self._run_background_task(event, images, processed_prompt, preset_name, deduction, uid, gid, 1, extra_rules)
            )
            
            return f"[TOOL_SUCCESS] 图生图任务已启动，预设：{preset_name}。图片将在后台生成并自动发送给用户。【重要】你不需要再回复任何内容，保持沉默即可，用户会直接收到图片。"
        
        return "未知任务类型"

    @filter.command("上下文状态", prefix_optional=True)
    async def on_context_status(self, event: AstrMessageEvent, ctx=None):
        """查看上下文状态（管理员）"""
        if not self.is_admin(event): return
        
        session_id = event.unified_msg_origin
        messages = await self.ctx_mgr.get_recent_messages(session_id, count=10)
        
        msg = f"📊 上下文状态:\n"
        msg += f"会话数: {self.ctx_mgr.get_session_count()}\n"
        msg += f"当前会话消息数: {len(messages)}\n"
        msg += f"LLM智能判断: {'已启用' if self._llm_auto_detect else '未启用'}\n"
        msg += f"上下文轮数: {self._context_rounds}\n"
        msg += f"置信度阈值: {self._auto_detect_confidence}\n"
        
        if messages:
            msg += f"\n最近 {len(messages)} 条消息:\n"
            for m in messages[-5:]:
                sender = "[Bot]" if m.is_bot else m.sender_name
                img_tag = " 📷" if m.has_image else ""
                content_preview = m.content[:30] + "..." if len(m.content) > 30 else m.content
                msg += f"  {sender}: {content_preview}{img_tag}\n"
        
        yield event.chain_result([Plain(msg)])

    @filter.command("清除上下文", prefix_optional=True)
    async def on_clear_context(self, event: AstrMessageEvent, ctx=None):
        """清除当前会话的上下文（管理员）"""
        if not self.is_admin(event): return
        
        session_id = event.unified_msg_origin
        count = await self.ctx_mgr.clear_session(session_id)
        
        yield event.chain_result([Plain(f"✅ 已清除 {count} 条上下文记录")])

    @filter.command("测试智能判断", prefix_optional=True)
    async def on_test_smart_detect(self, event: AstrMessageEvent, ctx=None):
        """测试智能判断功能（不实际生成）"""
        if not self.is_admin(event): return
        
        # 获取测试文本
        parts = event.message_str.split(maxsplit=1)
        test_text = parts[1] if len(parts) > 1 else event.message_str
        
        # 获取上下文
        session_id = event.unified_msg_origin
        context_messages = await self.ctx_mgr.get_recent_messages(session_id, count=self._context_rounds)
        
        # 提取当前消息信息
        msg_info = self._extract_message_info(event)
        
        # 分析
        analysis = LLMTaskAnalyzer.analyze_task_type(
            current_message=test_text,
            context_messages=context_messages,
            has_current_image=msg_info["has_image"]
        )
        
        msg = f"🔍 智能判断测试结果:\n"
        msg += f"任务类型: {analysis['task_type']}\n"
        msg += f"置信度: {analysis['confidence']:.0%}\n"
        msg += f"判断理由: {analysis['reason']}\n"
        msg += f"建议提示词: {analysis.get('suggested_prompt', '无')[:50]}...\n"
        msg += f"\n当前消息有图片: {'是' if msg_info['has_image'] else '否'}\n"
        msg += f"上下文消息数: {len(context_messages)}\n"
        
        if context_messages:
            has_ctx_img = self.ctx_mgr.has_recent_images(context_messages)
            msg += f"上下文有图片: {'是' if has_ctx_img else '否'}"
        
        yield event.chain_result([Plain(msg)])

    # ================= 预设参考图管理 =================

    @filter.command("预设参考图添加", aliases={"lmref添加", "添加参考图"}, prefix_optional=True)
    async def on_add_preset_ref(self, event: AstrMessageEvent, ctx=None):
        """为预设添加参考图（管理员）
        
        用法: #预设参考图添加 <预设名> [图片]
        """
        if not self.is_admin(event): return
        
        # 解析预设名
        parts = event.message_str.split()
        if len(parts) < 2:
            yield event.chain_result([Plain("用法: #预设参考图添加 <预设名> [图片]\n请同时发送或引用图片")])
            return
        
        preset_name = parts[1].strip()
        
        # 检查预设是否存在
        if preset_name not in self.data_mgr.prompt_map:
            yield event.chain_result([Plain(f"❌ 预设 [{preset_name}] 不存在\n请先使用 #lm添加 创建预设")])
            return
        
        # 提取图片
        bot_id = self._get_bot_id(event)
        images = await self.img_mgr.extract_images_from_event(event, ignore_id=bot_id, context=self.context)
        
        if not images:
            yield event.chain_result([Plain("❌ 未检测到图片，请发送或引用图片")])
            return
        
        # 保存参考图
        count = await self.data_mgr.add_preset_ref_images(preset_name, images)
        
        if count > 0:
            total = len(self.data_mgr.get_preset_ref_image_paths(preset_name))
            yield event.chain_result([Plain(f"✅ 已为预设 [{preset_name}] 添加 {count} 张参考图\n当前共 {total} 张参考图")])
        else:
            yield event.chain_result([Plain("❌ 保存参考图失败")])

    @filter.command("预设参考图查看", aliases={"lmref查看", "查看参考图"}, prefix_optional=True)
    async def on_view_preset_ref(self, event: AstrMessageEvent, ctx=None):
        """查看预设的参考图（管理员）
        
        用法: #预设参考图查看 <预设名>
        """
        if not self.is_admin(event): return
        
        parts = event.message_str.split()
        if len(parts) < 2:
            yield event.chain_result([Plain("用法: #预设参考图查看 <预设名>")])
            return
        
        preset_name = parts[1].strip()
        
        if not self.data_mgr.has_preset_ref_images(preset_name):
            yield event.chain_result([Plain(f"预设 [{preset_name}] 没有参考图")])
            return
        
        # 加载参考图
        ref_images = await self.data_mgr.load_preset_ref_images_bytes(preset_name)
        
        if not ref_images:
            yield event.chain_result([Plain(f"预设 [{preset_name}] 的参考图加载失败")])
            return
        
        # 发送参考图
        result = [Plain(f"📷 预设 [{preset_name}] 的参考图 ({len(ref_images)} 张):\n")]
        for i, img_bytes in enumerate(ref_images[:5]):  # 最多显示5张
            result.append(Image.fromBytes(img_bytes))
        
        if len(ref_images) > 5:
            result.append(Plain(f"\n... 还有 {len(ref_images) - 5} 张未显示"))
        
        yield event.chain_result(result)

    @filter.command("预设参考图清除", aliases={"lmref清除", "清除参考图"}, prefix_optional=True)
    async def on_clear_preset_ref(self, event: AstrMessageEvent, ctx=None):
        """清除预设的所有参考图（管理员）
        
        用法: #预设参考图清除 <预设名>
        """
        if not self.is_admin(event): return
        
        parts = event.message_str.split()
        if len(parts) < 2:
            yield event.chain_result([Plain("用法: #预设参考图清除 <预设名>")])
            return
        
        preset_name = parts[1].strip()
        
        count = await self.data_mgr.clear_preset_ref_images(preset_name)
        
        if count > 0:
            yield event.chain_result([Plain(f"✅ 已清除预设 [{preset_name}] 的 {count} 张参考图")])
        else:
            yield event.chain_result([Plain(f"预设 [{preset_name}] 没有参考图")])

    @filter.command("预设参考图删除", aliases={"lmref删除", "删除参考图"}, prefix_optional=True)
    async def on_remove_preset_ref(self, event: AstrMessageEvent, ctx=None):
        """删除预设的指定参考图（管理员）
        
        用法: #预设参考图删除 <预设名> <序号>
        """
        if not self.is_admin(event): return
        
        parts = event.message_str.split()
        if len(parts) < 3:
            yield event.chain_result([Plain("用法: #预设参考图删除 <预设名> <序号>\n序号从1开始")])
            return
        
        preset_name = parts[1].strip()
        
        if not parts[2].isdigit():
            yield event.chain_result([Plain("❌ 序号必须是数字")])
            return
        
        index = int(parts[2]) - 1  # 转为0开始的索引
        
        success = await self.data_mgr.remove_preset_ref_image(preset_name, index)
        
        if success:
            remaining = len(self.data_mgr.get_preset_ref_image_paths(preset_name))
            yield event.chain_result([Plain(f"✅ 已删除预设 [{preset_name}] 的第 {index + 1} 张参考图\n剩余 {remaining} 张")])
        else:
            yield event.chain_result([Plain(f"❌ 删除失败，请检查预设名和序号是否正确")])

    @filter.command("预设参考图统计", aliases={"lmref统计", "参考图统计"}, prefix_optional=True)
    async def on_preset_ref_stats(self, event: AstrMessageEvent, ctx=None):
        """查看预设参考图统计（管理员）"""
        if not self.is_admin(event): return
        
        stats = self.data_mgr.get_preset_ref_stats()
        
        msg = f"📊 预设参考图统计:\n"
        msg += f"有参考图的预设: {stats['total_presets']} 个\n"
        msg += f"总图片数: {stats['total_images']} 张\n"
        msg += f"总占用: {stats['total_size_mb']:.2f} MB\n"
        
        if stats['details']:
            msg += f"\n详情:\n"
            for preset, count in sorted(stats['details'].items(), key=lambda x: -x[1])[:10]:
                msg += f"  {preset}: {count} 张\n"
            
            if len(stats['details']) > 10:
                msg += f"  ... 还有 {len(stats['details']) - 10} 个预设"
        
        yield event.chain_result([Plain(msg)])

    @filter.command("预设参考图列表", aliases={"lmref列表", "参考图列表"}, prefix_optional=True)
    async def on_list_preset_refs(self, event: AstrMessageEvent, ctx=None):
        """列出所有有参考图的预设（管理员）"""
        if not self.is_admin(event): return
        
        stats = self.data_mgr.get_preset_ref_stats()
        
        if not stats['details']:
            yield event.chain_result([Plain("暂无预设参考图")])
            return
        
        msg = f"📋 有参考图的预设列表:\n"
        for preset, count in sorted(stats['details'].items()):
            has_prompt = "✓" if preset in self.data_mgr.prompt_map else "✗"
            msg += f"  [{preset}] {count}张 (预设{has_prompt})\n"
        
        yield event.chain_result([Plain(msg)])

    # ================= 批量处理图片功能 =================

    async def _collect_images_from_context(self, session_id: str, count: int = 10) -> List[Tuple[str, List[str]]]:
        """
        从上下文中收集图片
        
        Args:
            session_id: 会话ID
            count: 获取的消息数量
            
        Returns:
            [(消息ID, [图片URL列表]), ...]
        """
        messages = await self.ctx_mgr.get_recent_messages(session_id, count=count)
        
        result = []
        for msg in messages:
            if msg.has_image and msg.image_urls and not msg.is_bot:
                result.append((msg.msg_id, msg.image_urls))
        
        return result

    def _translate_error_to_chinese(self, error: str) -> str:
        """将错误信息翻译为中文"""
        error_lower = str(error).lower()
        
        # 网络相关错误
        if "timeout" in error_lower or "timed out" in error_lower:
            return "请求超时，API响应时间过长"
        if "connection" in error_lower and ("refused" in error_lower or "reset" in error_lower):
            return "连接被拒绝或重置，网络不稳定"
        if "connection" in error_lower:
            return "网络连接异常"
        if "ssl" in error_lower or "certificate" in error_lower:
            return "SSL证书验证失败"
        if "dns" in error_lower or "resolve" in error_lower:
            return "DNS解析失败，无法访问服务器"
        
        # API相关错误
        if "rate limit" in error_lower or "429" in error_lower:
            return "API请求频率过高，触发限流"
        if "quota" in error_lower or "exceeded" in error_lower:
            return "API配额已用尽"
        if "unauthorized" in error_lower or "401" in error_lower:
            return "API密钥无效或已过期"
        if "forbidden" in error_lower or "403" in error_lower:
            return "API访问被禁止"
        if "not found" in error_lower or "404" in error_lower:
            return "API接口不存在"
        if "500" in error_lower or "internal server" in error_lower:
            return "API服务器内部错误"
        if "502" in error_lower or "bad gateway" in error_lower:
            return "API网关错误"
        if "503" in error_lower or "service unavailable" in error_lower:
            return "API服务暂时不可用"
        if "524" in error_lower:
            return "Cloudflare超时，请求时间过长"
        
        # 图片相关错误
        if "image" in error_lower and ("invalid" in error_lower or "corrupt" in error_lower):
            return "图片格式无效或已损坏"
        if "image" in error_lower and "size" in error_lower:
            return "图片尺寸不符合要求"
        if "download" in error_lower:
            return "图片下载失败"
        if "base64" in error_lower:
            return "图片编码失败"
        
        # 内容相关错误
        if "content" in error_lower and ("policy" in error_lower or "filter" in error_lower):
            return "内容被安全策略过滤"
        if "nsfw" in error_lower or "inappropriate" in error_lower:
            return "内容不符合安全规范"
        
        # JSON相关错误
        if "json" in error_lower:
            return "API返回数据格式异常"
        
        # 默认返回原始错误（截断）
        error_str = str(error)
        if len(error_str) > 50:
            return f"未知错误: {error_str[:50]}..."
        return f"未知错误: {error_str}"

    async def _run_single_batch_task(self, event: AstrMessageEvent, image_bytes: bytes,
                                     prompt: str, preset_name: str, task_index: int, total_tasks: int,
                                     uid: str, gid: str, extra_rules: str = "", 
                                     image_source: str = "", hide_text: bool = False) -> Tuple[bool, str]:
        """
        执行单个批量任务（不扣费，由调用方统一扣费）
        
        Returns:
            (是否成功, 错误信息)
        """
        try:
            # 加载预设参考图（如果有）
            images = [image_bytes]
            if preset_name != "自定义" and self.conf.get("enable_preset_ref_images", True):
                ref_images = await self._load_preset_ref_images(preset_name)
                if ref_images:
                    images = ref_images + images

            # 调用 API
            model = self.conf.get("model", "nano-banana")
            start_time = datetime.now()

            res = await self.api_mgr.call_api(images, prompt, model, False, self.img_mgr.proxy)

            # 处理结果
            if isinstance(res, bytes):
                elapsed = (datetime.now() - start_time).total_seconds()
                await self.data_mgr.record_usage(uid, gid)

                chain_nodes = [Image.fromBytes(res)]
                if not hide_text:
                    # 构建成功文案
                    info_text = f"\n✅ [{task_index}/{total_tasks}] 生成成功 ({elapsed:.2f}s) | 预设: {preset_name}"
                    if extra_rules:
                        info_text += f" | 规则: {extra_rules[:15]}..."
                    chain_nodes.append(Plain(info_text))

                # 发送结果
                chain = event.chain_result(chain_nodes)
                await event.send(chain)
                return True, ""
            else:
                # API返回错误
                error_msg = self._translate_error_to_chinese(res)
                logger.error(f"Batch task {task_index} API error: {res}")
                return False, error_msg

        except Exception as e:
            # 系统异常
            error_msg = self._translate_error_to_chinese(str(e))
            logger.error(f"Batch task {task_index} exception: {e}", exc_info=True)
            return False, error_msg

    @filter.llm_tool(name="shoubanhua_batch_process")
    async def batch_process_tool(self, event: AstrMessageEvent, prompt: str, max_images: int = 10):
        '''批量处理上下文中的多张图片。仅在用户明确要求批量处理多张图片时才调用。
        
        调用前请严格判断：
        1. 用户是否明确要求批量处理/全部处理多张图片？
        2. 上下文中是否确实有多张图片需要处理？
        3. 如果用户只是发送了多张图片但没有要求处理，请不要调用此工具
        
        此工具会消耗用户大量使用次数（每张图片消耗1次），请谨慎调用。
        
        Args:
            prompt(string): 图片处理的提示词，可以是预设名+追加规则，如"手办化 皮肤白一点"
            max_images(int): 最多处理的图片数量，默认10张
        '''
        # 0. 读取配置中的限制
        conf_max_images = self.conf.get("batch_max_images", 10)
        max_images = min(max_images, conf_max_images) if max_images > 0 else conf_max_images

        # 0. 检查 LLM 工具开关
        if not self.conf.get("enable_llm_tool", True):
            return "❌ LLM 工具已禁用，请使用指令模式调用此功能。"
        
        # 0.1 检查图片生成冷却时间
        uid = norm_id(event.get_sender_id())
        in_cooldown, remaining = self._check_image_cooldown(uid)
        if in_cooldown:
            # 返回借口让LLM用自然语言拒绝
            excuse = self._get_cooldown_excuse(remaining)
            return f"【冷却中】{excuse}\n\n请用自然的方式告诉用户现在不方便处理图片，可以稍后再试。不要直接说'冷却'这个词。"
        
        # 1. 提取当前消息的图片 URL (可能还没存入上下文)
        current_urls = []
        for seg in event.message_obj.message:
            if hasattr(seg, 'url') and seg.url:
                current_urls.append(seg.url)
            elif hasattr(seg, 'file') and seg.file:
                current_urls.append(seg.file)
                
        # 2. 获取上下文中的图片
        session_id = event.unified_msg_origin
        image_sources = await self._collect_images_from_context(session_id, count=self._context_rounds)
        
        if not image_sources and not current_urls:
            return "❌ 未在上下文中找到图片。请先发送图片，然后再使用批量处理功能。"
        
        # 3. 收集所有图片URL（去重，优先获取最新图片）
        all_image_urls = []
        seen_urls = set()
        
        # 先添加当前消息的图片
        for url in current_urls:
            if url not in seen_urls:
                all_image_urls.append(url)
                seen_urls.add(url)
                
        # 倒序遍历上下文（从最新消息开始）
        for msg_id, urls in reversed(image_sources):
            # 一条消息内的图片按正常顺序遍历
            for url in urls:
                if url not in seen_urls:
                    all_image_urls.append(url)
                    seen_urls.add(url)
        
        # 限制数量（保留最新的 max_images 张）
        if max_images > 0:
            all_image_urls = all_image_urls[:max_images]
            
        # 提取完毕后将列表反转，以符合用户的正常阅读和发送顺序（从旧到新处理）
        all_image_urls.reverse()
        
        total_images = len(all_image_urls)
        if total_images == 0:
            return "❌ 未找到有效的图片URL。"
        
        # 3. 计算预设和追加规则
        final_prompt, preset_name, extra_rules = self._process_prompt_and_preset(prompt)
        
        # 4. 检查配额（批量任务需要足够的次数）
        gid = norm_id(event.get_group_id())
        total_cost = total_images
        
        deduction = await self._check_quota(event, uid, gid, total_cost)
        if not deduction["allowed"]:
            return f"❌ 次数不足。批量处理 {total_images} 张图片需要 {total_cost} 次。{deduction['msg']}"
        
        # 4.1 更新冷却时间
        self._update_image_cooldown(uid)
        
        # 根据配置决定是否隐藏进度提示（白名单用户和普通用户使用同一开关）
        hide_llm_progress = not self.conf.get("llm_show_progress", True)

        # 5. 发送开始提示
        if not hide_llm_progress:
            preset_display = "自定义" if preset_name in ["自定义", "编辑"] else preset_name
            feedback = f"📦 批量处理任务开始\n"
            feedback += f"📷 共 {total_images} 张图片\n"
            feedback += f"🎨 预设: {preset_display}"
            feedback += f"\n⏳ 每张图片将独立处理，请耐心等待..."
            await event.send(event.chain_result([Plain(feedback)]))
        
        # 6. 扣费
        if deduction["source"] == "user":
            await self.data_mgr.decrease_user_count(uid, total_cost)
        elif deduction["source"] == "group":
            await self.data_mgr.decrease_group_count(gid, total_cost)
        
        # 7. 启动批量处理任务
        async def process_all():
            success_count = 0
            fail_count = 0
            failed_details = []  # 记录失败详情
            max_retries = self.conf.get("batch_retries", 2)
            
            for i, url in enumerate(all_image_urls, 1):
                try:
                    # 下载图片
                    img_bytes = await self.img_mgr.load_bytes(url)
                    if not img_bytes:
                        error_msg = "图片下载失败，可能是链接已过期或网络问题"
                        logger.error(f"Batch process image {i} download failed: {url}")
                        failed_details.append({
                            "index": i,
                            "reason": error_msg,
                            "url_preview": url[:50] + "..." if len(url) > 50 else url
                        })
                        fail_count += 1
                        # 发送单条失败提示
                        await event.send(event.chain_result([
                            Plain(f"❌ 第 {i}/{total_images} 张图片处理失败\n📍 原因: {error_msg}")
                        ]))
                        continue
                    
                    # 处理单张图片（带重试机制）
                    retry_count = 0
                    success = False
                    error_msg = ""
                    
                    while retry_count <= max_retries:
                        success, error_msg = await self._run_single_batch_task(
                            event=event,
                            image_bytes=img_bytes,
                            prompt=final_prompt,
                            preset_name=preset_name,
                            task_index=i,
                            total_tasks=total_images,
                            uid=uid,
                            gid=gid,
                            extra_rules=extra_rules,
                            image_source=url,
                            hide_text=hide_llm_progress
                        )
                        
                        if success:
                            break
                            
                        retry_count += 1
                        if retry_count <= max_retries:
                            await event.send(event.chain_result([
                                Plain(f"⚠️ 第 {i}/{total_images} 张图片生成失败 ({error_msg})\n⏳ 正在进行第 {retry_count} 次重试...")
                            ]))
                            await asyncio.sleep(1.5)
                    
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                        failed_details.append({
                            "index": i,
                            "reason": error_msg,
                            "url_preview": url[:50] + "..." if len(url) > 50 else url
                        })
                        # 发送单条失败提示
                        await event.send(event.chain_result([
                            Plain(f"❌ 第 {i}/{total_images} 张图片最终处理失败\n📍 原因: {error_msg}")
                        ]))
                    
                    # 添加短暂延迟，避免API限流
                    if i < total_images:
                        await asyncio.sleep(0.5)
                        
                except Exception as e:
                    error_msg = self._translate_error_to_chinese(str(e))
                    logger.error(f"Batch process image {i} exception: {e}", exc_info=True)
                    failed_details.append({
                        "index": i,
                        "reason": error_msg,
                        "url_preview": url[:50] + "..." if len(url) > 50 else url
                    })
                    fail_count += 1
                    await event.send(event.chain_result([
                        Plain(f"❌ 第 {i}/{total_images} 张图片处理异常\n📍 原因: {error_msg}")
                    ]))
            
            if not hide_llm_progress:
                # 发送完成汇总
                quota_str = self._get_quota_str(deduction, uid)
                summary = f"\n📊 批量处理完成\n"
                summary += f"✅ 成功: {success_count} 张\n"
                summary += f"❌ 失败: {fail_count} 张\n"
                summary += f"💰 剩余次数: {quota_str}"
                
                # 如果有失败的，附加失败汇总
                if failed_details:
                    summary += f"\n\n📋 失败图片汇总:"
                    for detail in failed_details[:5]:  # 最多显示5条
                        summary += f"\n  • 第{detail['index']}张: {detail['reason']}"
                    if len(failed_details) > 5:
                        summary += f"\n  ... 还有 {len(failed_details) - 5} 张失败"
                
                await event.send(event.chain_result([Plain(summary)]))
        
        # 启动异步任务
        asyncio.create_task(process_all())
        
        return f"批量处理任务已启动，共 {total_images} 张图片，预设：{preset_name}。每张图片将独立处理并发送结果。"

    @filter.llm_tool(name="shoubanhua_batch_process_concurrent")
    async def batch_process_concurrent_tool(self, event: AstrMessageEvent, prompt: str, max_images: int = 10, concurrency: int = 3):
        '''并发批量处理上下文中的多张图片。仅在用户明确要求快速批量处理时才调用。
        
        调用前请严格判断：
        1. 用户是否明确要求批量处理/全部处理多张图片？
        2. 上下文中是否确实有多张图片需要处理？
        3. 如果用户只是发送了多张图片但没有要求处理，请不要调用此工具
        
        此工具会消耗用户大量使用次数（每张图片消耗1次），请谨慎调用。
        
        Args:
            prompt(string): 图片处理的提示词，可以是预设名+追加规则
            max_images(int): 最多处理的图片数量，默认10张
            concurrency(int): 并发数量，默认3（同时处理3张图片）
        '''
        # 读取配置中的限制，强制覆盖 LLM 参数
        conf_max_images = self.conf.get("batch_max_images", 10)
        conf_concurrency = self.conf.get("batch_concurrency", 3)
        max_images = min(max_images, conf_max_images) if max_images > 0 else conf_max_images
        concurrency = max(1, conf_concurrency)

        # 0. 检查 LLM 工具开关
        if not self.conf.get("enable_llm_tool", True):
            return "❌ LLM 工具已禁用，请使用指令模式调用此功能。"
        
        # 0.1 检查图片生成冷却时间
        uid = norm_id(event.get_sender_id())
        in_cooldown, remaining = self._check_image_cooldown(uid)
        if in_cooldown:
            # 返回借口让LLM用自然语言拒绝
            excuse = self._get_cooldown_excuse(remaining)
            return f"【冷却中】{excuse}\n\n请用自然的方式告诉用户现在不方便处理图片，可以稍后再试。不要直接说'冷却'这个词。"
        
        # 1. 提取当前消息的图片 URL (可能还没存入上下文)
        current_urls = []
        for seg in event.message_obj.message:
            if hasattr(seg, 'url') and seg.url:
                current_urls.append(seg.url)
            elif hasattr(seg, 'file') and seg.file:
                current_urls.append(seg.file)
                
        # 2. 获取上下文中的图片
        session_id = event.unified_msg_origin
        image_sources = await self._collect_images_from_context(session_id, count=self._context_rounds)
        
        if not image_sources and not current_urls:
            return "❌ 未在上下文中找到图片。请先发送图片，然后再使用批量处理功能。"
        
        # 3. 收集所有图片URL（去重，优先获取最新图片）
        all_image_urls = []
        seen_urls = set()
        
        # 先添加当前消息的图片
        for url in current_urls:
            if url not in seen_urls:
                all_image_urls.append(url)
                seen_urls.add(url)
                
        # 倒序遍历上下文（从最新消息开始）
        for msg_id, urls in reversed(image_sources):
            for url in urls:
                if url not in seen_urls:
                    all_image_urls.append(url)
                    seen_urls.add(url)
        
        # 限制数量（保留最新的 max_images 张）
        if max_images > 0:
            all_image_urls = all_image_urls[:max_images]
            
        # 反转列表以正常顺序处理
        all_image_urls.reverse()
        
        total_images = len(all_image_urls)
        if total_images == 0:
            return "❌ 未找到有效的图片URL。"
        
        # 3. 计算预设和追加规则
        final_prompt, preset_name, extra_rules = self._process_prompt_and_preset(prompt)
        
        # 4. 检查配额
        gid = norm_id(event.get_group_id())
        total_cost = total_images
        
        deduction = await self._check_quota(event, uid, gid, total_cost)
        if not deduction["allowed"]:
            return f"❌ 次数不足。批量处理 {total_images} 张图片需要 {total_cost} 次。{deduction['msg']}"
        
        # 4.1 更新冷却时间
        self._update_image_cooldown(uid)
        
        # 根据配置决定是否隐藏进度提示（白名单用户和普通用户使用同一开关）
        hide_llm_progress = not self.conf.get("llm_show_progress", True)

        # 5. 发送开始提示
        if not hide_llm_progress:
            preset_display = "自定义" if preset_name in ["自定义", "编辑"] else preset_name
            feedback = f"🚀 并发批量处理任务开始\n"
            feedback += f"📷 共 {total_images} 张图片 | 并发: {concurrency}\n"
            feedback += f"🎨 预设: {preset_display}"
            feedback += f"\n⏳ 图片将并发处理，请耐心等待..."
            await event.send(event.chain_result([Plain(feedback)]))
        
        # 6. 扣费
        if deduction["source"] == "user":
            await self.data_mgr.decrease_user_count(uid, total_cost)
        elif deduction["source"] == "group":
            await self.data_mgr.decrease_group_count(gid, total_cost)
        
        # 7. 使用信号量控制并发
        semaphore = asyncio.Semaphore(concurrency)
        results = {"success": 0, "fail": 0}
        failed_details = []
        results_lock = asyncio.Lock()
        max_retries = self.conf.get("batch_retries", 2)
        
        async def process_single(index: int, url: str):
            async with semaphore:
                try:
                    # 下载图片
                    img_bytes = await self.img_mgr.load_bytes(url)
                    if not img_bytes:
                        error_msg = "图片下载失败，可能是链接已过期或网络问题"
                        logger.error(f"Concurrent batch process image {index} download failed: {url}")
                        async with results_lock:
                            results["fail"] += 1
                            failed_details.append({
                                "index": index,
                                "reason": error_msg,
                                "url_preview": url[:50] + "..." if len(url) > 50 else url
                            })
                        await event.send(event.chain_result([
                            Plain(f"❌ 第 {index}/{total_images} 张图片处理失败\n📍 原因: {error_msg}")
                        ]))
                        return
                    
                    # 处理单张图片（带重试机制）
                    retry_count = 0
                    success = False
                    error_msg = ""
                    
                    while retry_count <= max_retries:
                        success, error_msg = await self._run_single_batch_task(
                            event=event,
                            image_bytes=img_bytes,
                            prompt=final_prompt,
                            preset_name=preset_name,
                            task_index=index,
                            total_tasks=total_images,
                            uid=uid,
                            gid=gid,
                            extra_rules=extra_rules,
                            image_source=url,
                            hide_text=hide_llm_progress
                        )
                        
                        if success:
                            break
                            
                        retry_count += 1
                        if retry_count <= max_retries:
                            await event.send(event.chain_result([
                                Plain(f"⚠️ 第 {index}/{total_images} 张图片生成失败 ({error_msg})\n⏳ 正在进行第 {retry_count} 次重试...")
                            ]))
                            await asyncio.sleep(1.5)
                    
                    async with results_lock:
                        if success:
                            results["success"] += 1
                        else:
                            results["fail"] += 1
                            failed_details.append({
                                "index": index,
                                "reason": error_msg,
                                "url_preview": url[:50] + "..." if len(url) > 50 else url
                            })
                            await event.send(event.chain_result([
                                Plain(f"❌ 第 {index}/{total_images} 张图片最终处理失败\n📍 原因: {error_msg}")
                            ]))
                            
                except Exception as e:
                    error_msg = self._translate_error_to_chinese(str(e))
                    logger.error(f"Concurrent batch process image {index} exception: {e}", exc_info=True)
                    async with results_lock:
                        results["fail"] += 1
                        failed_details.append({
                            "index": index,
                            "reason": error_msg,
                            "url_preview": url[:50] + "..." if len(url) > 50 else url
                        })
                    await event.send(event.chain_result([
                        Plain(f"❌ 第 {index}/{total_images} 张图片处理异常\n📍 原因: {error_msg}")
                    ]))
        
        async def process_all():
            # 创建所有任务
            tasks = [
                process_single(i, url) 
                for i, url in enumerate(all_image_urls, 1)
            ]
            
            # 等待所有任务完成
            await asyncio.gather(*tasks)
            
            if not hide_llm_progress:
                # 发送完成汇总
                quota_str = self._get_quota_str(deduction, uid)
                summary = f"\n📊 并发批量处理完成\n"
                summary += f"✅ 成功: {results['success']} 张\n"
                summary += f"❌ 失败: {results['fail']} 张\n"
                summary += f"💰 剩余次数: {quota_str}"
                
                # 如果有失败的，附加失败汇总
                if failed_details:
                    summary += f"\n\n📋 失败图片汇总:"
                    for detail in sorted(failed_details, key=lambda x: x['index'])[:5]:
                        summary += f"\n  • 第{detail['index']}张: {detail['reason']}"
                    if len(failed_details) > 5:
                        summary += f"\n  ... 还有 {len(failed_details) - 5} 张失败"
                
                await event.send(event.chain_result([Plain(summary)]))
        
        # 启动异步任务
        asyncio.create_task(process_all())
        
        return f"并发批量处理任务已启动，共 {total_images} 张图片，并发数 {concurrency}，预设：{preset_name}。"

    # ================= 日常人设功能 =================

    @filter.llm_tool(name="shoubanhua_persona_photo")
    async def persona_photo_tool(self, event: AstrMessageEvent, scene_hint: str = "", extra_request: str = ""):
        '''生成Bot人设角色的日常照片。
        
        【重要】调用条件（请严格遵守）：
        1. 用户明确要求看照片时才调用，例如：
           - "发张照片"、"看看你"、"自拍一张"、"给我看看你"
           - "发个自拍"、"拍张照片给我"、"让我看看你长什么样"
        
        2. 以下情况【不要】调用此工具：
           - 用户只是问"你在干嘛"、"你在做什么" → 用文字回答即可
           - 用户只是闲聊、打招呼 → 正常对话即可
           - 用户没有明确表达想看照片的意愿 → 不要主动发照片
        
        3. 偶尔（约10%概率）可以在以下情况主动发照片：
           - 用户表达想念、关心Bot时
           - 聊天氛围很好，想分享当前状态时
           - 但大多数情况下，请用文字描述而不是发照片
        
        此工具会消耗用户次数，请谨慎调用。
        
        Args:
            scene_hint(string): 场景提示（可选），如"咖啡店"、"公园"等，用于匹配预设场景
            extra_request(string): 用户的额外要求（可选），如"穿红色衣服"、"微笑"等
        '''
        # 0. 检查功能开关
        if not self._persona_mode:
            return "❌ 日常人设功能未启用。请在配置中开启 enable_persona_mode。"
        
        if not self.conf.get("enable_llm_tool", True):
            return "❌ LLM 工具已禁用，请使用指令模式调用此功能。"
        
        # 0.1 检查图片生成冷却时间
        uid = norm_id(event.get_sender_id())
        in_cooldown, remaining = self._check_image_cooldown(uid)
        if in_cooldown:
            # 返回借口让LLM用自然语言拒绝
            excuse = self._get_cooldown_excuse(remaining)
            return f"【冷却中】{excuse}\n\n请用自然的方式告诉用户现在不方便拍照，可以稍后再试。不要直接说'冷却'这个词。"
        
        # 1. 加载人设参考图
        ref_images = await self._load_persona_ref_images()
        if not ref_images:
            return "❌ 未配置人设参考图。请先使用 #人设参考图添加 命令添加参考图。"
        
        # 2. 获取上下文用于场景匹配
        session_id = event.unified_msg_origin
        context_messages = await self.ctx_mgr.get_recent_messages(session_id, count=10)
        
        # 构建上下文文本
        context_text = scene_hint
        if context_messages:
            for msg in context_messages[-5:]:
                if msg.is_bot:
                    context_text += " " + msg.content
        
        # 3. 匹配场景
        scene_name, scene_prompt = self._match_persona_scene(context_text)
        
        # 4. 构建完整提示词
        full_prompt = self._build_persona_prompt(scene_prompt, extra_request)
        
        # 5. 根据配置决定是否发送进度提示
        if self.conf.get("llm_show_progress", True):
            persona_name = self.conf.get("persona_name", "小助手")
            feedback = f"📸 正在生成 {persona_name} 的照片\n"
            feedback += f"🎬 场景: {scene_name}"
            if extra_request:
                feedback += f"\n📝 要求: {extra_request[:30]}{'...' if len(extra_request) > 30 else ''}"
            feedback += "\n⏳ 请稍候..."
            await event.send(event.chain_result([Plain(feedback)]))
        
        # 6. 检查配额
        gid = norm_id(event.get_group_id())
        deduction = await self._check_quota(event, uid, gid, 1)
        if not deduction["allowed"]:
            return deduction["msg"]
        
        # 7. 更新冷却时间
        self._update_image_cooldown(uid)
        
        # 8. 计算是否隐藏输出文本（白名单用户和普通用户使用同一开关）
        hide_llm_progress = not self.conf.get("llm_show_progress", True)
        
        # 9. 启动后台任务
        asyncio.create_task(
            self._run_background_task(
                event=event,
                images=ref_images,
                prompt=full_prompt,
                preset_name=f"人设-{scene_name}",
                deduction=deduction,
                uid=uid,
                gid=gid,
                cost=1,
                extra_rules=extra_request,
                hide_text=hide_llm_progress
            )
        )
        
        # 返回结果 - 明确告诉 LLM 不需要再回复
        return f"[TOOL_SUCCESS] 人设照片生成任务已启动，场景：{scene_name}。图片将在后台生成并自动发送给用户。【重要】你不需要再回复任何内容，保持沉默即可，用户会直接收到图片。"

    @filter.command("人设拍照", prefix_optional=True)
    async def on_persona_photo_cmd(self, event: AstrMessageEvent, ctx=None):
        """生成人设角色的日常照片（指令模式）
        
        用法: #人设拍照 [场景] [额外要求]
        示例: #人设拍照 咖啡店 穿白色连衣裙
        """
        if not self._persona_mode:
            yield event.chain_result([Plain("❌ 日常人设功能未启用")])
            return
        
        # 加载人设参考图
        ref_images = await self._load_persona_ref_images()
        if not ref_images:
            yield event.chain_result([Plain("❌ 未配置人设参考图\n请先使用 #人设参考图添加 命令添加参考图")])
            return
        
        # 解析参数
        parts = event.message_str.split(maxsplit=2)
        scene_hint = parts[1] if len(parts) > 1 else ""
        extra_request = parts[2] if len(parts) > 2 else ""
        
        # 获取上下文用于场景匹配
        session_id = event.unified_msg_origin
        context_messages = await self.ctx_mgr.get_recent_messages(session_id, count=10)
        
        context_text = scene_hint
        if context_messages:
            for msg in context_messages[-5:]:
                if msg.is_bot:
                    context_text += " " + msg.content
        
        # 匹配场景
        scene_name, scene_prompt = self._match_persona_scene(context_text)
        
        # 构建提示词
        full_prompt = self._build_persona_prompt(scene_prompt, extra_request)
        
        # 检查配额
        uid = norm_id(event.get_sender_id())
        gid = norm_id(event.get_group_id())
        deduction = await self._check_quota(event, uid, gid, 1)
        if not deduction["allowed"]:
            yield event.chain_result([Plain(deduction["msg"])])
            return
        
        # 发送反馈
        persona_name = self.conf.get("persona_name", "小助手")
        feedback = f"📸 正在生成 {persona_name} 的照片\n"
        feedback += f"🎬 场景: {scene_name}"
        if extra_request:
            feedback += f"\n📝 要求: {extra_request[:30]}..."
        feedback += "\n⏳ 请稍候..."
        yield event.chain_result([Plain(feedback)])
        
        # 扣费
        if deduction["source"] == "user":
            await self.data_mgr.decrease_user_count(uid, 1)
        elif deduction["source"] == "group":
            await self.data_mgr.decrease_group_count(gid, 1)
        
        # 调用 API
        model = self.conf.get("model", "nano-banana")
        start = datetime.now()
        res = await self.api_mgr.call_api(ref_images, full_prompt, model, False, self.img_mgr.proxy)
        
        if isinstance(res, bytes):
            elapsed = (datetime.now() - start).total_seconds()
            await self.data_mgr.record_usage(uid, gid)
            
            quota_str = self._get_quota_str(deduction, uid)
            info = f"\n✅ 生成成功 ({elapsed:.2f}s) | 场景: {scene_name} | 剩余: {quota_str}"
            yield event.chain_result([Image.fromBytes(res), Plain(info)])
        else:
            yield event.chain_result([Plain(f"❌ 生成失败: {res}")])

    @filter.command("人设参考图添加", aliases={"添加人设图"}, prefix_optional=True)
    async def on_add_persona_ref(self, event: AstrMessageEvent, ctx=None):
        """添加人设参考图（管理员）
        
        用法: #人设参考图添加 [图片]
        """
        if not self.is_admin(event): return
        
        # 提取图片
        bot_id = self._get_bot_id(event)
        images = await self.img_mgr.extract_images_from_event(event, ignore_id=bot_id, context=self.context)
        
        if not images:
            yield event.chain_result([Plain("❌ 未检测到图片，请发送或引用图片")])
            return
        
        # 保存到特殊预设 "_persona_"
        count = await self.data_mgr.add_preset_ref_images("_persona_", images)
        
        if count > 0:
            total = len(self.data_mgr.get_preset_ref_image_paths("_persona_"))
            yield event.chain_result([Plain(f"✅ 已添加 {count} 张人设参考图\n当前共 {total} 张参考图")])
        else:
            yield event.chain_result([Plain("❌ 保存参考图失败")])

    @filter.command("人设参考图查看", aliases={"查看人设图"}, prefix_optional=True)
    async def on_view_persona_ref(self, event: AstrMessageEvent, ctx=None):
        """查看人设参考图（管理员）"""
        if not self.is_admin(event): return
        
        if not self.data_mgr.has_preset_ref_images("_persona_"):
            yield event.chain_result([Plain("暂无人设参考图")])
            return
        
        ref_images = await self.data_mgr.load_preset_ref_images_bytes("_persona_")
        
        if not ref_images:
            yield event.chain_result([Plain("人设参考图加载失败")])
            return
        
        result = [Plain(f"📷 人设参考图 ({len(ref_images)} 张):\n")]
        for img_bytes in ref_images[:5]:
            result.append(Image.fromBytes(img_bytes))
        
        if len(ref_images) > 5:
            result.append(Plain(f"\n... 还有 {len(ref_images) - 5} 张未显示"))
        
        yield event.chain_result(result)

    @filter.command("人设参考图清除", aliases={"清除人设图"}, prefix_optional=True)
    async def on_clear_persona_ref(self, event: AstrMessageEvent, ctx=None):
        """清除所有人设参考图（管理员）"""
        if not self.is_admin(event): return
        
        count = await self.data_mgr.clear_preset_ref_images("_persona_")
        
        if count > 0:
            yield event.chain_result([Plain(f"✅ 已清除 {count} 张人设参考图")])
        else:
            yield event.chain_result([Plain("暂无人设参考图")])

    @filter.command("人设场景列表", aliases={"场景列表"}, prefix_optional=True)
    async def on_list_persona_scenes(self, event: AstrMessageEvent, ctx=None):
        """查看所有人设场景"""
        if not self._persona_scene_map:
            yield event.chain_result([Plain("暂无场景配置")])
            return
        
        msg = f"🎬 人设场景列表 ({len(self._persona_scene_map)} 个):\n"
        for scene_name, prompt in sorted(self._persona_scene_map.items()):
            prompt_preview = prompt[:40] + "..." if len(prompt) > 40 else prompt
            msg += f"\n• {scene_name}: {prompt_preview}"
        
        default_prompt = self.conf.get("persona_default_prompt", "一张日常自拍照")
        msg += f"\n\n📌 默认场景: {default_prompt[:40]}..."
        
        yield event.chain_result([Plain(msg)])

    @filter.command("人设状态", prefix_optional=True)
    async def on_persona_status(self, event: AstrMessageEvent, ctx=None):
        """查看人设功能状态（管理员）"""
        if not self.is_admin(event): return
        
        persona_name = self.conf.get("persona_name", "小助手")
        persona_desc = self.conf.get("persona_description", "未配置")
        photo_style = self.conf.get("persona_photo_style", "未配置")
        trigger_keywords = self.conf.get("persona_trigger_keywords", [])
        
        has_ref_images = self.data_mgr.has_preset_ref_images("_persona_")
        ref_count = len(self.data_mgr.get_preset_ref_image_paths("_persona_")) if has_ref_images else 0
        
        msg = f"👤 人设功能状态:\n"
        msg += f"启用状态: {'✅ 已启用' if self._persona_mode else '❌ 未启用'}\n"
        msg += f"人设名称: {persona_name}\n"
        msg += f"人设描述: {persona_desc[:50]}{'...' if len(persona_desc) > 50 else ''}\n"
        msg += f"照片风格: {photo_style[:30]}{'...' if len(photo_style) > 30 else ''}\n"
        msg += f"参考图: {ref_count} 张\n"
        msg += f"场景数: {len(self._persona_scene_map)} 个\n"
        msg += f"触发词: {', '.join(trigger_keywords[:5])}{'...' if len(trigger_keywords) > 5 else ''}"
        
        yield event.chain_result([Plain(msg)])

    @filter.event_message_type(filter.EventMessageType.ALL, priority=4)
    async def on_batch_process_cmd(self, event: AstrMessageEvent, ctx=None):
        """批量处理上下文中的图片（指令模式）
        
        用法: #批量<预设名> [追加规则]
        示例: #批量手办化 皮肤白一点
        """
        if self.conf.get("prefix", True) and not event.is_at_or_wake_command:
            return

        text = event.message_str.strip()
        if not text: return
        
        # 消息去重检查：防止多平台重复处理同一消息
        msg_id = str(event.message_obj.message_id)
        dedup_key = f"batch_{msg_id}"
        if self._is_message_processed(dedup_key):
            logger.debug(f"FigurinePro: 批量处理消息 {msg_id} 已被处理，跳过重复执行")
            return
        
        # 匹配 "批量xxx" 或 "全部xxx"
        match = re.match(r"^(?:#|/)?(批量|全部)(.+)$", text)
        if not match:
            # 单独的 "批量" 或 "全部"
            if re.match(r"^(?:#|/)?(批量|全部)$", text):
                yield event.chain_result([Plain("用法: #批量<预设名> [追加规则]\n示例: #批量手办化 皮肤白一点")])
                event.stop_event()
            return
            
        preset_and_rules = match.group(2).strip()
        
        if preset_and_rules.startswith("处理"):
            # 兼容旧版的 "批量处理 xxx"
            parts = text.split(maxsplit=1)
            if len(parts) < 2:
                yield event.chain_result([Plain("用法: #批量<预设名> [追加规则]\n示例: #批量手办化 皮肤白一点")])
                event.stop_event()
                return
            prompt = parts[1].strip()
        else:
            prompt = preset_and_rules
            
        # 阻止事件继续传递给 on_figurine_request
        event.stop_event()
        
        # 1. 提取当前消息的图片 URL
        current_urls = []
        for seg in event.message_obj.message:
            if hasattr(seg, 'url') and seg.url:
                current_urls.append(seg.url)
            elif hasattr(seg, 'file') and seg.file:
                current_urls.append(seg.file)
                
        # 2. 获取上下文中的图片
        session_id = event.unified_msg_origin
        image_sources = await self._collect_images_from_context(session_id, count=self._context_rounds)
        
        if not image_sources and not current_urls:
            yield event.chain_result([Plain("❌ 未在上下文中找到图片。请先发送图片，然后再使用批量处理功能。")])
            return
        
        # 3. 收集所有图片URL（去重，优先获取最新图片）
        all_image_urls = []
        seen_urls = set()
        
        # 先添加当前消息的图片
        for url in current_urls:
            if url not in seen_urls:
                all_image_urls.append(url)
                seen_urls.add(url)
                
        # 倒序遍历上下文
        for msg_id, urls in reversed(image_sources):
            for url in urls:
                if url not in seen_urls:
                    all_image_urls.append(url)
                    seen_urls.add(url)
        
        # 限制数量
        max_images = self.conf.get("batch_max_images", 10)
        if max_images > 0:
            all_image_urls = all_image_urls[:max_images]
            
        # 反转列表以正常顺序处理
        all_image_urls.reverse()
        
        total_images = len(all_image_urls)
        if total_images == 0:
            yield event.chain_result([Plain("❌ 未找到有效的图片。")])
            return
        
        # 计算预设和追加规则
        final_prompt, preset_name, extra_rules = self._process_prompt_and_preset(prompt)
        
        # 检查配额
        uid = norm_id(event.get_sender_id())
        gid = norm_id(event.get_group_id())
        total_cost = total_images
        
        deduction = await self._check_quota(event, uid, gid, total_cost)
        if not deduction["allowed"]:
            yield event.chain_result([Plain(f"❌ 次数不足。批量处理 {total_images} 张图片需要 {total_cost} 次。\n{deduction['msg']}")])
            return
        
        concurrency = max(1, self.conf.get("batch_concurrency", 3))

        # 发送开始提示
        preset_display = "自定义" if preset_name in ["自定义", "编辑"] else preset_name
        feedback = f"📦 批量处理任务开始\n"
        feedback += f"📷 共 {total_images} 张图片 | 并发: {concurrency}\n"
        feedback += f"🎨 预设: {preset_display}"
        feedback += f"\n⏳ 图片将并发处理，请耐心等待..."
        yield event.chain_result([Plain(feedback)])
        
        # 扣费
        if deduction["source"] == "user":
            await self.data_mgr.decrease_user_count(uid, total_cost)
        elif deduction["source"] == "group":
            await self.data_mgr.decrease_group_count(gid, total_cost)
        
        # 使用信号量控制并发
        semaphore = asyncio.Semaphore(concurrency)
        results = {"success": 0, "fail": 0}
        failed_details = []
        results_lock = asyncio.Lock()
        max_retries = self.conf.get("batch_retries", 2)
        
        async def process_single(index: int, url: str):
            async with semaphore:
                try:
                    # 下载图片
                    img_bytes = await self.img_mgr.load_bytes(url)
                    if not img_bytes:
                        error_msg = "图片下载失败，可能是链接已过期或网络问题"
                        async with results_lock:
                            results["fail"] += 1
                            failed_details.append({
                                "index": index,
                                "reason": error_msg,
                                "url_preview": url[:50] + "..." if len(url) > 50 else url
                            })
                        await event.send(event.chain_result([
                            Plain(f"❌ 第 {index}/{total_images} 张图片处理失败\n📍 原因: {error_msg}")
                        ]))
                        return
                    
                    # 处理单张图片（带重试机制）
                    retry_count = 0
                    success = False
                    error_msg = ""
                    
                    while retry_count <= max_retries:
                        success, error_msg = await self._run_single_batch_task(
                            event=event,
                            image_bytes=img_bytes,
                            prompt=final_prompt,
                            preset_name=preset_name,
                            task_index=index,
                            total_tasks=total_images,
                            uid=uid,
                            gid=gid,
                            extra_rules=extra_rules,
                            image_source=url,
                            hide_text=False
                        )
                        
                        if success:
                            break
                            
                        retry_count += 1
                        if retry_count <= max_retries:
                            await event.send(event.chain_result([
                                Plain(f"⚠️ 第 {index}/{total_images} 张图片生成失败 ({error_msg})\n⏳ 正在进行第 {retry_count} 次重试...")
                            ]))
                            await asyncio.sleep(1.5)
                    
                    async with results_lock:
                        if success:
                            results["success"] += 1
                        else:
                            results["fail"] += 1
                            failed_details.append({
                                "index": index,
                                "reason": error_msg,
                                "url_preview": url[:50] + "..." if len(url) > 50 else url
                            })
                            await event.send(event.chain_result([
                                Plain(f"❌ 第 {index}/{total_images} 张图片最终处理失败\n📍 原因: {error_msg}")
                            ]))
                            
                except Exception as e:
                    error_msg = self._translate_error_to_chinese(str(e))
                    logger.error(f"Batch process image {index} exception: {e}", exc_info=True)
                    async with results_lock:
                        results["fail"] += 1
                        failed_details.append({
                            "index": index,
                            "reason": error_msg,
                            "url_preview": url[:50] + "..." if len(url) > 50 else url
                        })
                    await event.send(event.chain_result([
                        Plain(f"❌ 第 {index}/{total_images} 张图片处理异常\n📍 原因: {error_msg}")
                    ]))
        
        async def process_all():
            tasks = [process_single(i, url) for i, url in enumerate(all_image_urls, 1)]
            await asyncio.gather(*tasks)
            
            # 发送完成汇总
            quota_str = self._get_quota_str(deduction, uid)
            summary = f"\n📊 批量处理完成\n"
            summary += f"✅ 成功: {results['success']} 张\n"
            summary += f"❌ 失败: {results['fail']} 张\n"
            summary += f"💰 剩余次数: {quota_str}"
            
            if failed_details:
                summary += f"\n\n📋 失败图片汇总:"
                for detail in sorted(failed_details, key=lambda x: x['index'])[:5]:
                    summary += f"\n  • 第{detail['index']}张: {detail['reason']}"
                if len(failed_details) > 5:
                    summary += f"\n  ... 还有 {len(failed_details) - 5} 张失败"
            
            await event.send(event.chain_result([Plain(summary)]))

        # 启动异步任务
        asyncio.create_task(process_all())
