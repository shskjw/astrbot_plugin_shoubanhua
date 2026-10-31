"""Exercise tool entry points without requiring a running AstrBot installation."""

import importlib
import pathlib
import sys
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_plugin(public_tools=True):
    package_name = "shoubanhua_tool_result_tests"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT)]
    modules = {package_name: package}
    tool_names = []

    def decorator(*args, **kwargs):
        return lambda function: function

    def tool_decorator(*args, **kwargs):
        tool_names.append(kwargs["name"])
        return lambda function: function

    logger = Mock()
    astrbot = types.ModuleType("astrbot")
    astrbot.logger = logger
    modules["astrbot"] = astrbot
    for name in (
        "astrbot.api", "astrbot.api.event", "astrbot.api.star",
        "astrbot.core", "astrbot.core.message", "astrbot.core.message.components",
        "astrbot.core.platform", "astrbot.core.platform.astr_message_event",
    ):
        modules[name] = types.ModuleType(name)
    filters = types.SimpleNamespace(
        command=decorator, event_message_type=decorator,
        EventMessageType=types.SimpleNamespace(ALL="all"),
    )
    if public_tools:
        modules["astrbot.api"].llm_tool = tool_decorator
    else:
        filters.llm_tool = tool_decorator
    modules["astrbot.api.event"].filter = filters
    for name in ("Context", "Star", "StarTools"):
        setattr(modules["astrbot.api.star"], name, type(name, (), {}))
    modules["astrbot.api.star"].register = decorator
    modules["astrbot.core"].AstrBotConfig = dict
    for name in ("Image", "Plain", "Node", "Nodes", "At", "Reply"):
        setattr(modules["astrbot.core.message.components"], name, type(name, (), {}))
    modules["astrbot.core.platform.astr_message_event"].AstrMessageEvent = type("Event", (), {})
    for name, exports in {
        "data_manager": ("DataManager",),
        "image_manager": ("ImageManager",),
        "api_manager": ("ApiManager",),
        "context_manager": ("ContextManager", "LLMTaskAnalyzer"),
    }.items():
        module = types.ModuleType(f"{package_name}.{name}")
        for export in exports:
            setattr(module, export, type(export, (), {}))
        modules[module.__name__] = module
    utils = types.ModuleType(f"{package_name}.utils")
    utils.norm_id = lambda value: str(value)
    utils.normalize_model_list = lambda value: value
    utils.extract_image_urls_from_text = lambda value: []
    modules[utils.__name__] = utils
    with patch.dict(sys.modules, modules):
        module = importlib.import_module(f"{package_name}.main")
    return module.FigurineProPlugin, tool_names, logger


class ToolRegistrationTest(unittest.TestCase):
    def test_public_registration_and_legacy_fallback_keep_tool_names(self):
        for public_tools in (True, False):
            with self.subTest(public_tools=public_tools):
                plugin_class, names, _ = load_plugin(public_tools)
                self.assertEqual(len(names), 8)
                self.assertIn("shoubanhua_persona_photo", names)
                self.assertIs(plugin_class.persona_photo_tool.__annotations__["return"], str)


class ToolResultTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.Plugin, _, cls.logger = load_plugin()

    def make_plugin(self):
        plugin = object.__new__(self.Plugin)
        plugin.conf = {"llm_show_progress": False, "enable_preset_ref_images": False}
        plugin._translate_error_to_chinese = lambda error, default_msg="": default_msg
        return plugin

    def make_persona(self, message):
        plugin = self.make_plugin()
        plugin._persona_mode = True
        plugin._context_rounds = 20
        plugin._check_image_cooldown = lambda uid: (False, 0)
        plugin._load_persona_ref_images = AsyncMock(return_value=[b"persona-reference"])
        plugin._get_bot_id = lambda event: "bot"
        plugin.img_mgr = types.SimpleNamespace(extract_images_from_event=AsyncMock(return_value=[]))
        plugin.ctx_mgr = types.SimpleNamespace(get_recent_messages=AsyncMock(return_value=[]))
        plugin._match_persona_scene = Mock(return_value=("default", "scene"))
        plugin._build_persona_prompt = lambda scene, extra: f"{scene}: {extra}"
        plugin._build_current_time_persona_hint = lambda: "time"
        plugin._check_quota = AsyncMock(return_value={"allowed": True, "source": "free"})
        plugin._update_image_cooldown = lambda uid: None
        plugin._clear_session_image_cache_if_idle = AsyncMock()
        plugin._register_pending_generation = AsyncMock()
        plugin._run_background_task = AsyncMock(return_value=(True, ""))
        plugin._run_batch_image_to_image = AsyncMock(return_value={"success": 3, "fail": 0})
        event = types.SimpleNamespace(
            message_str=message, unified_msg_origin="test-session",
            get_sender_id=lambda: "user", get_group_id=lambda: "group", send=AsyncMock(),
        )
        return plugin, event

    def test_failed_result_is_terminal_without_leaking_error(self):
        plugin = self.make_plugin()
        result = plugin._build_llm_tool_failure(
            "HTTP 500 https://provider.example/v1/images/edits?key=private-secret"
        )
        self.assertTrue(result.startswith("[TOOL_FAILED]"))
        self.assertNotIn("private-secret", result)
        self.assertNotIn("provider.example", result)
        self.assertNotIn("稍后我再", result)
        self.assertIn("不要自动重试", result)
        self.assertIn("不要调用其他工具", result)
        self.assertLess(len(result), 300)

    def test_empty_or_thought_only_result_cannot_claim_success(self):
        plugin = self.make_plugin()
        for text in ("", "<think>secret analysis</think>", "推理过程：内部信息"):
            with self.subTest(text=text):
                result = plugin._finalize_llm_tool_result(text)
                self.assertTrue(result.startswith("[TOOL_FAILED]"))
                self.assertNotIn("TOOL_SUCCESS", result)
                self.assertNotIn("secret analysis", result)

    async def test_count_only_persona_call_keeps_original_request(self):
        request = "在咖啡店穿红色裙子拍一张，4:5，4K"
        plugin, event = self.make_persona(request)
        result = await plugin.persona_photo_tool(event, count=1)
        kwargs = plugin._run_background_task.await_args.kwargs
        self.assertIn(request, kwargs["prompt"])
        self.assertEqual(kwargs["images"], [b"persona-reference"])
        plugin._match_persona_scene.assert_called_once_with(request)
        self.assertTrue(result.startswith("[TOOL_SUCCESS]"))

    async def test_count_only_persona_call_respects_explicit_user_count(self):
        plugin, event = self.make_persona("请给我拍3张穿红裙的照片")
        await plugin.persona_photo_tool(event, count=1)
        plugin._run_background_task.assert_not_awaited()
        self.assertEqual(plugin._run_batch_image_to_image.await_args.kwargs["count"], 3)

    async def test_explicit_persona_description_is_preserved(self):
        plugin, event = self.make_persona("原始消息不应重复写入")
        await plugin.persona_photo_tool(event, scene_hint="公园", extra_request="穿蓝裙")
        prompt = plugin._run_background_task.await_args.kwargs["prompt"]
        self.assertIn("穿蓝裙", prompt)
        self.assertNotIn(event.message_str, prompt)

    def test_persona_detection_does_not_steal_regular_poster_requests(self):
        plugin = self.make_plugin()
        plugin.conf.update({
            "persona_name": "菲比",
            "persona_trigger_keywords": ["菲比"],
        })

        self.assertFalse(
            plugin._looks_like_persona_photo_request(
                "菲比做一张原神联动金铲铲的宣传海报 4:3 4K"
            )
        )
        self.assertFalse(plugin._looks_like_persona_photo_request("菲比来一张宣传海报"))
        self.assertTrue(plugin._looks_like_persona_photo_request("菲比拍一张生活照 4:5 4K"))
        self.assertTrue(plugin._looks_like_persona_photo_request("菲比来一张自拍"))

    async def test_draw_tool_uses_original_message_for_persona_fallback(self):
        plugin = self.make_plugin()
        plugin._persona_mode = True
        plugin.conf.update({
            "persona_name": "菲比",
            "persona_trigger_keywords": ["菲比"],
        })
        plugin.persona_photo_tool = AsyncMock(side_effect=AssertionError("poster was misrouted"))
        plugin._check_image_cooldown = lambda uid: (False, 0)
        plugin._resolve_tool_requested_count = lambda *args, **kwargs: 1
        plugin._process_prompt_and_preset = lambda prompt: (prompt, "自定义", "")
        plugin._check_quota = AsyncMock(return_value={"allowed": True, "source": "free"})
        plugin._update_image_cooldown = lambda uid: None
        plugin._clear_session_image_cache_if_idle = AsyncMock()
        plugin._register_pending_generation = AsyncMock()
        plugin._run_background_task = AsyncMock(return_value=(True, ""))
        plugin._get_text_to_image_model = lambda: "image-model"
        plugin._is_vip_user = lambda uid, event: False
        plugin._get_rebellious_hint = lambda prompt, uid, event: ""
        event = types.SimpleNamespace(
            message_str="菲比做一张原神联动金铲铲的宣传海报 4:3 4K",
            unified_msg_origin="test-session",
            get_sender_id=lambda: "user",
            get_group_id=lambda: "group",
            send=AsyncMock(),
        )

        expanded_prompt = "菲比做一张原神联动金铲铲的宣传海报，CG级别渲染"
        result = await plugin.text_to_image_tool(
            event,
            prompt=expanded_prompt,
            aspect_ratio="4:3",
            resolution="4K",
        )

        plugin.persona_photo_tool.assert_not_awaited()
        self.assertEqual(plugin._run_background_task.await_args.args[1], [])
        self.assertIn(expanded_prompt, plugin._run_background_task.await_args.args[2])
        self.assertTrue(result.startswith("[TOOL_SUCCESS]"))

    async def test_draw_tool_does_not_classify_expanded_prompt_when_event_has_no_message(self):
        plugin = self.make_plugin()
        plugin._persona_mode = True
        plugin.conf.update({
            "persona_name": "菲比",
            "persona_trigger_keywords": ["菲比"],
        })
        plugin.persona_photo_tool = AsyncMock(side_effect=AssertionError("expanded prompt was misrouted"))
        plugin._check_image_cooldown = lambda uid: (False, 0)
        plugin._resolve_tool_requested_count = lambda *args, **kwargs: 1
        plugin._process_prompt_and_preset = lambda prompt: (prompt, "自定义", "")
        plugin._check_quota = AsyncMock(return_value={"allowed": True, "source": "free"})
        plugin._update_image_cooldown = lambda uid: None
        plugin._clear_session_image_cache_if_idle = AsyncMock()
        plugin._register_pending_generation = AsyncMock()
        plugin._run_background_task = AsyncMock(return_value=(True, ""))
        plugin._get_text_to_image_model = lambda: "image-model"
        plugin._is_vip_user = lambda uid, event: False
        plugin._get_rebellious_hint = lambda prompt, uid, event: ""
        event = types.SimpleNamespace(
            message_str="",
            unified_msg_origin="test-session",
            get_sender_id=lambda: "user",
            get_group_id=lambda: "group",
            send=AsyncMock(),
        )

        result = await plugin.text_to_image_tool(
            event,
            prompt="Generate a natural daily life photo of 菲比, preserve the persona reference image",
            aspect_ratio="4:3",
            resolution="4K",
        )

        plugin.persona_photo_tool.assert_not_awaited()
        self.assertEqual(plugin._run_background_task.await_args.args[1], [])
        self.assertTrue(result.startswith("[TOOL_SUCCESS]"))

    async def test_smart_tool_uses_original_message_for_persona_fallback(self):
        plugin = self.make_plugin()
        plugin._persona_mode = True
        plugin._context_rounds = 20
        plugin._auto_detect_confidence = 0.8
        plugin.conf.update({
            "persona_name": "菲比",
            "persona_trigger_keywords": ["菲比"],
        })
        plugin.persona_photo_tool = AsyncMock(side_effect=AssertionError("expanded prompt was misrouted"))
        plugin._check_image_cooldown = lambda uid: (False, 0)
        plugin._extract_message_info = lambda event: {
            "content": event.message_str,
            "has_image": False,
        }
        plugin.ctx_mgr = types.SimpleNamespace(get_recent_messages=AsyncMock(return_value=[]))
        plugin._has_recent_session_image_context = AsyncMock(return_value=False)
        plugin._get_conf_bool = lambda key, default=True: False if key == "llm_show_progress" else default
        plugin._process_prompt_and_preset = lambda prompt: (prompt, "自定义", "")
        plugin._check_quota = AsyncMock(return_value={"allowed": True, "source": "free"})
        plugin._update_image_cooldown = lambda uid: None
        plugin._register_pending_generation = AsyncMock()
        plugin._run_background_task = AsyncMock(return_value=(True, ""))
        plugin._is_vip_user = lambda uid, event: False
        plugin._get_rebellious_hint = lambda prompt, uid, event: ""
        event = types.SimpleNamespace(
            message_str="菲比做一张原神联动金铲铲的宣传海报 4:3 4K",
            unified_msg_origin="test-session",
            get_sender_id=lambda: "user",
            get_group_id=lambda: "group",
            send=AsyncMock(),
        )
        analyzer = plugin.smart_generate_tool.__func__.__globals__["LLMTaskAnalyzer"]
        with patch.object(
            analyzer,
            "analyze_task_type",
            return_value={
                "task_type": "text_to_image",
                "confidence": 0.95,
                "reason": "poster",
                "suggested_prompt": "菲比做一张原神联动金铲铲的宣传海报",
            },
            create=True,
        ):
            result = await plugin.smart_generate_tool(
                event,
                user_request="Generate a natural daily life photo of 菲比, preserve the persona reference image",
            )

        plugin.persona_photo_tool.assert_not_awaited()
        self.assertEqual(plugin._run_background_task.await_args.args[1], [])
        self.assertTrue(result.startswith("[TOOL_SUCCESS]"))

    async def test_persona_failure_returns_failure_without_sending_raw_error(self):
        plugin, event = self.make_persona("拍一张照片")
        plugin._run_background_task.return_value = (False, "HTTP 500 internal provider error")
        result = await plugin.persona_photo_tool(event, count=1)
        self.assertTrue(result.startswith("[TOOL_FAILED]"))
        self.assertNotIn("internal provider", result)
        event.send.assert_not_awaited()

    async def test_suppressed_background_failure_is_still_logged(self):
        plugin = self.make_plugin()
        plugin.api_mgr = types.SimpleNamespace(
            call_api=AsyncMock(return_value="HTTP 400 key=private-secret"),
            _summarize_api_error=Mock(return_value="HTTP 400 key=<redacted>"),
        )
        plugin.img_mgr = types.SimpleNamespace(proxy=None)
        plugin._is_transient_generation_error = lambda error: False
        plugin._complete_pending_generation = AsyncMock()
        event = types.SimpleNamespace(unified_msg_origin="test-session", send=AsyncMock())
        self.logger.reset_mock()
        success, _ = await plugin._run_background_task(
            event, [b"reference"], "prompt", "人设-default", {"source": "free"},
            "user", "group", 1, suppress_user_error=True,
        )
        self.assertFalse(success)
        self.assertIn("HTTP 400 key=<redacted>", self.logger.warning.call_args.args[0])
        event.send.assert_not_awaited()
        plugin._complete_pending_generation.assert_awaited_once_with("test-session", 1)


if __name__ == "__main__":
    unittest.main()
