import importlib
import json
import pathlib
import sys
import types
import unittest
from unittest.mock import AsyncMock


ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = "shoubanhua_test_package"


def load_api_manager():
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(ROOT)]
        sys.modules[PACKAGE] = package

    if "astrbot" not in sys.modules:
        astrbot = types.ModuleType("astrbot")
        astrbot.logger = types.SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
        )
        sys.modules["astrbot"] = astrbot

    return importlib.import_module(f"{PACKAGE}.api_manager").ApiManager


class ApiEndpointBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ApiManager = load_api_manager()

    def setUp(self):
        self.manager = self.ApiManager({})

    def test_chat_mode_replaces_version_and_endpoint(self):
        self.assertEqual(
            self.manager._normalize_generic_chat_url(
                "https://api.example.com/openai/v1beta/chat/completions"
            ),
            "https://api.example.com/openai/v1/chat/completions",
        )

    def test_image_mode_replaces_version_and_endpoint(self):
        self.assertEqual(
            self.manager._convert_to_images_api_url(
                "https://api.example.com/v1/images/generations"
            ),
            "https://api.example.com/v1/images/generations",
        )
        self.assertEqual(
            self.manager._convert_to_images_api_url(
                "https://api.example.com/v1/chat/completions",
                has_input_image=True,
            ),
            "https://api.example.com/v1/images/edits",
        )

    def test_image_mode_aliases_are_normalized(self):
        for alias in ("image", "images", "openai_images"):
            with self.subTest(alias=alias):
                self.manager.config["interface_mode"] = alias
                self.assertEqual(self.manager._get_interface_mode(), "openai_image")

    def test_gemini_mode_always_uses_v1beta(self):
        self.assertEqual(
            self.manager._build_gemini_api_url(
                "https://generativelanguage.googleapis.com/v1/models/old:generateContent",
                "models/gemini-2.5-flash-image",
            ),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent",
        )

    def test_request_timeout_leaves_margin_for_tool_executor(self):
        self.manager.config["timeout"] = 120
        self.assertEqual(self.manager._get_request_timeout(), 110)

        self.manager.config["timeout"] = 45
        self.assertEqual(self.manager._get_request_timeout(), 45)

    def test_result_download_timeout_cannot_exceed_request_timeout(self):
        self.manager.config.update({
            "timeout": 120,
            "result_image_download_timeout": 180,
        })
        self.assertEqual(self.manager._get_result_image_download_timeout(), 110)

        self.manager.config["result_image_download_timeout"] = 45
        self.assertEqual(self.manager._get_result_image_download_timeout(), 45)


class ApiRouteTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.ApiManager = load_api_manager()

    async def test_forced_images_route_keeps_multiple_input_images_off_chat(self):
        manager = self.ApiManager({
            "interface_mode": "openai_chat",
            "generic_prefer_images_api": True,
            "base_url": "https://api.example.com",
            "api_keys": ["test-key"],
        })
        manager.call_images_api = AsyncMock(return_value=b"result")

        result = await manager._call_api_once(
            [b"first-image", b"second-image"], "edit this", "image-model"
        )

        self.assertEqual(result, b"result")
        manager.call_images_api.assert_awaited_once()
        args = manager.call_images_api.await_args.args
        self.assertEqual(args[0], [b"first-image", b"second-image"])

    async def test_openai_image_mode_posts_image_to_edits_endpoint(self):
        requested_urls = []

        class FakeResponse:
            status = 200

            async def text(self):
                return json.dumps({"data": [{"b64_json": "cmVzdWx0"}]})

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        class FakeSession:
            closed = False

            def post(self, url, **kwargs):
                requested_urls.append(url)
                return FakeResponse()

        manager = self.ApiManager({
            "interface_mode": "openai_image",
            "base_url": "https://api.example.com/v1/chat/completions",
            "api_keys": ["test-key"],
        })
        manager._session = FakeSession()

        result = await manager._call_api_once(
            [b"input-image"], "edit this", "image-model"
        )

        self.assertEqual(result, b"result")
        self.assertEqual(requested_urls, ["https://api.example.com/v1/images/edits"])
        self.assertTrue(all("/chat/completions" not in url for url in requested_urls))

    async def test_forced_images_route_does_not_fallback_to_chat_on_edits_error(self):
        manager = self.ApiManager({
            "interface_mode": "openai_chat",
            "generic_prefer_images_api": True,
            "base_url": "https://api.example.com",
            "api_keys": ["test-key"],
        })
        edits_error = "Images API edits endpoint unsupported 404"
        manager.call_images_api = AsyncMock(return_value=edits_error)

        result = await manager._call_api_once(
            [b"input-image"], "edit this", "image-model"
        )

        self.assertEqual(result, edits_error)
        manager.call_images_api.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
