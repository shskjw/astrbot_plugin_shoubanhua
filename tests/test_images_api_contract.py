"""Exercise the Images API against a local HTTP server, without paid requests."""

import base64
import importlib
import io
import pathlib
import sys
import types
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = "shoubanhua_test_package"
API_KEY = "test-secret-image-api-key"


def load_api_manager_module():
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
    return importlib.import_module(f"{PACKAGE}.api_manager")


def make_png(color):
    output = io.BytesIO()
    Image.new("RGB", (40, 50), color).save(output, "PNG")
    return output.getvalue()


class ImagesApiContractTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.api_module = load_api_manager_module()

    async def asyncSetUp(self):
        self.png = make_png("red")
        self.second_png = make_png("blue")
        self.requests = []
        self.response_status = 200
        self.response_payload = {
            "data": [{"b64_json": base64.b64encode(self.png).decode("ascii")}]
        }
        app = web.Application()
        app.router.add_route("*", "/{path:.*}", self.handle_request)
        self.server = TestServer(app)
        await self.server.start_server()
        self.base_url = str(self.server.make_url("/")).rstrip("/")
        self.manager = self.api_module.ApiManager({
            "interface_mode": "openai_image",
            "base_url": self.base_url,
            "api_keys": [API_KEY],
            "image_resolution": "4K",
            "timeout": 3,
        })

    async def asyncTearDown(self):
        session = self.manager._session
        if session is not None and not session.closed:
            await session.close()
        await self.server.close()

    async def handle_request(self, request):
        observed = {
            "method": request.method,
            "path": request.path,
            "query": dict(request.query),
            "content_type": request.content_type,
            "authorization": request.headers.get("Authorization"),
            "fields": {},
            "files": [],
        }
        self.requests.append(observed)
        if request.method == "GET":
            return web.Response(body=self.png, content_type="image/png")
        if request.content_type == "multipart/form-data":
            reader = await request.multipart()
            async for part in reader:
                if part.filename:
                    observed["files"].append({
                        "name": part.name,
                        "filename": part.filename,
                        "content_type": part.headers.get("Content-Type"),
                        "data": bytes(await part.read()),
                    })
                else:
                    observed["fields"][part.name] = await part.text()
        else:
            observed["json"] = await request.json()
        return web.json_response(self.response_payload, status=self.response_status)

    async def test_persona_single_reference_posts_multipart_once_with_4k_portrait_size(self):
        result = await self.manager._call_api_once(
            [self.png], "take a photo", "gptimage2"
        )

        self.assertEqual(result, self.png)
        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        self.assertEqual(request["path"], "/v1/images/edits")
        self.assertEqual(request["content_type"], "multipart/form-data")
        self.assertEqual(request["authorization"], f"Bearer {API_KEY}")
        self.assertEqual(request["fields"]["model"], "gptimage2")
        self.assertEqual(request["fields"]["size"], "2560x3200")
        self.assertEqual(request["fields"]["n"], "1")
        self.assertNotIn("response_format", request["fields"])
        self.assertEqual(len(request["files"]), 1)
        self.assertEqual(request["files"][0]["name"], "image")
        self.assertEqual(request["files"][0]["content_type"], "image/png")
        self.assertEqual(request["files"][0]["data"], self.png)

    async def test_multiple_references_all_arrive_as_uploaded_files(self):
        result = await self.manager._call_api_once(
            [self.png, self.second_png], "group photo", "image-model"
        )

        self.assertEqual(result, self.png)
        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        self.assertEqual(request["path"], "/v1/images/edits")
        self.assertEqual(request["content_type"], "multipart/form-data")
        self.assertEqual([file["name"] for file in request["files"]], ["image[]", "image[]"])
        self.assertEqual([file["data"] for file in request["files"]], [self.png, self.second_png])

    async def test_custom_edits_preserves_exact_endpoint_and_uses_multipart(self):
        endpoint = f"{self.base_url}/tenant/images/edits?route=one"
        result = await self.manager.call_images_api(
            [self.png], "edit", "image-model", API_KEY, endpoint,
            exact_endpoint=True,
        )

        self.assertEqual(result, self.png)
        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        self.assertEqual(request["path"], "/tenant/images/edits")
        self.assertEqual(request["query"], {"route": "one"})
        self.assertEqual(request["content_type"], "multipart/form-data")
        self.assertEqual(request["files"][0]["data"], self.png)

    async def test_custom_non_edits_endpoint_keeps_json_reference_support(self):
        endpoint = f"{self.base_url}/tenant/generate?route=one"
        result = await self.manager.call_images_api(
            [self.png, self.second_png], "edit", "image-model", API_KEY, endpoint,
            exact_endpoint=True,
        )

        self.assertEqual(result, self.png)
        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        self.assertEqual(request["path"], "/tenant/generate")
        self.assertEqual(request["content_type"], "application/json")
        self.assertEqual(len(request["json"]["images"]), 2)
        images = [base64.b64decode(value.split(",", 1)[1]) for value in request["json"]["images"]]
        self.assertEqual(images, [self.png, self.second_png])

    async def test_custom_edits_without_reference_rejects_before_request(self):
        result = await self.manager.call_images_api(
            [], "edit", "image-model", API_KEY,
            f"{self.base_url}/tenant/images/edits?route=one",
            exact_endpoint=True,
        )

        self.assertIsInstance(result, str)
        self.assertIn("参考图", result)
        self.assertEqual(self.requests, [])

    async def test_text_to_image_keeps_json_and_omits_unrequested_legacy_fields(self):
        for model in ("gpt-image-1", "gptimage2", "gemini-image"):
            with self.subTest(model=model):
                self.requests.clear()
                result = await self.manager._call_api_once([], "a tree", model)
                self.assertEqual(result, self.png)
                self.assertEqual(len(self.requests), 1)
                request = self.requests[0]
                self.assertEqual(request["path"], "/v1/images/generations")
                self.assertEqual(request["content_type"], "application/json")
                self.assertNotIn("response_format", request["json"])
                self.assertEqual(request["json"]["n"], 1)

    async def test_dall_e_retains_supported_base64_response_format(self):
        result = await self.manager._call_api_once([], "a tree", "dall-e-3")

        self.assertEqual(result, self.png)
        self.assertEqual(self.requests[0]["json"]["response_format"], "b64_json")

    async def test_validation_error_is_logged_safely_without_extra_requests(self):
        encoded_png = base64.b64encode(self.png).decode("ascii")
        self.response_status = 422
        self.response_payload = {"error": {"message": (
            f"/v1/images/edits: expected file; token={API_KEY}; "
            f"source=data:image/png;base64,{encoded_png}"
        )}}

        with patch.object(self.api_module.logger, "error") as error_log:
            result = await self.manager._call_api_once(
                [self.png], "edit", "image-model"
            )

        self.assertIsInstance(result, str)
        self.assertIn("422", result)
        self.assertNotIn("endpoint unsupported", result.lower())
        self.assertEqual(len(self.requests), 1)
        diagnostic = " ".join(str(call) for call in error_log.call_args_list)
        self.assertIn("422", diagnostic)
        self.assertIn("expected file", diagnostic)
        self.assertNotIn(API_KEY, diagnostic)
        self.assertNotIn(encoded_png, diagnostic)

    def test_edits_endpoint_mention_does_not_mean_endpoint_is_unsupported(self):
        self.assertFalse(self.manager._is_images_edits_unsupported_error(
            "/v1/images/edits: image must be a file"
        ))
        self.assertTrue(self.manager._is_images_edits_unsupported_error(
            "unsupported endpoint: /v1/images/edits"
        ))

    async def test_relative_result_url_download_uses_api_host(self):
        self.manager.config["base_url"] = f"{self.base_url}/tenant/v1/images/generations"
        for image_url, expected_path in (
            ("/results/output.png", "/results/output.png"),
            ("results/output.png", "/tenant/results/output.png"),
        ):
            with self.subTest(image_url=image_url):
                self.requests.clear()
                self.response_payload = {"data": [{"url": image_url}]}

                result = await self.manager._call_api_once([], "a tree", "image-model")

                self.assertEqual(result, self.png)
                self.assertEqual(len(self.requests), 2)
                self.assertEqual(self.requests[0]["path"], "/tenant/v1/images/generations")
                self.assertEqual(self.requests[1]["method"], "GET")
                self.assertEqual(self.requests[1]["path"], expected_path)

    async def test_http_success_with_error_or_missing_image_remains_logged_failure(self):
        for payload, expected_reason in (
            ({"error": {"message": "model not available"}}, "model not available"),
            ({"data": []}, "未找到图片"),
        ):
            with self.subTest(payload=payload):
                self.requests.clear()
                self.response_payload = payload
                with patch.object(self.api_module.logger, "error") as error_log:
                    result = await self.manager._call_api_once([], "a tree", "image-model")

                self.assertIsInstance(result, str)
                self.assertIn(expected_reason, result)
                self.assertEqual(len(self.requests), 1)
                self.assertTrue(error_log.called)

    async def test_empty_or_invalid_base64_is_not_reported_as_image_bytes(self):
        for images in ([], [self.png]):
            for encoded_image in ("", "  ", "!!!!", "a", None, 1234):
                with self.subTest(has_reference=bool(images), encoded_image=encoded_image):
                    self.requests.clear()
                    self.response_payload = {"data": [{"b64_json": encoded_image}]}
                    with patch.object(self.api_module.logger, "error") as error_log:
                        result = await self.manager._call_api_once(images, "a tree", "image-model")

                    self.assertIsInstance(result, str)
                    self.assertEqual(len(self.requests), 1)
                    self.assertTrue(error_log.called)


if __name__ == "__main__":
    unittest.main()
