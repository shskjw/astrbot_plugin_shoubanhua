import io
import unittest

from PIL import Image

from generation_params import (
    compute_size_for_aspect_ratio,
    detect_aspect_ratio_from_image,
    find_closest_standard_aspect_ratio,
    resolve_image_generation_params,
)


class GenerationParamsTest(unittest.TestCase):
    def test_prompt_ratio_and_resolution(self):
        params = resolve_image_generation_params(
            "请生成一张 16:9 小学数学 PPT 正文页成图，4K",
            default_aspect_ratio="4:3",
        )
        self.assertEqual(params["aspect_ratio"], "16:9")
        self.assertEqual(params["resolution"], "4K")
        self.assertEqual(params["size"], "3840x2160")

    def test_exact_dimensions_can_be_reversed(self):
        params = resolve_image_generation_params("输出 2336x3504")
        self.assertEqual(params["aspect_ratio"], "2:3")
        self.assertEqual(params["resolution"], "4K")

    def test_image_ratio_is_exact(self):
        output = io.BytesIO()
        Image.new("RGB", (1080, 1920)).save(output, "PNG")
        self.assertEqual(detect_aspect_ratio_from_image(output.getvalue(), "4:3"), "9:16")

    def test_image_ratio_keeps_non_standard_value(self):
        output = io.BytesIO()
        Image.new("RGB", (1080, 2400)).save(output, "PNG")
        detected = detect_aspect_ratio_from_image(output.getvalue(), "4:3")
        self.assertEqual(detected, "9:20")

        params = resolve_image_generation_params("手办化", default_aspect_ratio=detected)
        self.assertEqual(params["aspect_ratio"], "9:20")
        self.assertEqual(params["size"], "688x1520")

    def test_explicit_ratio_accepts_arbitrary_value(self):
        for value, expected in (
            ("9:20", "9:20"),
            ("20:9", "20:9"),
            ("9：20", "9:20"),
            ("9*20", "9:20"),
            ("18:40", "9:20"),
        ):
            with self.subTest(value=value):
                params = resolve_image_generation_params("画一张手机壁纸", aspect_ratio=value)
                self.assertEqual(params["aspect_ratio"], expected)

    def test_non_standard_ratio_size_is_16_aligned(self):
        size = compute_size_for_aspect_ratio("2K", "9:20")
        width, height = (int(part) for part in size.split("x"))
        self.assertEqual(size, "1184x2640")
        self.assertEqual(width % 16, 0)
        self.assertEqual(height % 16, 0)

    def test_standard_ratio_still_uses_preset_size(self):
        self.assertEqual(compute_size_for_aspect_ratio("1K", "16:9"), "1280x720")
        self.assertEqual(compute_size_for_aspect_ratio("2K", "4:3"), "2048x1536")

    def test_closest_standard_ratio_for_gemini_enum(self):
        self.assertEqual(find_closest_standard_aspect_ratio("9:20"), "9:16")
        self.assertEqual(find_closest_standard_aspect_ratio("20:9"), "21:9")
        self.assertEqual(find_closest_standard_aspect_ratio("7:3"), "21:9")

    def test_prompt_ratio_parsing_stays_backward_compatible(self):
        # 提示词解析不扩大范围，避免把时间/比分当成画幅
        self.assertEqual(resolve_image_generation_params("20:30 的时钟海报")["aspect_ratio"], "1:1")
        self.assertEqual(resolve_image_generation_params("比分3:1")["aspect_ratio"], "1:1")

    def test_config_defaults_are_used_when_prompt_has_no_parameters(self):
        params = resolve_image_generation_params(
            "生成一张课程插图",
            default_resolution="2K",
            default_aspect_ratio="4:3",
        )
        self.assertEqual(params["size"], "2048x1536")

    def test_star_ratio_notation_is_supported(self):
        params = resolve_image_generation_params("小学课件正文页，4*3，1K")
        self.assertEqual(params["aspect_ratio"], "4:3")
        self.assertEqual(params["size"], "1024x768")


if __name__ == "__main__":
    unittest.main()
