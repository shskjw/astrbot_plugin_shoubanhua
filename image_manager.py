import io
import asyncio
import aiohttp
import base64
import ssl
import re
import inspect
from typing import List, Tuple
from pathlib import Path
from PIL import Image as PILImage, ImageFont, ImageDraw
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.message.components import At, Image, Reply
from astrbot import logger


class ImageManager:
    def __init__(self, config: dict):
        self.proxy = config.get("proxy_url") if config.get("use_proxy") else None
        self.max_retries = config.get("download_retries", 3)
        self.timeout = config.get("timeout", 60)
        self.table_quality = config.get("preset_table_quality", "高清")
        self.table_columns = config.get("preset_table_columns", 5)

    async def _download_image(self, url: str) -> bytes | None:
        """通用下载逻辑"""
        for i in range(self.max_retries + 1):
            try:
                ssl_ctx = ssl.create_default_context()
                ssl_ctx.check_hostname = False
                ssl_ctx.verify_mode = ssl.CERT_NONE

                async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_ctx)) as session:
                    async with session.get(url, proxy=self.proxy, timeout=self.timeout) as resp:
                        resp.raise_for_status()
                        return await resp.read()
            except Exception as e:
                if i < self.max_retries:
                    await asyncio.sleep(1)
        return None

    def _optimize_output_image_sync(self, raw: bytes) -> bytes:
        """对结果图做更积极的发送前压缩，降低平台上传耗时。"""
        if not raw or len(raw) < 700_000:
            return raw

        try:
            with PILImage.open(io.BytesIO(raw)) as img:
                working = img.copy()
                width, height = working.size

                # 更积极地下采样，优先缩短上传时间
                max_side = 1600 if len(raw) < 3_000_000 else 1280
                if width > max_side or height > max_side:
                    working.thumbnail((max_side, max_side), PILImage.Resampling.LANCZOS)

                # 判断是否真的需要透明通道
                has_alpha_band = "A" in working.getbands()
                needs_alpha = False
                if has_alpha_band:
                    try:
                        alpha = working.getchannel("A")
                        alpha_min, alpha_max = alpha.getextrema()
                        needs_alpha = alpha_min < 255 or alpha_max < 255
                    except Exception:
                        needs_alpha = True

                candidates = []

                # 候选1：如果不需要透明，优先转 JPEG
                if not needs_alpha:
                    rgb = working.convert("RGB")
                    for quality in (85, 78, 72):
                        out = io.BytesIO()
                        rgb.save(
                            out,
                            format="JPEG",
                            quality=quality,
                            optimize=True,
                            progressive=True,
                            subsampling=1 if quality >= 80 else 2,
                        )
                        candidates.append(("JPEG", quality, out.getvalue()))
                else:
                    # 候选2：保留透明时，尽量压缩 PNG
                    rgba = working.convert("RGBA")
                    for compress_level in (9,):
                        out = io.BytesIO()
                        rgba.save(
                            out,
                            format="PNG",
                            optimize=True,
                            compress_level=compress_level,
                        )
                        candidates.append(("PNG", compress_level, out.getvalue()))

                if not candidates:
                    return raw

                # 先取最小候选
                best_format, best_level, best_data = min(candidates, key=lambda x: len(x[2]))

                # 如果仍然偏大，再做一次更强缩放兜底
                target_limit = 900_000
                if len(best_data) > target_limit:
                    fallback = working.copy()
                    fallback.thumbnail((1280, 1280), PILImage.Resampling.LANCZOS)

                    if not needs_alpha:
                        fallback = fallback.convert("RGB")
                        out = io.BytesIO()
                        fallback.save(
                            out,
                            format="JPEG",
                            quality=68,
                            optimize=True,
                            progressive=True,
                            subsampling=2,
                        )
                        best_format, best_level, best_data = "JPEG", 68, out.getvalue()
                    else:
                        out = io.BytesIO()
                        fallback.convert("RGBA").save(
                            out,
                            format="PNG",
                            optimize=True,
                            compress_level=9,
                        )
                        best_format, best_level, best_data = "PNG", 9, out.getvalue()

                if best_data and len(best_data) < len(raw):
                    logger.info(
                        f"结果图发送前已压缩: {len(raw) / 1024:.1f}KB -> {len(best_data) / 1024:.1f}KB "
                        f"| format={best_format} level={best_level}"
                    )
                    return best_data
        except Exception as e:
            logger.warning(f"Output image optimize failed: {e}")

        return raw

    async def optimize_output_image(self, raw: bytes) -> bytes:
        return await asyncio.to_thread(self._optimize_output_image_sync, raw)

    def _extract_first_frame_sync(self, raw: bytes) -> bytes:
        """(同步) 提取第一帧、转PNG并压缩"""
        try:
            with PILImage.open(io.BytesIO(raw)) as img:
                # 统一转换为 RGBA PNG，解决很多兼容性问题
                if getattr(img, "is_animated", False):
                    img.seek(0)
                img_conv = img.convert("RGBA")
                
                # [新增] 限制最大边长为 1568 (Gemini/OpenAI Pro Vision 推荐值)
                # 超过此尺寸只会增加 Tokens 消耗和延迟，对生成质量帮助不大
                max_side = 1568
                w, h = img_conv.size
                if w > max_side or h > max_side:
                    ratio = min(max_side / w, max_side / h)
                    new_size = (int(w * ratio), int(h * ratio))
                    img_conv = img_conv.resize(new_size, PILImage.Resampling.LANCZOS)
                
                out = io.BytesIO()
                # 使用 PNG 压缩优化 (optimize=True 比较耗时，为了速度改为 False，仅依靠Resize减小体积)
                img_conv.save(out, format="PNG")
                return out.getvalue()
        except Exception as e:
            logger.warning(f"Image conversion/resize failed: {e}")
            return raw

    def _is_avatar_source(self, src: str) -> bool:
        """判断一个来源是否明显是平台头像，而不是用户主动发送的图片"""
        if src is None:
            return False

        try:
            text = str(src).strip().lower()
        except Exception:
            return False

        if not text:
            return False

        avatar_keywords = [
            "qlogo.cn",
            "q1.qlogo.cn",
            "q2.qlogo.cn",
            "q3.qlogo.cn",
            "q4.qlogo.cn",
            "thirdqq.qlogo.cn",
            "headimg",
            "/avatar",
            "useravatar",
            "getavatar",
        ]
        return any(keyword in text for keyword in avatar_keywords)

    def _is_probably_valid_source(self, src: str) -> bool:
        """判断一个图片来源字符串是否值得继续尝试加载"""
        if src is None:
            return False

        try:
            src = str(src).strip()
        except Exception:
            return False

        if not src:
            return False

        if self._is_avatar_source(src):
            return False

        if src.startswith("http://") or src.startswith("https://") or src.startswith("base64://"):
            return True

        if len(src) < 512:
            try:
                return Path(src).is_file()
            except Exception:
                return False

        return False

    async def _resolve_bot_for_context(self, context):
        """兼容不同 AstrBot 版本的 Context/Bot 获取方式"""
        if not context:
            return None

        candidate_attrs = [
            "get_bot",
            "get_robot",
            "get_adapter",
            "get_client",
            "bot",
            "robot",
            "adapter",
            "client",
        ]

        for attr_name in candidate_attrs:
            if not hasattr(context, attr_name):
                continue

            try:
                target = getattr(context, attr_name)
                value = target() if callable(target) else target
                if inspect.isawaitable(value):
                    value = await value
                if value:
                    return value
            except Exception:
                continue

        return None

    async def _fetch_reply_components(self, context, reply_id) -> list:
        """尝试从上下文/机器人中获取被回复消息的组件列表"""
        if not context or not reply_id:
            return []

        bot = await self._resolve_bot_for_context(context)
        if not bot:
            return []

        for method_name in ("get_message", "fetch_message", "get_msg", "get_reply_message"):
            if not hasattr(bot, method_name):
                continue

            try:
                method = getattr(bot, method_name)
                result = method(reply_id)
                if inspect.isawaitable(result):
                    result = await result

                if not result:
                    continue

                if hasattr(result, "message_obj") and hasattr(result.message_obj, "message"):
                    return list(result.message_obj.message)
                if hasattr(result, "message"):
                    return list(result.message)
                if isinstance(result, list):
                    return result
            except Exception:
                continue

        return []

    async def load_raw_bytes(self, src: str) -> bytes | None:
        """加载原始字节（本地/URL/Base64），不做图片解析或格式转换"""
        raw = None
        loop = asyncio.get_running_loop()

        try:
            src = str(src).strip()
            if not src:
                return None

            is_local_file = False
            if len(src) < 512 and not src.startswith("http") and not src.startswith("base64://"):
                try:
                    if Path(src).is_file():
                        is_local_file = True
                except:
                    pass

            if src.startswith("http"):
                raw = await self._download_image(src)
            elif src.startswith("base64://"):
                raw = await loop.run_in_executor(None, base64.b64decode, src[9:])
            elif is_local_file:
                raw = await loop.run_in_executor(None, Path(src).read_bytes)
            else:
                logger.debug(f"跳过无效图片来源: {src[:80]}")
                return None

            return raw
        except Exception as e:
            logger.error(f"Failed to load raw bytes from {src[:50]}...: {e}")
            return None

    async def load_bytes(self, src: str) -> bytes | None:
        """加载图片数据（本地/URL/Base64）- 纯异步封装"""
        loop = asyncio.get_running_loop()

        try:
            raw = await self.load_raw_bytes(src)
            if raw:
                # 图片处理(PIL)放入线程池，防止阻塞
                return await loop.run_in_executor(None, self._extract_first_frame_sync, raw)
        except Exception as e:
            logger.error(f"Failed to load bytes from {src[:50]}...: {e}")

        return None

    async def get_avatar(self, user_id: str) -> bytes | None:
        if not user_id.isdigit(): return None
        return await self._download_image(f"https://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640")

    def images_to_pdf(self, images_bytes: List[bytes]) -> bytes:
        """将多张图片字节打包为单个 PDF"""
        import io
        from PIL import Image as PILImage
        
        if not images_bytes:
            return b""
            
        pil_images = []
        for img_bytes in images_bytes:
            try:
                img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                pil_images.append(img)
            except Exception as e:
                logger.error(f"Error opening image for PDF: {e}")
                
        if not pil_images:
            return b""
            
        out = io.BytesIO()
        if len(pil_images) == 1:
            pil_images[0].save(out, format="PDF", resolution=100.0)
        else:
            pil_images[0].save(out, format="PDF", resolution=100.0, save_all=True, append_images=pil_images[1:])
        
        return out.getvalue()
        
    def pdf_to_images(self, pdf_bytes: bytes) -> List[bytes]:
        """将 PDF 字节解析为图片列表（每页一张）"""
        try:
            import fitz # PyMuPDF
        except ImportError:
            raise ImportError("请先安装 PyMuPDF 库以支持解析 PDF：在终端运行 pip install PyMuPDF")
            
        images = []
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                images.append(pix.tobytes("png"))
        except Exception as e:
            raise Exception(f"解析 PDF 失败: {e}")
            
        return images

    def _collect_pdf_sources_from_components(self, components: list) -> List[str]:
        """从组件列表中提取 PDF 来源（url/file），保持顺序并去重"""
        if not components:
            return []

        sources = []
        seen = set()

        for seg in components:
            seg_class = type(seg).__name__
            # 兼容 AstrBot 的 File/Document 等文件组件
            if seg_class in ["File", "Document", "Record"]:
                url = getattr(seg, "url", None)
                file = getattr(seg, "file", None)
                name = str(getattr(seg, "name", "") or "").lower()

                is_pdf = False
                if name.endswith(".pdf"):
                    is_pdf = True
                elif isinstance(url, str) and ".pdf" in url.lower():
                    is_pdf = True
                elif isinstance(file, str) and ".pdf" in file.lower():
                    is_pdf = True

                if not is_pdf:
                    continue

                source = None
                if url:
                    source = str(url).strip()
                elif file:
                    source = str(file).strip()

                if source and source not in seen:
                    seen.add(source)
                    sources.append(source)

        return sources

    async def extract_pdfs_from_event(self, event: AstrMessageEvent, context=None) -> List[bytes]:
        """从消息中提取 PDF 文件，支持当前消息、引用链和主动拉取被引用消息"""
        pdf_sources = []
        seen_sources = set()

        def add_sources(sources: List[str]):
            for src in sources:
                if src and src not in seen_sources:
                    seen_sources.add(src)
                    pdf_sources.append(src)

        # 1. 当前消息中的 PDF
        add_sources(self._collect_pdf_sources_from_components(list(event.message_obj.message)))

        # 2. Reply 中的 PDF（优先使用 chain；如 chain 为空则主动拉取被引用消息）
        for seg in event.message_obj.message:
            if not isinstance(seg, Reply):
                continue

            found_in_chain = False
            if seg.chain:
                chain_sources = self._collect_pdf_sources_from_components(list(seg.chain))
                if chain_sources:
                    found_in_chain = True
                    add_sources(chain_sources)

            if not found_in_chain and context and hasattr(seg, "id") and seg.id:
                try:
                    logger.debug(f"Reply chain empty/no-pdf, fetching message_id: {seg.id}")
                    components = await self._fetch_reply_components(context, seg.id)
                    add_sources(self._collect_pdf_sources_from_components(components))
                except Exception as e:
                    logger.warning(f"Failed to fetch reply PDF message {seg.id}: {e}")

        if not pdf_sources:
            return []

        tasks = [self.load_raw_bytes(src) for src in pdf_sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        pdf_bytes = []
        for res in results:
            if isinstance(res, bytes):
                pdf_bytes.append(res)
            elif isinstance(res, Exception):
                logger.warning(f"PDF extraction error: {res}")

        return pdf_bytes

    async def extract_images_from_event(self, event: AstrMessageEvent, ignore_id: str = None, context=None,
                                        include_at_avatar: bool = True) -> List[bytes]:
        """从消息事件中提取所有图片 - 并发加速"""
        tasks = []
        at_users = set()  # 使用集合去重

        # 1. 规范化 ignore_id，确保是字符串且去除空白
        if ignore_id:
            ignore_id = str(ignore_id).strip()

        logger.debug(f"extract_images_from_event: ignore_id={ignore_id}")

        # 2. 收集所有待下载/读取的任务
        for seg in event.message_obj.message:
            # 回复链
            if isinstance(seg, Reply):
                # 优先尝试使用 chain (AstrBot 可能会自动填充)
                found_in_chain = False
                if seg.chain:
                    for s_chain in seg.chain:
                        if isinstance(s_chain, Image):
                            found_in_chain = True
                            if s_chain.url:
                                tasks.append(self.load_bytes(s_chain.url))
                            elif s_chain.file:
                                tasks.append(self.load_bytes(s_chain.file))
                
                # 如果 chain 中没有图片，且有 context 和 message_id，尝试主动获取消息
                if not found_in_chain and context and hasattr(seg, "id") and seg.id:
                    try:
                        logger.debug(f"Reply chain empty/no-image, fetching message_id: {seg.id}")
                        components = await self._fetch_reply_components(context, seg.id)

                        for comp in components:
                            if isinstance(comp, Image):
                                if self._is_probably_valid_source(getattr(comp, "url", None)):
                                    tasks.append(self.load_bytes(comp.url))
                                elif self._is_probably_valid_source(getattr(comp, "file", None)):
                                    tasks.append(self.load_bytes(comp.file))
                    except Exception as e:
                        logger.warning(f"Failed to fetch reply message {seg.id}: {e}")

            # 当前消息图片
            elif isinstance(seg, Image):
                if self._is_probably_valid_source(getattr(seg, "url", None)):
                    tasks.append(self.load_bytes(seg.url))
                elif self._is_probably_valid_source(getattr(seg, "file", None)):
                    tasks.append(self.load_bytes(seg.file))
            # @用户
            elif isinstance(seg, At):
                qq = str(seg.qq).strip()
                # 过滤机器人自身的 ID
                if ignore_id and qq == ignore_id:
                    continue
                at_users.add(qq)

        # 3. 文本中正则匹配的@
        # 有些平台 At 可能表现为纯文本
        text_ats = re.findall(r'@(\d+)', event.message_str)
        for qq in text_ats:
            qq = str(qq).strip()
            # 过滤机器人自身的 ID
            if ignore_id and qq == ignore_id:
                continue
            at_users.add(qq)

        # 4. 头像任务 (去重后)
        if include_at_avatar and at_users:
            logger.debug(f"At users to fetch avatars: {at_users}")
            for uid in at_users:
                tasks.append(self.get_avatar(uid))

        # 5. 并发执行所有任务
        if not tasks: return []
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 6. 过滤有效结果
        img_bytes = []
        for res in results:
            if isinstance(res, bytes):
                img_bytes.append(res)
            elif isinstance(res, Exception):
                logger.warning(f"Image extraction error: {res}")

        return img_bytes

    def _create_table_sync(self, presets: List[Tuple[str, bool]], data_mgr, font_path: str) -> bytes:
        """(同步) 绘制表格"""
        q_map = {
            "标准": (200, 250, 16),
            "高清": (300, 380, 24),
            "超清": (400, 500, 30)
        }
        cw, ch, fs = q_map.get(self.table_quality, q_map["高清"])
        cols = self.table_columns
        padding = 15 if self.table_quality == "高清" else 10

        rows = (len(presets) + cols - 1) // cols
        width = cols * cw + (cols + 1) * padding
        height = rows * ch + (rows + 1) * padding

        canvas = PILImage.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(canvas)

        font = None
        # 尝试加载字体
        try:
            if font_path and Path(font_path).exists():
                font = ImageFont.truetype(font_path, fs)
        except:
            pass

        if not font:
            # 回退系统字体
            sys_fonts = ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc",
                         "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"]
            for p in sys_fonts:
                if Path(p).exists():
                    try:
                        font = ImageFont.truetype(p, fs)
                        break
                    except:
                        continue
        if not font: font = ImageFont.load_default()

        image_area_h = int(ch * 0.8)

        for i, (name, is_builtin) in enumerate(presets):
            row, col = divmod(i, cols)
            x = padding + col * (cw + padding)
            y = padding + row * (ch + padding)

            # 绘制图片 (同步读取本地缓存文件)
            img_path = data_mgr.get_preset_image_path(name)
            if img_path:
                try:
                    p_img = PILImage.open(img_path).convert('RGB')
                    p_img.thumbnail((cw - 2 * padding, image_area_h - 2 * padding), PILImage.Resampling.LANCZOS)
                    ix = x + (cw - p_img.width) // 2
                    iy = y + (image_area_h - p_img.height) // 2
                    canvas.paste(p_img, (ix, iy))
                except:
                    pass

            draw.rectangle([x, y, x + cw, y + ch], outline='black', width=1)
            disp_name = ("📌" if is_builtin else "✨") + name

            try:
                bbox = draw.textbbox((0, 0), disp_name, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except:
                text_w = len(disp_name) * (fs // 2)
                text_h = fs

            tx = x + (cw - text_w) // 2
            ty = y + image_area_h + (ch - image_area_h - text_h) // 2
            draw.text((tx, ty), disp_name, fill='black', font=font)

        out = io.BytesIO()
        canvas.save(out, format='PNG', optimize=True)
        return out.getvalue()

    async def create_preset_table(self, presets: List[Tuple[str, bool]], data_mgr) -> bytes:
        """异步生成预览表格"""
        loop = asyncio.get_running_loop()
        current_dir = Path(__file__).parent
        custom_font_path = str(current_dir / "fonts" / "text.ttf")

        return await loop.run_in_executor(
            None,
            self._create_table_sync,
            presets,
            data_mgr,
            custom_font_path
        )
