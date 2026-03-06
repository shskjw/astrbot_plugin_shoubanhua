import io
import asyncio
import aiohttp
import base64
import ssl
import re
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

    async def load_bytes(self, src: str) -> bytes | None:
        """加载图片数据（本地/URL/Base64）- 纯异步封装"""
        raw = None
        loop = asyncio.get_running_loop()

        try:
            # 过滤空字符串或纯空格
            src = str(src).strip()
            if not src:
                return None

            # 判断是否是本地文件路径（兼容Windows环境，但忽略超长字符串因为可能是base64）
            is_local_file = False
            if len(src) < 512 and not src.startswith("http") and not src.startswith("base64://"):
                try:
                    if Path(src).is_file():
                        is_local_file = True
                except:
                    pass

            if is_local_file:
                # 文件IO放入线程池
                raw = await loop.run_in_executor(None, Path(src).read_bytes)
            elif src.startswith("http"):
                # 网络IO本身就是异步
                raw = await self._download_image(src)
            elif src.startswith("base64://"):
                # Base64解码放入线程池
                raw = await loop.run_in_executor(None, base64.b64decode, src[9:])

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

    async def extract_pdfs_from_event(self, event: AstrMessageEvent) -> List[bytes]:
        """从消息中提取 PDF 文件"""
        tasks = []
        for seg in event.message_obj.message:
            seg_class = type(seg).__name__
            # 兼容 AstrBot 的 File/Document 等文件组件
            if seg_class in ['File', 'Document', 'Record']:
                url = getattr(seg, 'url', None)
                file = getattr(seg, 'file', None)
                name = getattr(seg, 'name', '').lower()
                
                is_pdf = False
                if name.endswith('.pdf'):
                    is_pdf = True
                elif url and '.pdf' in url.lower():
                    is_pdf = True
                elif file and '.pdf' in str(file).lower():
                    is_pdf = True
                    
                if is_pdf:
                    if url:
                        tasks.append(self.load_bytes(url))
                    elif file:
                        tasks.append(self.load_bytes(file))
                        
        if not tasks:
            return []
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        pdf_bytes = []
        for res in results:
            if isinstance(res, bytes):
                pdf_bytes.append(res)
            elif isinstance(res, Exception):
                logger.warning(f"PDF extraction error: {res}")
                
        return pdf_bytes

    async def extract_images_from_event(self, event: AstrMessageEvent, ignore_id: str = None, context=None) -> List[bytes]:
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
                        # 尝试获取原消息
                        bot = context.get_bot()
                        if bot:
                            # 这是一个异步调用，获取历史消息
                            target_msg = await bot.get_message(seg.id)
                            if target_msg:
                                # target_msg 可能是 AstrMessageEvent 或 组件列表
                                components = []
                                if hasattr(target_msg, "message_obj") and hasattr(target_msg.message_obj, "message"):
                                    components = target_msg.message_obj.message
                                elif isinstance(target_msg, list):
                                    components = target_msg
                                elif hasattr(target_msg, "message"):
                                    components = target_msg.message
                                
                                for comp in components:
                                    if isinstance(comp, Image):
                                        if comp.url:
                                            tasks.append(self.load_bytes(comp.url))
                                        elif comp.file:
                                            tasks.append(self.load_bytes(comp.file))
                    except Exception as e:
                        logger.warning(f"Failed to fetch reply message {seg.id}: {e}")

            # 当前消息图片
            elif isinstance(seg, Image):
                if seg.url:
                    tasks.append(self.load_bytes(seg.url))
                elif seg.file:
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
        if at_users:
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
