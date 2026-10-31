## v2.9.4
- 参考 OmniDraw 最新工具实现，使用 AstrBot 公开 `llm_tool` 注册方式，并保留旧版兼容
- Images 图生图直接以 multipart 上传全部参考图，不再依赖 JSON 失败文案决定是否上传文件
- 修复仅传数量的人设拍照调用丢失用户场景、换装、尺寸和参考意图的问题
- 精简工具失败返回，明确结束本轮操作，不再引导模型自行重试、执行命令或查日志
- 图片接口失败记录脱敏状态与原因，修复相对结果图下载及空结果误报成功的问题
- 对非 DALL-E 模型不再强制传入 `response_format`，兼容更多 GPT Image 和中转模型名称

## v2.9.3
- 修复选择 Images API 后，多参考图图生图仍被隐式改走 `chat/completions` 的问题
- 图生图在 Images API 模式下固定请求 `/v1/images/edits`，接口不支持时明确报错，不再静默回落 Chat API
- Images API 图生图保留全部参考图，并兼容 `image` / `image[]` multipart 上传
- 修复旧 `dynamic_config.json` 可能覆盖面板中新选择接口模式的问题，并增加运行时实际路由日志

## v2.9.0
- 修复 `#切换模型` 在管理员 ID 类型不一致或旧模型列表格式异常时无响应的问题
- 统一 URL 自动补全：基础地址中的 v1/v1beta 和完整接口尾部会被忽略，再按所选模式生成唯一请求地址
- 新增 OpenAI Images、OpenAI Chat、Gemini 官方、自定义完整路径四种接口模式
- 配置面板移除旧 Generic/Gemini 独立地址、Key 池和 api_mode，仅保留统一接口配置；旧配置仍可静默读取
- `#切换API模式 generic` 等旧指令写法继续兼容，并映射到新的接口模式
- 自动识别提示词中的常规宽高比与 1K/2K/4K，并填充 OpenAI `size` 或 Gemini `imageConfig`
- 文生图默认比例可配置为 4:3，图生图会自动识别原图比例
- 补充 aiohttp、Pillow、PyMuPDF 插件依赖声明

## v2.8.8
- 优化上下文逻辑修改配置文件写入逻辑
