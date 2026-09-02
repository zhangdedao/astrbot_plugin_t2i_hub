# =============================================================================
# astrbot_plugin_t2i_hub · T2I Hub 通用文生图（AstrBot 插件）
# 本仓库 : https://github.com/zhangdedao/astrbot_plugin_t2i_hub
# 许可证 : MIT —— 二开部分 Copyright (c) 2025 zhangdedao
#   SPDX : MIT
#
# 【来源声明】本文件派生自 MIT 开源项目 astrbot_plugin_txsc
#   上游仓库 : https://github.com/zhuiye8/astrbot_plugin_txsc
#   原作者   : zhuiye（上游代码版权归原作者所有，原版权声明予以保留）
#
#   ⚠️ 上游许可存在冲突，如实记录如下：
#      上游 README 声明「本项目采用MIT许可证」；但其随包 LICENSE 文件实为
#      GNU AGPL-3.0 全文（GitHub 官方检测亦判定该仓库为 AGPL-3.0）。
#      本二开项目按上游 README 的 MIT 声明执行，并已将上游 LICENSE 原样保留于
#      third_party/LICENSE-astrbot_plugin_txsc-AGPL-3.0.txt（未作任何修改）。
#      若原作者确认 AGPL-3.0 为其真实意图，本项目将相应变更许可。
#
# 【二开声明】维护者 : zhangdedao
#   改动范围   : 大幅修改 —— 新增多模型额度耗尽降级调度、自然语言触发词过滤器、副脑提示词优化器
# =============================================================================

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple

# 【原有】图片发送所需模块
import tempfile
import os
import base64

# 【改造】副脑已改为复用 AstrBot 内置的模型提供商，不再自建 aiohttp 会话，
# 因此本文件不再需要 aiohttp（各 Provider 内部仍自行依赖）。

from astrbot.api import logger
from astrbot.api.star import Star, Context, register
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain, Image

from .providers.base import BaseProvider, GenerationConfig, ImageGenerationResult


# ==================== 【新增】常量定义 ====================

# 功能1.5：全部模型尝试失败后返回给用户的提示
ALL_MODELS_FAILED_MESSAGE = "所有绘图模型调用失败，请检查免费额度或者API配置"

# 【修复】副脑优化结果的最大长度（字符）。
# 提示词过长会被上游判定为"参数非法"，而参数非法不属于额度耗尽、不会触发自动降级，
# 会直接导致绘图失败，因此在副脑输出侧做上限截断。
# 仅截断副脑优化结果，不改动用户原始输入。
MAX_OPTIMIZED_PROMPT_LENGTH = 2000

# 功能3：副脑默认系统提示词
DEFAULT_OPTIMIZER_SYSTEM_PROMPT = """你是AI绘图提示词优化专家，将用户中文描述转换为高质量英文绘图正向提示词。
只输出优化后的提示词正文，不要任何解释，不要多余对话。
丰富光影、画风、材质、细节描述，不要输出负面提示词。"""

# 功能2：触发词 handler 优先级。
# AstrBot 按 priority 降序执行 handler（默认 0）。这里设为 -1，
# 使本插件排在其它默认优先级插件（如 livingmemory 记忆插件）之后执行，
# 只有真正命中触发词时才 stop_event，最大程度避免与其它插件冲突。
TRIGGER_HANDLER_PRIORITY = -1

# 支持"多模型额度耗尽自动降级"的供应商 key
FALLBACK_CAPABLE_PROVIDER = "tongyi"

# 【体验优化】启动日志中模型队列最多展示的模型数量，超出部分用 ...(共N个) 缩写，避免刷屏
MODEL_LOG_DISPLAY_LIMIT = 4


def format_model_list(models: List[str], limit: int = MODEL_LOG_DISPLAY_LIMIT) -> str:
    """【体验优化】把模型队列格式化为可读字符串，过长时截断展示。

    例：wan2.7-image-pro, wan2.7-image, wan2.6-image, qwen-image-2.0-pro ...(共7个)
    注意：本函数只影响日志/展示，**不参与任何调度逻辑**，调度始终使用完整列表。
    """
    if not models:
        return "无"
    if len(models) <= limit:
        return ", ".join(models)
    return f"{', '.join(models[:limit])} ...(共{len(models)}个)"


# 【修复】触发词配置快照（纯数据，不含任何插件对象引用）。
#
# 背景：Filter 实例由 AstrBot 框架创建（TriggerWordFilter(raise_error)），既拿不到插件实例，
# 其 filter() 的 cfg 参数又是全局配置（不含插件配置），所以 Filter 无法直接读插件配置。
# 这里改为由插件实例把"开关 + 触发词列表"刷新到这份轻量快照中：
#   - 不再用类属性强引用插件实例，避免旧实例无法回收、以及热重载窗口期读到过期实例；
#   - 快照是纯数据，热重载后即使 Filter 实例尚未重建，也能读到当前配置，不会短暂失效。
_TRIGGER_SNAPSHOT: Dict[str, Any] = {"enabled": False, "words": []}


def update_trigger_snapshot(enabled: bool, words: List[str]) -> None:
    """由插件实例在初始化时调用，刷新 Filter 使用的触发词配置快照"""
    _TRIGGER_SNAPSHOT["enabled"] = bool(enabled)
    _TRIGGER_SNAPSHOT["words"] = list(words or [])
    logger.debug(
        f"[触发词] 配置快照已更新: 开关={_TRIGGER_SNAPSHOT['enabled']}, "
        f"触发词={_TRIGGER_SNAPSHOT['words'] or '无'}"
    )


def match_trigger_text(message: str) -> Optional[str]:
    """纯函数：判断消息是否命中触发词，命中则返回该触发词，否则返回 None。

    供 Filter（无状态）使用；插件实例方法 match_trigger_word 与之逻辑一致。
    """
    if not _TRIGGER_SNAPSHOT["enabled"] or not message:
        return None

    text = message.strip()
    for word in _TRIGGER_SNAPSHOT["words"]:
        if text.startswith(word):
            return word
    return None


class TriggerWordFilter(filter.CustomFilter):
    """【功能2】自定义自然语言触发词过滤器。

    采用 AstrBot 标准的 CustomFilter 事件机制，且保持**无状态**：
    - 仅在消息命中触发词时返回 True，激活对应的 handler；
    - 未命中时返回 False，不拦截、不消费任何事件，也不会强制唤醒机器人。

    ⚠️ 实现注意：Filter 内部必须完成**完整的触发词匹配**，不能图省事只判断开关或恒返回 True。
    因为 AstrBot 在 waking_check 阶段只要 Filter 通过就会置 is_wake=True 并激活 handler，
    若 Filter 对每条消息都返回 True，会导致机器人对所有消息都做出响应，属于严重回归。
    """

    def filter(self, event: AstrMessageEvent, cfg) -> bool:
        # 关于入参 cfg：它是 AstrBot 的**全局配置对象**（ctx.astrbot_config），
        # 只包含 wake_prefix、plugin_set 等全局项，**不是本插件的配置**。
        # 本插件的 enable_custom_trigger / trigger_words 无法从这里读取，请勿误用 cfg 读插件配置。
        #
        # ⚠️ 该方法运行在消息事件主流程中：一旦抛出异常，AstrBot 会把错误直接
        #    发送到聊天窗口并终止事件。因此这里必须全量捕获并返回 False。
        try:
            return match_trigger_text(event.message_str) is not None
        except Exception as e:
            logger.debug(f"[触发词] 匹配时发生异常，已忽略: {e}")
            return False


@register(
    "astrbot_plugin_t2i_hub",
    "zhangdedao",
    "通用文生图插件，支持多家AI图像生成服务商的统一调用，支持多模型额度耗尽自动降级、自然语言触发词、副脑提示词优化",
    "1.0.0"
)
class UniversalTextToImagePlugin(Star):
    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)
        self.config = config or {}
        self.providers: Dict[str, BaseProvider] = {}
        self.active_providers: List[str] = []

        self.plugin_name = "通用文生图插件"
        self.plugin_description = "支持多家供应商的文生图功能"
        self.plugin_version = "1.0.0"

        logger.info("初始化通用文生图插件")

        # 【修复】刷新触发词配置快照（Filter 无状态，只读快照，不再持有插件实例）
        update_trigger_snapshot(self._is_custom_trigger_enabled(), self._get_trigger_words())

        self._load_providers()
        self._initialize_providers()
        self._log_feature_status()

    def _load_providers(self):
        """动态加载所有供应商"""
        try:
            from .providers.ppio import PPIOProvider
            from .providers.qianfan import QianfanProvider
            from .providers.tongyi import TongyiProvider
            from .providers.volcengine import VolcengineProvider
            from .providers.xunfei import XunfeiProvider
            from .providers.zhipu import ZhipuProvider

            provider_mappings = {
                'zhipu': (ZhipuProvider, 'zhipu'),
                'qianfan': (QianfanProvider, 'qianfan'),
                'ppio': (PPIOProvider, 'ppio'),
                'tongyi': (TongyiProvider, 'tongyi'),
                'volcengine': (VolcengineProvider, 'volcengine'),
                'xunfei': (XunfeiProvider, 'xunfei')
            }

            for provider_name, (provider_class, config_prefix) in provider_mappings.items():
                try:
                    provider_config = self._get_provider_config(config_prefix)
                    if provider_config:
                        self.providers[provider_name] = provider_class(provider_config)
                        logger.info(f"加载供应商: {provider_name}")
                except Exception as e:
                    logger.warning(f"加载供应商 {provider_name} 失败: {e}")

        except ImportError as e:
            logger.error(f"导入供应商模块失败: {e}")

    def _get_provider_config(self, prefix: str) -> Dict[str, Any]:
        """从扁平化配置中提取供应商配置"""
        config = {}

        if prefix == 'zhipu':
            api_key = self.config.get('zhipu_api_key', '')
            if api_key:
                config = {
                    'api_key': api_key,
                    'base_url': self.config.get('zhipu_base_url'),
                    'model': self.config.get('zhipu_model')
                }
        elif prefix == 'qianfan':
            access_token = self.config.get('qianfan_access_token', '')
            if access_token:
                config = {
                    'access_token': access_token,
                    'model': self.config.get('qianfan_model'),
                    'steps': self.config.get('qianfan_steps')
                }
        elif prefix == 'ppio':
            api_key = self.config.get('ppio_api_key', '')
            if api_key:
                config = {
                    'api_key': api_key,
                    'base_url': self.config.get('ppio_base_url'),
                    'model': self.config.get('ppio_model'),
                    'steps': self.config.get('ppio_steps'),
                    'guidance_scale': self.config.get('ppio_guidance_scale')
                }
        elif prefix == 'tongyi':
            api_key = self.config.get('tongyi_api_key', '')
            if api_key:
                config = {
                    'api_key': api_key,
                    'base_url': self.config.get('tongyi_base_url'),
                    'model': self.config.get('tongyi_model'),
                    # 【配置精简】出图尺寸不再单独配置，统一由「默认图片宽高」映射（见 _map_size）
                    'watermark': self.config.get('tongyi_watermark', False),
                    'timeout': self.config.get('tongyi_timeout', 120),
                }
        elif prefix == 'volcengine':
            api_key = self.config.get('volcengine_api_key', '')
            if api_key:
                config = {
                    'api_key': api_key,
                    'base_url': self.config.get('volcengine_base_url'),
                    'model': self.config.get('volcengine_model')
                }
        elif prefix == 'xunfei':
            app_id = self.config.get('xunfei_app_id', '')
            api_key = self.config.get('xunfei_api_key', '')
            api_secret = self.config.get('xunfei_api_secret', '')
            if app_id and api_key and api_secret:
                config = {
                    'app_id': app_id,
                    'api_key': api_key,
                    'api_secret': api_secret
                }

        return config

    def _initialize_providers(self):
        """初始化可用的供应商"""
        for name, provider in self.providers.items():
            try:
                if provider.is_configured():
                    self.active_providers.append(name)
                    logger.info(f"供应商 {name} 已配置并可用")
                else:
                    logger.warning(f"供应商 {name} 配置不完整")
            except Exception as e:
                logger.error(f"初始化供应商 {name} 失败: {e}")

        if not self.active_providers:
            logger.warning("没有可用的文生图供应商")
        else:
            logger.info(f"已启用 {len(self.active_providers)} 个供应商: {', '.join(self.active_providers)}")

    # ==================== 【新增】功能状态日志 ====================

    def _log_feature_status(self):
        """启动时打印三大新功能的开关状态，便于排障"""
        logger.info("=" * 50)
        logger.info("【通用文生图插件 v1.0 功能状态】")
        # 【体验优化】模型队列过长时缩写，避免启动日志刷屏（仅影响展示，调度仍用完整列表）
        logger.info(f"  功能1 多模型自动降级: {'开启' if self._is_fallback_enabled() else '关闭'}"
                    f" | 模型队列: {format_model_list(self._get_model_priority_list())}")
        logger.info(f"  功能2 自定义触发词: {'开启' if self._is_custom_trigger_enabled() else '关闭'}"
                    f" | 触发词: {', '.join(self._get_trigger_words()) or '无'}")
        optimizer_desc = self._describe_optimizer_provider()
        logger.info(f"  功能3 副脑提示词优化: {'开启' if self._is_optimizer_enabled() else '关闭'}"
                    f" | 副脑模型: {optimizer_desc}")
        logger.info("=" * 50)

    def _describe_optimizer_provider(self) -> str:
        """功能3：描述副脑最终会用到的模型提供商，仅用于启动日志展示。

        不影响实际调度；AstrBot 尚未配置对话模型时也只返回说明文字，不抛异常。
        """
        provider_id = str(self.config.get("optimizer_provider_id") or "").strip()
        if provider_id:
            return f"{provider_id}（配置指定）"
        try:
            provider = self.context.get_using_provider()
        except Exception:
            return "自动跟随 AstrBot 当前对话模型"
        if provider is None:
            return "自动跟随 AstrBot 当前对话模型（当前暂无可用对话模型）"
        try:
            return f"{provider.meta().id} / {provider.get_model()}（自动跟随当前对话模型）"
        except Exception:
            return "自动跟随 AstrBot 当前对话模型"

    # ==================== 【新增】配置读取辅助方法 ====================

    def _is_fallback_enabled(self) -> bool:
        """功能1：自动降级总开关"""
        return bool(self.config.get("enable_auto_fallback", True))

    def _get_model_priority_list(self) -> List[str]:
        """功能1：文生图模型优先级列表（严格按此顺序逐个尝试）

        【修复】默认值只在本函数内硬编码一份，不再维护模块级 DEFAULT_MODEL_PRIORITY 常量，
        避免"常量一处、函数一处"两份默认值不同步的问题。
        业务行为保持不变：配置非空则用配置，配置缺失/为空/全是空白则用下面的兜底列表。
        """
        models = self.config.get("model_priority_list")
        if isinstance(models, (list, tuple)):
            cleaned = [str(m).strip() for m in models if str(m).strip()]
            if cleaned:
                return cleaned
        if models:
            # 配置里填了值但清洗后为空（例如全是空白字符），给一条 warning 方便排障
            logger.warning(
                f"[模型队列] 配置项 model_priority_list 内容无效（原始值: {models!r}），已回退到内置默认模型队列"
            )
        # 唯一的一处兜底默认值来源
        return [
            "wan2.7-image-pro",
            "wan2.7-image",
            "wan2.6-image",
            "qwen-image-2.0-pro",
            "qwen-image-2.0",
            "qwen-image-plus",
            "qwen-image-max",
        ]

    def _is_custom_trigger_enabled(self) -> bool:
        """功能2：自定义触发词总开关"""
        return bool(self.config.get("enable_custom_trigger", False))

    def _get_trigger_words(self) -> List[str]:
        """功能2：触发词列表

        【体验优化】逐条清洗，丢弃无效条目（None / 空字符串 / 纯空白）。
        WebUI 列表控件很容易留下空行；另外若直接用 str(w) 处理 None，
        会得到字符串 "None" 并被当成合法触发词，这里一并规避。
        """
        words = self.config.get("trigger_words")
        if isinstance(words, str):
            # 兼容：面板/配置被填成逗号分隔字符串的情况
            words = words.split(",")
        if not isinstance(words, (list, tuple)):
            return []

        cleaned: List[str] = []
        dropped = 0
        for word in words:
            if word is None:
                dropped += 1
                continue
            text = str(word).strip()
            if not text:
                dropped += 1
                continue
            cleaned.append(text)

        if dropped:
            logger.warning(
                f"[触发词] 已忽略 {dropped} 个无效触发词（空/纯空白），"
                f"当前生效触发词: {cleaned or '无'}"
            )
        return cleaned

    def _is_optimizer_enabled(self) -> bool:
        """功能3：副脑开关"""
        return bool(self.config.get("enable_optimizer", False))

    # ==================== 原有斜杠命令（全部保留） ====================

    @filter.command("tti", alias={"文生图"})
    async def text_to_image_command(self, event: AstrMessageEvent):
        """文生图命令"""
        async for result in self._handle_image_generation(event, None):
            yield result

    @filter.command("tti-zhipu")
    async def text_to_image_zhipu_command(self, event: AstrMessageEvent):
        """使用智谱AI生成图片"""
        async for result in self._handle_image_generation(event, "zhipu"):
            yield result

    @filter.command("tti-qianfan")
    async def text_to_image_qianfan_command(self, event: AstrMessageEvent):
        """使用百度千帆生成图片"""
        async for result in self._handle_image_generation(event, "qianfan"):
            yield result

    @filter.command("tti-tongyi")
    async def text_to_image_tongyi_command(self, event: AstrMessageEvent):
        """使用阿里百炼/通义万相生成图片"""
        async for result in self._handle_image_generation(event, "tongyi"):
            yield result

    @filter.command("tti-ppio")
    async def text_to_image_ppio_command(self, event: AstrMessageEvent):
        """使用PPIO生成图片"""
        async for result in self._handle_image_generation(event, "ppio"):
            yield result

    @filter.command("tti-huoshan")
    async def text_to_image_volcengine_command(self, event: AstrMessageEvent):
        """使用火山引擎生成图片"""
        async for result in self._handle_image_generation(event, "volcengine"):
            yield result

    @filter.command("tti-xunfei")
    async def text_to_image_xunfei_command(self, event: AstrMessageEvent):
        """使用科大讯飞生成图片"""
        async for result in self._handle_image_generation(event, "xunfei"):
            yield result

    # ==================== 【新增】功能2：自然语言触发词 ====================

    def match_trigger_word(self, message: str) -> Optional[str]:
        """判断消息是否命中触发词，命中则返回该触发词本身，否则返回 None"""
        if not self._is_custom_trigger_enabled():
            return None
        if not message:
            return None

        text = message.strip()
        for word in self._get_trigger_words():
            if text.startswith(word):
                return word
        return None

    @filter.custom_filter(TriggerWordFilter, priority=TRIGGER_HANDLER_PRIORITY)
    async def custom_trigger_handler(self, event: AstrMessageEvent):
        """功能2：命中自然语言触发词后直接绘图，无需斜杠命令"""
        matched_word = self.match_trigger_word(event.message_str)
        if matched_word is None:
            # 理论上不会走到这里（Filter 已过滤），保留防御
            return

        text = event.message_str.strip()
        prompt = text[len(matched_word):].strip()

        # 功能2.2：截取后 prompt 为空
        if not prompt:
            logger.info(f"[触发词] 命中「{matched_word}」但描述为空，已提示用户输入")
            yield event.plain_result("请输入图片描述")
            event.stop_event()
            return

        logger.info(f"[触发词] 命中触发词「{matched_word}」，绘图 prompt: {prompt}")

        async for result in self._handle_image_generation(
            event,
            specific_provider=None,
            prompt_override=prompt,
        ):
            yield result

        # 已由本插件处理完毕，终止事件传播，避免再走一次 LLM 对话
        event.stop_event()

    # ==================== 【新增】功能3：副脑提示词优化 ====================

    def _resolve_optimizer_provider(self, event: Optional[AstrMessageEvent]):
        """功能3：确定副脑使用的模型提供商。

        副脑直接复用 AstrBot「模型提供商」页面里已配置好的对话模型，
        因此无需在本插件里再填一遍 API 地址与密钥。

        选取顺序：
          1. 配置里显式指定的 optimizer_provider_id；
          2. 未指定（或该提供商已被删除/改名）时，回退到当前会话正在使用的对话模型；
          3. 再不行则用 AstrBot 全局默认的对话模型。
        """
        provider_id = str(self.config.get("optimizer_provider_id") or "").strip()
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
            if provider is not None:
                return provider
            logger.warning(
                f"[副脑优化] 配置指定的模型提供商「{provider_id}」不存在，可能已被删除或改名；"
                f"已自动回退到 AstrBot 当前使用的对话模型"
            )

        umo = getattr(event, "unified_msg_origin", None) if event is not None else None
        if umo:
            try:
                provider = self.context.get_using_provider(umo)
                if provider is not None:
                    return provider
            except Exception as e:
                logger.debug(
                    f"[副脑优化] 获取会话当前模型失败（{type(e).__name__}: {e}），改用全局默认模型"
                )
        try:
            return self.context.get_using_provider()
        except Exception as e:
            logger.debug(f"[副脑优化] 获取全局默认模型失败（{type(e).__name__}: {e}）")
            return None

    async def _optimize_prompt(self, prompt: str, event: Optional[AstrMessageEvent] = None) -> str:
        """功能3：调用副脑 LLM 将中文描述优化为高质量英文绘图提示词。

        任何异常（未找到提供商 / 超时 / 调用失败 / 返回为空）都只降级为原始 prompt，
        绝不中断绘图流程。
        """
        if not self._is_optimizer_enabled():
            return prompt

        provider = self._resolve_optimizer_provider(event)
        if provider is None:
            logger.warning(
                "[副脑优化] 未找到可用的模型提供商，跳过优化，使用原始 prompt。"
                "请到 AstrBot「模型提供商」页面至少配置一个对话模型，"
                "或在本插件的「副脑使用的模型提供商」中指定一个。"
            )
            return prompt

        try:
            provider_id = provider.meta().id
            provider_model = provider.get_model()
        except Exception as e:
            logger.warning(f"[副脑优化] 读取模型提供商信息失败（{type(e).__name__}: {e}），降级使用原始 prompt")
            return prompt

        system_prompt = str(self.config.get("optimizer_system_prompt") or "").strip() or DEFAULT_OPTIMIZER_SYSTEM_PROMPT
        timeout = int(self.config.get("optimizer_timeout", 30) or 30)
        temperature = float(self.config.get("optimizer_temperature", 0.8) or 0.8)

        # 功能3.4：打印原始 prompt 与本次使用的副脑
        logger.info(f"[副脑优化] 原始 prompt: {prompt}")
        logger.info(f"[副脑优化] 使用模型提供商: {provider_id}（模型: {provider_model}）")

        try:
            # 【改造】改走 AstrBot 内置的模型提供商，不再自建 aiohttp 会话。
            # 超时仍用 wait_for 兜底，超时后降级，不影响后续绘图。
            llm_resp = await asyncio.wait_for(
                self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[副脑优化] 调用超时（{timeout}s），降级使用原始 prompt")
            return prompt
        except Exception as e:
            logger.warning(f"[副脑优化] 调用失败（{type(e).__name__}: {e}），降级使用原始 prompt")
            return prompt

        try:
            optimized = (llm_resp.completion_text or "").strip()
        except Exception as e:
            logger.warning(f"[副脑优化] 解析返回结果失败（{type(e).__name__}: {e}），降级使用原始 prompt")
            return prompt

        if not optimized:
            logger.warning("[副脑优化] 模型返回内容为空，降级使用原始 prompt")
            return prompt

        # 【修复】副脑可能"越优化越长"甚至复述系统提示词，这里对优化结果做长度上限保护。
        # 只截断副脑的产出，绝不改动用户原始输入；截断后仍是一段可用的英文提示词。
        if len(optimized) > MAX_OPTIMIZED_PROMPT_LENGTH:
            logger.warning(
                f"[副脑优化] 提示词过长（{len(optimized)} 字符 > {MAX_OPTIMIZED_PROMPT_LENGTH}），已做截断"
            )
            optimized = optimized[:MAX_OPTIMIZED_PROMPT_LENGTH].rstrip()

        # 功能3.4：打印优化后的 prompt
        logger.info(f"[副脑优化] ✅ 优化后 prompt: {optimized}")
        return optimized

    # ==================== 【新增】功能1：多模型优先级调度 ====================

    async def _generate_with_model_fallback(
        self,
        provider: BaseProvider,
        config: GenerationConfig,
        models: List[str],
    ) -> ImageGenerationResult:
        """功能1：按模型优先级列表逐个尝试，仅在"额度耗尽"时切换到下一个模型。

        - 额度耗尽（quota exhausted）→ 记录日志并切换下一个模型
        - 401 鉴权失败 / 参数非法 / 模型不存在 / 其它错误 → 立即返回，禁止降级
        - 全部模型失败 → 返回统一失败提示
        """
        total = len(models)
        attempted: List[str] = []

        for index, model in enumerate(models):
            single_config = GenerationConfig(
                prompt=config.prompt,
                width=config.width,
                height=config.height,
                model=model,
                quality=config.quality,
                style=config.style,
            )

            # 功能1.6：打印每次尝试的模型名
            logger.info(f"[多模型调度] 第 {index + 1}/{total} 次尝试，使用模型: {model}")
            attempted.append(model)

            try:
                result = await provider.generate_image(single_config)
            except Exception as e:
                result = ImageGenerationResult(
                    success=False,
                    model=model,
                    error_message=f"请求异常: {type(e).__name__}: {e}"
                )

            if result.success and result.has_image:
                logger.info(f"[多模型调度] ✅ 模型 {model} 生成成功（第 {index + 1}/{total} 次尝试）")
                result.model = model
                return result

            # 功能1.6：打印失败原因
            reason = result.error_message or "未知错误"

            if result.is_quota_exhausted:
                attempted[-1] = f"{model}(额度耗尽)"
                if index + 1 < total:
                    # 功能1.6：打印切换到哪个备选模型
                    logger.warning(
                        f"[多模型调度] ⚠️ 模型 {model} 免费额度已耗尽（{reason}），"
                        f"自动切换到下一个模型: {models[index + 1]}"
                    )
                else:
                    logger.warning(f"[多模型调度] ⚠️ 模型 {model} 免费额度已耗尽，且已是最后一个模型")
                continue

            # 非额度问题：禁止降级，直接返回错误
            logger.error(
                f"[多模型调度] ❌ 模型 {model} 调用失败（非额度问题，终止降级）: {reason}"
            )
            return result

        # 功能1.5：全部模型失败
        logger.error(f"[多模型调度] ❌ 全部 {total} 个模型均调用失败，尝试记录: {', '.join(attempted)}")
        return ImageGenerationResult(
            success=False,
            error_message=ALL_MODELS_FAILED_MESSAGE
        )

    # ==================== 【改造】统一图像处理主流程 ====================

    def _parse_command_args(self, message_str: str) -> Tuple[str, Optional[str]]:
        """功能1.7：解析斜杠命令参数，支持 --model xxx / --model=xxx"""
        raw = (message_str or "").strip()
        tokens = raw.split()[1:]  # 去掉命令本身（如 /tti）

        model = None
        words: List[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            lower = token.lower()
            if lower == "--model" and index + 1 < len(tokens):
                model = tokens[index + 1].strip()
                index += 2
                continue
            if lower.startswith("--model="):
                model = token.split("=", 1)[1].strip()
                index += 1
                continue
            words.append(token)
            index += 1

        return " ".join(words).strip(), (model or None)

    async def _handle_image_generation(
        self,
        event: AstrMessageEvent,
        specific_provider: str = None,
        prompt_override: str = None,
        forced_model: str = None,
    ):
        """统一的图像生成处理方法。

        兼容原有斜杠命令调用，同时支持：
        - 功能1.7：--model 参数强制指定模型（绕过优先级队列）
        - 功能2  ：触发词路径传入 prompt_override
        - 功能3  ：全程执行副脑提示词优化
        """
        # 1. 解析 prompt 与 --model 参数
        if prompt_override is not None:
            prompt = prompt_override.strip()
        else:
            prompt, parsed_model = self._parse_command_args(event.message_str)
            if forced_model is None:
                forced_model = parsed_model

        if not prompt:
            yield event.plain_result(self._get_help_text())
            return

        # 2. 【功能3】副脑提示词优化（失败自动降级，不中断流程）
        prompt = await self._optimize_prompt(prompt, event)

        # 3. 选择供应商
        if specific_provider:
            if specific_provider not in self.active_providers:
                if specific_provider not in self.providers:
                    yield event.plain_result(f"供应商 {specific_provider} 未配置")
                else:
                    yield event.plain_result(f"供应商 {specific_provider} 配置无效或不可用")
                return
            available_providers = [specific_provider]
            tip = f"🎨 在画了，请稍等一会...\n🎯 正在使用 {specific_provider}"
        else:
            if not self.active_providers:
                yield event.plain_result("当前没有可用的文生图服务，请检查配置")
                return
            available_providers = self.active_providers
            tip = "🎨 在画了，请稍等一会..."

        if forced_model:
            tip += f"\n🎯 指定模型: {forced_model}（已绕过优先级队列）"
            logger.info(f"[绘图] 使用 --model 强制指定模型: {forced_model}")

        yield event.plain_result(tip)

        # 4. 生成
        config = GenerationConfig(
            prompt=prompt,
            width=self.config.get("default_width", 512),
            height=self.config.get("default_height", 512)
        )

        result = await self._generate_with_providers(
            config, available_providers, forced_model=forced_model
        )

        # 5. 结果处理
        if result.success and result.has_image:
            async for msg in self._yield_image(event, result):
                yield msg
            # 功能4：出图成功后追加一条提示语（留空则不发送）
            completion_msg = (self.config.get("completion_message") or "").strip()
            if completion_msg:
                yield event.plain_result(completion_msg)
        else:
            error_msg = result.error_message or "生成图片失败"
            yield event.plain_result(f"生成失败: {error_msg}")

    async def _yield_image(self, event: AstrMessageEvent, result: ImageGenerationResult):
        """发送图片结果：URL 直接发送，base64 落临时文件后发送"""
        if result.image_url:
            yield event.image_result(result.image_url)
        elif result.image_base64:
            tmp_file_path = None
            try:
                image_data = base64.b64decode(result.image_base64)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    tmp_file.write(image_data)
                    tmp_file_path = tmp_file.name
                yield event.image_result(tmp_file_path)
            except Exception as e:
                logger.error(f"处理base64图片并发送时出错: {e}")
                yield event.plain_result("图片已生成，但在发送时遇到问题。")
            finally:
                # 清理临时文件（yield 之后框架已完成本次结果消费）
                if tmp_file_path and os.path.exists(tmp_file_path):
                    try:
                        os.remove(tmp_file_path)
                    except Exception as e:
                        logger.debug(f"清理临时图片文件失败: {e}")

    async def _generate_with_providers(
        self,
        config: GenerationConfig,
        providers_list: list,
        forced_model: str = None,
    ) -> ImageGenerationResult:
        """使用指定的供应商列表生成图片。

        【改造】对支持降级的供应商（阿里百炼）启用多模型优先级调度；
                其它供应商保持原有的"逐个尝试"逻辑不变。
        """
        errors = []

        for provider_name in providers_list:
            if provider_name not in self.providers:
                errors.append(f"{provider_name}: 供应商未配置")
                continue

            provider = self.providers[provider_name]
            try:
                # 【新增】百炼：多模型额度耗尽自动降级
                if (
                    provider_name == FALLBACK_CAPABLE_PROVIDER
                    and self._is_fallback_enabled()
                    and not forced_model
                ):
                    models = self._get_model_priority_list()
                    logger.info(f"[多模型调度] 开始调度，共 {len(models)} 个模型，顺序: {' -> '.join(models)}")
                    result = await self._generate_with_model_fallback(provider, config, models)
                else:
                    if forced_model:
                        config.model = forced_model
                    logger.info(f"尝试使用供应商: {provider_name}"
                                + (f"（指定模型: {forced_model}）" if forced_model else ""))
                    result = await provider.generate_image(config)

                if result.success and result.has_image:
                    logger.info(f"供应商 {provider_name} 生成成功"
                                + (f"，使用模型: {result.model}" if result.model else ""))
                    return result
                else:
                    error_msg = result.error_message or "未知错误"
                    logger.warning(f"供应商 {provider_name} 生成失败: {error_msg}")
                    errors.append(f"{provider_name}: {error_msg}")
            except Exception as e:
                error_msg = f"请求异常: {str(e)}"
                logger.error(f"供应商 {provider_name} 异常: {error_msg}")
                errors.append(f"{provider_name}: {error_msg}")

        if len(providers_list) == 1:
            error_message = errors[0].split(": ", 1)[1] if errors else "生成失败"
        else:
            error_message = f"所有供应商都无法生成图片。详细错误: {'; '.join(errors)}"

        return ImageGenerationResult(success=False, error_message=error_message)

    # ==================== 【新增】Agent Function-call 工具 ====================

    @filter.llm_tool(name="t2i_generate")
    async def t2i_generate_tool(self, event: AstrMessageEvent, prompt: str, model: str = ""):
        '''调用 AI 文生图模型，根据文字描述生成图片并发送给用户。

        Args:
            prompt(string): 用于生成图片的详细描述文字，越具体越好
            model(string): 可选。指定绘图模型名称，留空则按优先级自动选择并在额度耗尽时自动降级
        '''
        if not prompt or not prompt.strip():
            yield "绘图失败：缺少图片描述内容，请提供具体的画面描述。"
            return

        if not self.active_providers:
            yield "绘图失败：当前没有可用的绘图服务，请检查插件配置（至少需要配置一个供应商的 API Key）。"
            return

        original_prompt = prompt.strip()
        # 【修复】model 可能是纯空白字符串（如 "   "），strip 后为空应当视为"未指定模型"，
        # 从而正常走优先级队列自动降级，而不是把空白当成模型名传给供应商。
        model_normalized = model.strip() if isinstance(model, str) else ""
        forced_model = model_normalized or None
        if model and not model_normalized:
            logger.info("[工具调用] model 参数为空白字符串，已按未指定模型处理，将走优先级自动降级队列")

        # 功能3.5：Agent 路径同样执行副脑优化
        final_prompt = await self._optimize_prompt(original_prompt, event)

        logger.info(f"[工具调用] t2i_generate 触发，原始 prompt: {original_prompt}"
                    + (f"，指定模型: {forced_model}" if forced_model else "，模型: 自动降级调度"))

        config = GenerationConfig(
            prompt=final_prompt,
            width=self.config.get("default_width", 512),
            height=self.config.get("default_height", 512)
        )

        # 功能1.8：Agent 路径同样完整走降级调度
        result = await self._generate_with_providers(
            config, list(self.active_providers), forced_model=forced_model
        )

        if result.success and result.has_image:
            async for msg in self._yield_image(event, result):
                yield msg
            model_info = f"，使用模型：{result.model}" if result.model else ""
            yield f"图片已成功生成并发送给用户{model_info}。请向用户简要描述这张图的内容。"
        else:
            yield f"绘图失败：{result.error_message or '未知错误'}"

    # ==================== 帮助文本 ====================

    def _get_help_text(self) -> str:
        """生成帮助文本"""
        provider_commands = []
        provider_display = {
            'zhipu': 'zhipu',
            'qianfan': 'qianfan',
            'tongyi': 'tongyi',
            'ppio': 'ppio',
            'volcengine': 'huoshan',
            'xunfei': 'xunfei'
        }

        for provider, cmd_name in provider_display.items():
            status = "✓" if provider in self.active_providers else "✗"
            provider_commands.append(f"  /tti-{cmd_name} <描述> - {status}")

        # 【新增】功能开关状态
        fallback_status = "开启" if self._is_fallback_enabled() else "关闭"
        trigger_status = "开启" if self._is_custom_trigger_enabled() else "关闭"
        optimizer_status = "开启" if self._is_optimizer_enabled() else "关闭"
        trigger_words = "、".join(self._get_trigger_words()) or "无"

        return f"""🎨 通用文生图插件使用帮助

📋 基本命令:
/tti <描述文字> - 自动选择供应商生成图片
/文生图 <描述文字> - 同上（中文别名）

🎯 指定供应商命令:
{chr(10).join(provider_commands)}

🔧 指定模型（功能1）:
/tti --model wan2.7-image-pro <描述文字>
/tti --model=qwen-image-max <描述文字>

💬 自然语言触发词（功能2）:
当前状态: {trigger_status} | 触发词: {trigger_words}
直接发送「触发词 + 描述」即可绘图，无需斜杠命令

🧠 副脑提示词优化（功能3）:
当前状态: {optimizer_status}（将中文描述自动优化为高质量英文提示词）

🔄 多模型额度自动降级（功能1）:
当前状态: {fallback_status}
模型队列: {' -> '.join(self._get_model_priority_list())}

⚠️ 阿里云百炼注意:
· 部分模型不支持 2048*2048 分辨率，报 InvalidParameter 时请切换为 1024*1024
· 阿里每个模型免费额度独立，有效期 90 天（建议去百炼控制台开额度告警）

📊 当前可用供应商: {', '.join(self.active_providers) if self.active_providers else '无'}

💡 使用示例:
/tti 一只可爱的橘色小猫咪，坐在阳光明媚的窗台上
/tti-tongyi 科技感的未来城市夜景，霓虹灯闪烁
/tti-huoshan 美丽的山水风景画，中国风格
/tti --model wan2.7-image 赛博朋克风格的城市街道

⚠️ 注意事项:
• PPIO使用异步任务机制，生成时间较长（30秒-2分钟）
• 请确保账户余额充足
• 额度耗尽会自动切换下一个模型；401/参数错误不会降级

📖 完整文档请参阅插件README.md
"""
