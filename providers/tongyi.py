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
#   改动范围   : 较大修改 —— 迁移至 multimodal-generation 接口，新增额度耗尽识别与错误分类
# =============================================================================

import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional, Tuple

from .base import BaseProvider, GenerationConfig, ImageGenerationResult
from astrbot.api import logger


# 【改造】阿里云百炼（DashScope）统一文生图端点（multimodal-generation）
# 该端点为 wan2.x-image / qwen-image 系列纯文生图模型的统一入口
DASHSCOPE_T2I_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
)

# 【新增】明确判定为"额度耗尽"的错误码（命中后触发自动降级到下一个模型）
QUOTA_EXHAUSTED_CODES = {
    "quota_exhausted",
    "quota_exceeded",
    "free_quota_exhausted",
    "allocationquotaexhausted",
    "throttling.quotaexceeded",
    "out_of_quota",
    "insufficient_quota",
}

# 【新增】额度类关键词，用于模糊匹配（兼容厂商文案变化）
QUOTA_KEYWORDS = ("quota", "free", "额度", "免费")

# 【新增】耗尽类关键词。已补充中文"用尽 / 已用尽 / 已用完"，
# 修复纯中文报错（如"您的免费额度已用尽"）因关键词缺失而漏判额度耗尽的问题。
EXHAUST_KEYWORDS = ("exhaust", "exceed", "insufficient", "not enough", "run out",
                    "耗尽", "用完", "用尽", "已用尽", "已用完", "超出", "不足")

# 【新增】中文强特征短语。阿里云部分报错为纯中文且不含 quota 英文字样，
# 命中其中任意一条即直接判定为额度耗尽（无需再搭配 quota 英文关键词），避免漏判。
QUOTA_EXHAUSTED_PHRASES = (
    "额度已用尽", "额度已用完", "额度用尽", "额度用完",
    "免费额度已用尽", "免费额度用完", "免费额度耗尽",
    "调用次数已用尽", "免费额度已用完",
    "额度不足", "余额不足", "账户余额不足",
)

# 【新增】鉴权类错误码：禁止降级，必须直接暴露给用户
AUTH_ERROR_CODES = {
    "invalidapikey", "unauthorized", "authenticationerror", "authentication_failed",
    "permissiondenied", "forbidden", "accessdenied", "invalidauthorization",
}

# 【体验优化】参数非法（InvalidParameter）相关的提示文案。
# 背景：百炼各模型支持的分辨率并不一致，2048*2048 并非全系可用；
# 而"参数非法"不属于额度耗尽，不会触发自动降级，用户只会看到一次失败，
# 因此这里在日志和用户可见文案上都显式引导其改小尺寸。
PARAM_ERROR_LOG_HINT = "提示：部分模型不支持当前分辨率，请尝试把「默认图片宽高」改为 1024x1024"
PARAM_ERROR_USER_HINT = "部分模型不支持该分辨率，可尝试将尺寸改为1024*1024"


class TongyiProvider(BaseProvider):
    """阿里云百炼 / 通义万相 文生图 Provider。

    v1.0 改造点：
    1. 统一使用 multimodal-generation 端点与标准文生图请求体；
    2. 支持按模型名逐个调用（供上层多模型降级调度）；
    3. 精确区分「额度耗尽 / 鉴权失败 / 参数非法 / 其它错误」。
       - 仅"额度耗尽"会标记 is_quota_exhausted=True 供上层降级；
       - 401、参数非法、模型不存在等一律不降级，直接返回错误。
    """

    @property
    def required_config_keys(self) -> list[str]:
        return ["api_key"]

    @property
    def default_model(self) -> str:
        return "wan2.7-image-pro"

    def validate_config(self) -> bool:
        api_key = self.get_config_value("api_key")
        return isinstance(api_key, str) and api_key.strip() != ""

    async def generate_image(self, config: GenerationConfig) -> ImageGenerationResult:
        api_key = self.get_config_value("api_key")
        base_url = self.get_config_value("base_url") or DASHSCOPE_T2I_ENDPOINT
        model = config.model or self.get_config_value("model") or self.default_model
        # 【配置精简】出图尺寸统一由「默认图片宽高」映射，不再单独提供 tongyi_size。
        # 映射规则见 _map_size：512x512→1024*1024、2048x2048→2048*2048、1280x720→1280*720
        size = self._map_size(config.width, config.height)
        watermark = bool(self.get_config_value("watermark", False))
        timeout = int(self.get_config_value("timeout", 120) or 120)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # 【改造】统一标准文生图请求体（所有模型通用）
        data = {
            "model": model,
            "input": {
                "messages": [
                    {"role": "user", "content": [{"text": config.prompt}]}
                ]
            },
            "parameters": {
                "size": size,
                "watermark": watermark
            }
        }

        logger.debug(f"[阿里百炼] 提交请求 model={model} size={size} endpoint={base_url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    base_url,
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    raw_text = await response.text()

                    if response.status == 200:
                        return self._parse_success(model, raw_text)

                    code, message = self._parse_error(raw_text, response.status)
                    return self._build_error(model, response.status, code, message, size=size)

        except asyncio.TimeoutError:
            msg = f"阿里百炼请求超时（超过 {timeout} 秒），模型 {model}"
            logger.warning(f"[阿里百炼] {msg}")
            return ImageGenerationResult(success=False, model=model, error_message=msg)
        except aiohttp.ClientError as e:
            msg = f"阿里百炼网络异常: {type(e).__name__}: {e}"
            logger.error(f"[阿里百炼] {msg}")
            return ImageGenerationResult(success=False, model=model, error_message=msg)
        except Exception as e:
            msg = f"阿里百炼请求异常: {type(e).__name__}: {e}"
            logger.error(f"[阿里百炼] {msg}")
            return ImageGenerationResult(success=False, model=model, error_message=msg)

    # ==================== 【新增】响应解析与错误分类 ====================

    def _parse_success(self, model: str, raw_text: str) -> ImageGenerationResult:
        """解析成功响应，兼容阿里云百炼两种返回结构：

        1. image-synthesis 专用接口风格：``$.output.results[*].url``
        2. multimodal-generation（对话）接口风格，wan2.x-image / qwen-image 系列
           走该端点时返回 OpenAI 对话风格：
           ``$.output.choices[*].message.content[*].image``
           （也可能为标准 OpenAI 格式的 ``image_url.url``）
        """
        try:
            result = json.loads(raw_text)
        except Exception:
            logger.error(f"[阿里百炼] 返回内容无法解析为 JSON: {raw_text[:300]}")
            return ImageGenerationResult(
                success=False,
                model=model,
                error_message=f"阿里百炼返回内容无法解析: {raw_text[:200]}"
            )

        output = result.get("output") or {}

        # 路径1：image-synthesis 风格 output.results[].url
        for item in (output.get("results") or []):
            url = (item or {}).get("url")
            if url:
                logger.info(f"[阿里百炼] 模型 {model} 生成成功")
                return ImageGenerationResult(success=True, image_url=url, model=model)

        # 路径2：multimodal-generation 对话风格
        # output.choices[].message.content[]，每个元素可能是
        #   {"image": "https://..."}           或
        #   {"image_url": {"url": "https://..."}}
        for choice in (output.get("choices") or []):
            message = (choice or {}).get("message") or {}
            for part in (message.get("content") or []):
                if not isinstance(part, dict):
                    continue
                url = part.get("image") or (part.get("image_url") or {}).get("url")
                if url:
                    logger.info(f"[阿里百炼] 模型 {model} 生成成功")
                    return ImageGenerationResult(success=True, image_url=url, model=model)

        _, message = self._parse_error(raw_text, 200)
        logger.warning(f"[阿里百炼] 成功响应中未找到图片 url: {raw_text[:300]}")
        return ImageGenerationResult(
            success=False,
            model=model,
            error_message=f"阿里百炼未返回图片地址: {message or '响应中缺少图片地址'}"
        )

    def _parse_error(self, raw_text: str, status: int) -> Tuple[str, str]:
        """从响应体中解析 (code, message)，解析失败时降级为纯文本"""
        try:
            data = json.loads(raw_text)
        except Exception:
            return "", (raw_text[:300] or f"HTTP {status}")

        if not isinstance(data, dict):
            return "", (raw_text[:300] or f"HTTP {status}")

        error = data.get("error") or {}
        code, message = "", ""
        if isinstance(error, dict):
            code = str(error.get("code") or "")
            message = str(error.get("message") or "")
        if not code:
            code = str(data.get("code") or "")
        if not message:
            message = str(data.get("message") or "")
        return code, (message or f"HTTP {status}")

    def _build_error(
        self,
        model: str,
        status: int,
        code: str,
        message: str,
        size: str = "",
    ) -> ImageGenerationResult:
        """按错误类型分类处理，决定是否可以降级"""
        detail = f"code={code or '未知'} message={message}"
        code_norm = (code or "").strip().lower().replace("-", "_").replace(".", "")

        # 1) 鉴权失败 / 权限不足 —— 禁止降级
        if status in (401, 403) or code_norm in AUTH_ERROR_CODES:
            logger.error(f"[阿里百炼] ❌ 鉴权失败（HTTP {status}）：{detail}。模型 {model}，不触发降级。")
            return ImageGenerationResult(
                success=False,
                model=model,
                error_code=code or str(status),
                error_message=(
                    f"阿里百炼鉴权失败（HTTP {status}）：{message}。"
                    f"请检查 API Key 是否正确、是否已开通该模型服务。"
                )
            )

        # 2) 额度耗尽 —— 允许降级
        if self._is_quota_exhausted(code, message):
            logger.warning(f"[阿里百炼] ⚠️ 模型 {model} 免费额度已耗尽：{detail}")
            return ImageGenerationResult(
                success=False,
                model=model,
                error_code=code or "quota_exhausted",
                is_quota_exhausted=True,
                error_message=f"模型 {model} 免费额度已耗尽（{code or 'quota_exhausted'}）"
            )

        # 3) 参数非法 / 模型不存在 / 服务异常 —— 禁止降级
        logger.error(f"[阿里百炼] ❌ 模型 {model} 调用失败（HTTP {status}）：{detail}。非额度问题，不触发降级。")

        error_message = f"阿里百炼调用失败（HTTP {status}）：{message}"
        # 【体验优化】参数非法时额外给出分辨率引导：
        # 这类错误不会自动降级，只提示"调用失败"用户无法自助排查。
        if self._is_invalid_parameter(code, message):
            logger.debug(f"[阿里百炼] 当前生效尺寸 size={size or '未取到'}")
            logger.warning(f"[阿里百炼] {PARAM_ERROR_LOG_HINT}")
            error_message = f"{error_message}。{PARAM_ERROR_USER_HINT}"

        return ImageGenerationResult(
            success=False,
            model=model,
            error_code=code or str(status),
            error_message=error_message
        )

    def _is_invalid_parameter(self, code: str, message: str) -> bool:
        """判断是否为"参数非法"错误（InvalidParameter）。

        归一化掉空格、连字符与下划线后再匹配，以兼容
        InvalidParameter / invalid_parameter / invalid parameter 等写法。
        """
        text = f"{code} {message}".lower()
        for ch in (" ", "-", "_"):
            text = text.replace(ch, "")
        return "invalidparameter" in text

    def _is_quota_exhausted(self, code: str, message: str) -> bool:
        """判断是否为额度耗尽错误。

        满足以下任意一种即判定为额度耗尽（触发自动降级）：
          1. 精确命中 QUOTA_EXHAUSTED_CODES 错误码；
          2. 【条件B】命中中文强特征短语 —— 纯中文报错无 quota 英文也可判定；
          3. 【条件A】同时包含额度类关键词与耗尽类关键词。
        """
        code_norm = (code or "").strip().lower().replace("-", "_").replace(".", "")
        if code_norm in {c.replace(".", "").replace("-", "_") for c in QUOTA_EXHAUSTED_CODES}:
            return True

        text = f"{code} {message}".lower()

        # 条件B：中文强特征短语，独立判定，不依赖 quota 英文关键词
        for phrase in QUOTA_EXHAUSTED_PHRASES:
            if phrase in text:
                logger.debug(f"[阿里百炼] 命中中文额度耗尽强特征短语: {phrase}")
                return True

        # 条件A：额度类关键词 + 耗尽类关键词
        if not any(k in text for k in QUOTA_KEYWORDS):
            return False
        return any(k in text for k in EXHAUST_KEYWORDS)

    def _map_size(self, width: int, height: int) -> str:
        """把「默认图片宽高」映射为阿里云百炼的 size 字符串（宽*高）。

        这是阿里云出图尺寸的唯一来源（原 tongyi_size 配置项已移除）。
        """
        if width == height:
            if width <= 1024:
                return "1024*1024"
            return "2048*2048"
        elif width > height:
            if width <= 1280:
                return "1280*720"
            return "2048*1152"
        else:
            if height <= 1280:
                return "720*1280"
            return "1152*2048"
