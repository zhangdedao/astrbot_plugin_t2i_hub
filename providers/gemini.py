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
#   改动范围   : 未修改 —— 完整沿用上游实现
# =============================================================================

import aiohttp
import json
from typing import Dict, Any

from .base import BaseProvider, GenerationConfig, ImageGenerationResult


class GeminiProvider(BaseProvider):
    @property
    def required_config_keys(self) -> list[str]:
        return ["api_key"]
    
    @property
    def default_model(self) -> str:
        return "gemini-2.0-flash-exp"
    
    def validate_config(self) -> bool:
        api_key = self.get_config_value("api_key")
        return isinstance(api_key, str) and api_key.strip() != ""
    
    async def generate_image(self, config: GenerationConfig) -> ImageGenerationResult:
        api_key = self.get_config_value("api_key")
        model = config.model or self.get_config_value("model", self.default_model)
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        params = {"key": api_key}
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": f"Generate an image: {config.prompt}"
                }]
            }],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"]
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=data,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        candidates = result.get("candidates", [])
                        if candidates and "content" in candidates[0]:
                            parts = candidates[0]["content"]["parts"]
                            for part in parts:
                                if "inlineData" in part:
                                    image_data = part["inlineData"]["data"]
                                    return ImageGenerationResult(
                                        success=True,
                                        image_base64=image_data
                                    )
                        
                        return ImageGenerationResult(
                            success=False,
                            error_message="Gemini响应中没有找到图片数据"
                        )
                    else:
                        error_text = await response.text()
                        try:
                            error_data = json.loads(error_text)
                            error_msg = error_data.get("error", {}).get("message", f"HTTP {response.status}")
                        except:
                            error_msg = f"HTTP {response.status}: {error_text}"
                        return ImageGenerationResult(
                            success=False,
                            error_message=f"Gemini API错误: {error_msg}"
                        )
        except Exception as e:
            return ImageGenerationResult(
                success=False,
                error_message=f"Gemini请求异常: {str(e)}"
            )