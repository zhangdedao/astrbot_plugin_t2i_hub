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


class OpenAIProvider(BaseProvider):
    @property
    def required_config_keys(self) -> list[str]:
        return ["api_key"]
    
    @property
    def default_model(self) -> str:
        return "dall-e-3"
    
    def validate_config(self) -> bool:
        api_key = self.get_config_value("api_key")
        return isinstance(api_key, str) and api_key.strip() != ""
    
    async def generate_image(self, config: GenerationConfig) -> ImageGenerationResult:
        api_key = self.get_config_value("api_key")
        base_url = self.get_config_value("base_url", "https://api.openai.com/v1")
        model = config.model or self.get_config_value("model", self.default_model)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 映射尺寸参数
        size = self._map_size(config.width, config.height)
        quality = config.quality or "standard"
        
        data = {
            "model": model,
            "prompt": config.prompt,
            "size": size,
            "quality": quality,
            "n": 1
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/images/generations",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        image_url = result["data"][0]["url"]
                        return ImageGenerationResult(
                            success=True,
                            image_url=image_url
                        )
                    else:
                        error_text = await response.text()
                        try:
                            error_data = json.loads(error_text) if error_text else {}
                            error_msg = error_data.get("error", {}).get("message", f"HTTP {response.status}")
                        except json.JSONDecodeError:
                            error_msg = f"HTTP {response.status}: {error_text[:200]}"
                        return ImageGenerationResult(
                            success=False,
                            error_message=f"OpenAI API错误: {error_msg}"
                        )
        except aiohttp.ClientTimeout:
            return ImageGenerationResult(
                success=False,
                error_message="OpenAI请求超时: 连接或响应超时"
            )
        except aiohttp.ClientConnectorError as e:
            return ImageGenerationResult(
                success=False,
                error_message=f"OpenAI连接错误: {str(e)}"
            )
        except aiohttp.ClientResponseError as e:
            return ImageGenerationResult(
                success=False,
                error_message=f"OpenAI响应错误: HTTP {e.status}"
            )
        except Exception as e:
            return ImageGenerationResult(
                success=False,
                error_message=f"OpenAI请求异常: {type(e).__name__}: {str(e)}"
            )
    
    def _map_size(self, width: int, height: int) -> str:
        """将宽高映射到OpenAI支持的尺寸"""
        if width == height:
            if width <= 512:
                return "512x512"
            elif width <= 1024:
                return "1024x1024"
            else:
                return "1024x1024"
        elif width > height:
            return "1792x1024"
        else:
            return "1024x1792"