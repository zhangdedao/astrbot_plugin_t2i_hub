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


class VolcengineProvider(BaseProvider):
    @property
    def required_config_keys(self) -> list[str]:
        return ["api_key"]
    
    @property
    def default_model(self) -> str:
        return "doubao-seedream-3-0-t2i-250415"
    
    def validate_config(self) -> bool:
        api_key = self.get_config_value("api_key")
        return isinstance(api_key, str) and api_key.strip() != ""
    
    async def generate_image(self, config: GenerationConfig) -> ImageGenerationResult:
        api_key = self.get_config_value("api_key")
        # 从配置中获取的base_url应为 https://ark.cn-beijing.volces.com/api/v3
        base_url = self.get_config_value("base_url", "https://ark.cn-beijing.volces.com/api/v3")
        model = config.model or self.get_config_value("model", self.default_model)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "prompt": config.prompt,
            "width": config.width,
            "height": config.height,
            "guidance_scale": self.get_config_value("guidance_scale", 7.5),
            "num_inference_steps": self.get_config_value("steps", 20),
            "enable_watermark": self.get_config_value("enable_watermark", False)
        }
        try:
            async with aiohttp.ClientSession() as session:
                # 【关键修复】去掉了硬编码的 "/api/v3"，直接在base_url后拼接接口名
                endpoint_url = f"{base_url}/images/generations"
                async with session.post(
                    endpoint_url,
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "data" in result and result["data"]:
                            image_url = result["data"][0]["url"]
                            return ImageGenerationResult(
                                success=True,
                                image_url=image_url
                            )
                        else:
                            # 兼容 "error" 字段
                            error_info = result.get("error", {})
                            error_msg = error_info.get("message", "未知错误")
                            return ImageGenerationResult(
                                success=False,
                                error_message=f"火山引擎API错误: {error_msg}"
                            )
                    else:
                        error_text = await response.text()
                        try:
                            error_data = json.loads(error_text)
                            # 兼容 "error" 字段
                            error_info = error_data.get("error", {})
                            error_msg = error_info.get("message", f"HTTP {response.status}")
                        except:
                            error_msg = f"HTTP {response.status}: {error_text}"
                        return ImageGenerationResult(
                            success=False,
                            error_message=f"火山引擎API错误: {error_msg}"
                        )
        except Exception as e:
            return ImageGenerationResult(
                success=False,
                error_message=f"火山引擎请求异常: {str(e)}"
            )