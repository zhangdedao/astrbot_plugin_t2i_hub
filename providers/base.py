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
#   改动范围   : 轻微修改 —— ImageGenerationResult 新增 3 个字段（均带默认值，向后兼容）
# =============================================================================

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass


@dataclass
class ImageGenerationResult:
    success: bool
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    error_message: Optional[str] = None

    # 【新增 v1.0】以下字段为"多模型额度耗尽自动降级"增量添加。
    # 均带默认值，旧 Provider（智谱/千帆/PPIO/火山/讯飞）无需任何改动即可兼容。
    is_quota_exhausted: bool = False  # 是否为"额度耗尽"错误 —— 决定是否触发自动降级
    error_code: Optional[str] = None  # 上游返回的原始错误码，便于日志排障
    model: Optional[str] = None       # 实际使用的模型名称

    @property
    def has_image(self) -> bool:
        return self.image_url is not None or self.image_base64 is not None


@dataclass
class GenerationConfig:
    prompt: str
    width: int = 512
    height: int = 512
    model: Optional[str] = None
    quality: Optional[str] = None
    style: Optional[str] = None


class BaseProvider(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = self.__class__.__name__.lower().replace('provider', '')

    @abstractmethod
    async def generate_image(self, config: GenerationConfig) -> ImageGenerationResult:
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        pass

    @property
    @abstractmethod
    def required_config_keys(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        pass

    def get_config_value(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def is_configured(self) -> bool:
        try:
            return all(
                self.get_config_value(key) is not None
                for key in self.required_config_keys
            ) and self.validate_config()
        except Exception:
            return False
