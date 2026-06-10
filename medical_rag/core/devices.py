"""统一设备选择工具。"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_torch_device(device_name: str):
    """解析 PyTorch 设备配置。

    支持：
    - ``auto``: CUDA 可用时使用 ``cuda``，否则使用 ``cpu``
    - ``cpu``
    - ``cuda`` / ``cuda:0`` / ``cuda:1`` ...
    """
    import torch

    normalized = (device_name or "auto").strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("配置了 %s，但当前 CUDA 不可用，回退到 cpu。", device_name)
        return torch.device("cpu")
    return torch.device(normalized)


def resolve_sentence_transformer_device(device_name: str) -> str | None:
    """解析 SentenceTransformer 设备配置。

    返回 None 表示让 SentenceTransformer 自行选择。
    """
    normalized = (device_name or "auto").strip().lower()
    if normalized == "auto":
        return None
    if normalized.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("配置了 %s，但当前 CUDA 不可用，embedding 回退到 cpu。", device_name)
                return "cpu"
        except ImportError:
            logger.warning("未安装 torch，embedding 回退到 cpu。")
            return "cpu"
    return normalized


def resolve_hf_device_map(device_name: str):
    """解析 HuggingFace ``device_map`` 配置。"""
    normalized = (device_name or "auto").strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("配置了 %s，但当前 CUDA 不可用，HuggingFace device_map 回退到 cpu。", device_name)
                return {"": "cpu"}
        except ImportError:
            logger.warning("未安装 torch，HuggingFace device_map 回退到 cpu。")
            return {"": "cpu"}
    return {"": normalized}
