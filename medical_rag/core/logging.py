"""统一日志初始化。

应用入口和离线脚本在 ``main`` 起始处调用 :func:`setup_logging`，模块级代码
``logger = logging.getLogger(__name__)`` 即可获得统一格式日志。
"""

from __future__ import annotations

import logging

from medical_rag.core.config import settings


_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"


def setup_logging(level: str | None = None) -> None:
    """初始化根 logger；重复调用不会重复添加 handler。

    :param level: 覆盖默认的 LOG_LEVEL 环境变量；为 None 时使用 settings.LOG_LEVEL
    """
    target_level = (level or settings.LOG_LEVEL).upper()
    root = logging.getLogger()
    root.setLevel(target_level)
    if root.handlers:
        # 已经初始化过（如 streamlit 自带 handler），只调整级别即可
        for handler in root.handlers:
            handler.setLevel(target_level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.setLevel(target_level)
    root.addHandler(handler)
