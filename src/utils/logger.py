"""
日志配置：同时输出到终端和文件。

用法：
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
    logger.info("...")
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str,
    log_dir: str = "./logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    创建并配置 logger。

    Args:
        name: logger 名称，通常传 __name__
        log_dir: 日志目录
        level: 日志级别

    Returns:
        配置好的 logger
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # 格式
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # handler 1：输出到终端
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # handler 2：写入文件
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "rag_system.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger