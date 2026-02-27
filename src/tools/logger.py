"""通用模块日志工具。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict


_LOGGER_CACHE: Dict[str, "Logger"] = {}
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class Logger:
    """模块日志包装器。

    Args:
        name: 模块名称，用于区分日志目录与 logger 实例。
        path: 日志文件绝对路径。
        logger: 标准 logging 实例。

    Returns:
        Logger: 可调用 `.info()` 与 `.error()` 的日志对象。

    Raises:
        RuntimeError: 当日志文件目录无法创建时抛出。

    Example:
        >>> logger = get_logger("chunking")
        >>> logger.info("开始处理")
        >>> logger.error("处理失败")
    """

    name: str
    path: Path
    logger: logging.Logger

    def info(self, message: str) -> None:
        """记录信息日志。

        Args:
            message: 日志正文。

        Returns:
            None
        """

        self.logger.info(f"[info] {message}")

    def error(self, message: str) -> None:
        """记录错误日志。

        Args:
            message: 日志正文。

        Returns:
            None
        """

        self.logger.error(f"[error] {message}")


def _build_logger(module_name: str, log_dir: Path) -> Logger:
    """创建模块日志实例。

    Args:
        module_name: 模块名称。
        log_dir: 模块日志目录。

    Returns:
        Logger: 初始化后的日志对象。

    Raises:
        RuntimeError: 日志目录创建失败时抛出。
    """

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RuntimeError(f"无法创建日志目录: {log_dir}") from error

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{module_name}_{timestamp}.log"

    logger = logging.getLogger(f"crsr.{module_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return Logger(name=module_name, path=log_path, logger=logger)


def get_logger(module_name: str, log_dir: Path | None = None) -> Logger:
    """获取模块日志对象（单进程内按模块名缓存）。

    Args:
        module_name: 模块名称，例如 `chunking`、`classification`。
        log_dir: 可选日志目录；为空时使用 `log/<module_name>/`。

    Returns:
        Logger: 缓存或新建的模块日志对象。

    Raises:
        ValueError: 模块名称为空时抛出。
        RuntimeError: 日志目录创建失败时抛出。

    Example:
        >>> logger = get_logger("classification")
        >>> logger.info("任务开始")
    """

    name = module_name.strip()
    if not name:
        raise ValueError("module_name 不能为空。")

    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    target_dir = log_dir or (PROJECT_ROOT / "log" / name)
    instance = _build_logger(name, target_dir)
    _LOGGER_CACHE[name] = instance
    return instance
