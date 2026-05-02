


import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path




def setup_logger(
    name: str,
    log_file: str = "pipeline.log",
    level: int = logging.INFO,
    max_bytes: int = 10485760,
    backup_count: int = 5,
    console: bool = True
) -> logging.Logger:
    """
    Setup  logger with rotation and console output.
    
    Args:
        name: Logger name (usually __name__)
        log_file: Path to log file
        level: Logging level
        max_bytes: Max size before rotation (default 10MB)
        backup_count: Number of backup files to keep
        console: Whether to also log to console
    Example:
    logger = setup_logger(__name__, log_file="logs/<log_file_name>.log")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )

    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


