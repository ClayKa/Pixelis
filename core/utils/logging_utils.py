"""
Logging utilities for the Pixelis project.
Enhanced with distributed tracing support.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from .context import TraceContext


# Global logger configuration
_LOGGER_CONFIGURED = False
_LOG_LEVEL = logging.INFO
_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_LOG_FORMAT_WITH_TRACE = "%(asctime)s - %(name)s - %(levelname)s - [%(trace_id)s|%(component)s] - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_USE_TRACING = False


def setup_logging(
    level: Optional[int] = None,
    log_file: Optional[Path] = None,
    format_str: Optional[str] = None,
    use_tracing: bool = False,
    use_colors: bool = False,
) -> None:
    """
    Configure global logging settings.
    
    Args:
        level: Logging level (e.g., logging.INFO)
        log_file: Optional file to write logs to
        format_str: Custom format string for log messages
        use_tracing: Enable trace ID in logs
        use_colors: Enable colored output for console
    """
    global _LOGGER_CONFIGURED, _LOG_LEVEL, _LOG_FORMAT, _USE_TRACING
    
    if _LOGGER_CONFIGURED:
        return
    
    if level is not None:
        _LOG_LEVEL = level
    
    _USE_TRACING = use_tracing
    
    # Determine format string
    if format_str is not None:
        log_format = format_str
    elif use_tracing:
        log_format = _LOG_FORMAT_WITH_TRACE
    else:
        log_format = _LOG_FORMAT
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(_LOG_LEVEL)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Choose appropriate formatter
    if use_tracing and use_colors:
        console_formatter = ColoredTracingFormatter(log_format, _DATE_FORMAT)
    elif use_tracing:
        console_formatter = TracingFormatter(log_format, _DATE_FORMAT)
    elif use_colors:
        console_formatter = ColoredFormatter(log_format, _DATE_FORMAT)
    else:
        console_formatter = logging.Formatter(log_format, _DATE_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(_LOG_LEVEL)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(_LOG_LEVEL)
        # File handler always uses TracingFormatter if tracing is enabled
        if use_tracing:
            file_formatter = TracingFormatter(log_format, _DATE_FORMAT)
        else:
            file_formatter = logging.Formatter(log_format, _DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Set levels for third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    _LOGGER_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    # Ensure logging is configured
    if not _LOGGER_CONFIGURED:
        setup_logging()
    
    return logging.getLogger(name)


def set_log_level(level: int) -> None:
    """
    Set the global logging level.
    
    Args:
        level: Logging level (e.g., logging.DEBUG)
    """
    global _LOG_LEVEL
    _LOG_LEVEL = level
    
    # Update all handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)


def add_file_handler(log_file: Path) -> None:
    """
    Add a file handler to the root logger.
    
    Args:
        log_file: Path to log file
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(_LOG_LEVEL)
    formatter = logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)
    file_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)


class TracingFormatter(logging.Formatter):
    """
    Custom formatter that includes trace ID from context.
    
    This formatter automatically enriches log messages with trace IDs
    and component information from the current context.
    """
    
    def format(self, record):
        """
        Format the log record with trace context.
        
        Args:
            record: LogRecord to format.
            
        Returns:
            Formatted log message with trace ID.
        """
        # Get trace ID from context
        trace_id = TraceContext.get_trace_id()
        component = TraceContext.get_component()
        
        # Add trace context to the record
        record.trace_id = trace_id if trace_id else "-"
        record.component = component if component else "-"
        
        # Format the message
        return super().format(record)


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for console output.
    """
    
    # Color codes
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",        # Green
        "WARNING": "\033[33m",     # Yellow
        "ERROR": "\033[31m",       # Red
        "CRITICAL": "\033[35m",    # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record):
        """Format the log record with colors."""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        # Format the message
        formatted = super().format(record)
        
        # Reset levelname for other handlers
        record.levelname = levelname
        
        return formatted


class ColoredTracingFormatter(TracingFormatter, ColoredFormatter):
    """
    Combined formatter with both tracing and color support.
    """
    
    def format(self, record):
        """Format with both trace context and colors."""
        # First add trace context
        trace_id = TraceContext.get_trace_id()
        component = TraceContext.get_component()
        record.trace_id = trace_id if trace_id else "-"
        record.component = component if component else "-"
        
        # Then apply colors
        return ColoredFormatter.format(self, record)


def enable_colored_logging() -> None:
    """
    Enable colored logging for console output.
    """
    root_logger = logging.getLogger()
    
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            colored_formatter = ColoredFormatter(_LOG_FORMAT, _DATE_FORMAT)
            handler.setFormatter(colored_formatter)


def disable_third_party_logs() -> None:
    """
    Disable verbose logging from third-party libraries.
    """
    third_party_loggers = [
        "urllib3",
        "requests",
        "transformers",
        "torch",
        "tensorflow",
        "matplotlib",
        "PIL",
        "wandb",
        "datasets",
        "accelerate",
        "peft",
    ]
    
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def log_system_info(logger: Optional[logging.Logger] = None) -> None:
    """
    Log system information for debugging.
    
    Args:
        logger: Logger to use (default: root logger)
    """
    if logger is None:
        logger = logging.getLogger()
    
    import platform
    import sys
    
    logger.info("System Information:")
    logger.info(f"  Python: {sys.version}")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Processor: {platform.processor()}")
    
    try:
        import torch
        logger.info(f"  PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"  CUDA: {torch.version.cuda}")
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass
    
    try:
        import transformers
        logger.info(f"  Transformers: {transformers.__version__}")
    except ImportError:
        pass