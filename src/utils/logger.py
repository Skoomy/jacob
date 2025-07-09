import logging
import sys
from pathlib import Path
from typing import Optional

import structlog


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_logs: bool = False,
    include_caller: bool = True,
) -> None:
    """Set up structured logging for the application."""

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if include_caller:
        processors.append(
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            )
        )

    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set up file logging if requested
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))

        if json_logs:
            file_formatter = logging.Formatter("%(message)s")
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class PerformanceLogger:
    """Context manager for performance monitoring."""

    def __init__(self, logger: structlog.stdlib.BoundLogger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time: Optional[float] = None

    def __enter__(self) -> "PerformanceLogger":
        import time

        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import time

        if self.start_time is not None:
            duration = time.time() - self.start_time
        else:
            duration = 0.0

        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation}",
                duration_seconds=round(duration, 3),
                status="success",
            )
        else:
            self.logger.error(
                f"Failed {self.operation}",
                duration_seconds=round(duration, 3),
                status="error",
                error_type=exc_type.__name__,
                error_message=str(exc_val),
            )


def log_function_call(func):
    """Decorator to log function calls with performance metrics."""
    import functools
    import time

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()

        logger.debug(
            f"Calling {func.__name__}",
            args_count=len(args),
            kwargs_keys=list(kwargs.keys()),
        )

        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            logger.debug(
                f"Completed {func.__name__}",
                duration_seconds=round(duration, 3),
                status="success",
            )

            return result

        except Exception as e:
            duration = time.time() - start_time

            logger.error(
                f"Failed {func.__name__}",
                duration_seconds=round(duration, 3),
                status="error",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

    return wrapper
