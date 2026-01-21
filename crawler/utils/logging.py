"""
Structured logging for the adaptive web crawler.

Provides JSON-formatted logging with context propagation.
"""

import logging
import sys
from datetime import datetime
from typing import Any

import structlog


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: str | None = None,
) -> None:
    """
    Configure structured logging for the crawler.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_type: Output format ('json' or 'console').
        log_file: Optional file path to write logs to.
    """
    # Set the log level
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        level=log_level,
        stream=sys.stdout,
        format="%(message)s",
    )

    # Determine processors based on format
    shared_processors: list[structlog.types.Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if format_type == "json":
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__).

    Returns:
        A bound structured logger.
    """
    return structlog.get_logger(name)


class CrawlerLogger:
    """
    Specialized logger for crawler operations with pre-defined event types.
    """

    def __init__(self, name: str = "crawler"):
        self._logger = get_logger(name)
        self._context: dict[str, Any] = {}

    def bind(self, **kwargs: Any) -> "CrawlerLogger":
        """Bind context to all subsequent log calls."""
        new_logger = CrawlerLogger.__new__(CrawlerLogger)
        new_logger._logger = self._logger.bind(**kwargs)
        new_logger._context = {**self._context, **kwargs}
        return new_logger

    def fetch_start(self, url: str, **kwargs: Any) -> None:
        """Log the start of a fetch operation."""
        self._logger.info(
            "fetch_start",
            event_type="fetch",
            url=url,
            **kwargs,
        )

    def fetch_success(
        self,
        url: str,
        status_code: int,
        duration_ms: float,
        content_length: int,
        **kwargs: Any,
    ) -> None:
        """Log a successful fetch."""
        self._logger.info(
            "fetch_success",
            event_type="fetch",
            url=url,
            status_code=status_code,
            duration_ms=duration_ms,
            content_length=content_length,
            **kwargs,
        )

    def fetch_blocked(
        self,
        url: str,
        reason: str,
        **kwargs: Any,
    ) -> None:
        """Log a blocked fetch."""
        self._logger.warning(
            "fetch_blocked",
            event_type="fetch",
            url=url,
            reason=reason,
            **kwargs,
        )

    def fetch_error(
        self,
        url: str,
        error: str,
        error_type: str,
        **kwargs: Any,
    ) -> None:
        """Log a fetch error."""
        self._logger.error(
            "fetch_error",
            event_type="fetch",
            url=url,
            error=error,
            error_type=error_type,
            **kwargs,
        )

    def robots_check(
        self,
        url: str,
        allowed: bool,
        directive: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log a robots.txt check."""
        self._logger.debug(
            "robots_check",
            event_type="compliance",
            url=url,
            allowed=allowed,
            directive=directive,
            **kwargs,
        )

    def rate_limit_wait(
        self,
        domain: str,
        delay_seconds: float,
        reason: str,
        **kwargs: Any,
    ) -> None:
        """Log rate limit waiting."""
        self._logger.debug(
            "rate_limit_wait",
            event_type="compliance",
            domain=domain,
            delay_seconds=delay_seconds,
            reason=reason,
            **kwargs,
        )

    def structure_change(
        self,
        domain: str,
        page_type: str,
        change_type: str,
        similarity: float,
        **kwargs: Any,
    ) -> None:
        """Log a detected structure change."""
        self._logger.warning(
            "structure_change",
            event_type="adaptive",
            domain=domain,
            page_type=page_type,
            change_type=change_type,
            similarity=similarity,
            **kwargs,
        )

    def pii_detected(
        self,
        url: str,
        pii_types: list[str],
        action: str,
        **kwargs: Any,
    ) -> None:
        """Log PII detection."""
        self._logger.warning(
            "pii_detected",
            event_type="legal",
            url=url,
            pii_types=pii_types,
            action=action,
            **kwargs,
        )

    def extraction_result(
        self,
        url: str,
        success: bool,
        strategy: str,
        fields_extracted: list[str],
        **kwargs: Any,
    ) -> None:
        """Log extraction result."""
        level = "info" if success else "warning"
        getattr(self._logger, level)(
            "extraction_result",
            event_type="extraction",
            url=url,
            success=success,
            strategy=strategy,
            fields_extracted=fields_extracted,
            **kwargs,
        )

    def crawl_progress(
        self,
        pages_crawled: int,
        pages_pending: int,
        pages_failed: int,
        domains_active: int,
        **kwargs: Any,
    ) -> None:
        """Log crawl progress."""
        self._logger.info(
            "crawl_progress",
            event_type="progress",
            pages_crawled=pages_crawled,
            pages_pending=pages_pending,
            pages_failed=pages_failed,
            domains_active=domains_active,
            **kwargs,
        )

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message."""
        self._logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message."""
        self._logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message."""
        self._logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message."""
        self._logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log a critical message."""
        self._logger.critical(message, **kwargs)
