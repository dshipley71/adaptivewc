"""Utility modules for the adaptive web crawler."""

from crawler.utils.logging import CrawlerLogger, get_logger, setup_logging
from crawler.utils.url_utils import (
    get_domain,
    get_path,
    get_scheme,
    is_same_domain,
    is_valid_url,
    normalize_url,
    resolve_url,
)

__all__ = [
    "CrawlerLogger",
    "get_domain",
    "get_logger",
    "get_path",
    "get_scheme",
    "is_same_domain",
    "is_valid_url",
    "normalize_url",
    "resolve_url",
    "setup_logging",
]
