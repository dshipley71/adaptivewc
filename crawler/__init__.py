"""
Adaptive Web Crawler

An intelligent, ethical web crawler with legal compliance, robots.txt respect,
rate limiting, and ML-based structure learning.
"""

__version__ = "0.1.0"

from crawler.config import CrawlerSettings, load_config
from crawler.exceptions import CrawlerError
from crawler.models import (
    ExtractionResult,
    ExtractedContent,
    FetchResult,
    PageStructure,
)

__all__ = [
    "CrawlerError",
    "CrawlerSettings",
    "ExtractionResult",
    "ExtractedContent",
    "FetchResult",
    "PageStructure",
    "load_config",
]
