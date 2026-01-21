"""
Adaptive Web Crawler

An intelligent, ethical web crawler with legal compliance, robots.txt respect,
rate limiting, and ML-based structure learning.
"""

__version__ = "0.1.0"

from crawler.config import CrawlerSettings, CrawlConfig, load_config
from crawler.exceptions import CrawlerError
from crawler.models import (
    ExtractionResult,
    ExtractedContent,
    FetchResult,
    FetchStatus,
    PageStructure,
)
from crawler.core.crawler import Crawler, run_crawl

__all__ = [
    "Crawler",
    "CrawlerError",
    "CrawlerSettings",
    "CrawlConfig",
    "ExtractionResult",
    "ExtractedContent",
    "FetchResult",
    "FetchStatus",
    "PageStructure",
    "load_config",
    "run_crawl",
]
