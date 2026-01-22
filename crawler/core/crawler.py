"""
Main crawler orchestrator for the adaptive web crawler.

Coordinates all components to perform the crawl.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import redis.asyncio as redis

from crawler.compliance.rate_limiter import RateLimiter
from crawler.compliance.robots_parser import RobotsChecker
from crawler.config import CrawlConfig, GDPRConfig, PIIHandlingConfig, RateLimitConfig
from crawler.core.fetcher import Fetcher, FetcherConfig
from crawler.core.scheduler import Scheduler, SchedulerConfig
from crawler.extraction.link_extractor import LinkExtractor
from crawler.legal.cfaa_checker import CFAAChecker
from crawler.legal.pii_detector import PIIDetector
from crawler.models import FetchResult, FetchStatus
from crawler.storage.robots_cache import RobotsCache
from crawler.storage.url_store import URLStore, URLEntry
from crawler.storage.factory import create_structure_store, AnyStructureStore
from crawler.adaptive.structure_analyzer import StructureAnalyzer
from crawler.utils.logging import CrawlerLogger, setup_logging
from crawler.utils import metrics


@dataclass
class CrawlerStats:
    """Statistics for the crawl."""

    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None
    pages_crawled: int = 0
    pages_failed: int = 0
    pages_blocked: int = 0
    bytes_downloaded: int = 0
    links_discovered: int = 0
    domains_crawled: set[str] = field(default_factory=set)
    structures_analyzed: int = 0
    structure_changes_detected: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": (
                (self.finished_at or datetime.utcnow()) - self.started_at
            ).total_seconds(),
            "pages_crawled": self.pages_crawled,
            "pages_failed": self.pages_failed,
            "pages_blocked": self.pages_blocked,
            "bytes_downloaded": self.bytes_downloaded,
            "links_discovered": self.links_discovered,
            "domains_crawled": len(self.domains_crawled),
            "structures_analyzed": self.structures_analyzed,
            "structure_changes_detected": self.structure_changes_detected,
        }


class Crawler:
    """
    Main crawler orchestrator.

    Coordinates fetching, extraction, and storage components
    to perform web crawling with compliance checks.
    """

    def __init__(
        self,
        config: CrawlConfig,
        redis_url: str = "redis://localhost:6379/0",
        user_agent: str = "AdaptiveCrawler/1.0",
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the crawler.

        Args:
            config: Crawl configuration.
            redis_url: Redis connection URL.
            user_agent: User agent string.
            logger: Logger instance.
        """
        self.config = config
        self.redis_url = redis_url
        self.user_agent = user_agent
        self.logger = logger or CrawlerLogger("crawler")

        # Components (initialized in start())
        self._redis: redis.Redis | None = None
        self._fetcher: Fetcher | None = None
        self._scheduler: Scheduler | None = None
        self._link_extractor: LinkExtractor | None = None
        self._structure_analyzer: StructureAnalyzer | None = None
        self._structure_store: AnyStructureStore | None = None

        # State
        self._running = False
        self._stats = CrawlerStats()
        self._output_dir = Path(config.output_dir)

        # Callbacks
        self._on_page_crawled: list[Callable] = []
        self._on_error: list[Callable] = []

    async def start(self) -> None:
        """Initialize all components."""
        self.logger.info("Starting crawler", output_dir=str(self._output_dir))

        # Ensure output directory exists
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Connect to Redis
        self._redis = redis.from_url(self.redis_url)

        # Initialize components
        rate_limiter = RateLimiter(
            config=self.config.rate_limit,
            logger=self.logger,
        )

        robots_cache = RobotsCache(
            redis_client=self._redis,
            logger=self.logger,
        )

        robots_checker = RobotsChecker(
            user_agent=self.user_agent,
        )

        cfaa_checker = CFAAChecker(
            user_agent=self.user_agent,
            blocklist_path=self._get_blocklist_path(),
            logger=self.logger,
        )

        pii_detector = PIIDetector(
            config=self.config.pii,
            logger=self.logger,
        ) if self.config.gdpr.enabled else None

        # Initialize fetcher
        self._fetcher = Fetcher(
            config=FetcherConfig(
                user_agent=self.user_agent,
                timeout_seconds=self.config.safety.request_timeout_seconds,
                max_redirects=self.config.security.max_redirects,
                max_content_size=int(self.config.safety.max_page_size_mb * 1024 * 1024),
                verify_ssl=self.config.security.verify_ssl,
                block_private_ips=self.config.security.block_private_ips,
            ),
            rate_limiter=rate_limiter,
            robots_checker=robots_checker,
            robots_cache=robots_cache,
            cfaa_checker=cfaa_checker,
            pii_detector=pii_detector,
            gdpr_config=self.config.gdpr,
            logger=self.logger,
        )
        await self._fetcher.start()

        # Initialize URL store
        url_store = URLStore(
            redis_client=self._redis,
            logger=self.logger,
        )

        # Initialize scheduler
        self._scheduler = Scheduler(
            url_store=url_store,
            rate_limiter=rate_limiter,
            config=SchedulerConfig(
                max_depth=self.config.max_depth,
                max_pages=self.config.max_pages,
                allowed_domains=self.config.allowed_domains,
            ),
            logger=self.logger,
        )

        # Initialize link extractor
        self._link_extractor = LinkExtractor(
            follow_external=False,
            allowed_domains=self.config.allowed_domains or None,
            exclude_patterns=self.config.exclude_patterns,
            logger=self.logger,
        )

        # Initialize structure analyzer and store
        self._structure_analyzer = StructureAnalyzer(logger=self.logger)
        self._structure_store = create_structure_store(
            redis_client=self._redis,
            config=self.config.structure_store,
            logger=self.logger,
        )

        self._running = True
        self.logger.info("Crawler started")

    async def stop(self) -> None:
        """Stop the crawler and cleanup."""
        self._running = False

        if self._fetcher:
            await self._fetcher.stop()

        if self._redis:
            await self._redis.aclose()

        self._stats.finished_at = datetime.utcnow()
        self.logger.info("Crawler stopped", stats=self._stats.to_dict())

    def _get_blocklist_path(self) -> str | None:
        """Get path to legal blocklist if it exists."""
        paths = [
            "/etc/crawler/legal_blocklist.txt",
            str(self._output_dir / "legal_blocklist.txt"),
        ]
        for path in paths:
            if os.path.exists(path):
                return path
        return None

    async def crawl(self) -> CrawlerStats:
        """
        Run the crawl.

        Returns:
            Crawl statistics.
        """
        if not self._running:
            await self.start()

        assert self._scheduler is not None
        assert self._fetcher is not None
        assert self._link_extractor is not None

        # Add seed URLs
        added = await self._scheduler.add_seeds(self.config.seed_urls)
        self.logger.info("Added seed URLs", count=added)

        # Main crawl loop
        while self._running and await self._scheduler.has_more():
            # Get next URL
            entry = await self._scheduler.get_next()
            if not entry:
                # No URLs available, wait a bit
                await asyncio.sleep(0.1)
                continue

            # Crawl the URL
            await self._crawl_url(entry)

            # Log progress periodically
            if self._stats.pages_crawled % 10 == 0:
                progress = await self._scheduler.get_progress()
                self.logger.crawl_progress(**progress)

        await self.stop()
        return self._stats

    async def _crawl_url(self, entry: URLEntry) -> None:
        """Crawl a single URL."""
        assert self._fetcher is not None
        assert self._scheduler is not None
        assert self._link_extractor is not None
        assert self._structure_analyzer is not None
        assert self._structure_store is not None

        url = entry.url

        # Fetch
        result = await self._fetcher.fetch(url)

        if result.is_success():
            # Success - extract links and save content
            self._stats.pages_crawled += 1
            self._stats.bytes_downloaded += len(result.content or b"")
            self._stats.domains_crawled.add(entry.domain)

            # Extract links
            if result.html:
                links = self._link_extractor.extract_urls(result.html, url)
                self._stats.links_discovered += len(links)

                # Add discovered URLs
                await self._scheduler.add_discovered_urls(
                    urls=links,
                    parent_url=url,
                    parent_depth=entry.depth,
                )

                # Analyze and store page structure
                await self._analyze_structure(url, result.html, entry.domain)

            # Save content
            await self._save_content(url, result)

            # Mark completed
            await self._scheduler.mark_completed(url)

            # Callbacks
            for callback in self._on_page_crawled:
                await self._safe_callback(callback, url, result)

        elif result.status in (
            FetchStatus.BLOCKED_ROBOTS,
            FetchStatus.BLOCKED_LEGAL,
            FetchStatus.BLOCKED_BOT_DETECTION,
            FetchStatus.BLOCKED_RATE_LIMIT,
        ):
            # Blocked - don't retry
            self._stats.pages_blocked += 1
            await self._scheduler.mark_failed(entry, result.error_message or "blocked", retry=False)

        else:
            # Error - maybe retry
            self._stats.pages_failed += 1
            retry = entry.retries < self.config.safety.max_retries
            await self._scheduler.mark_failed(entry, result.error_message or "error", retry=retry)

            # Error callbacks
            for callback in self._on_error:
                await self._safe_callback(callback, url, result)

    async def _analyze_structure(self, url: str, html: str, domain: str) -> None:
        """Analyze page structure and store if changed."""
        assert self._structure_analyzer is not None
        assert self._structure_store is not None

        try:
            # Analyze the page structure
            structure = self._structure_analyzer.analyze(html, url)
            self._stats.structures_analyzed += 1

            # Check if structure has changed
            if await self._structure_store.has_changed(structure, url):
                # Get previous structure for comparison logging
                previous = await self._structure_store.get_latest(domain, structure.page_type)
                
                if previous:
                    similarity = self._structure_analyzer.compare(previous, structure)
                    self.logger.info(
                        "Structure change detected",
                        domain=domain,
                        page_type=structure.page_type,
                        similarity=f"{similarity:.2%}",
                        previous_version=previous.version,
                    )
                    self._stats.structure_changes_detected += 1
                    
                    # Record metric
                    metrics.record_structure_change(
                        domain=domain,
                        change_type="content_change",
                        breaking=similarity < 0.5,
                    )
                else:
                    self.logger.debug(
                        "New structure captured",
                        domain=domain,
                        page_type=structure.page_type,
                    )

                # Save the new structure
                version = await self._structure_store.save(structure, url)
                self.logger.debug(
                    "Structure saved",
                    domain=domain,
                    page_type=structure.page_type,
                    version=version,
                )

        except Exception as e:
            self.logger.warning(
                "Structure analysis failed",
                url=url,
                error=str(e),
            )

    async def _save_content(self, url: str, result: FetchResult) -> None:
        """Save crawled content to disk."""
        # Create safe filename from URL
        safe_name = self._url_to_filename(url)
        filepath = self._output_dir / f"{safe_name}.json"

        data = {
            "url": url,
            "fetched_at": result.fetched_at.isoformat(),
            "status_code": result.status_code,
            "content_length": len(result.content or b""),
            "headers": result.headers,
        }

        # Save metadata
        async with asyncio.Lock():
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

        # Save HTML content
        if result.html:
            html_path = self._output_dir / f"{safe_name}.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(result.html)

    def _url_to_filename(self, url: str) -> str:
        """Convert URL to safe filename."""
        import hashlib
        # Use hash of URL for uniqueness
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        # Extract last path component
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path.rstrip("/").split("/")[-1] or "index"
        # Sanitize
        safe_path = "".join(c if c.isalnum() else "_" for c in path)[:50]
        return f"{safe_path}_{url_hash}"

    async def _safe_callback(self, callback: Callable, *args: Any) -> None:
        """Safely execute a callback."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            self.logger.error("Callback error", error=str(e))

    def on_page_crawled(self, callback: Callable) -> None:
        """Register a callback for when a page is crawled."""
        self._on_page_crawled.append(callback)

    def on_error(self, callback: Callable) -> None:
        """Register a callback for errors."""
        self._on_error.append(callback)

    def get_stats(self) -> dict[str, Any]:
        """Get current crawl statistics."""
        return self._stats.to_dict()

    async def __aenter__(self) -> "Crawler":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.stop()


async def run_crawl(
    seed_urls: list[str],
    output_dir: str,
    max_pages: int | None = None,
    max_depth: int = 10,
    allowed_domains: list[str] | None = None,
    redis_url: str = "redis://localhost:6379/0",
) -> CrawlerStats:
    """
    Convenience function to run a crawl.

    Args:
        seed_urls: Starting URLs.
        output_dir: Directory to save results.
        max_pages: Maximum pages to crawl.
        max_depth: Maximum crawl depth.
        allowed_domains: Allowed domains to crawl.
        redis_url: Redis connection URL.

    Returns:
        Crawl statistics.
    """
    config = CrawlConfig(
        seed_urls=seed_urls,
        output_dir=output_dir,
        max_pages=max_pages,
        max_depth=max_depth,
        allowed_domains=allowed_domains or [],
    )

    setup_logging(level="INFO", format_type="console")

    async with Crawler(config, redis_url=redis_url) as crawler:
        return await crawler.crawl()
