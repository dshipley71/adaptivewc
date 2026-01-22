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

from crawler.adaptive.change_detector import ChangeDetector
from crawler.adaptive.strategy_learner import StrategyLearner
from crawler.adaptive.structure_analyzer import StructureAnalyzer
from crawler.compliance.rate_limiter import RateLimiter
from crawler.compliance.robots_parser import RobotsChecker
from crawler.config import CrawlConfig, GDPRConfig, PIIHandlingConfig, RateLimitConfig
from crawler.core.fetcher import Fetcher, FetcherConfig
from crawler.core.scheduler import Scheduler, SchedulerConfig
from crawler.exceptions import StorageError
from crawler.extraction.link_extractor import LinkExtractor
from crawler.legal.cfaa_checker import CFAAChecker
from crawler.legal.pii_detector import PIIDetector
from crawler.models import FetchResult, FetchStatus
from crawler.storage.robots_cache import RobotsCache
from crawler.storage.structure_store import StructureStore
from crawler.storage.url_store import URLStore, URLEntry
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
    structures_learned: int = 0
    structures_adapted: int = 0
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
            "structures_learned": self.structures_learned,
            "structures_adapted": self.structures_adapted,
            "domains_crawled": len(self.domains_crawled),
            "structures_analyzed": self.structures_analyzed,
            "structure_changes_detected": self.structure_changes_detected,
        }


class Crawler:
    """
    Main crawler orchestrator.

    Coordinates fetching, extraction, and storage components
    to perform web crawling with compliance checks and
    adaptive structure learning.
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

        # Adaptive extraction components
        self._structure_analyzer: StructureAnalyzer | None = None
        self._change_detector: ChangeDetector | None = None
        self._strategy_learner: StrategyLearner | None = None
        self._structure_store: StructureStore | None = None

        # State
        self._running = False
        self._stats = CrawlerStats()
        self._output_dir = Path(config.output_dir)
        self._file_lock = asyncio.Lock()

        # Callbacks
        self._on_page_crawled: list[Callable] = []
        self._on_error: list[Callable] = []

    async def start(self) -> None:
        """Initialize all components."""
        self.logger.info("Starting crawler", output_dir=str(self._output_dir))

        # Ensure output directory exists
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.logger.error("Failed to create output directory", error=str(e))
            raise StorageError(f"Cannot create output directory: {e}")

        # Connect to Redis
        try:
            self._redis = redis.from_url(self.redis_url)
            # Test connection
            await self._redis.ping()
        except Exception as e:
            self.logger.error("Failed to connect to Redis", error=str(e))
            raise ConnectionError(f"Cannot connect to Redis at {self.redis_url}: {e}")

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

        # Initialize adaptive extraction components
        self._structure_analyzer = StructureAnalyzer(logger=self.logger)
        self._change_detector = ChangeDetector(logger=self.logger)
        self._strategy_learner = StrategyLearner(logger=self.logger)
        self._structure_store = StructureStore(
            redis_client=self._redis,
            logger=self.logger,
        )

        self._running = True
        self.logger.info("Crawler started")

    async def stop(self) -> None:
        """Stop the crawler and cleanup."""
        self._running = False

        try:
            if self._fetcher:
                await self._fetcher.stop()
        except Exception as e:
            self.logger.error("Error stopping fetcher", error=str(e))

        try:
            if self._redis:
                await self._redis.aclose()
        except Exception as e:
            self.logger.error("Error closing Redis connection", error=str(e))

        self._stats.finished_at = datetime.utcnow()
        self.logger.info("Crawler stopped", stats=self._stats.to_dict())

    def _get_blocklist_path(self) -> str | None:
        """Get path to legal blocklist if it exists."""
        paths = [
            "/etc/crawler/legal_blocklist.txt",
            str(self._output_dir / "legal_blocklist.txt"),
        ]
        for path in paths:
            try:
                if os.path.exists(path):
                    return path
            except OSError:
                continue
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
            try:
                await self._crawl_url(entry)
            except Exception as e:
                self.logger.error("Unexpected error crawling URL", url=entry.url, error=str(e))
                self._stats.pages_failed += 1

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

                # Perform adaptive structure analysis
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
        """
        Analyze page structure for adaptive extraction with variant support.

        Automatically detects structural variants within the same page type
        (e.g., video articles vs text articles on a news site).
        """
        if not self._structure_analyzer or not self._structure_store:
            return

        try:
            # Analyze current page structure
            page_type = self._classify_page_type(url)
            current_structure = self._structure_analyzer.analyze(html, url, page_type)

            # Find matching variant or create new one
            matching_variant, similarity = await self._structure_store.find_matching_variant(
                current_structure
            )

            if matching_variant:
                # Found matching variant - check if it needs updating
                stored_structure = await self._structure_store.get_structure(
                    domain, page_type, matching_variant
                )

                if stored_structure:
                    # Compare with stored structure for this variant
                    assert self._change_detector is not None
                    analysis = self._change_detector.detect_changes(
                        stored_structure, current_structure
                    )

                    if analysis.requires_relearning:
                        # Structure changed significantly - adapt strategy
                        assert self._strategy_learner is not None
                        old_strategy = await self._structure_store.get_strategy(
                            domain, page_type, matching_variant
                        )

                        if old_strategy:
                            new_strategy = self._strategy_learner.adapt(
                                old_strategy, current_structure, html
                            )
                            current_structure.variant_id = matching_variant
                            new_strategy.strategy.variant_id = matching_variant
                            await self._structure_store.save_structure(
                                current_structure, new_strategy.strategy, matching_variant
                            )
                            self._stats.structures_adapted += 1
                            self.logger.info(
                                "Adapted extraction strategy",
                                domain=domain,
                                page_type=page_type,
                                variant_id=matching_variant,
                                similarity=analysis.similarity_score,
                            )
            else:
                # No matching variant - learn new strategy with variant detection
                assert self._strategy_learner is not None
                learned = self._strategy_learner.infer(html, current_structure)

                # Save with automatic variant detection
                success, variant_id, is_new = await self._structure_store.save_with_variant_detection(
                    current_structure, learned.strategy
                )

                if success:
                    self._stats.structures_learned += 1
                    self.logger.info(
                        "Learned new extraction strategy",
                        domain=domain,
                        page_type=page_type,
                        variant_id=variant_id,
                        is_new_variant=is_new,
                        confidence=learned.confidence,
                    )

        except Exception as e:
            self.logger.debug("Structure analysis failed", url=url, error=str(e))

    def _classify_page_type(self, url: str) -> str:
        """Classify the page type from URL patterns."""
        url_lower = url.lower()

        if any(p in url_lower for p in ["/article", "/post", "/blog", "/news"]):
            return "article"
        elif any(p in url_lower for p in ["/product", "/item", "/shop"]):
            return "product"
        elif any(p in url_lower for p in ["/category", "/tag", "/archive"]):
            return "listing"
        elif url_lower.rstrip("/").endswith((".com", ".org", ".net", ".io")):
            return "homepage"
        else:
            return "general"

    async def _save_content(self, url: str, result: FetchResult) -> None:
        """Save crawled content to disk with proper error handling."""
        try:
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

            # Save metadata with file lock
            async with self._file_lock:
                try:
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                except IOError as e:
                    self.logger.error("Failed to save metadata", url=url, path=str(filepath), error=str(e))
                    return

            # Save HTML content
            if result.html:
                html_path = self._output_dir / f"{safe_name}.html"
                try:
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(result.html)
                except IOError as e:
                    self.logger.error("Failed to save HTML", url=url, path=str(html_path), error=str(e))

        except Exception as e:
            self.logger.error("Unexpected error saving content", url=url, error=str(e))

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
