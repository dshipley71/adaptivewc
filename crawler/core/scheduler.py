"""
URL scheduler for the adaptive web crawler.

Manages the URL frontier and coordinates crawling.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from crawler.compliance.rate_limiter import RateLimiter
from crawler.storage.url_store import URLStore, URLEntry
from crawler.utils.logging import CrawlerLogger
from crawler.utils.url_utils import get_domain
from crawler.utils import metrics


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""

    max_depth: int = 10
    max_pages: int | None = None
    max_pages_per_domain: int | None = None
    allowed_domains: list[str] = field(default_factory=list)
    priority_boost_patterns: list[str] = field(default_factory=list)


@dataclass
class DomainStats:
    """Statistics for a single domain."""

    domain: str
    pages_crawled: int = 0
    pages_pending: int = 0
    pages_failed: int = 0
    last_crawl_time: datetime | None = None


class Scheduler:
    """
    URL scheduler that manages the crawl frontier.

    Features:
    - Domain-aware scheduling for politeness
    - Priority-based URL ordering
    - Depth limiting
    - Allowed domain filtering
    """

    def __init__(
        self,
        url_store: URLStore,
        rate_limiter: RateLimiter,
        config: SchedulerConfig | None = None,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the scheduler.

        Args:
            url_store: URL storage backend.
            rate_limiter: Rate limiter instance.
            config: Scheduler configuration.
            logger: Logger instance.
        """
        self.url_store = url_store
        self.rate_limiter = rate_limiter
        self.config = config or SchedulerConfig()
        self.logger = logger or CrawlerLogger("scheduler")

        self._domain_stats: dict[str, DomainStats] = {}
        self._total_crawled = 0
        self._active = False
        self._lock = asyncio.Lock()

    def _get_domain_stats(self, domain: str) -> DomainStats:
        """Get or create domain stats."""
        if domain not in self._domain_stats:
            self._domain_stats[domain] = DomainStats(domain=domain)
        return self._domain_stats[domain]

    async def add_seeds(self, urls: list[str]) -> int:
        """
        Add seed URLs to the frontier.

        Args:
            urls: List of seed URLs.

        Returns:
            Number of URLs added.
        """
        added = 0
        for url in urls:
            if await self.add_url(url, depth=0, priority=1.0):
                added += 1
        return added

    async def add_url(
        self,
        url: str,
        depth: int = 0,
        priority: float = 0.5,
        parent_url: str | None = None,
    ) -> bool:
        """
        Add a URL to the frontier.

        Args:
            url: URL to add.
            depth: Crawl depth.
            priority: Priority (0-1).
            parent_url: Parent URL.

        Returns:
            True if URL was added.
        """
        # Check depth limit
        if depth > self.config.max_depth:
            return False

        # Check allowed domains
        domain = get_domain(url)
        if self.config.allowed_domains:
            if not any(
                domain == d or domain.endswith(f".{d}")
                for d in self.config.allowed_domains
            ):
                return False

        # Check per-domain limit
        stats = self._get_domain_stats(domain)
        if self.config.max_pages_per_domain:
            if stats.pages_crawled >= self.config.max_pages_per_domain:
                return False

        # Adjust priority based on patterns
        priority = self._adjust_priority(url, priority, depth)

        # Add to store
        added = await self.url_store.add(
            url=url,
            depth=depth,
            priority=priority,
            parent_url=parent_url,
        )

        if added:
            stats.pages_pending += 1

        return added

    async def add_discovered_urls(
        self,
        urls: list[str],
        parent_url: str,
        parent_depth: int,
    ) -> int:
        """
        Add discovered URLs from a crawled page.

        Args:
            urls: List of discovered URLs.
            parent_url: URL they were discovered on.
            parent_depth: Depth of parent URL.

        Returns:
            Number of URLs added.
        """
        added = 0
        depth = parent_depth + 1

        for url in urls:
            if await self.add_url(
                url=url,
                depth=depth,
                priority=self._calculate_priority(url, depth),
                parent_url=parent_url,
            ):
                added += 1

        return added

    def _adjust_priority(
        self,
        url: str,
        base_priority: float,
        depth: int,
    ) -> float:
        """Adjust URL priority based on patterns and depth."""
        priority = base_priority

        # Boost for priority patterns
        for pattern in self.config.priority_boost_patterns:
            if pattern in url:
                priority = min(1.0, priority + 0.2)

        # Penalize deep URLs
        priority *= (1.0 - (depth * 0.05))

        return max(0.1, min(1.0, priority))

    def _calculate_priority(self, url: str, depth: int) -> float:
        """Calculate priority for a new URL."""
        base = 0.5

        # Higher priority for common important paths
        important_paths = ["/article", "/post", "/news", "/blog"]
        for path in important_paths:
            if path in url.lower():
                base = 0.7
                break

        return self._adjust_priority(url, base, depth)

    async def get_next(self) -> URLEntry | None:
        """
        Get the next URL to crawl.

        Returns:
            URLEntry or None if nothing to crawl.
        """
        # Check global limit
        if self.config.max_pages and self._total_crawled >= self.config.max_pages:
            return None

        # Get active domains
        domains = await self.url_store.get_active_domains()
        if not domains:
            return None

        # Find domain with available capacity
        for domain in domains:
            # Check per-domain limit
            stats = self._get_domain_stats(domain)
            if self.config.max_pages_per_domain:
                if stats.pages_crawled >= self.config.max_pages_per_domain:
                    continue

            # Try to get URL from this domain
            entry = await self.url_store.pop(domain)
            if entry:
                stats.pages_pending = max(0, stats.pages_pending - 1)
                return entry

        return None

    async def mark_completed(self, url: str) -> None:
        """
        Mark a URL as successfully crawled.

        Args:
            url: The crawled URL.
        """
        async with self._lock:
            self._total_crawled += 1
            domain = get_domain(url)
            stats = self._get_domain_stats(domain)
            stats.pages_crawled += 1
            stats.last_crawl_time = datetime.utcnow()

    async def mark_failed(
        self,
        entry: URLEntry,
        error: str,
        retry: bool = True,
    ) -> None:
        """
        Mark a URL as failed.

        Args:
            entry: The URL entry that failed.
            error: Error message.
            retry: Whether to retry the URL.
        """
        domain = get_domain(entry.url)
        stats = self._get_domain_stats(domain)

        if retry:
            requeued = await self.url_store.requeue(entry, error)
            if not requeued:
                stats.pages_failed += 1
        else:
            stats.pages_failed += 1

    async def is_empty(self) -> bool:
        """Check if the frontier is empty."""
        return await self.url_store.get_total_queue_size() == 0

    async def has_more(self) -> bool:
        """Check if there are more URLs to crawl."""
        # Check global limit
        if self.config.max_pages and self._total_crawled >= self.config.max_pages:
            return False

        # If queue is empty, nothing more to crawl
        if await self.is_empty():
            return False

        # If per-domain limit is set, check if any domain can still be crawled
        if self.config.max_pages_per_domain:
            domains = await self.url_store.get_active_domains()
            for domain in domains:
                stats = self._get_domain_stats(domain)
                if stats.pages_crawled < self.config.max_pages_per_domain:
                    # At least one domain can still be crawled
                    return True
            # All domains with pending URLs have hit their per-domain limit
            return False

        return True

    def get_stats(self) -> dict[str, Any]:
        """
        Get scheduler statistics.

        Returns:
            Dictionary of statistics.
        """
        return {
            "total_crawled": self._total_crawled,
            "domains": {
                d: {
                    "pages_crawled": s.pages_crawled,
                    "pages_pending": s.pages_pending,
                    "pages_failed": s.pages_failed,
                    "last_crawl": s.last_crawl_time.isoformat() if s.last_crawl_time else None,
                }
                for d, s in self._domain_stats.items()
            },
        }

    async def get_progress(self) -> dict[str, int]:
        """
        Get crawl progress.

        Returns:
            Progress dictionary.
        """
        queue_size = await self.url_store.get_total_queue_size()
        total_failed = sum(s.pages_failed for s in self._domain_stats.values())

        return {
            "pages_crawled": self._total_crawled,
            "pages_pending": queue_size,
            "pages_failed": total_failed,
            "domains_active": len(self._domain_stats),
        }

    async def clear(self) -> None:
        """Clear the frontier and reset stats."""
        await self.url_store.clear()
        self._domain_stats.clear()
        self._total_crawled = 0
