"""
URL frontier storage using Redis.

Provides persistent URL queue with priority support and deduplication.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import redis.asyncio as redis

from crawler.utils.logging import CrawlerLogger
from crawler.utils.url_utils import get_domain, normalize_url
from crawler.utils import metrics


@dataclass
class URLEntry:
    """Entry in the URL frontier."""

    url: str
    domain: str
    depth: int = 0
    priority: float = 0.5
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    parent_url: str | None = None
    retries: int = 0
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "domain": self.domain,
            "depth": self.depth,
            "priority": self.priority,
            "discovered_at": self.discovered_at.isoformat(),
            "parent_url": self.parent_url,
            "retries": self.retries,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "URLEntry":
        """Create from dictionary."""
        discovered_at = data.get("discovered_at")
        if isinstance(discovered_at, str):
            discovered_at = datetime.fromisoformat(discovered_at)
        else:
            discovered_at = datetime.utcnow()

        return cls(
            url=data["url"],
            domain=data["domain"],
            depth=data.get("depth", 0),
            priority=data.get("priority", 0.5),
            discovered_at=discovered_at,
            parent_url=data.get("parent_url"),
            retries=data.get("retries", 0),
            last_error=data.get("last_error"),
        )


class URLStore:
    """
    Redis-based URL frontier storage.

    Features:
    - Priority queue per domain
    - URL deduplication (visited set)
    - Domain-level queuing for politeness
    - Persistent state for crash recovery
    """

    # Redis key prefixes
    QUEUE_PREFIX = "crawler:queue:"
    SEEN_PREFIX = "crawler:seen:"
    DOMAINS_KEY = "crawler:domains"
    STATS_KEY = "crawler:stats"

    def __init__(
        self,
        redis_client: redis.Redis,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the URL store.

        Args:
            redis_client: Redis async client.
            logger: Logger instance.
        """
        self.redis = redis_client
        self.logger = logger or CrawlerLogger("url_store")

    def _queue_key(self, domain: str) -> str:
        """Get Redis key for domain queue."""
        return f"{self.QUEUE_PREFIX}{domain.lower()}"

    def _seen_key(self, domain: str) -> str:
        """Get Redis key for domain's seen URLs."""
        return f"{self.SEEN_PREFIX}{domain.lower()}"

    async def add(
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
            priority: Priority (0-1, higher = more important).
            parent_url: URL that linked to this one.

        Returns:
            True if URL was added (not a duplicate).
        """
        # Normalize URL for deduplication
        normalized = normalize_url(url)
        domain = get_domain(url)

        # Check if already seen
        seen_key = self._seen_key(domain)
        if await self.redis.sismember(seen_key, normalized):
            return False

        # Add to seen set
        await self.redis.sadd(seen_key, normalized)

        # Create entry
        entry = URLEntry(
            url=url,
            domain=domain,
            depth=depth,
            priority=priority,
            parent_url=parent_url,
        )

        # Add to domain queue (sorted set, score = priority * -1 for highest first)
        queue_key = self._queue_key(domain)
        score = -priority + (depth * 0.001)  # Prioritize lower depth
        await self.redis.zadd(queue_key, {json.dumps(entry.to_dict()): score})

        # Track active domain
        await self.redis.sadd(self.DOMAINS_KEY, domain)

        # Update stats
        await self.redis.hincrby(self.STATS_KEY, "total_added", 1)

        return True

    async def add_many(
        self,
        urls: list[tuple[str, int, float, str | None]],
    ) -> int:
        """
        Add multiple URLs to the frontier.

        Args:
            urls: List of (url, depth, priority, parent_url) tuples.

        Returns:
            Number of URLs actually added.
        """
        added = 0
        for url, depth, priority, parent in urls:
            if await self.add(url, depth, priority, parent):
                added += 1
        return added

    async def pop(self, domain: str) -> URLEntry | None:
        """
        Pop the highest priority URL for a domain.

        Args:
            domain: Domain to pop URL from.

        Returns:
            URLEntry or None if queue is empty.
        """
        queue_key = self._queue_key(domain)

        # Get and remove highest priority item
        result = await self.redis.zpopmin(queue_key, 1)
        if not result:
            return None

        data_str, _score = result[0]
        data = json.loads(data_str)

        # Update stats
        await self.redis.hincrby(self.STATS_KEY, "total_popped", 1)

        return URLEntry.from_dict(data)

    async def peek(self, domain: str) -> URLEntry | None:
        """
        Peek at the highest priority URL without removing.

        Args:
            domain: Domain to peek.

        Returns:
            URLEntry or None if queue is empty.
        """
        queue_key = self._queue_key(domain)

        result = await self.redis.zrange(queue_key, 0, 0)
        if not result:
            return None

        data = json.loads(result[0])
        return URLEntry.from_dict(data)

    async def get_next_domain(self) -> str | None:
        """
        Get the next domain that has URLs to crawl.

        Returns:
            Domain name or None if all queues are empty.
        """
        domains = await self.redis.smembers(self.DOMAINS_KEY)

        for domain in domains:
            domain_str = domain.decode() if isinstance(domain, bytes) else domain
            queue_key = self._queue_key(domain_str)
            if await self.redis.zcard(queue_key) > 0:
                return domain_str

        return None

    async def is_seen(self, url: str) -> bool:
        """
        Check if a URL has been seen.

        Args:
            url: URL to check.

        Returns:
            True if URL has been seen.
        """
        normalized = normalize_url(url)
        domain = get_domain(url)
        seen_key = self._seen_key(domain)
        return await self.redis.sismember(seen_key, normalized)

    async def mark_seen(self, url: str) -> None:
        """
        Mark a URL as seen without adding to queue.

        Args:
            url: URL to mark.
        """
        normalized = normalize_url(url)
        domain = get_domain(url)
        seen_key = self._seen_key(domain)
        await self.redis.sadd(seen_key, normalized)

    async def requeue(
        self,
        entry: URLEntry,
        error: str | None = None,
        max_retries: int = 3,
    ) -> bool:
        """
        Requeue a URL after a failure.

        Args:
            entry: URL entry to requeue.
            error: Error message.
            max_retries: Maximum retry attempts.

        Returns:
            True if requeued, False if max retries exceeded.
        """
        if entry.retries >= max_retries:
            await self.redis.hincrby(self.STATS_KEY, "total_failed", 1)
            return False

        entry.retries += 1
        entry.last_error = error

        # Lower priority for retries
        new_priority = entry.priority * (0.5 ** entry.retries)

        queue_key = self._queue_key(entry.domain)
        score = -new_priority + (entry.depth * 0.001)
        await self.redis.zadd(queue_key, {json.dumps(entry.to_dict()): score})

        return True

    async def get_queue_size(self, domain: str) -> int:
        """
        Get the queue size for a domain.

        Args:
            domain: Domain to check.

        Returns:
            Number of URLs in queue.
        """
        queue_key = self._queue_key(domain)
        return await self.redis.zcard(queue_key)

    async def get_total_queue_size(self) -> int:
        """
        Get the total queue size across all domains.

        Returns:
            Total number of URLs in all queues.
        """
        total = 0
        domains = await self.redis.smembers(self.DOMAINS_KEY)

        for domain in domains:
            domain_str = domain.decode() if isinstance(domain, bytes) else domain
            total += await self.get_queue_size(domain_str)

        return total

    async def get_active_domains(self) -> list[str]:
        """
        Get list of domains with URLs in queue.

        Returns:
            List of domain names.
        """
        domains = await self.redis.smembers(self.DOMAINS_KEY)
        active = []

        for domain in domains:
            domain_str = domain.decode() if isinstance(domain, bytes) else domain
            if await self.get_queue_size(domain_str) > 0:
                active.append(domain_str)

        return active

    async def get_stats(self) -> dict[str, Any]:
        """
        Get frontier statistics.

        Returns:
            Dictionary of statistics.
        """
        stats = await self.redis.hgetall(self.STATS_KEY)
        domains = await self.get_active_domains()
        total_queue = await self.get_total_queue_size()

        return {
            "total_added": int(stats.get(b"total_added", 0)),
            "total_popped": int(stats.get(b"total_popped", 0)),
            "total_failed": int(stats.get(b"total_failed", 0)),
            "queue_size": total_queue,
            "active_domains": len(domains),
            "domains": domains,
        }

    async def clear(self) -> None:
        """Clear all frontier data."""
        # Get all keys
        queue_keys = await self.redis.keys(f"{self.QUEUE_PREFIX}*")
        seen_keys = await self.redis.keys(f"{self.SEEN_PREFIX}*")

        # Delete all
        if queue_keys:
            await self.redis.delete(*queue_keys)
        if seen_keys:
            await self.redis.delete(*seen_keys)
        await self.redis.delete(self.DOMAINS_KEY, self.STATS_KEY)

        self.logger.info("URL store cleared")

    async def export_state(self) -> dict[str, Any]:
        """
        Export frontier state for backup.

        Returns:
            State dictionary.
        """
        state = {
            "stats": await self.get_stats(),
            "queues": {},
            "seen": {},
        }

        domains = await self.redis.smembers(self.DOMAINS_KEY)
        for domain in domains:
            domain_str = domain.decode() if isinstance(domain, bytes) else domain

            # Export queue
            queue_key = self._queue_key(domain_str)
            queue_data = await self.redis.zrange(queue_key, 0, -1, withscores=True)
            state["queues"][domain_str] = [
                {"entry": json.loads(e), "score": s}
                for e, s in queue_data
            ]

            # Export seen URLs
            seen_key = self._seen_key(domain_str)
            seen = await self.redis.smembers(seen_key)
            state["seen"][domain_str] = [
                s.decode() if isinstance(s, bytes) else s
                for s in seen
            ]

        return state
