"""
Redis-based robots.txt cache.

Provides caching for parsed robots.txt files with TTL support.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import redis.asyncio as redis

from crawler.compliance.robots_parser import RobotsTxt, RobotsParser
from crawler.utils.logging import CrawlerLogger
from crawler.utils import metrics


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    data: RobotsTxt
    cached_at: datetime
    expires_at: datetime
    hits: int = 0


class RobotsCache:
    """
    Redis-based cache for robots.txt files.

    Features:
    - TTL-based expiration
    - Serialization/deserialization of RobotsTxt objects
    - Cache hit/miss tracking
    """

    KEY_PREFIX = "crawler:robots:"

    def __init__(
        self,
        redis_client: redis.Redis,
        ttl_seconds: int = 86400,  # 24 hours
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the robots cache.

        Args:
            redis_client: Redis async client.
            ttl_seconds: Cache TTL in seconds.
            logger: Logger instance.
        """
        self.redis = redis_client
        self.ttl = ttl_seconds
        self.logger = logger or CrawlerLogger("robots_cache")
        self._parser = RobotsParser()

    def _key(self, domain: str) -> str:
        """Generate Redis key for a domain."""
        return f"{self.KEY_PREFIX}{domain.lower()}"

    async def get(self, domain: str) -> RobotsTxt | None:
        """
        Get cached robots.txt for a domain.

        Args:
            domain: The domain to look up.

        Returns:
            Parsed RobotsTxt or None if not cached.
        """
        key = self._key(domain)

        try:
            data = await self.redis.get(key)
            if data is None:
                metrics.REDIS_OPERATIONS.labels(
                    operation="get", status="miss"
                ).inc()
                return None

            metrics.REDIS_OPERATIONS.labels(
                operation="get", status="hit"
            ).inc()

            parsed = json.loads(data)
            return self._deserialize(parsed, domain)

        except redis.RedisError as e:
            self.logger.error(
                "Redis error getting robots.txt",
                domain=domain,
                error=str(e),
            )
            metrics.REDIS_OPERATIONS.labels(
                operation="get", status="error"
            ).inc()
            return None

    async def set(
        self,
        domain: str,
        robots_txt: RobotsTxt,
        ttl: int | None = None,
    ) -> bool:
        """
        Cache robots.txt for a domain.

        Args:
            domain: The domain.
            robots_txt: Parsed RobotsTxt object.
            ttl: Optional custom TTL in seconds.

        Returns:
            True if successfully cached.
        """
        key = self._key(domain)
        ttl = ttl or self.ttl

        try:
            data = json.dumps(robots_txt.to_dict())
            await self.redis.setex(key, ttl, data)
            metrics.REDIS_OPERATIONS.labels(
                operation="set", status="success"
            ).inc()
            return True

        except redis.RedisError as e:
            self.logger.error(
                "Redis error caching robots.txt",
                domain=domain,
                error=str(e),
            )
            metrics.REDIS_OPERATIONS.labels(
                operation="set", status="error"
            ).inc()
            return False

    async def delete(self, domain: str) -> bool:
        """
        Delete cached robots.txt for a domain.

        Args:
            domain: The domain.

        Returns:
            True if successfully deleted.
        """
        key = self._key(domain)

        try:
            await self.redis.delete(key)
            metrics.REDIS_OPERATIONS.labels(
                operation="delete", status="success"
            ).inc()
            return True

        except redis.RedisError as e:
            self.logger.error(
                "Redis error deleting robots.txt",
                domain=domain,
                error=str(e),
            )
            metrics.REDIS_OPERATIONS.labels(
                operation="delete", status="error"
            ).inc()
            return False

    async def exists(self, domain: str) -> bool:
        """
        Check if robots.txt is cached for a domain.

        Args:
            domain: The domain.

        Returns:
            True if cached.
        """
        key = self._key(domain)

        try:
            return bool(await self.redis.exists(key))
        except redis.RedisError:
            return False

    async def get_or_fetch(
        self,
        domain: str,
        fetch_func,
    ) -> RobotsTxt:
        """
        Get from cache or fetch and cache.

        Args:
            domain: The domain.
            fetch_func: Async function to fetch robots.txt (returns content, status_code).

        Returns:
            Parsed RobotsTxt.
        """
        # Try cache first
        cached = await self.get(domain)
        if cached is not None:
            return cached

        # Fetch and parse
        content, status_code = await fetch_func(domain)
        robots_txt = self._parser.parse(content, domain, status_code)

        # Cache the result
        await self.set(domain, robots_txt)

        return robots_txt

    def _deserialize(self, data: dict[str, Any], domain: str) -> RobotsTxt:
        """Deserialize cached data to RobotsTxt."""
        from crawler.compliance.robots_parser import RobotsGroup, RobotsRule

        groups = []
        for g in data.get("groups", []):
            rules = [
                RobotsRule(pattern=r["pattern"], allow=r["allow"])
                for r in g.get("rules", [])
            ]
            group = RobotsGroup(
                user_agents=g.get("user_agents", []),
                rules=rules,
                crawl_delay=g.get("crawl_delay"),
            )
            groups.append(group)

        return RobotsTxt(
            domain=domain,
            groups=groups,
            sitemaps=data.get("sitemaps", []),
            fetch_status=data.get("fetch_status", 200),
        )

    async def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary of cache statistics.
        """
        try:
            keys = await self.redis.keys(f"{self.KEY_PREFIX}*")
            return {
                "cached_domains": len(keys),
                "key_prefix": self.KEY_PREFIX,
                "ttl_seconds": self.ttl,
            }
        except redis.RedisError:
            return {"error": "Failed to get cache stats"}

    async def clear(self) -> int:
        """
        Clear all cached robots.txt entries.

        Returns:
            Number of entries cleared.
        """
        try:
            keys = await self.redis.keys(f"{self.KEY_PREFIX}*")
            if keys:
                await self.redis.delete(*keys)
            return len(keys)
        except redis.RedisError as e:
            self.logger.error("Failed to clear robots cache", error=str(e))
            return 0
