"""
Redis-based storage for page structures and extraction strategies.

Provides persistent storage for adaptive extraction components.
"""

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any

import redis.asyncio as redis

from crawler.models import PageStructure, ExtractionStrategy, ContentRegion, PaginationInfo, IframeInfo
from crawler.utils.logging import CrawlerLogger
from crawler.utils import metrics


class StructureStore:
    """
    Redis-based storage for page structures and extraction strategies.

    Stores structure fingerprints and learned extraction strategies
    for each domain/page_type combination.
    """

    # Redis key prefixes
    STRUCTURE_PREFIX = "crawler:structure:"
    STRATEGY_PREFIX = "crawler:strategy:"
    HISTORY_PREFIX = "crawler:structure_history:"

    # Default TTL (7 days)
    DEFAULT_TTL = 604800

    def __init__(
        self,
        redis_client: redis.Redis,
        ttl: int = DEFAULT_TTL,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the structure store.

        Args:
            redis_client: Redis async client.
            ttl: Time-to-live for stored structures in seconds.
            logger: Logger instance.
        """
        self.redis = redis_client
        self.ttl = ttl
        self.logger = logger or CrawlerLogger("structure_store")

    def _structure_key(self, domain: str, page_type: str) -> str:
        """Get Redis key for a structure."""
        return f"{self.STRUCTURE_PREFIX}{domain}:{page_type}"

    def _strategy_key(self, domain: str, page_type: str) -> str:
        """Get Redis key for a strategy."""
        return f"{self.STRATEGY_PREFIX}{domain}:{page_type}"

    def _history_key(self, domain: str, page_type: str) -> str:
        """Get Redis key for structure history."""
        return f"{self.HISTORY_PREFIX}{domain}:{page_type}"

    async def get_structure(
        self,
        domain: str,
        page_type: str,
    ) -> PageStructure | None:
        """
        Get stored structure for a domain/page_type.

        Args:
            domain: Domain name.
            page_type: Page type identifier.

        Returns:
            PageStructure or None if not found.
        """
        key = self._structure_key(domain, page_type)

        try:
            data = await self.redis.get(key)
            if not data:
                return None

            return self._deserialize_structure(json.loads(data))

        except Exception as e:
            self.logger.error(
                "Failed to get structure",
                domain=domain,
                page_type=page_type,
                error=str(e),
            )
            return None

    async def save_structure(
        self,
        structure: PageStructure,
        strategy: ExtractionStrategy | None = None,
    ) -> bool:
        """
        Save a page structure.

        Args:
            structure: PageStructure to save.
            strategy: Optional associated extraction strategy.

        Returns:
            True if saved successfully.
        """
        structure_key = self._structure_key(structure.domain, structure.page_type)

        try:
            # Serialize and save structure
            data = self._serialize_structure(structure)
            await self.redis.setex(structure_key, self.ttl, json.dumps(data))

            # Save strategy if provided
            if strategy:
                await self.save_strategy(strategy)

            # Add to history
            await self._add_to_history(structure)

            self.logger.debug(
                "Saved structure",
                domain=structure.domain,
                page_type=structure.page_type,
            )
            return True

        except Exception as e:
            self.logger.error(
                "Failed to save structure",
                domain=structure.domain,
                page_type=structure.page_type,
                error=str(e),
            )
            return False

    async def update_structure(
        self,
        domain: str,
        page_type: str,
        new_structure: PageStructure,
        new_strategy: ExtractionStrategy,
        change_reason: str,
    ) -> bool:
        """
        Update a structure after detecting changes.

        Args:
            domain: Domain name.
            page_type: Page type identifier.
            new_structure: Updated structure.
            new_strategy: Updated extraction strategy.
            change_reason: Reason for the update.

        Returns:
            True if updated successfully.
        """
        try:
            # Archive old structure to history
            old_structure = await self.get_structure(domain, page_type)
            if old_structure:
                await self._add_to_history(old_structure)

            # Save new structure and strategy
            new_structure.version = (old_structure.version + 1) if old_structure else 1
            await self.save_structure(new_structure, new_strategy)

            self.logger.info(
                "Updated structure",
                domain=domain,
                page_type=page_type,
                version=new_structure.version,
                reason=change_reason,
            )

            # Record metric
            metrics.record_structure_change(
                domain=domain,
                change_type="update",
                breaking=True,
            )

            return True

        except Exception as e:
            self.logger.error(
                "Failed to update structure",
                domain=domain,
                page_type=page_type,
                error=str(e),
            )
            return False

    async def get_strategy(
        self,
        domain: str,
        page_type: str,
    ) -> ExtractionStrategy | None:
        """
        Get stored extraction strategy.

        Args:
            domain: Domain name.
            page_type: Page type identifier.

        Returns:
            ExtractionStrategy or None if not found.
        """
        key = self._strategy_key(domain, page_type)

        try:
            data = await self.redis.get(key)
            if not data:
                return None

            return self._deserialize_strategy(json.loads(data))

        except Exception as e:
            self.logger.error(
                "Failed to get strategy",
                domain=domain,
                page_type=page_type,
                error=str(e),
            )
            return None

    async def save_strategy(
        self,
        strategy: ExtractionStrategy,
    ) -> bool:
        """
        Save an extraction strategy.

        Args:
            strategy: Strategy to save.

        Returns:
            True if saved successfully.
        """
        key = self._strategy_key(strategy.domain, strategy.page_type)

        try:
            data = self._serialize_strategy(strategy)
            await self.redis.setex(key, self.ttl, json.dumps(data))

            self.logger.debug(
                "Saved strategy",
                domain=strategy.domain,
                page_type=strategy.page_type,
                version=strategy.version,
            )
            return True

        except Exception as e:
            self.logger.error(
                "Failed to save strategy",
                domain=strategy.domain,
                page_type=strategy.page_type,
                error=str(e),
            )
            return False

    async def get_history(
        self,
        domain: str,
        page_type: str,
        limit: int = 10,
    ) -> list[PageStructure]:
        """
        Get structure change history.

        Args:
            domain: Domain name.
            page_type: Page type identifier.
            limit: Maximum entries to return.

        Returns:
            List of historical structures (newest first).
        """
        key = self._history_key(domain, page_type)

        try:
            # Get from sorted set (newest first)
            entries = await self.redis.zrevrange(key, 0, limit - 1)
            structures = []

            for entry in entries:
                data = json.loads(entry)
                structures.append(self._deserialize_structure(data))

            return structures

        except Exception as e:
            self.logger.error(
                "Failed to get history",
                domain=domain,
                page_type=page_type,
                error=str(e),
            )
            return []

    async def _add_to_history(self, structure: PageStructure) -> None:
        """Add structure to history."""
        key = self._history_key(structure.domain, structure.page_type)
        data = self._serialize_structure(structure)
        timestamp = structure.captured_at.timestamp()

        # Add to sorted set with timestamp as score
        await self.redis.zadd(key, {json.dumps(data): timestamp})

        # Trim to keep only last 50 entries
        await self.redis.zremrangebyrank(key, 0, -51)

        # Set TTL on history key
        await self.redis.expire(key, self.ttl * 4)  # Keep history longer

    async def delete(
        self,
        domain: str,
        page_type: str,
    ) -> bool:
        """
        Delete structure and strategy for a domain/page_type.

        Args:
            domain: Domain name.
            page_type: Page type identifier.

        Returns:
            True if deleted.
        """
        try:
            structure_key = self._structure_key(domain, page_type)
            strategy_key = self._strategy_key(domain, page_type)
            history_key = self._history_key(domain, page_type)

            await self.redis.delete(structure_key, strategy_key, history_key)
            return True

        except Exception as e:
            self.logger.error(
                "Failed to delete",
                domain=domain,
                page_type=page_type,
                error=str(e),
            )
            return False

    async def list_domains(self) -> list[str]:
        """Get list of domains with stored structures."""
        try:
            keys = await self.redis.keys(f"{self.STRUCTURE_PREFIX}*")
            domains = set()

            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                # Extract domain from key
                parts = key_str.replace(self.STRUCTURE_PREFIX, "").split(":")
                if parts:
                    domains.add(parts[0])

            return list(domains)

        except Exception as e:
            self.logger.error("Failed to list domains", error=str(e))
            return []

    def _serialize_structure(self, structure: PageStructure) -> dict[str, Any]:
        """Serialize PageStructure to dict."""
        data = {
            "domain": structure.domain,
            "page_type": structure.page_type,
            "url": structure.url,
            "captured_at": structure.captured_at.isoformat(),
            "version": structure.version,
            "tag_hierarchy_hash": structure.tag_hierarchy_hash,
            "content_hash": structure.content_hash,
            "tag_counts": structure.tag_counts,
            "css_class_counts": structure.css_class_counts,
            "id_attributes": structure.id_attributes,
            "semantic_landmarks": structure.semantic_landmarks,
            "navigation_selectors": structure.navigation_selectors,
            "script_signatures": structure.script_signatures,
        }

        # Serialize content regions
        if structure.content_regions:
            data["content_regions"] = [
                {
                    "selector": r.selector,
                    "region_type": r.region_type,
                    "confidence": r.confidence,
                }
                for r in structure.content_regions
            ]

        # Serialize iframes
        if structure.iframes:
            data["iframes"] = [
                {
                    "src": i.src,
                    "sandbox": i.sandbox,
                    "position": i.position,
                }
                for i in structure.iframes
            ]

        # Serialize pagination
        if structure.pagination:
            data["pagination"] = {
                "has_pagination": structure.pagination.has_pagination,
                "pattern_type": structure.pagination.pattern_type,
                "next_selector": structure.pagination.next_selector,
                "page_param": structure.pagination.page_param,
            }

        return data

    def _deserialize_structure(self, data: dict[str, Any]) -> PageStructure:
        """Deserialize dict to PageStructure."""
        # Parse datetime
        captured_at = data.get("captured_at")
        if isinstance(captured_at, str):
            captured_at = datetime.fromisoformat(captured_at)
        else:
            captured_at = datetime.utcnow()

        # Parse content regions
        content_regions = None
        if "content_regions" in data:
            content_regions = [
                ContentRegion(
                    selector=r["selector"],
                    region_type=r["region_type"],
                    confidence=r.get("confidence", 0.0),
                )
                for r in data["content_regions"]
            ]

        # Parse iframes
        iframes = None
        if "iframes" in data:
            iframes = [
                IframeInfo(
                    src=i["src"],
                    sandbox=i.get("sandbox"),
                    position=i.get("position"),
                )
                for i in data["iframes"]
            ]

        # Parse pagination
        pagination = None
        if "pagination" in data:
            p = data["pagination"]
            pagination = PaginationInfo(
                has_pagination=p.get("has_pagination", False),
                pattern_type=p.get("pattern_type"),
                next_selector=p.get("next_selector"),
                page_param=p.get("page_param"),
            )

        return PageStructure(
            domain=data["domain"],
            page_type=data["page_type"],
            url=data.get("url", ""),
            captured_at=captured_at,
            version=data.get("version", 1),
            tag_hierarchy_hash=data.get("tag_hierarchy_hash", ""),
            content_hash=data.get("content_hash", ""),
            tag_counts=data.get("tag_counts"),
            css_class_counts=data.get("css_class_counts"),
            id_attributes=data.get("id_attributes"),
            semantic_landmarks=data.get("semantic_landmarks"),
            content_regions=content_regions,
            navigation_selectors=data.get("navigation_selectors"),
            iframes=iframes,
            pagination=pagination,
            script_signatures=data.get("script_signatures"),
        )

    def _serialize_strategy(self, strategy: ExtractionStrategy) -> dict[str, Any]:
        """Serialize ExtractionStrategy to dict."""
        return {
            "domain": strategy.domain,
            "page_type": strategy.page_type,
            "selectors": strategy.selectors,
            "confidence": strategy.confidence,
            "version": strategy.version,
            "created_at": strategy.created_at.isoformat(),
            "last_validated": strategy.last_validated.isoformat() if strategy.last_validated else None,
        }

    def _deserialize_strategy(self, data: dict[str, Any]) -> ExtractionStrategy:
        """Deserialize dict to ExtractionStrategy."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.utcnow()

        last_validated = data.get("last_validated")
        if isinstance(last_validated, str):
            last_validated = datetime.fromisoformat(last_validated)

        return ExtractionStrategy(
            domain=data["domain"],
            page_type=data["page_type"],
            selectors=data.get("selectors", {}),
            confidence=data.get("confidence", 0.0),
            version=data.get("version", 1),
            created_at=created_at,
            last_validated=last_validated,
        )
