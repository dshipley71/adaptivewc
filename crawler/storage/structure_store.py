"""
Redis-based storage for page structures and extraction strategies.

Provides persistent storage for adaptive extraction components.
"""

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any

import redis.asyncio as redis

from crawler.models import (
    PageStructure,
    ExtractionStrategy,
    ContentRegion,
    PaginationInfo,
    IframeInfo,
    SelectorRule,
)
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
    DOMAINS_KEY = "crawler:structure_domains"

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

    async def get(
        self,
        domain: str,
        page_type: str,
    ) -> tuple[PageStructure | None, ExtractionStrategy | None]:
        """
        Get stored structure and strategy for a domain/page_type.

        Args:
            domain: Domain name.
            page_type: Page type identifier.

        Returns:
            Tuple of (PageStructure, ExtractionStrategy) or (None, None).
        """
        structure = await self.get_structure(domain, page_type)
        strategy = await self.get_strategy(domain, page_type)
        return structure, strategy

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

    async def save(
        self,
        domain: str,
        page_type: str,
        structure: PageStructure,
        strategy: ExtractionStrategy,
    ) -> bool:
        """
        Save a page structure and strategy.

        Args:
            domain: Domain name.
            page_type: Page type identifier.
            structure: PageStructure to save.
            strategy: ExtractionStrategy to save.

        Returns:
            True if saved successfully.
        """
        return await self.save_structure(structure, strategy)

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
                saved = await self.save_strategy(strategy)
                if not saved:
                    self.logger.warning("Failed to save strategy with structure")

            # Track domain/page_type combination
            await self.redis.sadd(
                self.DOMAINS_KEY,
                f"{structure.domain}:{structure.page_type}"
            )

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

    async def update(
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
            serialized = json.dumps(data)
            await self.redis.setex(key, self.ttl, serialized)

            self.logger.debug(
                "Saved strategy to Redis",
                key=key,
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

    async def get_version(
        self,
        domain: str,
        page_type: str,
        version: int,
    ) -> PageStructure | None:
        """
        Get a specific version of a structure from history.

        Args:
            domain: Domain name.
            page_type: Page type identifier.
            version: Version number to retrieve.

        Returns:
            PageStructure or None if not found.
        """
        history = await self.get_history(domain, page_type, limit=50)
        for structure in history:
            if structure.version == version:
                return structure
        return None

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
            await self.redis.srem(self.DOMAINS_KEY, f"{domain}:{page_type}")
            return True

        except Exception as e:
            self.logger.error(
                "Failed to delete",
                domain=domain,
                page_type=page_type,
                error=str(e),
            )
            return False

    async def list_domains(self) -> list[tuple[str, str]]:
        """
        Get list of (domain, page_type) pairs with stored structures.

        Returns:
            List of (domain, page_type) tuples.
        """
        try:
            members = await self.redis.smembers(self.DOMAINS_KEY)
            result = []

            for member in members:
                member_str = member.decode() if isinstance(member, bytes) else member
                parts = member_str.split(":", 1)
                if len(parts) == 2:
                    result.append((parts[0], parts[1]))

            return sorted(result)

        except Exception as e:
            self.logger.error("Failed to list domains", error=str(e))
            return []

    async def get_stats(self) -> dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary of statistics.
        """
        try:
            domains = await self.list_domains()
            structure_keys = await self.redis.keys(f"{self.STRUCTURE_PREFIX}*")
            strategy_keys = await self.redis.keys(f"{self.STRATEGY_PREFIX}*")

            # Get unique domains
            unique_domains = set(d[0] for d in domains)

            return {
                "tracked_domains": len(unique_domains),
                "tracked_page_types": len(domains),
                "total_structures": len(structure_keys),
                "total_strategies": len(strategy_keys),
                "ttl_seconds": self.ttl,
            }

        except Exception as e:
            self.logger.error("Failed to get stats", error=str(e))
            return {"error": str(e)}

    async def clear(self) -> int:
        """
        Clear all stored structures and strategies.

        Returns:
            Number of entries cleared.
        """
        try:
            structure_keys = await self.redis.keys(f"{self.STRUCTURE_PREFIX}*")
            strategy_keys = await self.redis.keys(f"{self.STRATEGY_PREFIX}*")
            history_keys = await self.redis.keys(f"{self.HISTORY_PREFIX}*")

            all_keys = structure_keys + strategy_keys + history_keys
            if all_keys:
                await self.redis.delete(*all_keys)
            await self.redis.delete(self.DOMAINS_KEY)

            return len(all_keys)

        except Exception as e:
            self.logger.error("Failed to clear", error=str(e))
            return 0

    def _serialize_structure(self, structure: PageStructure) -> dict[str, Any]:
        """Serialize PageStructure to dict."""
        data = {
            "domain": structure.domain,
            "page_type": structure.page_type,
            "url_pattern": structure.url_pattern,
            "captured_at": structure.captured_at.isoformat(),
            "version": structure.version,
            "content_hash": structure.content_hash,
            "tag_hierarchy": structure.tag_hierarchy,
            "css_class_map": structure.css_class_map,
            "id_attributes": list(structure.id_attributes),
            "semantic_landmarks": structure.semantic_landmarks,
            "navigation_selectors": structure.navigation_selectors,
            "script_signatures": structure.script_signatures,
        }

        # Serialize content regions
        if structure.content_regions:
            data["content_regions"] = [
                {
                    "name": r.name,
                    "primary_selector": r.primary_selector,
                    "fallback_selectors": r.fallback_selectors,
                    "content_type": r.content_type,
                    "confidence": r.confidence,
                }
                for r in structure.content_regions
            ]

        # Serialize iframes
        if structure.iframe_locations:
            data["iframe_locations"] = [
                {
                    "selector": i.selector,
                    "src_pattern": i.src_pattern,
                    "position": i.position,
                    "dimensions": i.dimensions,
                    "is_dynamic": i.is_dynamic,
                }
                for i in structure.iframe_locations
            ]

        # Serialize pagination
        if structure.pagination_pattern:
            data["pagination_pattern"] = {
                "next_selector": structure.pagination_pattern.next_selector,
                "prev_selector": structure.pagination_pattern.prev_selector,
                "page_number_selector": structure.pagination_pattern.page_number_selector,
                "pattern": structure.pagination_pattern.pattern,
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
        content_regions = []
        if "content_regions" in data:
            content_regions = [
                ContentRegion(
                    name=r.get("name", ""),
                    primary_selector=r.get("primary_selector", ""),
                    fallback_selectors=r.get("fallback_selectors", []),
                    content_type=r.get("content_type", "text"),
                    confidence=r.get("confidence", 0.0),
                )
                for r in data["content_regions"]
            ]

        # Parse iframes
        iframe_locations = []
        if "iframe_locations" in data:
            iframe_locations = [
                IframeInfo(
                    selector=i.get("selector", ""),
                    src_pattern=i.get("src_pattern", ""),
                    position=i.get("position", "content"),
                    dimensions=tuple(i["dimensions"]) if i.get("dimensions") else None,
                    is_dynamic=i.get("is_dynamic", False),
                )
                for i in data["iframe_locations"]
            ]

        # Parse pagination
        pagination_pattern = None
        if "pagination_pattern" in data:
            p = data["pagination_pattern"]
            pagination_pattern = PaginationInfo(
                next_selector=p.get("next_selector"),
                prev_selector=p.get("prev_selector"),
                page_number_selector=p.get("page_number_selector"),
                pattern=p.get("pattern"),
            )

        return PageStructure(
            domain=data["domain"],
            page_type=data["page_type"],
            url_pattern=data.get("url_pattern", ""),
            captured_at=captured_at,
            version=data.get("version", 1),
            content_hash=data.get("content_hash", ""),
            tag_hierarchy=data.get("tag_hierarchy", {}),
            css_class_map=data.get("css_class_map", {}),
            id_attributes=set(data.get("id_attributes", [])),
            semantic_landmarks=data.get("semantic_landmarks", {}),
            content_regions=content_regions,
            navigation_selectors=data.get("navigation_selectors", []),
            iframe_locations=iframe_locations,
            pagination_pattern=pagination_pattern,
            script_signatures=data.get("script_signatures", []),
        )

    def _serialize_strategy(self, strategy: ExtractionStrategy) -> dict[str, Any]:
        """Serialize ExtractionStrategy to dict."""
        data = {
            "domain": strategy.domain,
            "page_type": strategy.page_type,
            "version": strategy.version,
            "learned_at": strategy.learned_at.isoformat(),
            "learning_source": strategy.learning_source,
            "wait_for_selectors": strategy.wait_for_selectors,
            "required_fields": strategy.required_fields,
            "min_content_length": strategy.min_content_length,
            "confidence_scores": strategy.confidence_scores,
        }

        # Serialize selector rules
        if strategy.title:
            data["title"] = self._serialize_selector_rule(strategy.title)
        if strategy.content:
            data["content"] = self._serialize_selector_rule(strategy.content)
        if strategy.images:
            data["images"] = self._serialize_selector_rule(strategy.images)
        if strategy.links:
            data["links"] = self._serialize_selector_rule(strategy.links)

        if strategy.metadata:
            data["metadata"] = {
                k: self._serialize_selector_rule(v)
                for k, v in strategy.metadata.items()
            }

        if strategy.iframe_extraction:
            data["iframe_extraction"] = {
                k: self._serialize_selector_rule(v)
                for k, v in strategy.iframe_extraction.items()
            }

        return data

    def _serialize_selector_rule(self, rule: SelectorRule) -> dict[str, Any]:
        """Serialize a SelectorRule."""
        return {
            "primary": rule.primary,
            "fallbacks": rule.fallbacks,
            "extraction_method": rule.extraction_method,
            "attribute_name": rule.attribute_name,
            "post_processors": rule.post_processors,
            "confidence": rule.confidence,
        }

    def _deserialize_strategy(self, data: dict[str, Any]) -> ExtractionStrategy:
        """Deserialize dict to ExtractionStrategy."""
        learned_at = data.get("learned_at")
        if isinstance(learned_at, str):
            learned_at = datetime.fromisoformat(learned_at)
        else:
            learned_at = datetime.utcnow()

        # Deserialize selector rules
        title = None
        if "title" in data:
            title = self._deserialize_selector_rule(data["title"])

        content = None
        if "content" in data:
            content = self._deserialize_selector_rule(data["content"])

        images = None
        if "images" in data:
            images = self._deserialize_selector_rule(data["images"])

        links = None
        if "links" in data:
            links = self._deserialize_selector_rule(data["links"])

        metadata = {}
        if "metadata" in data:
            metadata = {
                k: self._deserialize_selector_rule(v)
                for k, v in data["metadata"].items()
            }

        iframe_extraction = {}
        if "iframe_extraction" in data:
            iframe_extraction = {
                k: self._deserialize_selector_rule(v)
                for k, v in data["iframe_extraction"].items()
            }

        return ExtractionStrategy(
            domain=data["domain"],
            page_type=data["page_type"],
            version=data.get("version", 1),
            title=title,
            content=content,
            metadata=metadata,
            images=images,
            links=links,
            wait_for_selectors=data.get("wait_for_selectors", []),
            iframe_extraction=iframe_extraction,
            required_fields=data.get("required_fields", ["title", "content"]),
            min_content_length=data.get("min_content_length", 100),
            learned_at=learned_at,
            learning_source=data.get("learning_source", "initial"),
            confidence_scores=data.get("confidence_scores", {}),
        )

    def _deserialize_selector_rule(self, data: dict[str, Any]) -> SelectorRule:
        """Deserialize a SelectorRule."""
        return SelectorRule(
            primary=data.get("primary", ""),
            fallbacks=data.get("fallbacks", []),
            extraction_method=data.get("extraction_method", "text"),
            attribute_name=data.get("attribute_name"),
            post_processors=data.get("post_processors", []),
            confidence=data.get("confidence", 0.0),
        )
