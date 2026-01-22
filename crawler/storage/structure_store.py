"""
Redis-based storage for page structures and extraction strategies.

Provides persistent storage for adaptive extraction components.
Supports variant tracking to handle multiple structural templates
within the same page type (e.g., video articles vs text articles).
"""

import hashlib
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
    for each domain/page_type/variant combination.

    Variant Tracking:
        When multiple pages of the same type have different structures
        (e.g., video articles vs text articles on a news site), each
        distinct structure is stored as a separate variant.

        Storage keys:
        - crawler:structure:{domain}:{page_type}:{variant_id}
        - crawler:strategy:{domain}:{page_type}:{variant_id}
        - crawler:variants:{domain}:{page_type} -> set of variant_ids
    """

    # Redis key prefixes
    STRUCTURE_PREFIX = "crawler:structure:"
    STRATEGY_PREFIX = "crawler:strategy:"
    HISTORY_PREFIX = "crawler:structure_history:"
    VARIANTS_PREFIX = "crawler:variants:"
    DOMAINS_KEY = "crawler:structure_domains"

    # Default TTL (7 days)
    DEFAULT_TTL = 604800

    # Similarity threshold for matching variants (0-1)
    # Structures with similarity >= this threshold are considered the same variant
    VARIANT_SIMILARITY_THRESHOLD = 0.85

    def __init__(
        self,
        redis_client: redis.Redis,
        ttl: int = DEFAULT_TTL,
        logger: CrawlerLogger | None = None,
        variant_similarity_threshold: float = VARIANT_SIMILARITY_THRESHOLD,
    ):
        """
        Initialize the structure store.

        Args:
            redis_client: Redis async client.
            ttl: Time-to-live for stored structures in seconds.
            logger: Logger instance.
            variant_similarity_threshold: Similarity threshold for variant matching.
        """
        self.redis = redis_client
        self.ttl = ttl
        self.logger = logger or CrawlerLogger("structure_store")
        self.variant_similarity_threshold = variant_similarity_threshold

    def _structure_key(self, domain: str, page_type: str, variant_id: str = "default") -> str:
        """Get Redis key for a structure."""
        return f"{self.STRUCTURE_PREFIX}{domain}:{page_type}:{variant_id}"

    def _strategy_key(self, domain: str, page_type: str, variant_id: str = "default") -> str:
        """Get Redis key for a strategy."""
        return f"{self.STRATEGY_PREFIX}{domain}:{page_type}:{variant_id}"

    def _history_key(self, domain: str, page_type: str, variant_id: str = "default") -> str:
        """Get Redis key for structure history."""
        return f"{self.HISTORY_PREFIX}{domain}:{page_type}:{variant_id}"

    def _variants_key(self, domain: str, page_type: str) -> str:
        """Get Redis key for the set of variants."""
        return f"{self.VARIANTS_PREFIX}{domain}:{page_type}"

    # =========================================================================
    # Variant Tracking Methods
    # =========================================================================

    def compute_structure_fingerprint(self, structure: PageStructure) -> str:
        """
        Compute a fingerprint for a structure based on key structural elements.

        This fingerprint is used to generate variant IDs and for quick
        similarity comparisons.

        Args:
            structure: PageStructure to fingerprint.

        Returns:
            A short hash string representing the structure's key features.
        """
        # Extract key structural features
        tag_counts = structure.tag_hierarchy.get("tag_counts", {}) if structure.tag_hierarchy else {}

        # Create a feature vector of important structural elements
        features = [
            # Key semantic tags (presence and rough counts)
            f"article:{tag_counts.get('article', 0) > 0}",
            f"video:{tag_counts.get('video', 0) > 0}",
            f"iframe:{tag_counts.get('iframe', 0) > 0}",
            f"table:{tag_counts.get('table', 0) > 0}",
            f"form:{tag_counts.get('form', 0) > 0}",
            f"nav:{min(tag_counts.get('nav', 0), 5)}",  # Cap at 5
            # Content regions
            f"regions:{len(structure.content_regions or [])}",
            # Landmark presence
            f"landmarks:{','.join(sorted(structure.semantic_landmarks.keys())[:5]) if structure.semantic_landmarks else 'none'}",
            # Pagination
            f"pagination:{bool(structure.pagination_pattern)}",
            # Media richness
            f"img:{min(tag_counts.get('img', 0) // 5, 10)}",  # Buckets of 5, max 10
        ]

        # Create hash from features
        feature_str = "|".join(features)
        return hashlib.md5(feature_str.encode()).hexdigest()[:12]

    def compute_structure_similarity(
        self,
        struct1: PageStructure,
        struct2: PageStructure,
    ) -> float:
        """
        Compute similarity between two page structures.

        Uses a weighted combination of structural features to determine
        how similar two pages are.

        Args:
            struct1: First structure.
            struct2: Second structure.

        Returns:
            Similarity score between 0.0 (completely different) and 1.0 (identical).
        """
        scores = []
        weights = []

        # Tag hierarchy similarity (most important)
        tag_counts1 = struct1.tag_hierarchy.get("tag_counts", {}) if struct1.tag_hierarchy else {}
        tag_counts2 = struct2.tag_hierarchy.get("tag_counts", {}) if struct2.tag_hierarchy else {}

        if tag_counts1 or tag_counts2:
            all_tags = set(tag_counts1.keys()) | set(tag_counts2.keys())
            if all_tags:
                # Jaccard-like similarity for tags
                common_tags = set(tag_counts1.keys()) & set(tag_counts2.keys())
                tag_sim = len(common_tags) / len(all_tags)
                scores.append(tag_sim)
                weights.append(0.3)

                # Count similarity for common tags
                if common_tags:
                    count_diffs = []
                    for tag in common_tags:
                        c1, c2 = tag_counts1.get(tag, 0), tag_counts2.get(tag, 0)
                        max_c = max(c1, c2)
                        if max_c > 0:
                            count_diffs.append(1 - abs(c1 - c2) / max_c)
                    if count_diffs:
                        scores.append(sum(count_diffs) / len(count_diffs))
                        weights.append(0.2)

        # Content regions similarity
        regions1 = {r.name for r in (struct1.content_regions or [])}
        regions2 = {r.name for r in (struct2.content_regions or [])}
        if regions1 or regions2:
            all_regions = regions1 | regions2
            common_regions = regions1 & regions2
            region_sim = len(common_regions) / len(all_regions) if all_regions else 1.0
            scores.append(region_sim)
            weights.append(0.2)

        # Semantic landmarks similarity
        landmarks1 = set(struct1.semantic_landmarks.keys()) if struct1.semantic_landmarks else set()
        landmarks2 = set(struct2.semantic_landmarks.keys()) if struct2.semantic_landmarks else set()
        if landmarks1 or landmarks2:
            all_landmarks = landmarks1 | landmarks2
            common_landmarks = landmarks1 & landmarks2
            landmark_sim = len(common_landmarks) / len(all_landmarks) if all_landmarks else 1.0
            scores.append(landmark_sim)
            weights.append(0.15)

        # Pagination pattern match
        has_pagination1 = bool(struct1.pagination_pattern)
        has_pagination2 = bool(struct2.pagination_pattern)
        scores.append(1.0 if has_pagination1 == has_pagination2 else 0.5)
        weights.append(0.05)

        # Navigation complexity similarity
        nav_count1 = len(struct1.navigation_selectors or [])
        nav_count2 = len(struct2.navigation_selectors or [])
        max_nav = max(nav_count1, nav_count2)
        if max_nav > 0:
            nav_sim = 1 - abs(nav_count1 - nav_count2) / max_nav
            scores.append(nav_sim)
            weights.append(0.1)

        # Calculate weighted average
        if not scores:
            return 1.0  # No features to compare, assume same

        total_weight = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / total_weight

    async def find_matching_variant(
        self,
        structure: PageStructure,
    ) -> tuple[str | None, float]:
        """
        Find an existing variant that matches the given structure.

        Compares the structure against all existing variants for the same
        domain/page_type and returns the best match if similarity exceeds
        the threshold.

        Args:
            structure: Structure to find a match for.

        Returns:
            Tuple of (variant_id, similarity) or (None, 0.0) if no match.
        """
        domain = structure.domain
        page_type = structure.page_type

        variants = await self.get_all_variants(domain, page_type)

        if not variants:
            return None, 0.0

        best_match = None
        best_similarity = 0.0

        for variant_id in variants:
            existing = await self.get_structure(domain, page_type, variant_id)
            if existing:
                similarity = self.compute_structure_similarity(structure, existing)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = variant_id

        if best_similarity >= self.variant_similarity_threshold:
            return best_match, best_similarity

        return None, best_similarity

    async def get_all_variants(
        self,
        domain: str,
        page_type: str,
    ) -> list[str]:
        """
        Get all variant IDs for a domain/page_type.

        Args:
            domain: Domain name.
            page_type: Page type identifier.

        Returns:
            List of variant IDs.
        """
        variants_key = self._variants_key(domain, page_type)

        try:
            members = await self.redis.smembers(variants_key)
            return [
                m.decode() if isinstance(m, bytes) else m
                for m in members
            ]
        except Exception as e:
            self.logger.error(
                "Failed to get variants",
                domain=domain,
                page_type=page_type,
                error=str(e),
            )
            return []

    async def save_with_variant_detection(
        self,
        structure: PageStructure,
        strategy: ExtractionStrategy | None = None,
    ) -> tuple[bool, str, bool]:
        """
        Save a structure with automatic variant detection.

        If the structure matches an existing variant (based on similarity),
        updates that variant. Otherwise, creates a new variant.

        Args:
            structure: PageStructure to save.
            strategy: Optional associated extraction strategy.

        Returns:
            Tuple of (success, variant_id, is_new_variant).
        """
        domain = structure.domain
        page_type = structure.page_type

        # Check for matching variant
        matching_variant, similarity = await self.find_matching_variant(structure)

        if matching_variant:
            # Update existing variant
            variant_id = matching_variant
            is_new = False
            self.logger.debug(
                "Matched existing variant",
                domain=domain,
                page_type=page_type,
                variant_id=variant_id,
                similarity=f"{similarity:.2%}",
            )
        else:
            # Create new variant
            variant_id = self.compute_structure_fingerprint(structure)
            is_new = True
            self.logger.info(
                "Creating new variant",
                domain=domain,
                page_type=page_type,
                variant_id=variant_id,
            )

        # Update structure and strategy with variant ID
        structure.variant_id = variant_id
        if strategy:
            strategy.variant_id = variant_id

        # Save structure
        success = await self.save_structure(structure, strategy, variant_id)

        if success:
            # Track this variant
            variants_key = self._variants_key(domain, page_type)
            await self.redis.sadd(variants_key, variant_id)
            await self.redis.expire(variants_key, self.ttl)

        return success, variant_id, is_new

    async def get_variant_stats(
        self,
        domain: str,
        page_type: str,
    ) -> dict[str, Any]:
        """
        Get statistics about variants for a domain/page_type.

        Args:
            domain: Domain name.
            page_type: Page type identifier.

        Returns:
            Dictionary with variant statistics.
        """
        variants = await self.get_all_variants(domain, page_type)

        variant_info = []
        for variant_id in variants:
            structure = await self.get_structure(domain, page_type, variant_id)
            if structure:
                tag_counts = structure.tag_hierarchy.get("tag_counts", {}) if structure.tag_hierarchy else {}
                variant_info.append({
                    "variant_id": variant_id,
                    "url_pattern": structure.url_pattern,
                    "captured_at": structure.captured_at.isoformat(),
                    "version": structure.version,
                    "total_elements": sum(tag_counts.values()),
                    "has_video": tag_counts.get("video", 0) > 0,
                    "has_iframe": tag_counts.get("iframe", 0) > 0,
                    "has_table": tag_counts.get("table", 0) > 0,
                    "content_regions": len(structure.content_regions or []),
                })

        return {
            "domain": domain,
            "page_type": page_type,
            "variant_count": len(variants),
            "variants": variant_info,
        }

    # =========================================================================
    # Standard Get/Save Methods (with variant support)
    # =========================================================================

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
        variant_id: str = "default",
    ) -> PageStructure | None:
        """
        Get stored structure for a domain/page_type/variant.

        Args:
            domain: Domain name.
            page_type: Page type identifier.
            variant_id: Variant identifier (default: "default").

        Returns:
            PageStructure or None if not found.
        """
        key = self._structure_key(domain, page_type, variant_id)

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
                variant_id=variant_id,
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
        variant_id: str = "default",
    ) -> bool:
        """
        Save a page structure.

        Args:
            structure: PageStructure to save.
            strategy: Optional associated extraction strategy.
            variant_id: Variant identifier (default: "default").

        Returns:
            True if saved successfully.
        """
        structure_key = self._structure_key(structure.domain, structure.page_type, variant_id)

        try:
            # Serialize and save structure
            data = self._serialize_structure(structure)
            data["variant_id"] = variant_id  # Store variant_id in the data
            await self.redis.setex(structure_key, self.ttl, json.dumps(data))

            # Save strategy if provided
            if strategy:
                saved = await self.save_strategy(strategy, variant_id)
                if not saved:
                    self.logger.warning("Failed to save strategy with structure")

            # Track domain/page_type combination
            await self.redis.sadd(
                self.DOMAINS_KEY,
                f"{structure.domain}:{structure.page_type}"
            )

            # Track variant
            variants_key = self._variants_key(structure.domain, structure.page_type)
            await self.redis.sadd(variants_key, variant_id)
            await self.redis.expire(variants_key, self.ttl)

            # Add to history
            await self._add_to_history(structure, variant_id)

            self.logger.debug(
                "Saved structure",
                domain=structure.domain,
                page_type=structure.page_type,
                variant_id=variant_id,
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
        variant_id: str = "default",
    ) -> ExtractionStrategy | None:
        """
        Get stored extraction strategy.

        Args:
            domain: Domain name.
            page_type: Page type identifier.
            variant_id: Variant identifier (default: "default").

        Returns:
            ExtractionStrategy or None if not found.
        """
        key = self._strategy_key(domain, page_type, variant_id)

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
                variant_id=variant_id,
                error=str(e),
            )
            return None

    async def save_strategy(
        self,
        strategy: ExtractionStrategy,
        variant_id: str = "default",
    ) -> bool:
        """
        Save an extraction strategy.

        Args:
            strategy: Strategy to save.
            variant_id: Variant identifier (default: "default").

        Returns:
            True if saved successfully.
        """
        key = self._strategy_key(strategy.domain, strategy.page_type, variant_id)

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

    async def _add_to_history(
        self,
        structure: PageStructure,
        variant_id: str = "default",
    ) -> None:
        """Add structure to history."""
        key = self._history_key(structure.domain, structure.page_type, variant_id)
        data = self._serialize_structure(structure)
        data["variant_id"] = variant_id
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
