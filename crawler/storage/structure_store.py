"""
Redis-based storage for page structure fingerprints.

Provides persistent storage for PageStructure objects with versioning
and quick signature comparison for change detection.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import redis.asyncio as redis

from crawler.models import (
    ContentRegion,
    IframeInfo,
    PageStructure,
    PaginationInfo,
)
from crawler.utils.logging import CrawlerLogger
from crawler.utils import metrics


@dataclass
class StructureSignature:
    """Lightweight signature for quick change detection."""

    domain: str
    page_type: str
    version: int

    # Quick comparison hashes
    content_hash: str
    tag_count_hash: str
    class_set_hash: str
    landmark_hash: str

    # Metadata
    captured_at: datetime
    sample_url: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "domain": self.domain,
            "page_type": self.page_type,
            "version": self.version,
            "content_hash": self.content_hash,
            "tag_count_hash": self.tag_count_hash,
            "class_set_hash": self.class_set_hash,
            "landmark_hash": self.landmark_hash,
            "captured_at": self.captured_at.isoformat(),
            "sample_url": self.sample_url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructureSignature":
        """Create from dictionary."""
        captured_at = data.get("captured_at")
        if isinstance(captured_at, str):
            captured_at = datetime.fromisoformat(captured_at)
        else:
            captured_at = datetime.utcnow()

        return cls(
            domain=data["domain"],
            page_type=data["page_type"],
            version=data.get("version", 1),
            content_hash=data["content_hash"],
            tag_count_hash=data["tag_count_hash"],
            class_set_hash=data["class_set_hash"],
            landmark_hash=data["landmark_hash"],
            captured_at=captured_at,
            sample_url=data.get("sample_url", ""),
        )

    @classmethod
    def from_structure(cls, structure: PageStructure, url: str = "") -> "StructureSignature":
        """Create signature from full PageStructure."""
        # Hash tag counts
        tag_counts = structure.tag_hierarchy.get("tag_counts", {})
        tag_count_hash = hashlib.md5(
            json.dumps(sorted(tag_counts.items())).encode()
        ).hexdigest()[:16]

        # Hash class names (not counts, just presence)
        class_set_hash = hashlib.md5(
            json.dumps(sorted(structure.css_class_map.keys())).encode()
        ).hexdigest()[:16]

        # Hash landmarks
        landmark_hash = hashlib.md5(
            json.dumps(sorted(structure.semantic_landmarks.items())).encode()
        ).hexdigest()[:16]

        return cls(
            domain=structure.domain,
            page_type=structure.page_type,
            version=structure.version,
            content_hash=structure.content_hash[:16] if structure.content_hash else "",
            tag_count_hash=tag_count_hash,
            class_set_hash=class_set_hash,
            landmark_hash=landmark_hash,
            captured_at=structure.captured_at,
            sample_url=url,
        )

    def matches(self, other: "StructureSignature") -> bool:
        """Quick check if two signatures match."""
        return (
            self.content_hash == other.content_hash
            and self.tag_count_hash == other.tag_count_hash
            and self.class_set_hash == other.class_set_hash
        )


class StructureStore:
    """
    Redis-based storage for page structure fingerprints.

    Features:
    - Version history for structures
    - Quick signature comparison
    - TTL-based expiration
    - Domain/page_type indexing
    """

    # Redis key patterns
    STRUCTURE_PREFIX = "crawler:structure:"
    SIGNATURE_PREFIX = "crawler:sig:"
    VERSION_PREFIX = "crawler:structver:"
    INDEX_KEY = "crawler:structure:index"
    STATS_KEY = "crawler:structure:stats"

    def __init__(
        self,
        redis_client: redis.Redis,
        ttl_seconds: int = 604800,  # 7 days
        max_versions: int = 10,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the structure store.

        Args:
            redis_client: Redis async client.
            ttl_seconds: TTL for structure data.
            max_versions: Maximum versions to keep per domain/page_type.
            logger: Logger instance.
        """
        self.redis = redis_client
        self.ttl = ttl_seconds
        self.max_versions = max_versions
        self.logger = logger or CrawlerLogger("structure_store")

    def _structure_key(self, domain: str, page_type: str, version: int) -> str:
        """Get Redis key for a specific structure version."""
        return f"{self.STRUCTURE_PREFIX}{domain}:{page_type}:v{version}"

    def _latest_key(self, domain: str, page_type: str) -> str:
        """Get Redis key for latest version pointer."""
        return f"{self.VERSION_PREFIX}{domain}:{page_type}:latest"

    def _signature_key(self, domain: str, page_type: str) -> str:
        """Get Redis key for quick signature."""
        return f"{self.SIGNATURE_PREFIX}{domain}:{page_type}"

    def _index_entry(self, domain: str, page_type: str) -> str:
        """Create index entry string."""
        return f"{domain}:{page_type}"

    async def save(
        self,
        structure: PageStructure,
        url: str = "",
    ) -> int:
        """
        Save a page structure to Redis.

        Args:
            structure: PageStructure to save.
            url: Sample URL this structure was captured from.

        Returns:
            Version number assigned.
        """
        domain = structure.domain
        page_type = structure.page_type

        try:
            # Get current version
            latest_key = self._latest_key(domain, page_type)
            current_version = await self.redis.get(latest_key)
            if current_version:
                current_version = int(current_version)
            else:
                current_version = 0

            # Increment version
            new_version = current_version + 1
            structure.version = new_version

            # Save full structure
            structure_key = self._structure_key(domain, page_type, new_version)
            structure_data = json.dumps(structure.to_dict())
            await self.redis.setex(structure_key, self.ttl, structure_data)

            # Update latest pointer
            await self.redis.set(latest_key, str(new_version))

            # Save signature for quick comparison
            signature = StructureSignature.from_structure(structure, url)
            signature_key = self._signature_key(domain, page_type)
            await self.redis.setex(
                signature_key, self.ttl, json.dumps(signature.to_dict())
            )

            # Add to index
            await self.redis.sadd(self.INDEX_KEY, self._index_entry(domain, page_type))

            # Update stats
            await self.redis.hincrby(self.STATS_KEY, "total_saved", 1)
            await self.redis.hset(
                self.STATS_KEY, "last_save", datetime.utcnow().isoformat()
            )

            # Cleanup old versions
            await self._cleanup_old_versions(domain, page_type, new_version)

            self.logger.debug(
                "Saved structure",
                domain=domain,
                page_type=page_type,
                version=new_version,
            )

            metrics.REDIS_OPERATIONS.labels(operation="structure_save", status="success").inc()

            return new_version

        except redis.RedisError as e:
            self.logger.error(
                "Failed to save structure",
                domain=domain,
                page_type=page_type,
                error=str(e),
            )
            metrics.REDIS_OPERATIONS.labels(operation="structure_save", status="error").inc()
            raise

    async def get_latest(
        self,
        domain: str,
        page_type: str,
    ) -> PageStructure | None:
        """
        Get the most recent structure for a domain/page_type.

        Args:
            domain: Domain name.
            page_type: Page type.

        Returns:
            PageStructure or None if not found.
        """
        try:
            # Get latest version number
            latest_key = self._latest_key(domain, page_type)
            version = await self.redis.get(latest_key)
            if not version:
                return None

            version = int(version)
            return await self.get_version(domain, page_type, version)

        except redis.RedisError as e:
            self.logger.error(
                "Failed to get latest structure",
                domain=domain,
                page_type=page_type,
                error=str(e),
            )
            return None

    async def get_version(
        self,
        domain: str,
        page_type: str,
        version: int,
    ) -> PageStructure | None:
        """
        Get a specific version of a structure.

        Args:
            domain: Domain name.
            page_type: Page type.
            version: Version number.

        Returns:
            PageStructure or None if not found.
        """
        try:
            structure_key = self._structure_key(domain, page_type, version)
            data = await self.redis.get(structure_key)

            if not data:
                return None

            return self._deserialize_structure(json.loads(data))

        except redis.RedisError as e:
            self.logger.error(
                "Failed to get structure version",
                domain=domain,
                page_type=page_type,
                version=version,
                error=str(e),
            )
            return None

    async def get_signature(
        self,
        domain: str,
        page_type: str,
    ) -> StructureSignature | None:
        """
        Get lightweight signature for quick comparison.

        Args:
            domain: Domain name.
            page_type: Page type.

        Returns:
            StructureSignature or None if not found.
        """
        try:
            signature_key = self._signature_key(domain, page_type)
            data = await self.redis.get(signature_key)

            if not data:
                return None

            return StructureSignature.from_dict(json.loads(data))

        except redis.RedisError as e:
            self.logger.error(
                "Failed to get signature",
                domain=domain,
                page_type=page_type,
                error=str(e),
            )
            return None

    async def has_changed(
        self,
        structure: PageStructure,
        url: str = "",
    ) -> bool:
        """
        Quick check if a structure differs from the stored version.

        Args:
            structure: New structure to compare.
            url: URL the structure was captured from.

        Returns:
            True if structure has changed or no previous version exists.
        """
        stored_sig = await self.get_signature(structure.domain, structure.page_type)

        if stored_sig is None:
            return True  # No previous version, treat as "changed"

        new_sig = StructureSignature.from_structure(structure, url)
        return not stored_sig.matches(new_sig)

    async def get_history(
        self,
        domain: str,
        page_type: str,
        limit: int = 10,
    ) -> list[PageStructure]:
        """
        Get version history for a domain/page_type.

        Args:
            domain: Domain name.
            page_type: Page type.
            limit: Maximum versions to return.

        Returns:
            List of PageStructure objects, newest first.
        """
        try:
            # Get latest version
            latest_key = self._latest_key(domain, page_type)
            latest = await self.redis.get(latest_key)
            if not latest:
                return []

            latest_version = int(latest)
            structures = []

            # Fetch versions in reverse order
            for version in range(latest_version, max(0, latest_version - limit), -1):
                structure = await self.get_version(domain, page_type, version)
                if structure:
                    structures.append(structure)

            return structures

        except redis.RedisError as e:
            self.logger.error(
                "Failed to get structure history",
                domain=domain,
                page_type=page_type,
                error=str(e),
            )
            return []

    async def list_domains(self) -> list[tuple[str, str]]:
        """
        List all tracked domain/page_type pairs.

        Returns:
            List of (domain, page_type) tuples.
        """
        try:
            entries = await self.redis.smembers(self.INDEX_KEY)
            result = []

            for entry in entries:
                entry_str = entry.decode() if isinstance(entry, bytes) else entry
                parts = entry_str.split(":", 1)
                if len(parts) == 2:
                    result.append((parts[0], parts[1]))

            return sorted(result)

        except redis.RedisError as e:
            self.logger.error("Failed to list domains", error=str(e))
            return []

    async def get_stats(self) -> dict[str, Any]:
        """
        Get store statistics.

        Returns:
            Dictionary of statistics.
        """
        try:
            stats = await self.redis.hgetall(self.STATS_KEY)
            domains = await self.list_domains()

            return {
                "total_saved": int(stats.get(b"total_saved", 0)),
                "last_save": stats.get(b"last_save", b"").decode(),
                "tracked_domains": len(set(d[0] for d in domains)),
                "tracked_page_types": len(domains),
                "ttl_seconds": self.ttl,
                "max_versions": self.max_versions,
            }

        except redis.RedisError as e:
            self.logger.error("Failed to get stats", error=str(e))
            return {"error": str(e)}

    async def delete(
        self,
        domain: str,
        page_type: str,
    ) -> bool:
        """
        Delete all structure data for a domain/page_type.

        Args:
            domain: Domain name.
            page_type: Page type.

        Returns:
            True if deleted successfully.
        """
        try:
            # Get all version keys
            latest_key = self._latest_key(domain, page_type)
            latest = await self.redis.get(latest_key)

            keys_to_delete = [
                latest_key,
                self._signature_key(domain, page_type),
            ]

            if latest:
                latest_version = int(latest)
                for version in range(1, latest_version + 1):
                    keys_to_delete.append(
                        self._structure_key(domain, page_type, version)
                    )

            # Delete all keys
            if keys_to_delete:
                await self.redis.delete(*keys_to_delete)

            # Remove from index
            await self.redis.srem(
                self.INDEX_KEY, self._index_entry(domain, page_type)
            )

            return True

        except redis.RedisError as e:
            self.logger.error(
                "Failed to delete structure",
                domain=domain,
                page_type=page_type,
                error=str(e),
            )
            return False

    async def clear(self) -> int:
        """
        Clear all structure data.

        Returns:
            Number of entries cleared.
        """
        try:
            # Get all keys
            structure_keys = await self.redis.keys(f"{self.STRUCTURE_PREFIX}*")
            signature_keys = await self.redis.keys(f"{self.SIGNATURE_PREFIX}*")
            version_keys = await self.redis.keys(f"{self.VERSION_PREFIX}*")

            all_keys = structure_keys + signature_keys + version_keys
            if all_keys:
                await self.redis.delete(*all_keys)

            await self.redis.delete(self.INDEX_KEY, self.STATS_KEY)

            count = len(all_keys)
            self.logger.info("Cleared structure store", entries=count)
            return count

        except redis.RedisError as e:
            self.logger.error("Failed to clear structure store", error=str(e))
            return 0

    async def _cleanup_old_versions(
        self,
        domain: str,
        page_type: str,
        current_version: int,
    ) -> None:
        """Remove old versions beyond max_versions limit."""
        if current_version <= self.max_versions:
            return

        try:
            oldest_to_keep = current_version - self.max_versions + 1
            for version in range(1, oldest_to_keep):
                key = self._structure_key(domain, page_type, version)
                await self.redis.delete(key)

        except redis.RedisError as e:
            self.logger.warning(
                "Failed to cleanup old versions",
                domain=domain,
                page_type=page_type,
                error=str(e),
            )

    def _deserialize_structure(self, data: dict[str, Any]) -> PageStructure:
        """Deserialize structure from dictionary."""
        # Parse captured_at
        captured_at = data.get("captured_at")
        if isinstance(captured_at, str):
            captured_at = datetime.fromisoformat(captured_at)
        else:
            captured_at = datetime.utcnow()

        # Parse iframe_locations
        iframe_locations = [
            IframeInfo(
                selector=i["selector"],
                src_pattern=i["src_pattern"],
                position=i["position"],
                dimensions=tuple(i["dimensions"]) if i.get("dimensions") else None,
                is_dynamic=i.get("is_dynamic", False),
            )
            for i in data.get("iframe_locations", [])
        ]

        # Parse content_regions
        content_regions = [
            ContentRegion(
                name=r["name"],
                primary_selector=r["primary_selector"],
                fallback_selectors=r.get("fallback_selectors", []),
                content_type=r.get("content_type", "text"),
                confidence=r.get("confidence", 0.0),
            )
            for r in data.get("content_regions", [])
        ]

        # Parse pagination_pattern
        pagination_data = data.get("pagination_pattern")
        pagination_pattern = None
        if pagination_data:
            pagination_pattern = PaginationInfo(
                next_selector=pagination_data.get("next_selector"),
                prev_selector=pagination_data.get("prev_selector"),
                page_number_selector=pagination_data.get("page_number_selector"),
                pattern=pagination_data.get("pattern"),
            )

        return PageStructure(
            domain=data["domain"],
            page_type=data["page_type"],
            url_pattern=data.get("url_pattern", ""),
            tag_hierarchy=data.get("tag_hierarchy", {}),
            iframe_locations=iframe_locations,
            script_signatures=data.get("script_signatures", []),
            css_class_map=data.get("css_class_map", {}),
            id_attributes=set(data.get("id_attributes", [])),
            semantic_landmarks=data.get("semantic_landmarks", {}),
            content_regions=content_regions,
            navigation_selectors=data.get("navigation_selectors", []),
            pagination_pattern=pagination_pattern,
            captured_at=captured_at,
            version=data.get("version", 1),
            content_hash=data.get("content_hash", ""),
        )
