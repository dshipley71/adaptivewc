# AGENTS.md - Storage Module

Complete specification for the Redis storage layer module.

---

## Module Purpose

The storage module provides:
- Structure storage with versioning
- Embedding persistence
- Caching utilities
- Volatile domain tracking

---

## Files to Generate

```
fingerprint/storage/
├── __init__.py
├── structure_store.py    # Structure storage
├── embedding_store.py    # Embedding storage
└── cache.py              # Caching utilities
```

---

## fingerprint/storage/__init__.py

```python
"""
Storage module - Redis persistence layer.
"""

from fingerprint.storage.structure_store import StructureStore
from fingerprint.storage.embedding_store import EmbeddingStore
from fingerprint.storage.cache import Cache

__all__ = [
    "StructureStore",
    "EmbeddingStore",
    "Cache",
]
```

---

## fingerprint/storage/structure_store.py

```python
"""
Redis storage for page structures.

Key patterns:
- {prefix}:structure:{domain}:{page_type}:{variant_id} - Current structure
- {prefix}:structure:{domain}:{page_type}:{variant_id}:v{n} - Version history
- {prefix}:volatile:{domain} - Volatile domain flag

Verbose logging pattern:
[STORE:OPERATION] Message
"""

import json
from datetime import datetime
from dataclasses import asdict
from typing import Any

import redis.asyncio as redis

from fingerprint.config import RedisConfig
from fingerprint.core.verbose import get_logger
from fingerprint.exceptions import RedisConnectionError, SerializationError
from fingerprint.models import PageStructure, TagHierarchy, ContentRegion


class StructureStore:
    """
    Persistent storage for page structures.

    Features:
    - Current version access
    - Version history (configurable depth)
    - Volatile domain tracking
    - TTL-based expiration

    Usage:
        store = StructureStore(config.redis)
        await store.save(structure)
        structure = await store.get(domain, page_type)
    """

    def __init__(self, config: RedisConfig):
        self.config = config
        self.logger = get_logger()
        self._client: redis.Redis | None = None

        self.logger.info(
            "STORE", "INIT",
            "Structure store initialized",
            prefix=config.key_prefix,
            ttl_days=config.ttl_seconds // 86400,
        )

    async def _get_client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            try:
                self._client = redis.from_url(self.config.url)
                await self._client.ping()

                self.logger.debug(
                    "STORE", "CONNECT",
                    "Connected to Redis",
                    url=self.config.url.split("@")[-1],  # Hide credentials
                )
            except Exception as e:
                self.logger.error("STORE", "CONNECT_ERROR", str(e))
                raise RedisConnectionError(f"Failed to connect to Redis: {e}")

        return self._client

    def _key(self, domain: str, page_type: str, variant_id: str = "default") -> str:
        """Generate Redis key for structure."""
        return f"{self.config.key_prefix}:structure:{domain}:{page_type}:{variant_id}"

    def _version_key(
        self,
        domain: str,
        page_type: str,
        variant_id: str,
        version: int,
    ) -> str:
        """Generate Redis key for specific version."""
        return f"{self._key(domain, page_type, variant_id)}:v{version}"

    def _volatile_key(self, domain: str) -> str:
        """Generate Redis key for volatile domain flag."""
        return f"{self.config.key_prefix}:volatile:{domain}"

    async def save(self, structure: PageStructure) -> None:
        """
        Save structure to Redis.

        Automatically:
        - Archives previous version
        - Increments version number
        - Sets TTL

        Args:
            structure: PageStructure to save

        Verbose output:
            [STORE:SAVE] Saving structure
              - domain: example.com
              - page_type: article
              - version: 4
            [STORE:ARCHIVE] Archived previous version
              - archived_version: 3
        """
        client = await self._get_client()
        key = self._key(structure.domain, structure.page_type, structure.variant_id)

        self.logger.info(
            "STORE", "SAVE",
            f"Saving structure for {structure.domain}/{structure.page_type}",
            version=structure.version,
        )

        # Check for existing structure to archive
        existing_data = await client.get(key)
        if existing_data:
            try:
                existing = json.loads(existing_data)
                old_version = existing.get("version", 1)

                # Archive previous version
                archive_key = self._version_key(
                    structure.domain,
                    structure.page_type,
                    structure.variant_id,
                    old_version,
                )
                await client.setex(
                    archive_key,
                    self.config.ttl_seconds,
                    existing_data,
                )

                self.logger.debug(
                    "STORE", "ARCHIVE",
                    f"Archived version {old_version}",
                )

                # Increment version
                structure.version = old_version + 1

                # Cleanup old versions if needed
                await self._cleanup_old_versions(
                    structure.domain,
                    structure.page_type,
                    structure.variant_id,
                )

            except json.JSONDecodeError:
                self.logger.warn("STORE", "DECODE_ERROR", "Could not decode existing structure")

        # Serialize and save
        try:
            data = self._serialize(structure)
        except Exception as e:
            self.logger.error("STORE", "SERIALIZE_ERROR", str(e))
            raise SerializationError(f"Failed to serialize structure: {e}")

        await client.setex(key, self.config.ttl_seconds, data)

        self.logger.info(
            "STORE", "SAVED",
            f"Structure saved (version {structure.version})",
            key=key,
            size_bytes=len(data),
        )

    async def get(
        self,
        domain: str,
        page_type: str,
        variant_id: str = "default",
        version: int | None = None,
    ) -> PageStructure | None:
        """
        Retrieve structure from Redis.

        Args:
            domain: Domain name
            page_type: Page type
            variant_id: Variant identifier
            version: Specific version (None for current)

        Returns:
            PageStructure or None if not found

        Verbose output:
            [STORE:GET] Retrieving structure
              - domain: example.com
              - page_type: article
            [STORE:FOUND] Structure found
              - version: 4
              - captured_at: 2024-01-15T10:30:00Z
        """
        client = await self._get_client()

        if version is not None:
            key = self._version_key(domain, page_type, variant_id, version)
        else:
            key = self._key(domain, page_type, variant_id)

        self.logger.debug(
            "STORE", "GET",
            f"Retrieving {domain}/{page_type}",
            version=version or "current",
        )

        data = await client.get(key)

        if data is None:
            self.logger.debug(
                "STORE", "NOT_FOUND",
                f"No structure found for {domain}/{page_type}",
            )
            return None

        try:
            structure = self._deserialize(data)

            self.logger.info(
                "STORE", "FOUND",
                f"Structure found for {domain}/{page_type}",
                version=structure.version,
                captured_at=structure.captured_at.isoformat(),
            )

            return structure

        except Exception as e:
            self.logger.error("STORE", "DESERIALIZE_ERROR", str(e))
            raise SerializationError(f"Failed to deserialize structure: {e}")

    async def delete(
        self,
        domain: str,
        page_type: str,
        variant_id: str = "default",
    ) -> bool:
        """
        Delete structure and all versions.

        Returns:
            True if deleted, False if not found
        """
        client = await self._get_client()
        key = self._key(domain, page_type, variant_id)

        self.logger.info(
            "STORE", "DELETE",
            f"Deleting {domain}/{page_type}",
        )

        # Delete main key
        deleted = await client.delete(key)

        # Delete version history
        pattern = f"{key}:v*"
        version_keys = []
        async for k in client.scan_iter(pattern):
            version_keys.append(k)

        if version_keys:
            await client.delete(*version_keys)
            self.logger.debug(
                "STORE", "DELETE_VERSIONS",
                f"Deleted {len(version_keys)} versions",
            )

        return deleted > 0

    async def list_versions(
        self,
        domain: str,
        page_type: str,
        variant_id: str = "default",
    ) -> list[int]:
        """
        List available versions for a structure.

        Returns:
            List of version numbers (sorted descending)
        """
        client = await self._get_client()
        key = self._key(domain, page_type, variant_id)
        pattern = f"{key}:v*"

        versions = []
        async for k in client.scan_iter(pattern):
            # Extract version number from key
            try:
                version_str = k.decode() if isinstance(k, bytes) else k
                version_num = int(version_str.split(":v")[-1])
                versions.append(version_num)
            except (ValueError, IndexError):
                continue

        # Check if current version exists
        if await client.exists(key):
            data = await client.get(key)
            if data:
                try:
                    structure = json.loads(data)
                    current_version = structure.get("version", 1)
                    if current_version not in versions:
                        versions.append(current_version)
                except json.JSONDecodeError:
                    pass

        return sorted(versions, reverse=True)

    async def is_volatile(self, domain: str) -> bool:
        """
        Check if domain is flagged as volatile.

        Volatile domains have frequent CSS changes and should
        prefer ML-based fingerprinting.
        """
        client = await self._get_client()
        key = self._volatile_key(domain)

        exists = await client.exists(key)
        return bool(exists)

    async def mark_volatile(self, domain: str, ttl_days: int = 30) -> None:
        """
        Mark domain as volatile.

        Args:
            domain: Domain to mark
            ttl_days: How long to keep the flag
        """
        client = await self._get_client()
        key = self._volatile_key(domain)

        await client.setex(key, ttl_days * 86400, "1")

        self.logger.info(
            "STORE", "VOLATILE",
            f"Marked {domain} as volatile",
            ttl_days=ttl_days,
        )

    async def _cleanup_old_versions(
        self,
        domain: str,
        page_type: str,
        variant_id: str,
    ) -> None:
        """Remove versions beyond max_versions limit."""
        client = await self._get_client()
        versions = await self.list_versions(domain, page_type, variant_id)

        if len(versions) > self.config.max_versions:
            # Delete oldest versions
            to_delete = versions[self.config.max_versions:]

            for version in to_delete:
                key = self._version_key(domain, page_type, variant_id, version)
                await client.delete(key)

            self.logger.debug(
                "STORE", "CLEANUP",
                f"Removed {len(to_delete)} old versions",
            )

    def _serialize(self, structure: PageStructure) -> str:
        """Serialize structure to JSON."""
        data = asdict(structure)

        # Handle datetime
        data["captured_at"] = structure.captured_at.isoformat()

        # Handle sets
        data["id_attributes"] = list(structure.id_attributes)

        return json.dumps(data)

    def _deserialize(self, data: bytes | str) -> PageStructure:
        """Deserialize JSON to structure."""
        if isinstance(data, bytes):
            data = data.decode("utf-8")

        obj = json.loads(data)

        # Reconstruct nested objects
        if obj.get("tag_hierarchy"):
            obj["tag_hierarchy"] = TagHierarchy(**obj["tag_hierarchy"])

        if obj.get("content_regions"):
            obj["content_regions"] = [
                ContentRegion(**r) for r in obj["content_regions"]
            ]

        # Handle datetime
        if obj.get("captured_at"):
            obj["captured_at"] = datetime.fromisoformat(obj["captured_at"])

        # Handle sets
        if obj.get("id_attributes"):
            obj["id_attributes"] = set(obj["id_attributes"])

        return PageStructure(**obj)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self.logger.debug("STORE", "CLOSE", "Connection closed")
```

---

## fingerprint/storage/embedding_store.py

```python
"""
Redis storage for structure embeddings.

Key pattern: {prefix}:embedding:{domain}:{page_type}:{variant_id}

Verbose logging pattern:
[EMBED_STORE:OPERATION] Message
"""

import json
from datetime import datetime
from dataclasses import asdict

import redis.asyncio as redis

from fingerprint.config import RedisConfig
from fingerprint.core.verbose import get_logger
from fingerprint.exceptions import RedisConnectionError, SerializationError
from fingerprint.models import StructureEmbedding


class EmbeddingStore:
    """
    Persistent storage for structure embeddings.

    Usage:
        store = EmbeddingStore(config.redis)
        await store.save(embedding)
        embedding = await store.get(domain, page_type)
    """

    def __init__(self, config: RedisConfig):
        self.config = config
        self.logger = get_logger()
        self._client: redis.Redis | None = None

        self.logger.info(
            "EMBED_STORE", "INIT",
            "Embedding store initialized",
        )

    async def _get_client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            try:
                self._client = redis.from_url(self.config.url)
                await self._client.ping()
            except Exception as e:
                raise RedisConnectionError(f"Failed to connect to Redis: {e}")

        return self._client

    def _key(self, domain: str, page_type: str, variant_id: str = "default") -> str:
        """Generate Redis key for embedding."""
        return f"{self.config.key_prefix}:embedding:{domain}:{page_type}:{variant_id}"

    async def save(self, embedding: StructureEmbedding) -> None:
        """
        Save embedding to Redis.

        Args:
            embedding: StructureEmbedding to save

        Verbose output:
            [EMBED_STORE:SAVE] Saving embedding
              - domain: example.com
              - dimensions: 384
        """
        client = await self._get_client()
        key = self._key(embedding.domain, embedding.page_type, embedding.variant_id)

        self.logger.info(
            "EMBED_STORE", "SAVE",
            f"Saving embedding for {embedding.domain}/{embedding.page_type}",
            dimensions=embedding.dimensions,
        )

        data = self._serialize(embedding)
        await client.setex(key, self.config.ttl_seconds, data)

        self.logger.debug(
            "EMBED_STORE", "SAVED",
            "Embedding saved",
            size_bytes=len(data),
        )

    async def get(
        self,
        domain: str,
        page_type: str,
        variant_id: str = "default",
    ) -> StructureEmbedding | None:
        """
        Retrieve embedding from Redis.

        Returns:
            StructureEmbedding or None if not found
        """
        client = await self._get_client()
        key = self._key(domain, page_type, variant_id)

        self.logger.debug(
            "EMBED_STORE", "GET",
            f"Retrieving embedding for {domain}/{page_type}",
        )

        data = await client.get(key)

        if data is None:
            self.logger.debug(
                "EMBED_STORE", "NOT_FOUND",
                "Embedding not found",
            )
            return None

        try:
            embedding = self._deserialize(data)
            self.logger.debug(
                "EMBED_STORE", "FOUND",
                "Embedding found",
                dimensions=embedding.dimensions,
            )
            return embedding

        except Exception as e:
            self.logger.error("EMBED_STORE", "DESERIALIZE_ERROR", str(e))
            raise SerializationError(f"Failed to deserialize embedding: {e}")

    async def delete(
        self,
        domain: str,
        page_type: str,
        variant_id: str = "default",
    ) -> bool:
        """Delete embedding from Redis."""
        client = await self._get_client()
        key = self._key(domain, page_type, variant_id)

        deleted = await client.delete(key)
        return deleted > 0

    def _serialize(self, embedding: StructureEmbedding) -> str:
        """Serialize embedding to JSON."""
        data = asdict(embedding)
        data["generated_at"] = embedding.generated_at.isoformat()
        return json.dumps(data)

    def _deserialize(self, data: bytes | str) -> StructureEmbedding:
        """Deserialize JSON to embedding."""
        if isinstance(data, bytes):
            data = data.decode("utf-8")

        obj = json.loads(data)

        if obj.get("generated_at"):
            obj["generated_at"] = datetime.fromisoformat(obj["generated_at"])

        return StructureEmbedding(**obj)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
```

---

## fingerprint/storage/cache.py

```python
"""
In-memory caching utilities.

Provides TTL-based caching for expensive operations:
- Embedding generation
- Ollama Cloud responses
- Structure comparisons

Verbose logging pattern:
[CACHE:OPERATION] Message
"""

import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from fingerprint.core.verbose import get_logger

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with TTL."""
    value: T
    created_at: float
    ttl_seconds: float

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > (self.created_at + self.ttl_seconds)


class Cache(Generic[T]):
    """
    TTL-based in-memory cache.

    Usage:
        cache = Cache[str](default_ttl=300)
        cache.set("key", "value")
        value = cache.get("key")
    """

    def __init__(self, default_ttl: float = 300):
        self.default_ttl = default_ttl
        self.logger = get_logger()
        self._entries: dict[str, CacheEntry[T]] = {}

        self.logger.debug(
            "CACHE", "INIT",
            f"Cache initialized (TTL: {default_ttl}s)",
        )

    def set(self, key: str, value: T, ttl: float | None = None) -> None:
        """
        Set cache entry.

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses default if None)

        Verbose output:
            [CACHE:SET] Cached key: example_key
              - ttl: 300
        """
        entry = CacheEntry(
            value=value,
            created_at=time.time(),
            ttl_seconds=ttl or self.default_ttl,
        )

        self._entries[key] = entry

        self.logger.debug(
            "CACHE", "SET",
            f"Cached: {key}",
            ttl=entry.ttl_seconds,
        )

    def get(self, key: str) -> T | None:
        """
        Get cache entry.

        Returns:
            Cached value or None if not found/expired

        Verbose output:
            [CACHE:HIT] Cache hit: example_key
            [CACHE:MISS] Cache miss: example_key
            [CACHE:EXPIRED] Cache expired: example_key
        """
        entry = self._entries.get(key)

        if entry is None:
            self.logger.debug("CACHE", "MISS", f"Miss: {key}")
            return None

        if entry.is_expired:
            self.logger.debug("CACHE", "EXPIRED", f"Expired: {key}")
            del self._entries[key]
            return None

        self.logger.debug("CACHE", "HIT", f"Hit: {key}")
        return entry.value

    def delete(self, key: str) -> bool:
        """Delete cache entry."""
        if key in self._entries:
            del self._entries[key]
            self.logger.debug("CACHE", "DELETE", f"Deleted: {key}")
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self._entries)
        self._entries.clear()
        self.logger.debug("CACHE", "CLEAR", f"Cleared {count} entries")

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        expired_keys = [
            key for key, entry in self._entries.items()
            if entry.is_expired
        ]

        for key in expired_keys:
            del self._entries[key]

        if expired_keys:
            self.logger.debug(
                "CACHE", "CLEANUP",
                f"Removed {len(expired_keys)} expired entries",
            )

        return len(expired_keys)

    @property
    def size(self) -> int:
        """Get number of entries in cache."""
        return len(self._entries)

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        expired_count = sum(1 for e in self._entries.values() if e.is_expired)

        return {
            "total_entries": len(self._entries),
            "expired_entries": expired_count,
            "active_entries": len(self._entries) - expired_count,
        }
```

---

## Redis Key Schema

### Structure Storage

```
{prefix}:structure:{domain}:{page_type}:{variant_id}
```

Example:
```
fingerprint:structure:example.com:article:default
```

### Version History

```
{prefix}:structure:{domain}:{page_type}:{variant_id}:v{n}
```

Example:
```
fingerprint:structure:example.com:article:default:v3
fingerprint:structure:example.com:article:default:v2
fingerprint:structure:example.com:article:default:v1
```

### Embeddings

```
{prefix}:embedding:{domain}:{page_type}:{variant_id}
```

Example:
```
fingerprint:embedding:example.com:article:default
```

### Volatile Domains

```
{prefix}:volatile:{domain}
```

Example:
```
fingerprint:volatile:frequently-changing-site.com
```

---

## Data Format

### Structure JSON

```json
{
    "domain": "example.com",
    "page_type": "article",
    "url_pattern": "/blog/{date}/{slug}",
    "variant_id": "default",
    "tag_hierarchy": {
        "tag_counts": {"div": 150, "p": 45, "a": 89},
        "depth_distribution": {"0": 1, "1": 5, "2": 15},
        "parent_child_pairs": {"div>div": 89, "div>p": 32},
        "max_depth": 12
    },
    "css_class_map": {
        "article-content": 1,
        "post-title": 1,
        "nav-link": 5
    },
    "id_attributes": ["header", "main", "footer"],
    "semantic_landmarks": {
        "header": "header.site-header",
        "nav": "nav.main-nav",
        "main": "main.content",
        "footer": "footer.site-footer"
    },
    "content_regions": [
        {
            "name": "title",
            "primary_selector": "h1.post-title",
            "fallback_selectors": ["h1", ".title"],
            "confidence": 0.92
        }
    ],
    "navigation_selectors": ["nav.main-nav", ".pagination"],
    "script_signatures": ["/_next/static/chunks/main.*.js"],
    "detected_framework": "next",
    "captured_at": "2024-01-15T10:30:00Z",
    "version": 4,
    "content_hash": "a1b2c3d4e5f6..."
}
```

### Embedding JSON

```json
{
    "domain": "example.com",
    "page_type": "article",
    "variant_id": "default",
    "vector": [0.123, -0.456, 0.789, ...],
    "dimensions": 384,
    "model_name": "all-MiniLM-L6-v2",
    "description": "Page type: article. Framework: next...",
    "generated_at": "2024-01-15T10:30:00Z"
}
```

---

## Verbose Logging

All storage operations use consistent logging:

| Operation | Description |
|-----------|-------------|
| STORE:INIT | Store initialized |
| STORE:CONNECT | Connected to Redis |
| STORE:SAVE | Saving structure |
| STORE:SAVED | Structure saved |
| STORE:ARCHIVE | Previous version archived |
| STORE:GET | Retrieving structure |
| STORE:FOUND | Structure found |
| STORE:NOT_FOUND | Structure not found |
| STORE:DELETE | Deleting structure |
| STORE:CLEANUP | Old versions removed |
| STORE:VOLATILE | Domain marked volatile |
| EMBED_STORE:SAVE | Saving embedding |
| EMBED_STORE:GET | Retrieving embedding |
| CACHE:SET | Cache entry set |
| CACHE:HIT | Cache hit |
| CACHE:MISS | Cache miss |
| CACHE:EXPIRED | Cache entry expired |

### Example Output

```
[2024-01-15T10:30:00Z] [STORE:INIT] Structure store initialized
  - prefix: fingerprint
  - ttl_days: 7

[2024-01-15T10:30:00Z] [STORE:CONNECT] Connected to Redis
  - url: localhost:6379/0

[2024-01-15T10:30:01Z] [STORE:SAVE] Saving structure for example.com/article
  - version: 4

[2024-01-15T10:30:01Z] [STORE:ARCHIVE] Archived version 3

[2024-01-15T10:30:01Z] [STORE:SAVED] Structure saved (version 4)
  - key: fingerprint:structure:example.com:article:default
  - size_bytes: 2456

[2024-01-15T10:30:02Z] [STORE:GET] Retrieving example.com/article
  - version: current

[2024-01-15T10:30:02Z] [STORE:FOUND] Structure found for example.com/article
  - version: 4
  - captured_at: 2024-01-15T10:30:01Z
```
