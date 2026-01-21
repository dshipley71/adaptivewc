"""
Redis-based storage for page structure fingerprints.

Provides persistent storage for PageStructure objects with versioning,
quick signature comparison, and semantic embeddings for ML-based change detection.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import redis.asyncio as redis

from crawler.models import (
    ContentRegion,
    IframeInfo,
    PageStructure,
    PaginationInfo,
)
from crawler.utils.logging import CrawlerLogger
from crawler.utils import metrics


# =============================================================================
# Semantic Description Generator
# =============================================================================


class SemanticDescriptionGenerator:
    """
    Generates natural language descriptions of page structures.
    
    These descriptions are suitable for embedding with sentence transformers
    like all-MiniLM-L6-v2 for semantic similarity comparison.
    """
    
    # Framework indicators in CSS classes
    FRAMEWORK_PATTERNS = {
        "react": ["react", "jsx", "__react"],
        "vue": ["vue", "v-", "__vue"],
        "angular": ["ng-", "angular", "_ng"],
        "nextjs": ["next", "__next"],
        "tailwind": ["tw-", "tailwind"],
        "bootstrap": ["btn-", "col-", "row", "container", "navbar"],
        "material": ["mat-", "mdc-", "MuiButton"],
        "nfl": ["nfl-", "d3-o-", "d3-l-"],  # NFL team sites
    }
    
    # Common content indicator classes
    CONTENT_INDICATORS = [
        "article", "post", "content", "entry", "story", "news",
        "blog", "text", "body", "main", "primary"
    ]
    
    # Navigation indicator classes  
    NAV_INDICATORS = [
        "nav", "menu", "header", "footer", "sidebar", "toolbar",
        "breadcrumb", "pagination", "tabs"
    ]

    def generate(self, structure: PageStructure, url: str = "") -> str:
        """
        Generate a semantic description of the page structure.
        
        Args:
            structure: PageStructure to describe.
            url: Optional URL for additional context.
            
        Returns:
            Natural language description suitable for embedding.
        """
        parts = []
        
        # 1. Basic identification
        parts.append(self._describe_identity(structure, url))
        
        # 2. Content structure
        parts.append(self._describe_content_regions(structure))
        
        # 3. Semantic landmarks
        parts.append(self._describe_landmarks(structure))
        
        # 4. Framework and technology
        parts.append(self._describe_technology(structure))
        
        # 5. Navigation structure
        parts.append(self._describe_navigation(structure))
        
        # 6. Special elements (iframes, pagination)
        parts.append(self._describe_special_elements(structure))
        
        # Filter empty parts and join
        description = " ".join(filter(None, parts))
        
        return description
    
    def generate_compact(self, structure: PageStructure) -> str:
        """
        Generate a shorter description focused on key differentiators.
        
        Useful for quick comparisons or when embedding dimension is limited.
        """
        parts = []
        
        # Domain and page type
        parts.append(f"{structure.domain} {structure.page_type}")
        
        # URL pattern
        if structure.url_pattern:
            parts.append(f"pattern:{structure.url_pattern}")
        
        # Top content regions
        if structure.content_regions:
            regions = [r.primary_selector for r in structure.content_regions[:3]]
            parts.append(f"content:{','.join(regions)}")
        
        # Key landmarks
        key_landmarks = ["main", "article", "nav", "header"]
        found = [k for k in key_landmarks if k in structure.semantic_landmarks]
        if found:
            parts.append(f"landmarks:{','.join(found)}")
        
        # Framework hint
        framework = self._detect_framework(structure)
        if framework:
            parts.append(f"framework:{framework}")
        
        return " | ".join(parts)

    def _describe_identity(self, structure: PageStructure, url: str) -> str:
        """Describe basic page identity."""
        parts = [f"Page on {structure.domain}"]
        
        if structure.page_type and structure.page_type != "unknown":
            parts.append(f"of type {structure.page_type}")
        
        if structure.url_pattern:
            # Make pattern more readable
            pattern = structure.url_pattern.replace("{slug}", "article-name")
            pattern = pattern.replace("{id}", "123")
            parts.append(f"with URL structure like {pattern}")
        
        return " ".join(parts) + "."
    
    def _describe_content_regions(self, structure: PageStructure) -> str:
        """Describe content extraction regions."""
        if not structure.content_regions:
            return ""
        
        descriptions = []
        for region in structure.content_regions[:5]:
            conf = f"{region.confidence:.0%}" if region.confidence else "unknown"
            selector = region.primary_selector
            
            # Make selector more readable
            if selector.startswith("div."):
                selector = selector[4:]  # Remove "div."
            
            descriptions.append(f"{region.name} ({selector}, {conf} confidence)")
        
        if len(descriptions) == 1:
            return f"Main content is in {descriptions[0]}."
        else:
            return f"Content regions include: {', '.join(descriptions)}."
    
    def _describe_landmarks(self, structure: PageStructure) -> str:
        """Describe semantic landmarks."""
        if not structure.semantic_landmarks:
            return "No semantic HTML5 landmarks found."
        
        landmark_names = {
            "main": "main content area",
            "nav": "navigation",
            "header": "page header",
            "footer": "page footer",
            "article": "article content",
            "aside": "sidebar",
            "section": "content section",
            "figure": "media figure",
            "aria-banner": "ARIA banner",
            "aria-navigation": "ARIA navigation",
            "aria-main": "ARIA main content",
            "aria-contentinfo": "ARIA content info",
        }
        
        found = []
        for landmark, selector in structure.semantic_landmarks.items():
            readable = landmark_names.get(landmark, landmark)
            # Extract meaningful part of selector
            if "#" in selector:
                id_part = selector.split("#")[1].split(".")[0].split("[")[0]
                found.append(f"{readable} (#{id_part})")
            elif "." in selector:
                class_part = selector.split(".")[1].split("[")[0]
                found.append(f"{readable} (.{class_part})")
            else:
                found.append(readable)
        
        return f"Semantic structure includes: {', '.join(found)}."
    
    def _describe_technology(self, structure: PageStructure) -> str:
        """Describe detected frameworks and technologies."""
        parts = []
        
        # Detect framework from classes
        framework = self._detect_framework(structure)
        if framework:
            parts.append(f"Uses {framework} framework")
        
        # Describe scripts
        if structure.script_signatures:
            notable_scripts = []
            for sig in structure.script_signatures:
                if "react" in sig.lower():
                    notable_scripts.append("React")
                elif "vue" in sig.lower():
                    notable_scripts.append("Vue")
                elif "angular" in sig.lower():
                    notable_scripts.append("Angular")
                elif "jquery" in sig.lower():
                    notable_scripts.append("jQuery")
                elif "gpt" in sig.lower():
                    notable_scripts.append("Google Publisher Tags (ads)")
                elif "analytics" in sig.lower() or "gtag" in sig.lower():
                    notable_scripts.append("Google Analytics")
                elif "gigya" in sig.lower():
                    notable_scripts.append("Gigya (social login)")
            
            if notable_scripts:
                parts.append(f"with {', '.join(set(notable_scripts))}")
        
        # Tag distribution insights
        tag_counts = structure.tag_hierarchy.get("tag_counts", {})
        if tag_counts:
            svg_count = tag_counts.get("svg", 0) + tag_counts.get("path", 0)
            if svg_count > 50:
                parts.append("heavy use of SVG graphics")
            
            if tag_counts.get("picture", 0) > 10:
                parts.append("responsive images")
            
            if tag_counts.get("iframe", 0) > 0:
                parts.append(f"{tag_counts.get('iframe', 0)} embedded iframes")
        
        if parts:
            return " ".join(parts) + "."
        return ""
    
    def _detect_framework(self, structure: PageStructure) -> str | None:
        """Detect frontend framework from CSS classes."""
        class_names = " ".join(structure.css_class_map.keys()).lower()
        
        for framework, patterns in self.FRAMEWORK_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in class_names:
                    return framework
        
        return None
    
    def _describe_navigation(self, structure: PageStructure) -> str:
        """Describe navigation structure."""
        if not structure.navigation_selectors:
            return ""
        
        nav_types = []
        for selector in structure.navigation_selectors[:5]:
            selector_lower = selector.lower()
            if "mobile" in selector_lower:
                nav_types.append("mobile menu")
            elif "secondary" in selector_lower:
                nav_types.append("secondary navigation")
            elif "action" in selector_lower:
                nav_types.append("action menu")
            elif "footer" in selector_lower:
                nav_types.append("footer navigation")
            else:
                nav_types.append("navigation")
        
        unique_types = list(dict.fromkeys(nav_types))  # Preserve order, remove dupes
        
        if len(unique_types) == 1:
            return f"Has {unique_types[0]}."
        else:
            return f"Navigation includes: {', '.join(unique_types)}."
    
    def _describe_special_elements(self, structure: PageStructure) -> str:
        """Describe iframes, pagination, and other special elements."""
        parts = []
        
        # Iframes
        if structure.iframe_locations:
            iframe_types = []
            for iframe in structure.iframe_locations:
                if iframe.is_dynamic:
                    iframe_types.append(f"dynamic {iframe.position} embed")
                else:
                    iframe_types.append(f"{iframe.position} iframe")
            
            parts.append(f"Embeds: {', '.join(iframe_types)}")
        
        # Pagination
        if structure.pagination_pattern:
            pag = structure.pagination_pattern
            if pag.pattern:
                parts.append(f"Paginated with pattern {pag.pattern}")
            elif pag.next_selector:
                parts.append("Has pagination controls")
        
        return ". ".join(parts) + "." if parts else ""


# =============================================================================
# Embedding Manager
# =============================================================================


class EmbeddingManager:
    """
    Manages semantic embeddings for page structures.
    
    Uses sentence-transformers (all-MiniLM-L6-v2) for generating embeddings
    that can be compared for semantic similarity.
    """
    
    # Embedding dimension for all-MiniLM-L6-v2
    EMBEDDING_DIM = 384
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Sentence transformer model to use.
        """
        self.model_name = model_name
        self._model = None
        self._description_generator = SemanticDescriptionGenerator()
    
    @property
    def model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for embeddings. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def generate_embedding(self, structure: PageStructure, url: str = "") -> list[float]:
        """
        Generate embedding vector for a page structure.
        
        Args:
            structure: PageStructure to embed.
            url: Optional URL for additional context.
            
        Returns:
            List of floats representing the embedding vector.
        """
        description = self._description_generator.generate(structure, url)
        embedding = self.model.encode(description, convert_to_numpy=True)
        return embedding.tolist()
    
    def generate_embedding_from_text(self, text: str) -> list[float]:
        """
        Generate embedding from raw text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector as list of floats.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def compute_similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.
            
        Returns:
            Similarity score between -1 and 1 (1 = identical).
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def batch_generate_embeddings(
        self,
        structures: list[PageStructure],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple structures efficiently.
        
        Args:
            structures: List of PageStructure objects.
            
        Returns:
            List of embedding vectors.
        """
        descriptions = [
            self._description_generator.generate(s) for s in structures
        ]
        embeddings = self.model.encode(descriptions, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]


# =============================================================================
# Structure Signature (with embedding support)
# =============================================================================


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

    # Semantic description and embedding
    semantic_description: str = ""
    embedding: list[float] = field(default_factory=list)

    # Metadata
    captured_at: datetime = field(default_factory=datetime.utcnow)
    sample_url: str = ""

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
            "semantic_description": self.semantic_description,
            "embedding": self.embedding,
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
            semantic_description=data.get("semantic_description", ""),
            embedding=data.get("embedding", []),
            captured_at=captured_at,
            sample_url=data.get("sample_url", ""),
        )

    @classmethod
    def from_structure(
        cls,
        structure: PageStructure,
        url: str = "",
        include_embedding: bool = False,
        embedding_manager: "EmbeddingManager | None" = None,
    ) -> "StructureSignature":
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

        # Generate semantic description
        desc_generator = SemanticDescriptionGenerator()
        semantic_description = desc_generator.generate(structure, url)
        
        # Generate embedding if requested
        embedding = []
        if include_embedding and embedding_manager:
            embedding = embedding_manager.generate_embedding(structure, url)

        return cls(
            domain=structure.domain,
            page_type=structure.page_type,
            version=structure.version,
            content_hash=structure.content_hash[:16] if structure.content_hash else "",
            tag_count_hash=tag_count_hash,
            class_set_hash=class_set_hash,
            landmark_hash=landmark_hash,
            semantic_description=semantic_description,
            embedding=embedding,
            captured_at=structure.captured_at,
            sample_url=url,
        )

    def matches(self, other: "StructureSignature") -> bool:
        """Quick check if two signatures match (hash-based)."""
        return (
            self.content_hash == other.content_hash
            and self.tag_count_hash == other.tag_count_hash
            and self.class_set_hash == other.class_set_hash
        )
    
    def semantic_similarity(self, other: "StructureSignature") -> float | None:
        """
        Compute semantic similarity using embeddings.
        
        Returns:
            Similarity score (0-1) or None if embeddings not available.
        """
        if not self.embedding or not other.embedding:
            return None
        
        vec1 = np.array(self.embedding)
        vec2 = np.array(other.embedding)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))


# =============================================================================
# Structure Store
# =============================================================================


class StructureStore:
    """
    Redis-based storage for page structure fingerprints.

    Features:
    - Version history for structures
    - Quick signature comparison (hash-based)
    - Semantic descriptions for ML embedding
    - Optional embedding storage for similarity search
    - TTL-based expiration
    - Domain/page_type indexing
    """

    # Redis key patterns
    STRUCTURE_PREFIX = "crawler:structure:"
    SIGNATURE_PREFIX = "crawler:sig:"
    EMBEDDING_PREFIX = "crawler:emb:"
    VERSION_PREFIX = "crawler:structver:"
    INDEX_KEY = "crawler:structure:index"
    STATS_KEY = "crawler:structure:stats"

    def __init__(
        self,
        redis_client: redis.Redis,
        ttl_seconds: int = 604800,  # 7 days
        max_versions: int = 10,
        enable_embeddings: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the structure store.

        Args:
            redis_client: Redis async client.
            ttl_seconds: TTL for structure data.
            max_versions: Maximum versions to keep per domain/page_type.
            enable_embeddings: Whether to generate and store embeddings.
            embedding_model: Sentence transformer model for embeddings.
            logger: Logger instance.
        """
        self.redis = redis_client
        self.ttl = ttl_seconds
        self.max_versions = max_versions
        self.enable_embeddings = enable_embeddings
        self.logger = logger or CrawlerLogger("structure_store")
        
        # Initialize embedding manager lazily
        self._embedding_manager: EmbeddingManager | None = None
        self._embedding_model = embedding_model
        
        # Description generator (always available)
        self._description_generator = SemanticDescriptionGenerator()

    @property
    def embedding_manager(self) -> EmbeddingManager | None:
        """Lazy load embedding manager."""
        if self.enable_embeddings and self._embedding_manager is None:
            try:
                self._embedding_manager = EmbeddingManager(self._embedding_model)
            except ImportError as e:
                self.logger.warning(
                    "Embeddings disabled - sentence-transformers not installed",
                    error=str(e),
                )
                self.enable_embeddings = False
        return self._embedding_manager

    def _structure_key(self, domain: str, page_type: str, version: int) -> str:
        """Get Redis key for a specific structure version."""
        return f"{self.STRUCTURE_PREFIX}{domain}:{page_type}:v{version}"

    def _latest_key(self, domain: str, page_type: str) -> str:
        """Get Redis key for latest version pointer."""
        return f"{self.VERSION_PREFIX}{domain}:{page_type}:latest"

    def _signature_key(self, domain: str, page_type: str) -> str:
        """Get Redis key for quick signature."""
        return f"{self.SIGNATURE_PREFIX}{domain}:{page_type}"

    def _embedding_key(self, domain: str, page_type: str) -> str:
        """Get Redis key for embedding vector."""
        return f"{self.EMBEDDING_PREFIX}{domain}:{page_type}"

    def _index_entry(self, domain: str, page_type: str) -> str:
        """Create index entry string."""
        return f"{domain}:{page_type}"

    def generate_description(self, structure: PageStructure, url: str = "") -> str:
        """
        Generate semantic description for a structure.
        
        This can be used independently of storage for ML purposes.
        
        Args:
            structure: PageStructure to describe.
            url: Optional URL for context.
            
        Returns:
            Natural language description.
        """
        return self._description_generator.generate(structure, url)

    def generate_compact_description(self, structure: PageStructure) -> str:
        """
        Generate compact semantic description.
        
        Args:
            structure: PageStructure to describe.
            
        Returns:
            Compact description string.
        """
        return self._description_generator.generate_compact(structure)

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

            # Create and save signature with semantic description
            signature = StructureSignature.from_structure(
                structure,
                url,
                include_embedding=self.enable_embeddings,
                embedding_manager=self.embedding_manager,
            )
            signature_key = self._signature_key(domain, page_type)
            await self.redis.setex(
                signature_key, self.ttl, json.dumps(signature.to_dict())
            )

            # Store embedding separately for efficient retrieval
            if self.enable_embeddings and signature.embedding:
                embedding_key = self._embedding_key(domain, page_type)
                await self.redis.setex(
                    embedding_key,
                    self.ttl,
                    json.dumps(signature.embedding),
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
                has_embedding=bool(signature.embedding),
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

    async def get_embedding(
        self,
        domain: str,
        page_type: str,
    ) -> list[float] | None:
        """
        Get embedding vector for a structure.

        Args:
            domain: Domain name.
            page_type: Page type.

        Returns:
            Embedding vector or None if not found.
        """
        try:
            embedding_key = self._embedding_key(domain, page_type)
            data = await self.redis.get(embedding_key)

            if not data:
                return None

            return json.loads(data)

        except redis.RedisError as e:
            self.logger.error(
                "Failed to get embedding",
                domain=domain,
                page_type=page_type,
                error=str(e),
            )
            return None

    async def get_description(
        self,
        domain: str,
        page_type: str,
    ) -> str | None:
        """
        Get semantic description for a structure.

        Args:
            domain: Domain name.
            page_type: Page type.

        Returns:
            Semantic description or None if not found.
        """
        signature = await self.get_signature(domain, page_type)
        if signature:
            return signature.semantic_description
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

    async def compute_semantic_similarity(
        self,
        domain1: str,
        page_type1: str,
        domain2: str,
        page_type2: str,
    ) -> float | None:
        """
        Compute semantic similarity between two stored structures.

        Args:
            domain1: First domain.
            page_type1: First page type.
            domain2: Second domain.
            page_type2: Second page type.

        Returns:
            Similarity score (0-1) or None if embeddings not available.
        """
        emb1 = await self.get_embedding(domain1, page_type1)
        emb2 = await self.get_embedding(domain2, page_type2)

        if not emb1 or not emb2:
            return None

        # Compute cosine similarity
        vec1 = np.array(emb1)
        vec2 = np.array(emb2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    async def find_similar_structures(
        self,
        structure: PageStructure,
        threshold: float = 0.8,
        limit: int = 10,
    ) -> list[tuple[str, str, float]]:
        """
        Find structures similar to the given one using embeddings.

        Args:
            structure: Structure to find similar ones for.
            threshold: Minimum similarity threshold (0-1).
            limit: Maximum results to return.

        Returns:
            List of (domain, page_type, similarity) tuples.
        """
        if not self.enable_embeddings or not self.embedding_manager:
            return []

        # Generate embedding for query structure
        query_embedding = self.embedding_manager.generate_embedding(structure)

        # Get all tracked structures
        all_structures = await self.list_domains()
        
        results = []
        for domain, page_type in all_structures:
            stored_embedding = await self.get_embedding(domain, page_type)
            if not stored_embedding:
                continue

            similarity = self.embedding_manager.compute_similarity(
                query_embedding, stored_embedding
            )

            if similarity >= threshold:
                results.append((domain, page_type, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[2], reverse=True)

        return results[:limit]

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

            # Count structures with embeddings
            embedding_count = 0
            if self.enable_embeddings:
                for domain, page_type in domains:
                    if await self.get_embedding(domain, page_type):
                        embedding_count += 1

            return {
                "total_saved": int(stats.get(b"total_saved", 0)),
                "last_save": stats.get(b"last_save", b"").decode() if stats.get(b"last_save") else "",
                "tracked_domains": len(set(d[0] for d in domains)),
                "tracked_page_types": len(domains),
                "structures_with_embeddings": embedding_count,
                "embeddings_enabled": self.enable_embeddings,
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
                self._embedding_key(domain, page_type),
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
            embedding_keys = await self.redis.keys(f"{self.EMBEDDING_PREFIX}*")
            version_keys = await self.redis.keys(f"{self.VERSION_PREFIX}*")

            all_keys = structure_keys + signature_keys + embedding_keys + version_keys
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
