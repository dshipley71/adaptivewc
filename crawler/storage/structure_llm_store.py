"""
Redis-based storage for page structure fingerprints with LLM-powered descriptions.

Uses an LLM to generate rich semantic descriptions of page structures,
which are then embedded with sentence transformers for similarity comparison.
"""

import hashlib
import json
from abc import ABC, abstractmethod
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
# LLM Provider Abstraction
# =============================================================================


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens in response.
            
        Returns:
            Generated text.
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier."""
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            model: Model to use.
        """
        self.model = model
        self._client = None
        self._api_key = api_key
    
    @property
    def client(self):
        """Lazy load the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "anthropic is required for AnthropicProvider. "
                    "Install with: pip install anthropic"
                )
        return self._client
    
    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text using Claude."""
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    
    def get_model_name(self) -> str:
        return f"anthropic/{self.model}"


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Model to use.
        """
        self.model = model
        self._client = None
        self._api_key = api_key
    
    @property
    def client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "openai is required for OpenAIProvider. "
                    "Install with: pip install openai"
                )
        return self._client
    
    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text using OpenAI."""
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    
    def get_model_name(self) -> str:
        return f"openai/{self.model}"


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize Ollama provider.
        
        Args:
            model: Model to use.
            base_url: Ollama server URL.
        """
        self.model = model
        self.base_url = base_url
        self._client = None
    
    @property
    def client(self):
        """Lazy load the Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.AsyncClient(host=self.base_url)
            except ImportError:
                raise ImportError(
                    "ollama is required for OllamaProvider. "
                    "Install with: pip install ollama"
                )
        return self._client
    
    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text using Ollama."""
        response = await self.client.generate(
            model=self.model,
            prompt=prompt,
            options={"num_predict": max_tokens}
        )
        return response["response"]
    
    def get_model_name(self) -> str:
        return f"ollama/{self.model}"


# =============================================================================
# LLM Semantic Description Generator
# =============================================================================


class LLMSemanticDescriptionGenerator:
    """
    Uses an LLM to generate rich semantic descriptions of page structures.
    
    These descriptions capture the meaning and purpose of page elements
    in natural language, making them ideal for semantic embedding and
    similarity comparison.
    """
    
    # System prompt for description generation
    SYSTEM_CONTEXT = """You are a web structure analyst. Your task is to create 
concise, semantic descriptions of web page structures that capture the essential 
characteristics for detecting when a website has changed its layout or design.

Focus on:
- The purpose and type of the page (article, product listing, homepage, etc.)
- Key content areas and their roles
- Navigation patterns
- Technology stack indicators
- Unique structural characteristics

Output a single paragraph (3-5 sentences) that would help identify this page 
structure even if CSS class names changed. Do not use bullet points or lists.
Be specific but concise."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the LLM description generator.
        
        Args:
            llm_provider: LLM provider to use for generation.
            logger: Logger instance.
        """
        self.llm = llm_provider
        self.logger = logger or CrawlerLogger("llm_description_generator")
    
    def _build_structure_summary(self, structure: PageStructure) -> str:
        """Build a structured summary of the page for the LLM."""
        parts = []
        
        # Basic info
        parts.append(f"Domain: {structure.domain}")
        parts.append(f"Page Type: {structure.page_type}")
        if structure.url_pattern:
            parts.append(f"URL Pattern: {structure.url_pattern}")
        
        # Tag distribution
        tag_counts = structure.tag_hierarchy.get("tag_counts", {})
        if tag_counts:
            top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:10]
            tag_summary = ", ".join(f"{tag}:{count}" for tag, count in top_tags)
            parts.append(f"Top Tags: {tag_summary}")
        
        # CSS classes (top 15)
        if structure.css_class_map:
            top_classes = sorted(structure.css_class_map.items(), key=lambda x: -x[1])[:15]
            class_names = [cls for cls, _ in top_classes]
            parts.append(f"Key CSS Classes: {', '.join(class_names)}")
        
        # Semantic landmarks
        if structure.semantic_landmarks:
            landmarks = list(structure.semantic_landmarks.keys())
            parts.append(f"Semantic Landmarks: {', '.join(landmarks)}")
        
        # Content regions
        if structure.content_regions:
            regions = [f"{r.name}({r.primary_selector})" for r in structure.content_regions[:5]]
            parts.append(f"Content Regions: {', '.join(regions)}")
        
        # Navigation
        if structure.navigation_selectors:
            parts.append(f"Navigation Count: {len(structure.navigation_selectors)}")
        
        # Iframes
        if structure.iframe_locations:
            iframe_positions = [f.position for f in structure.iframe_locations]
            parts.append(f"Iframes: {', '.join(iframe_positions)}")
        
        # Scripts
        if structure.script_signatures:
            parts.append(f"Scripts: {', '.join(structure.script_signatures[:10])}")
        
        # Pagination
        if structure.pagination_pattern:
            parts.append("Has Pagination: Yes")
        
        return "\n".join(parts)
    
    async def generate(self, structure: PageStructure, url: str = "") -> str:
        """
        Generate a semantic description using the LLM.
        
        Args:
            structure: PageStructure to describe.
            url: Optional URL for additional context.
            
        Returns:
            Natural language description suitable for embedding.
        """
        # Build the structure summary
        summary = self._build_structure_summary(structure)
        
        # Build the prompt
        prompt = f"""{self.SYSTEM_CONTEXT}

Analyze this web page structure and generate a semantic description:

{summary}

{f"Sample URL: {url}" if url else ""}

Generate a concise paragraph describing this page's structure for change detection:"""

        try:
            description = await self.llm.generate(prompt, max_tokens=300)
            
            # Clean up the response
            description = description.strip()
            
            # Remove any leading/trailing quotes if present
            if description.startswith('"') and description.endswith('"'):
                description = description[1:-1]
            
            self.logger.debug(
                "Generated LLM description",
                domain=structure.domain,
                model=self.llm.get_model_name(),
                description_length=len(description),
            )
            
            return description
            
        except Exception as e:
            self.logger.error(
                "LLM description generation failed",
                domain=structure.domain,
                error=str(e),
            )
            # Fall back to a basic description
            return self._generate_fallback(structure)
    
    def _generate_fallback(self, structure: PageStructure) -> str:
        """Generate a basic fallback description without LLM."""
        parts = [f"Page on {structure.domain}"]
        
        if structure.page_type != "unknown":
            parts.append(f"of type {structure.page_type}")
        
        if structure.content_regions:
            regions = [r.name for r in structure.content_regions[:3]]
            parts.append(f"with content in {', '.join(regions)}")
        
        if structure.semantic_landmarks:
            landmarks = list(structure.semantic_landmarks.keys())[:4]
            parts.append(f"using {', '.join(landmarks)} landmarks")
        
        return " ".join(parts) + "."
    
    async def generate_comparison_description(
        self,
        old_structure: PageStructure,
        new_structure: PageStructure,
    ) -> str:
        """
        Generate a description of changes between two structures.
        
        Args:
            old_structure: Previous structure.
            new_structure: New structure.
            
        Returns:
            Description of the changes.
        """
        old_summary = self._build_structure_summary(old_structure)
        new_summary = self._build_structure_summary(new_structure)
        
        prompt = f"""Compare these two versions of a web page structure and describe what changed.
Focus on structural changes that would affect content extraction.

PREVIOUS VERSION:
{old_summary}

NEW VERSION:
{new_summary}

Describe the key structural changes in 2-3 sentences:"""

        try:
            return await self.llm.generate(prompt, max_tokens=200)
        except Exception as e:
            self.logger.error("LLM comparison generation failed", error=str(e))
            return "Unable to generate change description."


# =============================================================================
# Embedding Manager (same as structure_store.py)
# =============================================================================


class EmbeddingManager:
    """
    Manages semantic embeddings for page structures.
    
    Uses sentence-transformers (all-MiniLM-L6-v2) for generating embeddings
    that can be compared for semantic similarity.
    """
    
    EMBEDDING_DIM = 384
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Sentence transformer model to use.
        """
        self.model_name = model_name
        self._model = None
    
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
    
    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding vector from text.
        
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
    
    def batch_generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]


# =============================================================================
# Structure Signature with LLM Description
# =============================================================================


@dataclass
class LLMStructureSignature:
    """Signature with LLM-generated semantic description."""

    domain: str
    page_type: str
    version: int

    # Quick comparison hashes
    content_hash: str
    tag_count_hash: str
    class_set_hash: str
    landmark_hash: str

    # LLM-generated description and embedding
    llm_description: str = ""
    llm_model: str = ""
    embedding: list[float] = field(default_factory=list)

    # Metadata
    captured_at: datetime = field(default_factory=datetime.utcnow)
    sample_url: str = ""
    generation_time_ms: float = 0.0

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
            "llm_description": self.llm_description,
            "llm_model": self.llm_model,
            "embedding": self.embedding,
            "captured_at": self.captured_at.isoformat(),
            "sample_url": self.sample_url,
            "generation_time_ms": self.generation_time_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMStructureSignature":
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
            content_hash=data.get("content_hash", ""),
            tag_count_hash=data.get("tag_count_hash", ""),
            class_set_hash=data.get("class_set_hash", ""),
            landmark_hash=data.get("landmark_hash", ""),
            llm_description=data.get("llm_description", ""),
            llm_model=data.get("llm_model", ""),
            embedding=data.get("embedding", []),
            captured_at=captured_at,
            sample_url=data.get("sample_url", ""),
            generation_time_ms=data.get("generation_time_ms", 0.0),
        )

    @classmethod
    def create_hashes(cls, structure: PageStructure) -> dict[str, str]:
        """Create hash values from structure."""
        # Hash tag counts
        tag_counts = structure.tag_hierarchy.get("tag_counts", {})
        tag_count_hash = hashlib.md5(
            json.dumps(sorted(tag_counts.items())).encode()
        ).hexdigest()[:16]

        # Hash class names
        class_set_hash = hashlib.md5(
            json.dumps(sorted(structure.css_class_map.keys())).encode()
        ).hexdigest()[:16]

        # Hash landmarks
        landmark_hash = hashlib.md5(
            json.dumps(sorted(structure.semantic_landmarks.items())).encode()
        ).hexdigest()[:16]

        return {
            "content_hash": structure.content_hash[:16] if structure.content_hash else "",
            "tag_count_hash": tag_count_hash,
            "class_set_hash": class_set_hash,
            "landmark_hash": landmark_hash,
        }

    def matches(self, other: "LLMStructureSignature") -> bool:
        """Quick check if two signatures match (hash-based)."""
        return (
            self.content_hash == other.content_hash
            and self.tag_count_hash == other.tag_count_hash
            and self.class_set_hash == other.class_set_hash
        )
    
    def semantic_similarity(self, other: "LLMStructureSignature") -> float | None:
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
# LLM Structure Store
# =============================================================================


class LLMStructureStore:
    """
    Redis-based storage for page structures with LLM-generated descriptions.

    Features:
    - LLM-powered semantic descriptions
    - Version history for structures
    - Quick signature comparison (hash-based)
    - Embedding storage for similarity search
    - TTL-based expiration
    - Support for multiple LLM providers
    """

    # Redis key patterns
    STRUCTURE_PREFIX = "crawler:llmstruct:"
    SIGNATURE_PREFIX = "crawler:llmsig:"
    EMBEDDING_PREFIX = "crawler:llmemb:"
    VERSION_PREFIX = "crawler:llmver:"
    INDEX_KEY = "crawler:llmstructure:index"
    STATS_KEY = "crawler:llmstructure:stats"

    def __init__(
        self,
        redis_client: redis.Redis,
        llm_provider: LLMProvider,
        ttl_seconds: int = 604800,  # 7 days
        max_versions: int = 10,
        enable_embeddings: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the LLM structure store.

        Args:
            redis_client: Redis async client.
            llm_provider: LLM provider for description generation.
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
        self.logger = logger or CrawlerLogger("llm_structure_store")
        
        # Initialize LLM description generator
        self._description_generator = LLMSemanticDescriptionGenerator(
            llm_provider=llm_provider,
            logger=self.logger,
        )
        self._llm_model = llm_provider.get_model_name()
        
        # Initialize embedding manager lazily
        self._embedding_manager: EmbeddingManager | None = None
        self._embedding_model = embedding_model

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
        """Get Redis key for signature."""
        return f"{self.SIGNATURE_PREFIX}{domain}:{page_type}"

    def _embedding_key(self, domain: str, page_type: str) -> str:
        """Get Redis key for embedding vector."""
        return f"{self.EMBEDDING_PREFIX}{domain}:{page_type}"

    def _index_entry(self, domain: str, page_type: str) -> str:
        """Create index entry string."""
        return f"{domain}:{page_type}"

    async def save(
        self,
        structure: PageStructure,
        url: str = "",
    ) -> int:
        """
        Save a page structure with LLM-generated description.

        Args:
            structure: PageStructure to save.
            url: Sample URL this structure was captured from.

        Returns:
            Version number assigned.
        """
        import time
        
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

            # Generate LLM description
            start_time = time.time()
            llm_description = await self._description_generator.generate(structure, url)
            generation_time_ms = (time.time() - start_time) * 1000

            # Create signature with hashes
            hashes = LLMStructureSignature.create_hashes(structure)
            
            # Generate embedding if enabled
            embedding = []
            if self.enable_embeddings and self.embedding_manager:
                embedding = self.embedding_manager.generate_embedding(llm_description)

            # Create signature
            signature = LLMStructureSignature(
                domain=domain,
                page_type=page_type,
                version=new_version,
                content_hash=hashes["content_hash"],
                tag_count_hash=hashes["tag_count_hash"],
                class_set_hash=hashes["class_set_hash"],
                landmark_hash=hashes["landmark_hash"],
                llm_description=llm_description,
                llm_model=self._llm_model,
                embedding=embedding,
                captured_at=structure.captured_at,
                sample_url=url,
                generation_time_ms=generation_time_ms,
            )

            # Save signature
            signature_key = self._signature_key(domain, page_type)
            await self.redis.setex(
                signature_key, self.ttl, json.dumps(signature.to_dict())
            )

            # Store embedding separately for efficient retrieval
            if embedding:
                embedding_key = self._embedding_key(domain, page_type)
                await self.redis.setex(
                    embedding_key,
                    self.ttl,
                    json.dumps(embedding),
                )

            # Add to index
            await self.redis.sadd(self.INDEX_KEY, self._index_entry(domain, page_type))

            # Update stats
            await self.redis.hincrby(self.STATS_KEY, "total_saved", 1)
            await self.redis.hset(
                self.STATS_KEY, "last_save", datetime.utcnow().isoformat()
            )
            await self.redis.hset(self.STATS_KEY, "llm_model", self._llm_model)

            # Cleanup old versions
            await self._cleanup_old_versions(domain, page_type, new_version)

            self.logger.info(
                "Saved structure with LLM description",
                domain=domain,
                page_type=page_type,
                version=new_version,
                llm_model=self._llm_model,
                generation_time_ms=f"{generation_time_ms:.1f}",
                has_embedding=bool(embedding),
            )

            metrics.REDIS_OPERATIONS.labels(operation="llm_structure_save", status="success").inc()

            return new_version

        except redis.RedisError as e:
            self.logger.error(
                "Failed to save structure",
                domain=domain,
                page_type=page_type,
                error=str(e),
            )
            metrics.REDIS_OPERATIONS.labels(operation="llm_structure_save", status="error").inc()
            raise

    async def get_latest(
        self,
        domain: str,
        page_type: str,
    ) -> PageStructure | None:
        """Get the most recent structure for a domain/page_type."""
        try:
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
        """Get a specific version of a structure."""
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
    ) -> LLMStructureSignature | None:
        """Get signature with LLM description."""
        try:
            signature_key = self._signature_key(domain, page_type)
            data = await self.redis.get(signature_key)

            if not data:
                return None

            return LLMStructureSignature.from_dict(json.loads(data))

        except redis.RedisError as e:
            self.logger.error(
                "Failed to get signature",
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
        """Get LLM-generated description."""
        signature = await self.get_signature(domain, page_type)
        if signature:
            return signature.llm_description
        return None

    async def get_embedding(
        self,
        domain: str,
        page_type: str,
    ) -> list[float] | None:
        """Get embedding vector."""
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

    async def has_changed(
        self,
        structure: PageStructure,
        url: str = "",
    ) -> bool:
        """Quick check if structure has changed (hash-based)."""
        stored_sig = await self.get_signature(structure.domain, structure.page_type)

        if stored_sig is None:
            return True

        new_hashes = LLMStructureSignature.create_hashes(structure)
        
        return not (
            stored_sig.content_hash == new_hashes["content_hash"]
            and stored_sig.tag_count_hash == new_hashes["tag_count_hash"]
            and stored_sig.class_set_hash == new_hashes["class_set_hash"]
        )

    async def compute_semantic_similarity(
        self,
        domain1: str,
        page_type1: str,
        domain2: str,
        page_type2: str,
    ) -> float | None:
        """Compute semantic similarity between two structures."""
        emb1 = await self.get_embedding(domain1, page_type1)
        emb2 = await self.get_embedding(domain2, page_type2)

        if not emb1 or not emb2:
            return None

        if self.embedding_manager:
            return self.embedding_manager.compute_similarity(emb1, emb2)
        
        return None

    async def generate_change_description(
        self,
        domain: str,
        page_type: str,
        old_version: int,
        new_version: int,
    ) -> str | None:
        """Generate LLM description of changes between versions."""
        old_structure = await self.get_version(domain, page_type, old_version)
        new_structure = await self.get_version(domain, page_type, new_version)

        if not old_structure or not new_structure:
            return None

        return await self._description_generator.generate_comparison_description(
            old_structure, new_structure
        )

    async def find_similar_structures(
        self,
        structure: PageStructure,
        threshold: float = 0.8,
        limit: int = 10,
    ) -> list[tuple[str, str, float]]:
        """Find structures similar to the given one."""
        if not self.enable_embeddings or not self.embedding_manager:
            return []

        # Generate description and embedding for query
        description = await self._description_generator.generate(structure)
        query_embedding = self.embedding_manager.generate_embedding(description)

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

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:limit]

    async def list_domains(self) -> list[tuple[str, str]]:
        """List all tracked domain/page_type pairs."""
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
        """Get store statistics."""
        try:
            stats = await self.redis.hgetall(self.STATS_KEY)
            domains = await self.list_domains()

            embedding_count = 0
            if self.enable_embeddings:
                for domain, page_type in domains:
                    if await self.get_embedding(domain, page_type):
                        embedding_count += 1

            return {
                "total_saved": int(stats.get(b"total_saved", 0)),
                "last_save": stats.get(b"last_save", b"").decode() if stats.get(b"last_save") else "",
                "llm_model": stats.get(b"llm_model", b"").decode() if stats.get(b"llm_model") else "",
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

    async def get_history(
        self,
        domain: str,
        page_type: str,
        limit: int = 10,
    ) -> list[PageStructure]:
        """Get version history for a domain/page_type."""
        try:
            latest_key = self._latest_key(domain, page_type)
            latest = await self.redis.get(latest_key)
            if not latest:
                return []

            latest_version = int(latest)
            structures = []

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

    async def delete(
        self,
        domain: str,
        page_type: str,
    ) -> bool:
        """Delete all structure data for a domain/page_type."""
        try:
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

            if keys_to_delete:
                await self.redis.delete(*keys_to_delete)

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
        """Clear all structure data."""
        try:
            structure_keys = await self.redis.keys(f"{self.STRUCTURE_PREFIX}*")
            signature_keys = await self.redis.keys(f"{self.SIGNATURE_PREFIX}*")
            embedding_keys = await self.redis.keys(f"{self.EMBEDDING_PREFIX}*")
            version_keys = await self.redis.keys(f"{self.VERSION_PREFIX}*")

            all_keys = structure_keys + signature_keys + embedding_keys + version_keys
            if all_keys:
                await self.redis.delete(*all_keys)

            await self.redis.delete(self.INDEX_KEY, self.STATS_KEY)

            count = len(all_keys)
            self.logger.info("Cleared LLM structure store", entries=count)
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
        captured_at = data.get("captured_at")
        if isinstance(captured_at, str):
            captured_at = datetime.fromisoformat(captured_at)
        else:
            captured_at = datetime.utcnow()

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
