# AGENTS.md - ML Module

Complete specification for the ML embeddings and Ollama Cloud integration module.

---

## Module Purpose

The ML module provides:
- Structure embedding generation using sentence transformers
- Ollama Cloud API client for LLM descriptions
- Page type classification using embeddings

---

## Files to Generate

```
fingerprint/ml/
├── __init__.py
├── embeddings.py       # Embedding generation
├── ollama_client.py    # Ollama Cloud API
└── classifier.py       # Page classification
```

---

## fingerprint/ml/__init__.py

```python
"""
ML module - Embeddings and Ollama Cloud integration.
"""

from fingerprint.ml.embeddings import EmbeddingGenerator
from fingerprint.ml.ollama_client import OllamaCloudClient
from fingerprint.ml.classifier import PageClassifier

__all__ = [
    "EmbeddingGenerator",
    "OllamaCloudClient",
    "PageClassifier",
]
```

---

## fingerprint/ml/embeddings.py

```python
"""
Embedding generation for semantic fingerprinting.

Uses sentence-transformers to generate embeddings from structure descriptions.
Default model: all-MiniLM-L6-v2 (384 dimensions)

Verbose logging pattern:
[ML:OPERATION] Message
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from fingerprint.config import EmbeddingsConfig
from fingerprint.core.verbose import get_logger
from fingerprint.exceptions import EmbeddingError, ModelLoadError
from fingerprint.models import PageStructure, StructureEmbedding


class EmbeddingGenerator:
    """
    Generates embeddings from page structures.

    Uses sentence transformers to create semantic vectors from
    structure descriptions, enabling similarity comparisons.

    Usage:
        generator = EmbeddingGenerator(config.embeddings)
        embedding = await generator.generate(structure)
        similarity = generator.cosine_similarity(emb1.vector, emb2.vector)
    """

    def __init__(self, config: EmbeddingsConfig):
        self.config = config
        self.logger = get_logger()
        self._model: SentenceTransformer | None = None

        self.logger.info(
            "ML", "INIT",
            f"Embedding generator initialized",
            model=config.model,
        )

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            self.logger.info(
                "ML", "LOAD",
                f"Loading embedding model: {self.config.model}",
            )
            try:
                self._model = SentenceTransformer(self.config.model)
                self.logger.info(
                    "ML", "LOADED",
                    "Model loaded successfully",
                    dimensions=self._model.get_sentence_embedding_dimension(),
                )
            except Exception as e:
                self.logger.error("ML", "LOAD_ERROR", str(e))
                raise ModelLoadError(f"Failed to load model {self.config.model}: {e}")

        return self._model

    async def generate(self, structure: PageStructure) -> StructureEmbedding:
        """
        Generate embedding from page structure.

        Args:
            structure: PageStructure to embed

        Returns:
            StructureEmbedding with vector representation

        Verbose output:
            [ML:EMBED] Generating embedding for example.com/article
            [ML:DESCRIBE] Creating structure description
              - tags: 45
              - classes: 89
              - landmarks: 5
            [ML:VECTOR] Embedding generated
              - dimensions: 384
              - norm: 1.0
        """
        self.logger.info(
            "ML", "EMBED",
            f"Generating embedding for {structure.domain}/{structure.page_type}",
        )

        # Create text description of structure
        description = self._create_description(structure)

        self.logger.debug(
            "ML", "DESCRIBE",
            "Created structure description",
            description_length=len(description),
            tags=len(structure.tag_hierarchy.tag_counts) if structure.tag_hierarchy else 0,
            classes=len(structure.css_class_map),
            landmarks=len(structure.semantic_landmarks),
        )

        # Generate embedding
        try:
            vector = self.model.encode(description, normalize_embeddings=True)
            vector_list = vector.tolist()
        except Exception as e:
            self.logger.error("ML", "EMBED_ERROR", str(e))
            raise EmbeddingError(f"Failed to generate embedding: {e}")

        self.logger.info(
            "ML", "VECTOR",
            "Embedding generated",
            dimensions=len(vector_list),
            norm=f"{np.linalg.norm(vector):.3f}",
        )

        return StructureEmbedding(
            domain=structure.domain,
            page_type=structure.page_type,
            variant_id=structure.variant_id,
            vector=vector_list,
            dimensions=len(vector_list),
            model_name=self.config.model,
            description=description,
        )

    def cosine_similarity(
        self,
        vector1: list[float],
        vector2: list[float],
    ) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vector1: First embedding vector
            vector2: Second embedding vector

        Returns:
            Similarity score between 0 and 1

        Verbose output:
            [ML:SIMILARITY] Calculating cosine similarity
              - result: 0.923
        """
        v1 = np.array(vector1)
        v2 = np.array(vector2)

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            similarity = 0.0
        else:
            similarity = float(dot_product / (norm1 * norm2))

        self.logger.debug(
            "ML", "SIMILARITY",
            f"Cosine similarity: {similarity:.3f}",
        )

        return similarity

    def _create_description(self, structure: PageStructure) -> str:
        """
        Create text description of structure for embedding.

        Format:
        "Page type: {type}. Framework: {framework}.
        Landmarks: {landmarks}. Main tags: {tags}.
        Key classes: {classes}."
        """
        parts = []

        # Page type
        parts.append(f"Page type: {structure.page_type}")

        # Framework
        if structure.detected_framework:
            parts.append(f"Framework: {structure.detected_framework}")

        # Landmarks
        if structure.semantic_landmarks:
            landmarks = ", ".join(structure.semantic_landmarks.keys())
            parts.append(f"Landmarks: {landmarks}")

        # Main tags (top 10 by frequency)
        if structure.tag_hierarchy and structure.tag_hierarchy.tag_counts:
            top_tags = sorted(
                structure.tag_hierarchy.tag_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            tags_str = ", ".join(f"{tag}({count})" for tag, count in top_tags)
            parts.append(f"Main tags: {tags_str}")

        # Key classes (top 15)
        if structure.css_class_map:
            top_classes = sorted(
                structure.css_class_map.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:15]
            classes_str = ", ".join(cls for cls, _ in top_classes)
            parts.append(f"Key classes: {classes_str}")

        # Content regions
        if structure.content_regions:
            regions = ", ".join(r.name for r in structure.content_regions)
            parts.append(f"Content regions: {regions}")

        return ". ".join(parts) + "."
```

---

## fingerprint/ml/ollama_client.py

```python
"""
Ollama Cloud API client for LLM-powered descriptions.

Endpoint: POST https://ollama.com/api/chat
Authentication: Bearer token

Verbose logging pattern:
[OLLAMA:OPERATION] Message
"""

import httpx

from fingerprint.config import OllamaCloudConfig
from fingerprint.core.verbose import get_logger
from fingerprint.exceptions import (
    OllamaAuthError,
    OllamaCloudError,
    OllamaRateLimitError,
    OllamaTimeoutError,
)
from fingerprint.models import PageStructure


class OllamaCloudClient:
    """
    Client for Ollama Cloud API.

    Generates rich, human-readable descriptions of page structures
    using LLM capabilities.

    Usage:
        client = OllamaCloudClient(config.ollama_cloud)
        description = await client.describe_structure(structure)
    """

    API_URL = "https://ollama.com/api/chat"

    def __init__(self, config: OllamaCloudConfig):
        self.config = config
        self.logger = get_logger()

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout),
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
        )

        self.logger.info(
            "OLLAMA", "INIT",
            "Ollama Cloud client initialized",
            model=config.model,
            enabled=config.enabled,
        )

    async def describe_structure(self, structure: PageStructure) -> str:
        """
        Generate LLM description of page structure.

        Args:
            structure: PageStructure to describe

        Returns:
            Human-readable description from LLM

        Raises:
            OllamaAuthError: Invalid API key
            OllamaTimeoutError: Request timed out
            OllamaRateLimitError: Rate limited
            OllamaCloudError: Other API errors

        Verbose output:
            [OLLAMA:REQUEST] Generating description
              - model: gemma3:12b
              - domain: example.com
              - page_type: article
            [OLLAMA:RESPONSE] Description received
              - tokens: 156
              - duration_ms: 1234
        """
        if not self.config.enabled:
            self.logger.warn("OLLAMA", "DISABLED", "Ollama Cloud is disabled")
            return "Ollama Cloud disabled"

        if not self.config.api_key:
            self.logger.error("OLLAMA", "NO_KEY", "No API key configured")
            raise OllamaAuthError("OLLAMA_CLOUD_API_KEY not set")

        self.logger.info(
            "OLLAMA", "REQUEST",
            "Generating description",
            model=self.config.model,
            domain=structure.domain,
            page_type=structure.page_type,
        )

        # Build prompt
        prompt = self._build_prompt(structure)

        # Build request payload
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": self.config.max_tokens,
                "temperature": self.config.temperature,
            },
        }

        # Make request with retries
        response = await self._make_request(payload)

        # Extract response
        description = response.get("message", {}).get("content", "")

        self.logger.info(
            "OLLAMA", "RESPONSE",
            "Description received",
            content_length=len(description),
        )

        return description

    async def analyze_change(
        self,
        old_structure: PageStructure,
        new_structure: PageStructure,
    ) -> str:
        """
        Generate LLM analysis of structural changes.

        Args:
            old_structure: Previous structure
            new_structure: Current structure

        Returns:
            Human-readable change analysis

        Verbose output:
            [OLLAMA:ANALYZE] Analyzing structural changes
            [OLLAMA:RESPONSE] Analysis received
        """
        self.logger.info(
            "OLLAMA", "ANALYZE",
            "Analyzing structural changes",
            domain=new_structure.domain,
        )

        prompt = self._build_change_prompt(old_structure, new_structure)

        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": self.config.max_tokens,
                "temperature": self.config.temperature,
            },
        }

        response = await self._make_request(payload)
        analysis = response.get("message", {}).get("content", "")

        self.logger.info(
            "OLLAMA", "RESPONSE",
            "Change analysis received",
            content_length=len(analysis),
        )

        return analysis

    async def _make_request(self, payload: dict) -> dict:
        """Make API request with retry logic."""
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                self.logger.debug(
                    "OLLAMA", "ATTEMPT",
                    f"Request attempt {attempt + 1}/{self.config.max_retries}",
                )

                response = await self._client.post(self.API_URL, json=payload)

                # Handle errors
                if response.status_code == 401:
                    self.logger.error("OLLAMA", "AUTH_ERROR", "Invalid API key")
                    raise OllamaAuthError("Invalid API key")

                if response.status_code == 429:
                    self.logger.warn("OLLAMA", "RATE_LIMIT", "Rate limited")
                    raise OllamaRateLimitError("Rate limited by Ollama Cloud")

                if response.status_code >= 400:
                    self.logger.error(
                        "OLLAMA", "API_ERROR",
                        f"API error: {response.status_code}",
                    )
                    raise OllamaCloudError(f"API error: {response.status_code}")

                return response.json()

            except httpx.TimeoutException as e:
                last_error = OllamaTimeoutError(f"Request timed out: {e}")
                self.logger.warn(
                    "OLLAMA", "TIMEOUT",
                    f"Timeout, retry {attempt + 1}/{self.config.max_retries}",
                )

            except (OllamaAuthError, OllamaRateLimitError):
                raise  # Don't retry auth/rate limit errors

            except Exception as e:
                last_error = OllamaCloudError(f"Request failed: {e}")
                self.logger.warn(
                    "OLLAMA", "ERROR",
                    f"Error, retry {attempt + 1}/{self.config.max_retries}",
                    error=str(e),
                )

        raise last_error or OllamaCloudError("Request failed after retries")

    def _build_prompt(self, structure: PageStructure) -> str:
        """Build prompt for structure description."""
        # Summarize structure
        tag_count = len(structure.tag_hierarchy.tag_counts) if structure.tag_hierarchy else 0
        class_count = len(structure.css_class_map)
        landmark_list = ", ".join(structure.semantic_landmarks.keys()) if structure.semantic_landmarks else "none"

        top_tags = ""
        if structure.tag_hierarchy and structure.tag_hierarchy.tag_counts:
            sorted_tags = sorted(
                structure.tag_hierarchy.tag_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            top_tags = ", ".join(f"{tag}({count})" for tag, count in sorted_tags)

        top_classes = ""
        if structure.css_class_map:
            sorted_classes = sorted(
                structure.css_class_map.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:15]
            top_classes = ", ".join(cls for cls, _ in sorted_classes)

        return f"""Describe this web page structure concisely:

Domain: {structure.domain}
Page Type: {structure.page_type}
Framework: {structure.detected_framework or "unknown"}
Semantic Landmarks: {landmark_list}
Tag Count: {tag_count}
Top Tags: {top_tags}
Class Count: {class_count}
Top Classes: {top_classes}

Provide a 2-3 sentence description of the page's structure and purpose. Focus on:
1. What type of content this page likely contains
2. How the page is organized (landmarks, regions)
3. Any notable structural patterns"""

    def _build_change_prompt(
        self,
        old_structure: PageStructure,
        new_structure: PageStructure,
    ) -> str:
        """Build prompt for change analysis."""
        old_classes = set(old_structure.css_class_map.keys())
        new_classes = set(new_structure.css_class_map.keys())
        added = new_classes - old_classes
        removed = old_classes - new_classes

        old_landmarks = set(old_structure.semantic_landmarks.keys())
        new_landmarks = set(new_structure.semantic_landmarks.keys())

        return f"""Analyze these structural changes to a web page:

Domain: {new_structure.domain}
Page Type: {new_structure.page_type}

CSS Classes:
- Added: {len(added)} ({', '.join(list(added)[:10])})
- Removed: {len(removed)} ({', '.join(list(removed)[:10])})

Landmarks:
- Old: {', '.join(old_landmarks)}
- New: {', '.join(new_landmarks)}

Framework:
- Old: {old_structure.detected_framework or "unknown"}
- New: {new_structure.detected_framework or "unknown"}

Provide a 2-3 sentence analysis of:
1. What likely changed (redesign, CSS refactor, framework update)
2. Impact on content extraction (breaking or compatible)
3. Recommended action (adapt selectors, learn new strategy, flag for review)"""

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
        self.logger.debug("OLLAMA", "CLOSE", "Client closed")
```

---

## fingerprint/ml/classifier.py

```python
"""
Page type classification using embeddings.

Uses cosine similarity with reference embeddings to classify page types.

Verbose logging pattern:
[CLASSIFY:OPERATION] Message
"""

from fingerprint.config import Config
from fingerprint.core.verbose import get_logger
from fingerprint.ml.embeddings import EmbeddingGenerator
from fingerprint.models import PageStructure


class PageClassifier:
    """
    Classifies pages using embedding similarity.

    Maintains reference embeddings for known page types and
    classifies new pages by finding the most similar reference.

    Usage:
        classifier = PageClassifier(config)
        page_type = await classifier.classify(structure)
    """

    # Reference descriptions for page types
    REFERENCE_DESCRIPTIONS = {
        "article": "Page type: article. Landmarks: header, nav, main, footer, article. "
                   "Main tags: article, p, h1, h2, section. Content regions: title, content, author, date.",
        "listing": "Page type: listing. Landmarks: header, nav, main, footer. "
                   "Main tags: ul, li, a, div, article. Content regions: list items, pagination.",
        "product": "Page type: product. Landmarks: header, nav, main, footer. "
                   "Main tags: div, img, span, button, form. Content regions: title, price, description, images.",
        "home": "Page type: home. Landmarks: header, nav, main, footer. "
                "Main tags: div, section, a, img, h2. Content regions: hero, featured, categories.",
    }

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger()
        self.embedding_generator = EmbeddingGenerator(config.embeddings)
        self._reference_embeddings: dict[str, list[float]] | None = None

    async def _load_references(self) -> dict[str, list[float]]:
        """Load or generate reference embeddings."""
        if self._reference_embeddings is None:
            self.logger.info(
                "CLASSIFY", "LOAD",
                "Generating reference embeddings",
            )

            self._reference_embeddings = {}

            for page_type, description in self.REFERENCE_DESCRIPTIONS.items():
                # Create a minimal structure for embedding
                vector = self.embedding_generator.model.encode(
                    description,
                    normalize_embeddings=True,
                )
                self._reference_embeddings[page_type] = vector.tolist()

                self.logger.debug(
                    "CLASSIFY", "REFERENCE",
                    f"Generated reference for {page_type}",
                )

        return self._reference_embeddings

    async def classify(self, structure: PageStructure) -> str:
        """
        Classify page type using embedding similarity.

        Args:
            structure: PageStructure to classify

        Returns:
            Classified page type

        Verbose output:
            [CLASSIFY:START] Classifying page structure
              - domain: example.com
            [CLASSIFY:COMPARE] Comparing with references
              - article: 0.82
              - listing: 0.45
              - product: 0.38
              - home: 0.52
            [CLASSIFY:RESULT] Classification: article (0.82)
        """
        self.logger.info(
            "CLASSIFY", "START",
            "Classifying page structure",
            domain=structure.domain,
        )

        # Generate embedding for structure
        embedding = await self.embedding_generator.generate(structure)

        # Load references
        references = await self._load_references()

        # Find most similar reference
        similarities: dict[str, float] = {}

        for page_type, ref_vector in references.items():
            similarity = self.embedding_generator.cosine_similarity(
                embedding.vector,
                ref_vector,
            )
            similarities[page_type] = similarity

        self.logger.debug(
            "CLASSIFY", "COMPARE",
            "Reference similarities",
            **{k: f"{v:.2f}" for k, v in similarities.items()},
        )

        # Get best match
        best_type = max(similarities, key=similarities.get)  # type: ignore
        best_score = similarities[best_type]

        self.logger.info(
            "CLASSIFY", "RESULT",
            f"Classification: {best_type} ({best_score:.2f})",
        )

        return best_type

    async def classify_with_confidence(
        self,
        structure: PageStructure,
    ) -> tuple[str, float]:
        """
        Classify page type and return confidence score.

        Returns:
            Tuple of (page_type, confidence)
        """
        page_type = await self.classify(structure)

        # Confidence is the similarity score
        references = await self._load_references()
        embedding = await self.embedding_generator.generate(structure)

        confidence = self.embedding_generator.cosine_similarity(
            embedding.vector,
            references[page_type],
        )

        return page_type, confidence
```

---

## Ollama Cloud API Reference

### Endpoint

```
POST https://ollama.com/api/chat
```

### Authentication

```
Authorization: Bearer {OLLAMA_CLOUD_API_KEY}
```

### Request Format

```json
{
    "model": "gemma3:12b",
    "messages": [
        {
            "role": "user",
            "content": "Describe this page structure..."
        }
    ],
    "stream": false,
    "options": {
        "num_predict": 500,
        "temperature": 0.3
    }
}
```

### Response Format

```json
{
    "message": {
        "role": "assistant",
        "content": "This is an article page with a standard blog layout..."
    }
}
```

### Error Codes

| Status | Error | Action |
|--------|-------|--------|
| 401 | Invalid API key | Raise OllamaAuthError |
| 429 | Rate limited | Raise OllamaRateLimitError |
| 500+ | Server error | Retry with backoff |

---

## Verbose Logging

All ML module operations use consistent logging:

| Operation | Description |
|-----------|-------------|
| ML:INIT | Embedding generator initialized |
| ML:LOAD | Loading embedding model |
| ML:LOADED | Model loaded successfully |
| ML:EMBED | Generating embedding |
| ML:DESCRIBE | Creating structure description |
| ML:VECTOR | Embedding generated |
| ML:SIMILARITY | Calculating cosine similarity |
| OLLAMA:INIT | Client initialized |
| OLLAMA:REQUEST | Making API request |
| OLLAMA:ATTEMPT | Request attempt |
| OLLAMA:RESPONSE | Response received |
| OLLAMA:ANALYZE | Analyzing changes |
| OLLAMA:ERROR | API error |
| CLASSIFY:START | Starting classification |
| CLASSIFY:COMPARE | Comparing with references |
| CLASSIFY:RESULT | Classification result |

### Example Output

```
[2024-01-15T10:30:00Z] [ML:INIT] Embedding generator initialized
  - model: all-MiniLM-L6-v2

[2024-01-15T10:30:01Z] [ML:LOAD] Loading embedding model: all-MiniLM-L6-v2

[2024-01-15T10:30:03Z] [ML:LOADED] Model loaded successfully
  - dimensions: 384

[2024-01-15T10:30:03Z] [ML:EMBED] Generating embedding for example.com/article

[2024-01-15T10:30:03Z] [ML:DESCRIBE] Created structure description
  - description_length: 256
  - tags: 45
  - classes: 89

[2024-01-15T10:30:03Z] [ML:VECTOR] Embedding generated
  - dimensions: 384
  - norm: 1.000

[2024-01-15T10:30:04Z] [OLLAMA:REQUEST] Generating description
  - model: gemma3:12b
  - domain: example.com
  - page_type: article

[2024-01-15T10:30:06Z] [OLLAMA:RESPONSE] Description received
  - content_length: 312
```
