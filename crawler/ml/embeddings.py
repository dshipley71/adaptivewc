"""
ML embeddings integration for adaptive web crawler.

Uses sentence transformers (like all-MiniLM-L6-v2) to create embeddings
of page structures for similarity search, clustering, and classification.

Supports multiple classifier backends:
- LogisticRegression (default, fast)
- XGBoost (gradient boosting, good for tabular features)
- LightGBM (fast gradient boosting, handles large datasets)

Supports multiple description generators:
- Rules-based (default, deterministic)
- LLM-based (uses OpenAI/Anthropic for richer descriptions)
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np

from crawler.models import PageStructure, ExtractionStrategy


class ClassifierType(str, Enum):
    """Supported classifier types."""
    LOGISTIC_REGRESSION = "logistic_regression"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


class DescriptionMode(str, Enum):
    """Description generation modes."""
    RULES = "rules"  # Rules-based (deterministic)
    LLM = "llm"  # LLM-based (requires API key)


@dataclass
class StructureEmbedding:
    """Embedding representation of a page structure."""

    domain: str
    page_type: str
    embedding: list[float]
    text_description: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    model_name: str = "all-MiniLM-L6-v2"

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "page_type": self.page_type,
            "embedding": self.embedding,
            "text_description": self.text_description,
            "created_at": self.created_at.isoformat(),
            "model_name": self.model_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StructureEmbedding":
        return cls(
            domain=data["domain"],
            page_type=data["page_type"],
            embedding=data["embedding"],
            text_description=data["text_description"],
            created_at=datetime.fromisoformat(data["created_at"]),
            model_name=data.get("model_name", "all-MiniLM-L6-v2"),
        )


class StructureDescriptionGenerator:
    """Generates ML-friendly text descriptions of page structures."""

    @staticmethod
    def generate(structure: PageStructure) -> str:
        """
        Generate a semantic text description suitable for embedding.

        This description is designed to capture the essential characteristics
        of a page structure in natural language that embedding models can process.
        """
        lines = []

        # Core identity
        lines.append(f"Website structure for {structure.domain}")
        lines.append(f"Page type: {structure.page_type}")

        # DOM structure summary
        tag_counts = structure.tag_hierarchy.get("tag_counts", {}) if structure.tag_hierarchy else {}
        if tag_counts:
            total_elements = sum(tag_counts.values())
            lines.append(f"DOM contains {total_elements} elements with {len(tag_counts)} unique tag types")

            # Key structural elements
            structural_tags = []
            if tag_counts.get("article", 0) > 0:
                structural_tags.append("article elements")
            if tag_counts.get("section", 0) > 0:
                structural_tags.append(f"{tag_counts['section']} sections")
            if tag_counts.get("nav", 0) > 0:
                structural_tags.append(f"{tag_counts['nav']} navigation areas")
            if tag_counts.get("header", 0) > 0:
                structural_tags.append("header")
            if tag_counts.get("footer", 0) > 0:
                structural_tags.append("footer")
            if tag_counts.get("aside", 0) > 0:
                structural_tags.append("sidebar content")
            if tag_counts.get("form", 0) > 0:
                structural_tags.append(f"{tag_counts['form']} forms")
            if tag_counts.get("table", 0) > 0:
                structural_tags.append(f"{tag_counts['table']} data tables")

            if structural_tags:
                lines.append(f"Structure includes: {', '.join(structural_tags)}")

            # Content density indicators
            content_tags = sum(tag_counts.get(t, 0) for t in ["p", "span", "div", "li"])
            interactive_tags = sum(tag_counts.get(t, 0) for t in ["a", "button", "input", "select"])
            media_tags = sum(tag_counts.get(t, 0) for t in ["img", "video", "audio", "iframe"])

            if content_tags > 100:
                lines.append("High text content density")
            if interactive_tags > 50:
                lines.append("Many interactive elements")
            if media_tags > 10:
                lines.append("Rich media content")

        # Semantic landmarks
        if structure.semantic_landmarks:
            landmarks = list(structure.semantic_landmarks.keys())
            lines.append(f"Semantic landmarks: {', '.join(landmarks)}")

        # Content regions
        if structure.content_regions:
            region_names = [r.name for r in structure.content_regions]
            lines.append(f"Content regions: {', '.join(region_names)}")

        # Navigation complexity
        if structure.navigation_selectors:
            nav_count = len(structure.navigation_selectors)
            if nav_count > 10:
                lines.append("Complex navigation structure")
            elif nav_count > 5:
                lines.append("Moderate navigation structure")
            else:
                lines.append("Simple navigation structure")

        # Pagination
        if structure.pagination_pattern:
            lines.append("Has pagination support")

        # Dynamic content indicators
        if structure.iframe_locations:
            lines.append(f"Contains {len(structure.iframe_locations)} embedded iframes")

        if structure.script_signatures:
            # Identify common frameworks
            scripts = " ".join(structure.script_signatures).lower()
            frameworks = []
            if "react" in scripts:
                frameworks.append("React")
            if "vue" in scripts:
                frameworks.append("Vue")
            if "angular" in scripts:
                frameworks.append("Angular")
            if "jquery" in scripts:
                frameworks.append("jQuery")
            if frameworks:
                lines.append(f"Uses: {', '.join(frameworks)}")

        # CSS complexity
        if structure.css_class_map:
            class_count = len(structure.css_class_map)
            if class_count > 200:
                lines.append("Complex CSS styling with many classes")
            elif class_count > 50:
                lines.append("Moderate CSS complexity")

        return " ".join(lines)

    @staticmethod
    def generate_for_similarity(structure: PageStructure) -> str:
        """
        Generate a concise description optimized for similarity matching.

        Focuses on structural characteristics that indicate similar page types.
        """
        features = []

        # Page type
        features.append(f"type:{structure.page_type}")

        # DOM size category
        tag_counts = structure.tag_hierarchy.get("tag_counts", {}) if structure.tag_hierarchy else {}
        total = sum(tag_counts.values()) if tag_counts else 0
        if total > 1000:
            features.append("size:large")
        elif total > 200:
            features.append("size:medium")
        else:
            features.append("size:small")

        # Key structural patterns
        if tag_counts.get("article", 0) > 0:
            features.append("has:article")
        if tag_counts.get("table", 0) > 0:
            features.append("has:tables")
        if tag_counts.get("form", 0) > 0:
            features.append("has:forms")
        if tag_counts.get("nav", 0) > 2:
            features.append("has:complex-nav")

        # Content type indicators
        if structure.pagination_pattern:
            features.append("has:pagination")
        if structure.iframe_locations:
            features.append("has:iframes")

        # Landmarks
        if structure.semantic_landmarks:
            for landmark in structure.semantic_landmarks.keys():
                features.append(f"landmark:{landmark}")

        return " ".join(features)


class StrategyDescriptionGenerator:
    """Generates ML-friendly text descriptions of extraction strategies."""

    @staticmethod
    def generate(strategy: ExtractionStrategy) -> str:
        """Generate a text description of an extraction strategy."""
        lines = []

        lines.append(f"Extraction strategy for {strategy.domain} {strategy.page_type} pages")
        lines.append(f"Learning source: {strategy.learning_source}")

        # Describe selectors
        if strategy.title:
            conf = f"({strategy.title.confidence:.0%} confidence)" if strategy.title.confidence else ""
            lines.append(f"Title extracted using {strategy.title.primary} selector {conf}")

        if strategy.content:
            conf = f"({strategy.content.confidence:.0%} confidence)" if strategy.content.confidence else ""
            lines.append(f"Content extracted using {strategy.content.primary} selector {conf}")

        if strategy.metadata:
            meta_fields = list(strategy.metadata.keys())
            lines.append(f"Metadata fields: {', '.join(meta_fields)}")

        # Extraction requirements
        lines.append(f"Required fields: {', '.join(strategy.required_fields)}")
        lines.append(f"Minimum content length: {strategy.min_content_length} characters")

        return " ".join(lines)


class BaseDescriptionGenerator(ABC):
    """Abstract base class for description generators."""

    @abstractmethod
    def generate(self, structure: PageStructure) -> str:
        """Generate a text description of a page structure."""
        pass

    @abstractmethod
    def generate_for_change_detection(
        self,
        old_structure: PageStructure,
        new_structure: PageStructure,
    ) -> str:
        """Generate description focused on detecting changes."""
        pass


class RulesBasedDescriptionGenerator(BaseDescriptionGenerator):
    """
    Rules-based description generator (deterministic).

    Uses handcrafted rules to generate consistent, structured descriptions.
    Fast, no external dependencies, deterministic output.
    """

    def generate(self, structure: PageStructure) -> str:
        """Generate description using rules."""
        return StructureDescriptionGenerator.generate(structure)

    def generate_for_change_detection(
        self,
        old_structure: PageStructure,
        new_structure: PageStructure,
    ) -> str:
        """Generate change-focused description."""
        lines = []
        lines.append("BEFORE:")
        lines.append(self.generate(old_structure))
        lines.append("AFTER:")
        lines.append(self.generate(new_structure))
        return "\n".join(lines)


class LLMDescriptionGenerator(BaseDescriptionGenerator):
    """
    LLM-based description generator.

    Uses an LLM (OpenAI or Anthropic) to generate rich, semantic descriptions
    that capture nuances rules might miss.

    Requires API key:
    - OPENAI_API_KEY for OpenAI models
    - ANTHROPIC_API_KEY for Claude models
    """

    def __init__(
        self,
        provider: Literal["openai", "anthropic"] = "openai",
        model: str | None = None,
    ):
        """
        Initialize LLM description generator.

        Args:
            provider: "openai" or "anthropic"
            model: Model name (defaults to gpt-4o-mini or claude-3-haiku)
        """
        self.provider = provider
        self.model = model or (
            "gpt-4o-mini" if provider == "openai" else "claude-3-haiku-20240307"
        )
        self._client = None

    def _get_client(self):
        """Lazy initialize the API client."""
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            try:
                from openai import OpenAI
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable required")
                self._client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package required: pip install openai")

        elif self.provider == "anthropic":
            try:
                import anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY environment variable required")
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")

        return self._client

    def _structure_to_json(self, structure: PageStructure) -> str:
        """Convert structure to JSON for LLM context."""
        data = {
            "domain": structure.domain,
            "page_type": structure.page_type,
            "tag_hierarchy": structure.tag_hierarchy,
            "content_regions": [
                {"name": r.name, "selector": r.selector}
                for r in (structure.content_regions or [])
            ],
            "navigation_count": len(structure.navigation_selectors or []),
            "semantic_landmarks": structure.semantic_landmarks,
            "has_pagination": bool(structure.pagination_pattern),
            "css_class_count": len(structure.css_class_map or {}),
            "script_frameworks": self._detect_frameworks(structure),
        }
        return json.dumps(data, indent=2)

    def _detect_frameworks(self, structure: PageStructure) -> list[str]:
        """Detect JavaScript frameworks from script signatures."""
        if not structure.script_signatures:
            return []
        scripts = " ".join(structure.script_signatures).lower()
        frameworks = []
        for fw in ["react", "vue", "angular", "jquery", "next", "nuxt", "svelte"]:
            if fw in scripts:
                frameworks.append(fw)
        return frameworks

    def generate(self, structure: PageStructure) -> str:
        """Generate description using LLM."""
        client = self._get_client()
        structure_json = self._structure_to_json(structure)

        prompt = f"""Analyze this webpage structure and generate a concise semantic description
suitable for ML embedding. Focus on:
- Page type and purpose
- Content organization and layout
- Navigation complexity
- Technical characteristics (frameworks, dynamic content)

Structure:
{structure_json}

Generate a 2-3 sentence description that captures the essence of this page structure."""

        if self.provider == "openai":
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()

        else:  # anthropic
            response = client.messages.create(
                model=self.model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()

    def generate_for_change_detection(
        self,
        old_structure: PageStructure,
        new_structure: PageStructure,
    ) -> str:
        """Generate change-focused description using LLM."""
        client = self._get_client()

        old_json = self._structure_to_json(old_structure)
        new_json = self._structure_to_json(new_structure)

        prompt = f"""Compare these two webpage structures and describe the changes.
Focus on structural differences that would affect web scraping/extraction:
- DOM structure changes
- Navigation changes
- Content region changes
- CSS class changes
- Framework changes

BEFORE:
{old_json}

AFTER:
{new_json}

Generate a 2-3 sentence description of the key changes and their potential impact on extraction."""

        if self.provider == "openai":
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()

        else:  # anthropic
            response = client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()


def get_description_generator(
    mode: DescriptionMode | str = DescriptionMode.RULES,
    **kwargs,
) -> BaseDescriptionGenerator:
    """
    Factory function to get the appropriate description generator.

    Args:
        mode: "rules" for deterministic, "llm" for LLM-based
        **kwargs: Additional arguments for LLM generator (provider, model)

    Returns:
        BaseDescriptionGenerator instance

    Examples:
        # Rules-based (default)
        gen = get_description_generator("rules")

        # LLM-based with OpenAI
        gen = get_description_generator("llm", provider="openai")

        # LLM-based with Anthropic
        gen = get_description_generator("llm", provider="anthropic", model="claude-3-sonnet-20240229")
    """
    if isinstance(mode, str):
        mode = DescriptionMode(mode)

    if mode == DescriptionMode.RULES:
        return RulesBasedDescriptionGenerator()
    elif mode == DescriptionMode.LLM:
        return LLMDescriptionGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown description mode: {mode}")


class StructureEmbeddingModel:
    """
    Creates embeddings of page structures using sentence transformers.

    Usage:
        model = StructureEmbeddingModel()
        embedding = model.embed_structure(page_structure)
        similar = model.find_similar(embedding, all_embeddings, top_k=5)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model name. Options:
                - "all-MiniLM-L6-v2" (fast, good quality, 384 dims)
                - "all-mpnet-base-v2" (slower, best quality, 768 dims)
                - "paraphrase-MiniLM-L6-v2" (optimized for paraphrase)
        """
        self.model_name = model_name
        self._model = None
        self._description_generator = StructureDescriptionGenerator()

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Install with: "
                    "pip install sentence-transformers"
                )
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Create embedding from text."""
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_structure(self, structure: PageStructure) -> StructureEmbedding:
        """
        Create an embedding from a page structure.

        Args:
            structure: PageStructure to embed.

        Returns:
            StructureEmbedding with the vector representation.
        """
        # Generate text description
        text = self._description_generator.generate(structure)

        # Create embedding
        embedding = self.embed_text(text)

        return StructureEmbedding(
            domain=structure.domain,
            page_type=structure.page_type,
            embedding=embedding,
            text_description=text,
            model_name=self.model_name,
        )

    def embed_structures_batch(
        self,
        structures: list[PageStructure],
    ) -> list[StructureEmbedding]:
        """
        Create embeddings for multiple structures efficiently.

        Args:
            structures: List of PageStructure objects.

        Returns:
            List of StructureEmbedding objects.
        """
        model = self._load_model()

        # Generate all descriptions
        texts = [
            self._description_generator.generate(s) for s in structures
        ]

        # Batch encode
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        # Create result objects
        results = []
        for structure, text, embedding in zip(structures, texts, embeddings):
            results.append(StructureEmbedding(
                domain=structure.domain,
                page_type=structure.page_type,
                embedding=embedding.tolist(),
                text_description=text,
                model_name=self.model_name,
            ))

        return results

    def compute_similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Returns:
            Similarity score between 0 and 1.
        """
        a = np.array(embedding1)
        b = np.array(embedding2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def find_similar(
        self,
        query_embedding: list[float],
        embeddings: list[StructureEmbedding],
        top_k: int = 5,
    ) -> list[tuple[StructureEmbedding, float]]:
        """
        Find most similar structures to a query.

        Args:
            query_embedding: The query embedding vector.
            embeddings: List of StructureEmbeddings to search.
            top_k: Number of results to return.

        Returns:
            List of (StructureEmbedding, similarity_score) tuples.
        """
        similarities = []
        for emb in embeddings:
            sim = self.compute_similarity(query_embedding, emb.embedding)
            similarities.append((emb, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]


class StructureClassifier:
    """
    Classifies page structures using embeddings + a classifier.

    This can be trained to predict:
    - Page type (article, product, listing, etc.)
    - Content quality
    - Extraction difficulty
    - Similar known sites

    Supports multiple classifier backends:
    - LogisticRegression (default, fast, interpretable)
    - XGBoost (gradient boosting, handles non-linear patterns)
    - LightGBM (fast gradient boosting, good for large datasets)
    """

    def __init__(
        self,
        embedding_model: StructureEmbeddingModel | None = None,
        classifier_type: ClassifierType | str = ClassifierType.LOGISTIC_REGRESSION,
    ):
        """
        Initialize the classifier.

        Args:
            embedding_model: Embedding model to use.
            classifier_type: Type of classifier backend:
                - "logistic_regression" (default)
                - "xgboost"
                - "lightgbm"
        """
        self.embedding_model = embedding_model or StructureEmbeddingModel()
        if isinstance(classifier_type, str):
            classifier_type = ClassifierType(classifier_type)
        self.classifier_type = classifier_type
        self._classifier = None
        self._label_encoder = None

    def _create_classifier(self):
        """Create the appropriate classifier based on type."""
        if self.classifier_type == ClassifierType.LOGISTIC_REGRESSION:
            try:
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(max_iter=1000)
            except ImportError:
                raise ImportError(
                    "scikit-learn required. Install with: pip install scikit-learn"
                )

        elif self.classifier_type == ClassifierType.XGBOOST:
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    use_label_encoder=False,
                )
            except ImportError:
                raise ImportError(
                    "XGBoost required. Install with: pip install xgboost"
                )

        elif self.classifier_type == ClassifierType.LIGHTGBM:
            try:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    objective="multiclass",
                    verbose=-1,
                )
            except ImportError:
                raise ImportError(
                    "LightGBM required. Install with: pip install lightgbm"
                )

        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

    def train(
        self,
        structures: list[PageStructure],
        labels: list[str],
    ) -> dict[str, float]:
        """
        Train a classifier on structure embeddings.

        Args:
            structures: Training structures.
            labels: Labels for each structure.

        Returns:
            Training metrics.
        """
        try:
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import cross_val_score
        except ImportError:
            raise ImportError(
                "scikit-learn required. Install with: pip install scikit-learn"
            )

        # Create embeddings
        embeddings = self.embedding_model.embed_structures_batch(structures)
        X = np.array([e.embedding for e in embeddings])

        # Encode labels
        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(labels)

        # Create and train classifier
        self._classifier = self._create_classifier()
        self._classifier.fit(X, y)

        # Cross-validation score
        cv_folds = min(5, len(set(labels)))
        if cv_folds >= 2:
            scores = cross_val_score(self._classifier, X, y, cv=cv_folds)
            accuracy = float(scores.mean())
            std = float(scores.std())
        else:
            accuracy = 0.0
            std = 0.0

        return {
            "accuracy": accuracy,
            "std": std,
            "num_samples": len(structures),
            "num_classes": len(set(labels)),
            "classifier_type": self.classifier_type.value,
        }

    def predict(self, structure: PageStructure) -> tuple[str, float]:
        """
        Predict the label for a structure.

        Returns:
            Tuple of (predicted_label, confidence).
        """
        if self._classifier is None:
            raise ValueError("Classifier not trained. Call train() first.")

        embedding = self.embedding_model.embed_structure(structure)
        X = np.array([embedding.embedding])

        # Get prediction and probability
        pred_idx = self._classifier.predict(X)[0]
        proba = self._classifier.predict_proba(X)[0]

        label = self._label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx])

        return label, confidence

    def predict_batch(
        self,
        structures: list[PageStructure],
    ) -> list[tuple[str, float]]:
        """
        Predict labels for multiple structures.

        Args:
            structures: List of structures to classify.

        Returns:
            List of (predicted_label, confidence) tuples.
        """
        if self._classifier is None:
            raise ValueError("Classifier not trained. Call train() first.")

        embeddings = self.embedding_model.embed_structures_batch(structures)
        X = np.array([e.embedding for e in embeddings])

        pred_indices = self._classifier.predict(X)
        probas = self._classifier.predict_proba(X)

        results = []
        for pred_idx, proba in zip(pred_indices, probas):
            label = self._label_encoder.inverse_transform([pred_idx])[0]
            confidence = float(proba[pred_idx])
            results.append((label, confidence))

        return results

    def get_feature_importance(self) -> dict[str, float] | None:
        """
        Get feature importance (for XGBoost/LightGBM).

        Returns:
            Dictionary mapping feature index to importance, or None if not available.
        """
        if self._classifier is None:
            return None

        if hasattr(self._classifier, "feature_importances_"):
            importances = self._classifier.feature_importances_
            return {f"dim_{i}": float(imp) for i, imp in enumerate(importances)}

        return None

    def save(self, path: str) -> None:
        """Save the trained classifier."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "classifier": self._classifier,
                "label_encoder": self._label_encoder,
                "model_name": self.embedding_model.model_name,
                "classifier_type": self.classifier_type.value,
            }, f)

    def load(self, path: str) -> None:
        """Load a trained classifier."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
            self._classifier = data["classifier"]
            self._label_encoder = data["label_encoder"]
            if "classifier_type" in data:
                self.classifier_type = ClassifierType(data["classifier_type"])


class MLChangeDetector:
    """
    ML-based change detection using embeddings.

    Uses embedding similarity to detect structural changes between page versions.
    Can be trained on site-specific data to learn what constitutes a breaking change.

    This complements the rules-based ChangeDetector in crawler/adaptive/change_detector.py
    by using learned representations instead of hand-crafted similarity metrics.
    """

    def __init__(
        self,
        embedding_model: StructureEmbeddingModel | None = None,
        description_generator: BaseDescriptionGenerator | None = None,
        breaking_threshold: float = 0.85,
        classifier_type: ClassifierType | str = ClassifierType.LOGISTIC_REGRESSION,
    ):
        """
        Initialize the ML change detector.

        Args:
            embedding_model: Model for creating embeddings.
            description_generator: Generator for text descriptions.
            breaking_threshold: Similarity below this indicates breaking changes.
            classifier_type: Classifier type for training change prediction.
        """
        self.embedding_model = embedding_model or StructureEmbeddingModel()
        self.description_generator = description_generator or RulesBasedDescriptionGenerator()
        self.breaking_threshold = breaking_threshold
        self.classifier_type = (
            ClassifierType(classifier_type)
            if isinstance(classifier_type, str)
            else classifier_type
        )
        self._change_classifier = None
        self._site_baselines: dict[str, StructureEmbedding] = {}

    def compute_similarity(
        self,
        old_structure: PageStructure,
        new_structure: PageStructure,
    ) -> float:
        """
        Compute embedding similarity between two structures.

        Args:
            old_structure: Previous structure.
            new_structure: Current structure.

        Returns:
            Similarity score between 0 and 1.
        """
        old_emb = self.embedding_model.embed_structure(old_structure)
        new_emb = self.embedding_model.embed_structure(new_structure)

        return self.embedding_model.compute_similarity(
            old_emb.embedding, new_emb.embedding
        )

    def detect_change(
        self,
        old_structure: PageStructure,
        new_structure: PageStructure,
    ) -> dict[str, Any]:
        """
        Detect changes between structures using embeddings.

        Args:
            old_structure: Previous structure.
            new_structure: Current structure.

        Returns:
            Dictionary with change analysis:
            - similarity: Embedding similarity score
            - is_breaking: Whether change is breaking
            - change_description: Text description of changes
            - predicted_impact: If classifier is trained, predicted impact
        """
        # Compute embedding similarity
        similarity = self.compute_similarity(old_structure, new_structure)

        # Generate change description
        change_desc = self.description_generator.generate_for_change_detection(
            old_structure, new_structure
        )

        result = {
            "similarity": similarity,
            "is_breaking": similarity < self.breaking_threshold,
            "change_description": change_desc,
            "domain": new_structure.domain,
            "page_type": new_structure.page_type,
        }

        # If classifier is trained, get predicted impact
        if self._change_classifier is not None:
            impact, confidence = self._predict_change_impact(
                old_structure, new_structure
            )
            result["predicted_impact"] = impact
            result["impact_confidence"] = confidence

        return result

    def set_site_baseline(
        self,
        domain: str,
        structure: PageStructure,
    ) -> None:
        """
        Set baseline structure for a specific site.

        Future comparisons will be against this baseline.

        Args:
            domain: Site domain.
            structure: Baseline structure.
        """
        embedding = self.embedding_model.embed_structure(structure)
        key = f"{domain}:{structure.page_type}"
        self._site_baselines[key] = embedding

    def detect_drift_from_baseline(
        self,
        structure: PageStructure,
    ) -> dict[str, Any] | None:
        """
        Detect drift from site baseline.

        Args:
            structure: Current structure to compare.

        Returns:
            Drift analysis or None if no baseline exists.
        """
        key = f"{structure.domain}:{structure.page_type}"
        baseline = self._site_baselines.get(key)

        if baseline is None:
            return None

        current_emb = self.embedding_model.embed_structure(structure)
        similarity = self.embedding_model.compute_similarity(
            baseline.embedding, current_emb.embedding
        )

        return {
            "similarity_to_baseline": similarity,
            "is_drifted": similarity < self.breaking_threshold,
            "baseline_created": baseline.created_at.isoformat(),
            "domain": structure.domain,
            "page_type": structure.page_type,
        }

    def train_change_classifier(
        self,
        change_pairs: list[tuple[PageStructure, PageStructure]],
        labels: list[str],  # e.g., "breaking", "minor", "cosmetic"
    ) -> dict[str, float]:
        """
        Train a classifier to predict change impact.

        Args:
            change_pairs: List of (old_structure, new_structure) tuples.
            labels: Impact labels for each pair.

        Returns:
            Training metrics.
        """
        try:
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import cross_val_score
        except ImportError:
            raise ImportError(
                "scikit-learn required. Install with: pip install scikit-learn"
            )

        # Create feature vectors from pairs
        features = []
        for old_struct, new_struct in change_pairs:
            old_emb = self.embedding_model.embed_structure(old_struct)
            new_emb = self.embedding_model.embed_structure(new_struct)

            # Combine embeddings: [old, new, diff, similarity]
            old_vec = np.array(old_emb.embedding)
            new_vec = np.array(new_emb.embedding)
            diff_vec = new_vec - old_vec
            similarity = self.embedding_model.compute_similarity(
                old_emb.embedding, new_emb.embedding
            )

            # Feature vector: concatenation + similarity
            feature_vec = np.concatenate([
                old_vec, new_vec, diff_vec, [similarity]
            ])
            features.append(feature_vec)

        X = np.array(features)

        # Create classifier
        self._change_label_encoder = LabelEncoder()
        y = self._change_label_encoder.fit_transform(labels)

        classifier = StructureClassifier(
            classifier_type=self.classifier_type
        )
        self._change_classifier = classifier._create_classifier()
        self._change_classifier.fit(X, y)

        # Cross-validation
        cv_folds = min(5, len(set(labels)))
        if cv_folds >= 2:
            scores = cross_val_score(self._change_classifier, X, y, cv=cv_folds)
            accuracy = float(scores.mean())
            std = float(scores.std())
        else:
            accuracy = 0.0
            std = 0.0

        return {
            "accuracy": accuracy,
            "std": std,
            "num_samples": len(change_pairs),
            "num_classes": len(set(labels)),
            "classifier_type": self.classifier_type.value,
        }

    def _predict_change_impact(
        self,
        old_structure: PageStructure,
        new_structure: PageStructure,
    ) -> tuple[str, float]:
        """Predict change impact using trained classifier."""
        old_emb = self.embedding_model.embed_structure(old_structure)
        new_emb = self.embedding_model.embed_structure(new_structure)

        old_vec = np.array(old_emb.embedding)
        new_vec = np.array(new_emb.embedding)
        diff_vec = new_vec - old_vec
        similarity = self.embedding_model.compute_similarity(
            old_emb.embedding, new_emb.embedding
        )

        feature_vec = np.concatenate([old_vec, new_vec, diff_vec, [similarity]])
        X = feature_vec.reshape(1, -1)

        pred_idx = self._change_classifier.predict(X)[0]
        proba = self._change_classifier.predict_proba(X)[0]

        label = self._change_label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx])

        return label, confidence

    def save(self, path: str) -> None:
        """Save the change detector state."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "classifier": self._change_classifier,
                "label_encoder": getattr(self, "_change_label_encoder", None),
                "baselines": {
                    k: v.to_dict() for k, v in self._site_baselines.items()
                },
                "breaking_threshold": self.breaking_threshold,
                "classifier_type": self.classifier_type.value,
            }, f)

    def load(self, path: str) -> None:
        """Load change detector state."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
            self._change_classifier = data.get("classifier")
            self._change_label_encoder = data.get("label_encoder")
            self._site_baselines = {
                k: StructureEmbedding.from_dict(v)
                for k, v in data.get("baselines", {}).items()
            }
            self.breaking_threshold = data.get("breaking_threshold", 0.85)
            if "classifier_type" in data:
                self.classifier_type = ClassifierType(data["classifier_type"])


def export_training_data(
    structures: list[PageStructure],
    strategies: list[ExtractionStrategy],
    output_path: str,
) -> None:
    """
    Export structure/strategy pairs as training data for fine-tuning.

    Creates a JSONL file suitable for fine-tuning sentence transformers
    or other models.

    Format:
    {"text": "structure description", "label": "page_type", "domain": "..."}
    """
    desc_gen = StructureDescriptionGenerator()

    with open(output_path, "w") as f:
        for structure, strategy in zip(structures, strategies):
            record = {
                "text": desc_gen.generate(structure),
                "label": structure.page_type,
                "domain": structure.domain,
                "has_title_selector": strategy.title is not None,
                "has_content_selector": strategy.content is not None,
                "confidence": sum(strategy.confidence_scores.values()) / len(strategy.confidence_scores)
                if strategy.confidence_scores else 0.0,
            }
            f.write(json.dumps(record) + "\n")


def create_similarity_pairs(
    structures: list[PageStructure],
    same_type_score: float = 1.0,
    diff_type_score: float = 0.0,
) -> list[dict]:
    """
    Create training pairs for contrastive learning.

    Pairs structures of the same page_type as similar,
    different page_type as dissimilar.

    Returns pairs suitable for training sentence-transformers
    with MultipleNegativesRankingLoss.
    """
    desc_gen = StructureDescriptionGenerator()
    pairs = []

    for i, s1 in enumerate(structures):
        for s2 in structures[i + 1:]:
            text1 = desc_gen.generate(s1)
            text2 = desc_gen.generate(s2)

            if s1.page_type == s2.page_type:
                score = same_type_score
            else:
                score = diff_type_score

            pairs.append({
                "sentence1": text1,
                "sentence2": text2,
                "score": score,
            })

    return pairs
