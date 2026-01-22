"""
ML embeddings integration for adaptive web crawler.

Uses sentence transformers (like all-MiniLM-L6-v2) to create embeddings
of page structures for similarity search, clustering, and classification.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from crawler.models import PageStructure, ExtractionStrategy


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
    """

    def __init__(self, embedding_model: StructureEmbeddingModel | None = None):
        self.embedding_model = embedding_model or StructureEmbeddingModel()
        self._classifier = None
        self._label_encoder = None

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
            from sklearn.linear_model import LogisticRegression
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

        # Train classifier
        self._classifier = LogisticRegression(max_iter=1000)
        self._classifier.fit(X, y)

        # Cross-validation score
        scores = cross_val_score(self._classifier, X, y, cv=min(5, len(labels)))

        return {
            "accuracy": float(scores.mean()),
            "std": float(scores.std()),
            "num_samples": len(structures),
            "num_classes": len(set(labels)),
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

    def save(self, path: str) -> None:
        """Save the trained classifier."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "classifier": self._classifier,
                "label_encoder": self._label_encoder,
                "model_name": self.embedding_model.model_name,
            }, f)

    def load(self, path: str) -> None:
        """Load a trained classifier."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
            self._classifier = data["classifier"]
            self._label_encoder = data["label_encoder"]


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
