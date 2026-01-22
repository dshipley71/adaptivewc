"""ML integration for adaptive web crawler."""

from crawler.ml.embeddings import (
    StructureEmbedding,
    StructureEmbeddingModel,
    StructureClassifier,
    StructureDescriptionGenerator,
    StrategyDescriptionGenerator,
    export_training_data,
    create_similarity_pairs,
)

__all__ = [
    "StructureEmbedding",
    "StructureEmbeddingModel",
    "StructureClassifier",
    "StructureDescriptionGenerator",
    "StrategyDescriptionGenerator",
    "export_training_data",
    "create_similarity_pairs",
]
