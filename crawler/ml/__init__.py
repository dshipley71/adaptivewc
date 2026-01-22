"""ML integration for adaptive web crawler."""

from crawler.ml.embeddings import (
    # Enums
    ClassifierType,
    DescriptionMode,
    # Data classes
    StructureEmbedding,
    # Embedding model
    StructureEmbeddingModel,
    # Classifiers
    StructureClassifier,
    # Description generators
    StructureDescriptionGenerator,
    StrategyDescriptionGenerator,
    BaseDescriptionGenerator,
    RulesBasedDescriptionGenerator,
    LLMDescriptionGenerator,
    get_description_generator,
    # Change detection
    MLChangeDetector,
    # Training utilities
    export_training_data,
    create_similarity_pairs,
)

__all__ = [
    # Enums
    "ClassifierType",
    "DescriptionMode",
    # Data classes
    "StructureEmbedding",
    # Embedding model
    "StructureEmbeddingModel",
    # Classifiers
    "StructureClassifier",
    # Description generators
    "StructureDescriptionGenerator",
    "StrategyDescriptionGenerator",
    "BaseDescriptionGenerator",
    "RulesBasedDescriptionGenerator",
    "LLMDescriptionGenerator",
    "get_description_generator",
    # Change detection
    "MLChangeDetector",
    # Training utilities
    "export_training_data",
    "create_similarity_pairs",
]
