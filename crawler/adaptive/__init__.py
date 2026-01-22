"""Adaptive extraction modules for ML-based structure learning."""

from crawler.adaptive.change_detector import (
    ChangeAnalysis,
    ChangeClassification,
    ChangeDetector,
)
from crawler.adaptive.strategy_learner import (
    LearnedStrategy,
    SelectorCandidate,
    StrategyLearner,
)
from crawler.adaptive.structure_analyzer import (
    AnalysisConfig,
    StructureAnalyzer,
)

__all__ = [
    "AnalysisConfig",
    "ChangeAnalysis",
    "ChangeClassification",
    "ChangeDetector",
    "LearnedStrategy",
    "SelectorCandidate",
    "StrategyLearner",
    "StructureAnalyzer",
]
