"""
Change detector for adaptive extraction.

Detects and classifies changes between page structures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from crawler.models import ChangeType, PageStructure, StructureChange, Severity
from crawler.utils.logging import CrawlerLogger


class ChangeClassification(str, Enum):
    """Classification of detected changes."""

    COSMETIC = "cosmetic"  # CSS/styling changes only
    MINOR = "minor"  # Small structural changes
    MODERATE = "moderate"  # Significant but adaptable changes
    BREAKING = "breaking"  # Requires re-learning extraction strategy


@dataclass
class ChangeAnalysis:
    """Detailed analysis of structure changes."""

    has_changes: bool
    classification: ChangeClassification
    similarity_score: float
    changes: list[StructureChange] = field(default_factory=list)
    requires_relearning: bool = False
    analyzed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "has_changes": self.has_changes,
            "classification": self.classification.value,
            "similarity_score": self.similarity_score,
            "changes": [c.to_dict() for c in self.changes],
            "requires_relearning": self.requires_relearning,
            "analyzed_at": self.analyzed_at.isoformat(),
        }


class ChangeDetector:
    """
    Detects changes between page structures.

    Compares stored structure fingerprints with current page
    analysis to identify what has changed and whether it
    affects extraction strategies.
    """

    # Thresholds for change classification
    COSMETIC_THRESHOLD = 0.95  # >95% similar = cosmetic
    MINOR_THRESHOLD = 0.85  # >85% similar = minor
    MODERATE_THRESHOLD = 0.70  # >70% similar = moderate
    BREAKING_THRESHOLD = 0.70  # <70% similar = breaking

    # Weights for different structure components
    COMPONENT_WEIGHTS = {
        "tag_hierarchy": 0.30,
        "content_regions": 0.25,
        "navigation": 0.15,
        "landmarks": 0.15,
        "css_classes": 0.10,
        "ids": 0.05,
    }

    def __init__(
        self,
        breaking_threshold: float = 0.70,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the change detector.

        Args:
            breaking_threshold: Similarity below this triggers re-learning.
            logger: Logger instance.
        """
        self.breaking_threshold = breaking_threshold
        self.logger = logger or CrawlerLogger("change_detector")

    def detect_changes(
        self,
        old_structure: PageStructure,
        new_structure: PageStructure,
    ) -> ChangeAnalysis:
        """
        Detect changes between two page structures.

        Args:
            old_structure: Previously stored structure.
            new_structure: Current page structure.

        Returns:
            ChangeAnalysis with detailed breakdown.
        """
        changes: list[StructureChange] = []

        # Compare tag hierarchy
        hierarchy_sim = self._compare_hierarchy(old_structure, new_structure)
        if hierarchy_sim < 1.0:
            changes.append(self._create_change(
                ChangeType.HIERARCHY_CHANGED,
                f"Tag hierarchy similarity: {hierarchy_sim:.2%}",
                1.0 - hierarchy_sim,
            ))

        # Compare content regions
        regions_sim = self._compare_regions(old_structure, new_structure)
        if regions_sim < 1.0:
            changes.append(self._create_change(
                ChangeType.CONTENT_REGION_MOVED,
                f"Content regions similarity: {regions_sim:.2%}",
                1.0 - regions_sim,
            ))

        # Compare navigation
        nav_sim = self._compare_navigation(old_structure, new_structure)
        if nav_sim < 1.0:
            changes.append(self._create_change(
                ChangeType.NAVIGATION_CHANGED,
                f"Navigation similarity: {nav_sim:.2%}",
                1.0 - nav_sim,
            ))

        # Compare landmarks
        landmarks_sim = self._compare_landmarks(old_structure, new_structure)
        if landmarks_sim < 1.0:
            changes.append(self._create_change(
                ChangeType.LANDMARK_REMOVED,
                f"Landmarks similarity: {landmarks_sim:.2%}",
                1.0 - landmarks_sim,
            ))

        # Compare CSS classes
        css_sim = self._compare_css_classes(old_structure, new_structure)
        if css_sim < 1.0:
            changes.append(self._create_change(
                ChangeType.CSS_CLASS_RENAMED,
                f"CSS classes similarity: {css_sim:.2%}",
                1.0 - css_sim,
            ))

        # Compare IDs
        ids_sim = self._compare_ids(old_structure, new_structure)
        if ids_sim < 1.0:
            changes.append(self._create_change(
                ChangeType.ID_CHANGED,
                f"IDs similarity: {ids_sim:.2%}",
                1.0 - ids_sim,
            ))

        # Calculate weighted overall similarity
        overall_similarity = (
            hierarchy_sim * self.COMPONENT_WEIGHTS["tag_hierarchy"] +
            regions_sim * self.COMPONENT_WEIGHTS["content_regions"] +
            nav_sim * self.COMPONENT_WEIGHTS["navigation"] +
            landmarks_sim * self.COMPONENT_WEIGHTS["landmarks"] +
            css_sim * self.COMPONENT_WEIGHTS["css_classes"] +
            ids_sim * self.COMPONENT_WEIGHTS["ids"]
        )

        # Classify the changes
        classification = self._classify_changes(overall_similarity)
        requires_relearning = overall_similarity < self.breaking_threshold

        analysis = ChangeAnalysis(
            has_changes=len(changes) > 0,
            classification=classification,
            similarity_score=overall_similarity,
            changes=changes,
            requires_relearning=requires_relearning,
        )

        if analysis.has_changes:
            self.logger.structure_change(
                domain=new_structure.domain,
                page_type=new_structure.page_type,
                similarity=overall_similarity,
                classification=classification.value,
                requires_relearning=requires_relearning,
            )

        return analysis

    def has_breaking_changes(
        self,
        old_structure: PageStructure,
        new_structure: PageStructure,
    ) -> bool:
        """
        Quick check if structures have breaking changes.

        Args:
            old_structure: Previously stored structure.
            new_structure: Current page structure.

        Returns:
            True if changes require re-learning.
        """
        analysis = self.detect_changes(old_structure, new_structure)
        return analysis.requires_relearning

    def _compare_hierarchy(
        self,
        old: PageStructure,
        new: PageStructure,
    ) -> float:
        """Compare tag hierarchy fingerprints."""
        if old.tag_hierarchy_hash == new.tag_hierarchy_hash:
            return 1.0

        # Compare tag counts
        old_tags = old.tag_counts or {}
        new_tags = new.tag_counts or {}

        if not old_tags and not new_tags:
            return 1.0

        all_tags = set(old_tags.keys()) | set(new_tags.keys())
        if not all_tags:
            return 1.0

        matches = 0
        total = 0
        for tag in all_tags:
            old_count = old_tags.get(tag, 0)
            new_count = new_tags.get(tag, 0)
            total += max(old_count, new_count)
            matches += min(old_count, new_count)

        return matches / total if total > 0 else 1.0

    def _compare_regions(
        self,
        old: PageStructure,
        new: PageStructure,
    ) -> float:
        """Compare content regions."""
        old_regions = {r.region_type for r in (old.content_regions or [])}
        new_regions = {r.region_type for r in (new.content_regions or [])}

        if not old_regions and not new_regions:
            return 1.0

        intersection = old_regions & new_regions
        union = old_regions | new_regions

        return len(intersection) / len(union) if union else 1.0

    def _compare_navigation(
        self,
        old: PageStructure,
        new: PageStructure,
    ) -> float:
        """Compare navigation selectors."""
        old_nav = set(old.navigation_selectors or [])
        new_nav = set(new.navigation_selectors or [])

        if not old_nav and not new_nav:
            return 1.0

        intersection = old_nav & new_nav
        union = old_nav | new_nav

        return len(intersection) / len(union) if union else 1.0

    def _compare_landmarks(
        self,
        old: PageStructure,
        new: PageStructure,
    ) -> float:
        """Compare semantic landmarks."""
        old_landmarks = set(old.semantic_landmarks or [])
        new_landmarks = set(new.semantic_landmarks or [])

        if not old_landmarks and not new_landmarks:
            return 1.0

        intersection = old_landmarks & new_landmarks
        union = old_landmarks | new_landmarks

        return len(intersection) / len(union) if union else 1.0

    def _compare_css_classes(
        self,
        old: PageStructure,
        new: PageStructure,
    ) -> float:
        """Compare CSS class usage."""
        old_classes = set((old.css_class_counts or {}).keys())
        new_classes = set((new.css_class_counts or {}).keys())

        if not old_classes and not new_classes:
            return 1.0

        # Focus on top classes by frequency
        old_top = set(
            sorted(
                (old.css_class_counts or {}).items(),
                key=lambda x: x[1],
                reverse=True,
            )[:50]
        )
        new_top = set(
            sorted(
                (new.css_class_counts or {}).items(),
                key=lambda x: x[1],
                reverse=True,
            )[:50]
        )

        old_names = {name for name, _ in old_top}
        new_names = {name for name, _ in new_top}

        intersection = old_names & new_names
        union = old_names | new_names

        return len(intersection) / len(union) if union else 1.0

    def _compare_ids(
        self,
        old: PageStructure,
        new: PageStructure,
    ) -> float:
        """Compare element IDs."""
        old_ids = set(old.id_attributes or [])
        new_ids = set(new.id_attributes or [])

        if not old_ids and not new_ids:
            return 1.0

        intersection = old_ids & new_ids
        union = old_ids | new_ids

        return len(intersection) / len(union) if union else 1.0

    def _classify_changes(self, similarity: float) -> ChangeClassification:
        """Classify changes based on similarity score."""
        if similarity >= self.COSMETIC_THRESHOLD:
            return ChangeClassification.COSMETIC
        elif similarity >= self.MINOR_THRESHOLD:
            return ChangeClassification.MINOR
        elif similarity >= self.MODERATE_THRESHOLD:
            return ChangeClassification.MODERATE
        else:
            return ChangeClassification.BREAKING

    def _create_change(
        self,
        change_type: ChangeType,
        description: str,
        magnitude: float,
    ) -> StructureChange:
        """Create a StructureChange record."""
        # Determine severity based on magnitude
        if magnitude < 0.1:
            severity = Severity.LOW
        elif magnitude < 0.3:
            severity = Severity.MEDIUM
        elif magnitude < 0.5:
            severity = Severity.HIGH
        else:
            severity = Severity.CRITICAL

        return StructureChange(
            change_type=change_type,
            severity=severity,
            old_value=None,
            new_value=None,
            selector=None,
            description=description,
        )
