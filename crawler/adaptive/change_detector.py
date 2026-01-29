"""
Change detector for adaptive extraction.

Detects and classifies changes between page structures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from crawler.models import ChangeType, PageStructure, StructureChange
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
    diff_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "has_changes": self.has_changes,
            "classification": self.classification.value,
            "similarity_score": self.similarity_score,
            "change_count": len(self.changes),
            "requires_relearning": self.requires_relearning,
            "analyzed_at": self.analyzed_at.isoformat(),
        }

        # Include detailed changes
        if self.changes:
            result["changes"] = [
                {
                    "change_type": c.change_type.value,
                    "affected_components": c.affected_components,
                    "reason": c.reason,
                    "breaking": c.breaking,
                    "evidence": c.evidence,
                }
                for c in self.changes
            ]

        # Include diff details
        if self.diff_details:
            result["diff_details"] = self.diff_details

        return result


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
        all_diffs: dict[str, Any] = {}
        now = datetime.utcnow()

        # Compare tag hierarchy
        hierarchy_sim, hierarchy_diff = self._compare_hierarchy(old_structure, new_structure)
        if hierarchy_diff:
            all_diffs["tag_hierarchy"] = hierarchy_diff
        if hierarchy_sim < 0.9:
            changes.append(StructureChange(
                domain=new_structure.domain,
                page_type=new_structure.page_type,
                detected_at=now,
                previous_version=old_structure.version,
                new_version=new_structure.version,
                change_type=ChangeType.STRUCTURE_REORGANIZED,
                affected_components=["tag_hierarchy"],
                reason=f"Tag hierarchy similarity: {hierarchy_sim:.2%}",
                evidence=hierarchy_diff,
                breaking=hierarchy_sim < self.breaking_threshold,
                confidence=1.0 - hierarchy_sim,
            ))

        # Compare content regions
        regions_sim, regions_diff = self._compare_regions(old_structure, new_structure)
        if regions_diff:
            all_diffs["content_regions"] = regions_diff
        if regions_sim < 0.9:
            changes.append(StructureChange(
                domain=new_structure.domain,
                page_type=new_structure.page_type,
                detected_at=now,
                previous_version=old_structure.version,
                new_version=new_structure.version,
                change_type=ChangeType.CONTENT_RELOCATED,
                affected_components=["content_regions"],
                reason=f"Content regions similarity: {regions_sim:.2%}",
                evidence=regions_diff,
                breaking=regions_sim < self.breaking_threshold,
                confidence=1.0 - regions_sim,
            ))

        # Compare navigation
        nav_sim, nav_diff = self._compare_navigation(old_structure, new_structure)
        if nav_diff:
            all_diffs["navigation"] = nav_diff
        if nav_sim < 0.9:
            changes.append(StructureChange(
                domain=new_structure.domain,
                page_type=new_structure.page_type,
                detected_at=now,
                previous_version=old_structure.version,
                new_version=new_structure.version,
                change_type=ChangeType.URL_PATTERN_CHANGED,
                affected_components=["navigation"],
                reason=f"Navigation similarity: {nav_sim:.2%}",
                evidence=nav_diff,
                breaking=nav_sim < self.breaking_threshold,
                confidence=1.0 - nav_sim,
            ))

        # Compare landmarks
        landmarks_sim, landmarks_diff = self._compare_landmarks(old_structure, new_structure)
        if landmarks_diff:
            all_diffs["landmarks"] = landmarks_diff
        if landmarks_sim < 0.9:
            changes.append(StructureChange(
                domain=new_structure.domain,
                page_type=new_structure.page_type,
                detected_at=now,
                previous_version=old_structure.version,
                new_version=new_structure.version,
                change_type=ChangeType.STRUCTURE_REORGANIZED,
                affected_components=["semantic_landmarks"],
                reason=f"Landmarks similarity: {landmarks_sim:.2%}",
                evidence=landmarks_diff,
                breaking=landmarks_sim < self.breaking_threshold,
                confidence=1.0 - landmarks_sim,
            ))

        # Compare CSS classes
        css_sim, css_diff = self._compare_css_classes(old_structure, new_structure)
        if css_diff:
            all_diffs["css_classes"] = css_diff
        if css_sim < 0.9:
            changes.append(StructureChange(
                domain=new_structure.domain,
                page_type=new_structure.page_type,
                detected_at=now,
                previous_version=old_structure.version,
                new_version=new_structure.version,
                change_type=ChangeType.CLASS_RENAMED,
                affected_components=["css_classes"],
                reason=f"CSS classes similarity: {css_sim:.2%}",
                evidence=css_diff,
                breaking=css_sim < self.breaking_threshold,
                confidence=1.0 - css_sim,
            ))

        # Compare IDs
        ids_sim, ids_diff = self._compare_ids(old_structure, new_structure)
        if ids_diff:
            all_diffs["ids"] = ids_diff
        if ids_sim < 0.9:
            changes.append(StructureChange(
                domain=new_structure.domain,
                page_type=new_structure.page_type,
                detected_at=now,
                previous_version=old_structure.version,
                new_version=new_structure.version,
                change_type=ChangeType.ID_CHANGED,
                affected_components=["id_attributes"],
                reason=f"IDs similarity: {ids_sim:.2%}",
                evidence=ids_diff,
                breaking=ids_sim < self.breaking_threshold,
                confidence=1.0 - ids_sim,
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
            diff_details=all_diffs,
        )

        if analysis.has_changes:
            self.logger.info(
                "Structure change detected",
                domain=new_structure.domain,
                page_type=new_structure.page_type,
                similarity=f"{overall_similarity:.2%}",
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
    ) -> tuple[float, dict[str, Any]]:
        """Compare tag hierarchy fingerprints. Returns (similarity, diff_details)."""
        diff_details: dict[str, Any] = {}

        # Compare content hash first for quick check
        if old.content_hash and new.content_hash:
            if old.content_hash == new.content_hash:
                return 1.0, diff_details

        # Compare tag counts from tag_hierarchy
        old_tags = old.tag_hierarchy.get("tag_counts", {}) if old.tag_hierarchy else {}
        new_tags = new.tag_hierarchy.get("tag_counts", {}) if new.tag_hierarchy else {}

        if not old_tags and not new_tags:
            return 1.0, diff_details

        all_tags = set(old_tags.keys()) | set(new_tags.keys())
        if not all_tags:
            return 1.0, diff_details

        matches = 0
        total = 0
        added_tags = {}
        removed_tags = {}
        changed_tags = {}

        for tag in all_tags:
            old_count = old_tags.get(tag, 0)
            new_count = new_tags.get(tag, 0)
            total += max(old_count, new_count)
            matches += min(old_count, new_count)

            if old_count == 0 and new_count > 0:
                added_tags[tag] = new_count
            elif new_count == 0 and old_count > 0:
                removed_tags[tag] = old_count
            elif old_count != new_count:
                changed_tags[tag] = {"old": old_count, "new": new_count, "delta": new_count - old_count}

        if added_tags:
            diff_details["added_tags"] = added_tags
        if removed_tags:
            diff_details["removed_tags"] = removed_tags
        if changed_tags:
            diff_details["changed_tags"] = changed_tags

        return (matches / total if total > 0 else 1.0), diff_details

    def _compare_regions(
        self,
        old: PageStructure,
        new: PageStructure,
    ) -> tuple[float, dict[str, Any]]:
        """Compare content regions. Returns (similarity, diff_details)."""
        diff_details: dict[str, Any] = {}

        old_regions = {r.name for r in (old.content_regions or [])}
        new_regions = {r.name for r in (new.content_regions or [])}

        if not old_regions and not new_regions:
            return 1.0, diff_details

        added = new_regions - old_regions
        removed = old_regions - new_regions

        if added:
            diff_details["added_regions"] = list(added)
        if removed:
            diff_details["removed_regions"] = list(removed)

        intersection = old_regions & new_regions
        union = old_regions | new_regions

        return (len(intersection) / len(union) if union else 1.0), diff_details

    def _compare_navigation(
        self,
        old: PageStructure,
        new: PageStructure,
    ) -> tuple[float, dict[str, Any]]:
        """Compare navigation selectors. Returns (similarity, diff_details)."""
        diff_details: dict[str, Any] = {}

        old_nav = set(old.navigation_selectors or [])
        new_nav = set(new.navigation_selectors or [])

        if not old_nav and not new_nav:
            return 1.0, diff_details

        added = new_nav - old_nav
        removed = old_nav - new_nav

        if added:
            diff_details["added_navigation"] = list(added)[:20]  # Limit to top 20
        if removed:
            diff_details["removed_navigation"] = list(removed)[:20]

        intersection = old_nav & new_nav
        union = old_nav | new_nav

        return (len(intersection) / len(union) if union else 1.0), diff_details

    def _compare_landmarks(
        self,
        old: PageStructure,
        new: PageStructure,
    ) -> tuple[float, dict[str, Any]]:
        """Compare semantic landmarks. Returns (similarity, diff_details)."""
        diff_details: dict[str, Any] = {}

        old_landmarks = old.semantic_landmarks or {}
        new_landmarks = new.semantic_landmarks or {}

        old_keys = set(old_landmarks.keys())
        new_keys = set(new_landmarks.keys())

        if not old_keys and not new_keys:
            return 1.0, diff_details

        added = new_keys - old_keys
        removed = old_keys - new_keys
        changed = {}

        for key in old_keys & new_keys:
            if old_landmarks[key] != new_landmarks[key]:
                changed[key] = {"old": old_landmarks[key], "new": new_landmarks[key]}

        if added:
            diff_details["added_landmarks"] = {k: new_landmarks[k] for k in added}
        if removed:
            diff_details["removed_landmarks"] = {k: old_landmarks[k] for k in removed}
        if changed:
            diff_details["changed_landmarks"] = changed

        intersection = old_keys & new_keys
        union = old_keys | new_keys

        return (len(intersection) / len(union) if union else 1.0), diff_details

    def _compare_css_classes(
        self,
        old: PageStructure,
        new: PageStructure,
    ) -> tuple[float, dict[str, Any]]:
        """Compare CSS class usage. Returns (similarity, diff_details)."""
        diff_details: dict[str, Any] = {}

        old_classes = set((old.css_class_map or {}).keys())
        new_classes = set((new.css_class_map or {}).keys())

        if not old_classes and not new_classes:
            return 1.0, diff_details

        # Focus on top classes by frequency
        old_sorted = sorted(
            (old.css_class_map or {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )[:50]
        new_sorted = sorted(
            (new.css_class_map or {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )[:50]

        old_names = {name for name, _ in old_sorted}
        new_names = {name for name, _ in new_sorted}

        added = new_names - old_names
        removed = old_names - new_names

        if added:
            new_map = new.css_class_map or {}
            diff_details["added_classes"] = {c: new_map.get(c, 0) for c in list(added)[:20]}
        if removed:
            old_map = old.css_class_map or {}
            diff_details["removed_classes"] = {c: old_map.get(c, 0) for c in list(removed)[:20]}

        intersection = old_names & new_names
        union = old_names | new_names

        return (len(intersection) / len(union) if union else 1.0), diff_details

    def _compare_ids(
        self,
        old: PageStructure,
        new: PageStructure,
    ) -> tuple[float, dict[str, Any]]:
        """Compare element IDs. Returns (similarity, diff_details)."""
        diff_details: dict[str, Any] = {}

        old_ids = old.id_attributes or set()
        new_ids = new.id_attributes or set()

        if not old_ids and not new_ids:
            return 1.0, diff_details

        added = new_ids - old_ids
        removed = old_ids - new_ids

        if added:
            diff_details["added_ids"] = list(added)[:30]
        if removed:
            diff_details["removed_ids"] = list(removed)[:30]

        intersection = old_ids & new_ids
        union = old_ids | new_ids

        return (len(intersection) / len(union) if union else 1.0), diff_details

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
