# AGENTS.md - Adaptive Module

Complete specification for the rules-based fingerprinting and change detection module.

---

## Module Purpose

The adaptive module provides:
- DOM structure analysis (rules-based fingerprinting)
- Change detection and classification
- CSS selector inference for extraction strategies

---

## Files to Generate

```
fingerprint/adaptive/
├── __init__.py
├── structure_analyzer.py   # DOM analysis
├── change_detector.py      # Change detection
└── strategy_learner.py     # Selector inference
```

---

## fingerprint/adaptive/__init__.py

```python
"""
Adaptive module - Rules-based fingerprinting and change detection.
"""

from fingerprint.adaptive.structure_analyzer import DOMStructureAnalyzer
from fingerprint.adaptive.change_detector import ChangeDetector
from fingerprint.adaptive.strategy_learner import StrategyLearner

__all__ = [
    "DOMStructureAnalyzer",
    "ChangeDetector",
    "StrategyLearner",
]
```

---

## fingerprint/adaptive/structure_analyzer.py

```python
"""
DOM structure analyzer for rules-based fingerprinting.

Analyzes HTML to extract structural fingerprint including:
- Tag hierarchy and counts
- CSS class distribution
- Semantic landmarks (header, nav, main, footer)
- Content regions
- Script signatures

Verbose logging pattern:
[STRUCTURE:OPERATION] Message
"""

import hashlib
import re
from collections import Counter
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

from fingerprint.core.verbose import get_logger
from fingerprint.models import (
    ContentRegion,
    PageStructure,
    TagHierarchy,
)


class DOMStructureAnalyzer:
    """
    Analyzes DOM structure to create fingerprint.

    Usage:
        analyzer = DOMStructureAnalyzer()
        structure = analyzer.analyze(html, "example.com")
    """

    # Page type patterns
    PAGE_TYPE_PATTERNS = {
        "article": [
            r"/article/",
            r"/post/",
            r"/blog/",
            r"/news/",
            r"\d{4}/\d{2}/\d{2}/",  # Date pattern
        ],
        "listing": [
            r"/category/",
            r"/tag/",
            r"/search",
            r"/archive",
            r"/page/\d+",
        ],
        "product": [
            r"/product/",
            r"/item/",
            r"/p/",
            r"/dp/",
        ],
        "home": [
            r"^/$",
            r"/index",
            r"/home",
        ],
    }

    # Framework detection patterns
    FRAMEWORK_PATTERNS = {
        "react": [r"__REACT", r"data-reactroot", r"_reactRootContainer"],
        "vue": [r"__vue__", r"data-v-", r"v-cloak"],
        "angular": [r"ng-version", r"_ngcontent", r"ng-"],
        "next": [r"__NEXT_DATA__", r"_next/"],
        "nuxt": [r"__NUXT__", r"_nuxt/"],
    }

    def __init__(self):
        self.logger = get_logger()

    def analyze(self, html: str, domain: str, url: str = "") -> PageStructure:
        """
        Analyze HTML and create structure fingerprint.

        Args:
            html: HTML content
            domain: Domain name
            url: Original URL (for page type detection)

        Returns:
            PageStructure with complete fingerprint

        Verbose output:
            [STRUCTURE:PARSE] Parsing HTML
              - length: 45230
            [STRUCTURE:TAGS] Analyzing tag hierarchy
              - unique_tags: 45
              - max_depth: 12
            [STRUCTURE:CLASSES] Analyzing CSS classes
              - unique_classes: 89
              - total_usages: 456
            [STRUCTURE:LANDMARKS] Identifying semantic landmarks
              - found: header, nav, main, footer
            [STRUCTURE:HASH] Generated content hash
              - hash: a1b2c3d4...
        """
        self.logger.info("STRUCTURE", "PARSE", f"Parsing HTML ({len(html)} bytes)")

        soup = BeautifulSoup(html, "lxml")

        # Detect page type
        page_type = self._detect_page_type(url)
        self.logger.debug("STRUCTURE", "TYPE", f"Page type: {page_type}")

        # Analyze tag hierarchy
        tag_hierarchy = self._analyze_tags(soup)
        self.logger.info(
            "STRUCTURE", "TAGS",
            "Tag hierarchy analyzed",
            unique_tags=len(tag_hierarchy.tag_counts),
            max_depth=tag_hierarchy.max_depth,
        )

        # Analyze CSS classes
        css_class_map = self._analyze_classes(soup)
        self.logger.info(
            "STRUCTURE", "CLASSES",
            "CSS classes analyzed",
            unique_classes=len(css_class_map),
            total_usages=sum(css_class_map.values()),
        )

        # Extract IDs
        id_attributes = self._extract_ids(soup)

        # Identify semantic landmarks
        landmarks = self._identify_landmarks(soup)
        self.logger.info(
            "STRUCTURE", "LANDMARKS",
            f"Found {len(landmarks)} landmarks",
            landmarks=list(landmarks.keys()),
        )

        # Identify content regions
        regions = self._identify_regions(soup)

        # Extract navigation selectors
        nav_selectors = self._extract_navigation(soup)

        # Detect framework
        framework = self._detect_framework(html, soup)
        if framework:
            self.logger.debug("STRUCTURE", "FRAMEWORK", f"Detected: {framework}")

        # Extract script signatures
        scripts = self._extract_script_signatures(soup)

        # Generate content hash
        content_hash = self._generate_hash(soup)
        self.logger.debug("STRUCTURE", "HASH", f"Hash: {content_hash[:16]}...")

        return PageStructure(
            domain=domain,
            page_type=page_type,
            url_pattern=self._extract_url_pattern(url) if url else "",
            tag_hierarchy=tag_hierarchy,
            css_class_map=css_class_map,
            id_attributes=id_attributes,
            semantic_landmarks=landmarks,
            content_regions=regions,
            navigation_selectors=nav_selectors,
            script_signatures=scripts,
            detected_framework=framework,
            content_hash=content_hash,
        )

    def _detect_page_type(self, url: str) -> str:
        """Detect page type from URL patterns."""
        if not url:
            return "unknown"

        path = urlparse(url).path

        for page_type, patterns in self.PAGE_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, path, re.IGNORECASE):
                    return page_type

        return "content"  # Default

    def _analyze_tags(self, soup: BeautifulSoup) -> TagHierarchy:
        """Analyze tag hierarchy and distribution."""
        tag_counts: Counter[str] = Counter()
        depth_distribution: Counter[int] = Counter()
        parent_child_pairs: Counter[str] = Counter()
        max_depth = 0

        def traverse(element: Tag, depth: int = 0) -> None:
            nonlocal max_depth

            if not isinstance(element, Tag):
                return

            tag_name = element.name
            tag_counts[tag_name] += 1
            depth_distribution[depth] += 1
            max_depth = max(max_depth, depth)

            # Track parent-child relationships
            for child in element.children:
                if isinstance(child, Tag):
                    pair = f"{tag_name}>{child.name}"
                    parent_child_pairs[pair] += 1
                    traverse(child, depth + 1)

        if soup.body:
            traverse(soup.body)

        return TagHierarchy(
            tag_counts=dict(tag_counts),
            depth_distribution=dict(depth_distribution),
            parent_child_pairs=dict(parent_child_pairs),
            max_depth=max_depth,
        )

    def _analyze_classes(self, soup: BeautifulSoup) -> dict[str, int]:
        """Extract and count CSS classes."""
        class_counts: Counter[str] = Counter()

        for element in soup.find_all(class_=True):
            classes = element.get("class", [])
            if isinstance(classes, list):
                for cls in classes:
                    class_counts[cls] += 1

        return dict(class_counts)

    def _extract_ids(self, soup: BeautifulSoup) -> set[str]:
        """Extract all ID attributes."""
        ids = set()
        for element in soup.find_all(id=True):
            id_value = element.get("id")
            if id_value:
                ids.add(id_value)
        return ids

    def _identify_landmarks(self, soup: BeautifulSoup) -> dict[str, str]:
        """
        Identify semantic landmarks in the page.

        Looks for:
        - HTML5 semantic elements (header, nav, main, footer, article, aside)
        - ARIA landmarks (role="navigation", etc.)
        - Common class/id patterns (header, nav, content, footer)
        """
        landmarks: dict[str, str] = {}

        # HTML5 semantic elements
        semantic_tags = ["header", "nav", "main", "footer", "article", "aside", "section"]
        for tag in semantic_tags:
            element = soup.find(tag)
            if element:
                # Build selector
                classes = element.get("class", [])
                if classes:
                    selector = f"{tag}.{classes[0]}"
                else:
                    selector = tag
                landmarks[tag] = selector

        # ARIA roles
        role_mapping = {
            "banner": "header",
            "navigation": "nav",
            "main": "main",
            "contentinfo": "footer",
        }
        for role, landmark_name in role_mapping.items():
            if landmark_name not in landmarks:
                element = soup.find(attrs={"role": role})
                if element:
                    landmarks[landmark_name] = f"[role='{role}']"

        # Common patterns as fallback
        fallback_patterns = {
            "header": ["#header", ".header", "#masthead", ".masthead"],
            "nav": ["#nav", ".nav", "#navigation", ".navigation", "#menu", ".menu"],
            "main": ["#main", ".main", "#content", ".content", "#primary", ".primary"],
            "footer": ["#footer", ".footer", "#colophon", ".colophon"],
        }

        for landmark_name, selectors in fallback_patterns.items():
            if landmark_name not in landmarks:
                for selector in selectors:
                    element = soup.select_one(selector)
                    if element:
                        landmarks[landmark_name] = selector
                        break

        return landmarks

    def _identify_regions(self, soup: BeautifulSoup) -> list[ContentRegion]:
        """Identify content extraction regions."""
        regions: list[ContentRegion] = []

        # Title region
        title_element = soup.find("h1")
        if title_element:
            classes = title_element.get("class", [])
            if classes:
                selector = f"h1.{classes[0]}"
            else:
                parent = title_element.parent
                if parent and parent.get("class"):
                    selector = f".{parent['class'][0]} h1"
                else:
                    selector = "h1"

            regions.append(ContentRegion(
                name="title",
                primary_selector=selector,
                fallback_selectors=["h1", "title", ".title", "#title"],
                confidence=0.9,
            ))

        # Main content region
        article = soup.find("article")
        if article:
            classes = article.get("class", [])
            selector = f"article.{classes[0]}" if classes else "article"
            regions.append(ContentRegion(
                name="content",
                primary_selector=selector,
                fallback_selectors=["article", ".content", "#content", "main"],
                confidence=0.85,
            ))

        return regions

    def _extract_navigation(self, soup: BeautifulSoup) -> list[str]:
        """Extract navigation element selectors."""
        nav_selectors: list[str] = []

        # Nav elements
        for nav in soup.find_all("nav"):
            classes = nav.get("class", [])
            if classes:
                nav_selectors.append(f"nav.{classes[0]}")
            else:
                nav_selectors.append("nav")

        # Pagination
        pagination_patterns = [".pagination", ".pager", ".page-numbers", "[class*='pagination']"]
        for pattern in pagination_patterns:
            if soup.select_one(pattern):
                nav_selectors.append(pattern)
                break

        return nav_selectors

    def _detect_framework(self, html: str, soup: BeautifulSoup) -> str | None:
        """Detect JavaScript framework from signatures."""
        for framework, patterns in self.FRAMEWORK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, html):
                    return framework

        return None

    def _extract_script_signatures(self, soup: BeautifulSoup) -> list[str]:
        """Extract script src patterns for fingerprinting."""
        signatures: list[str] = []

        for script in soup.find_all("script", src=True):
            src = script.get("src", "")
            # Extract meaningful part (remove hash/version)
            clean_src = re.sub(r'[?#].*$', '', src)
            clean_src = re.sub(r'\.[a-f0-9]{8,}\.', '.*.', clean_src)

            if clean_src:
                signatures.append(clean_src)

        return signatures[:20]  # Limit to 20

    def _extract_url_pattern(self, url: str) -> str:
        """Extract URL pattern for matching similar pages."""
        parsed = urlparse(url)
        path = parsed.path

        # Replace IDs and dates with placeholders
        pattern = re.sub(r'/\d+/', '/{id}/', path)
        pattern = re.sub(r'/\d{4}/\d{2}/\d{2}/', '/{date}/', pattern)
        pattern = re.sub(r'/[a-f0-9]{24,}/', '/{hash}/', pattern)

        return pattern

    def _generate_hash(self, soup: BeautifulSoup) -> str:
        """Generate structural hash of the page."""
        # Hash based on tag structure, not content
        structure_parts = []

        def collect_structure(element: Tag, depth: int = 0) -> None:
            if not isinstance(element, Tag) or depth > 5:
                return

            classes = element.get("class", [])
            class_str = ".".join(sorted(classes[:3])) if classes else ""
            structure_parts.append(f"{element.name}:{class_str}")

            for child in element.children:
                if isinstance(child, Tag):
                    collect_structure(child, depth + 1)

        if soup.body:
            collect_structure(soup.body)

        content = "|".join(structure_parts)
        return hashlib.sha256(content.encode()).hexdigest()[:32]
```

---

## fingerprint/adaptive/change_detector.py

```python
"""
Change detection and classification.

Compares two PageStructure instances and classifies changes:
- COSMETIC: > 0.95 similarity
- MINOR: 0.85 - 0.95 similarity
- MODERATE: 0.70 - 0.85 similarity
- BREAKING: < 0.70 similarity

Verbose logging pattern:
[CHANGE:OPERATION] Message
"""

from fingerprint.config import ThresholdsConfig
from fingerprint.core.verbose import get_logger
from fingerprint.models import (
    ChangeAnalysis,
    ChangeClassification,
    ChangeType,
    FingerprintMode,
    PageStructure,
    StructureChange,
)


class ChangeDetector:
    """
    Detects and classifies structural changes.

    Usage:
        detector = ChangeDetector(config.thresholds)
        analysis = detector.detect_changes(old_structure, new_structure)
    """

    def __init__(self, thresholds: ThresholdsConfig):
        self.thresholds = thresholds
        self.logger = get_logger()

    def detect_changes(
        self,
        old: PageStructure,
        new: PageStructure,
    ) -> ChangeAnalysis:
        """
        Detect changes between two structures.

        Args:
            old: Previous structure
            new: Current structure

        Returns:
            ChangeAnalysis with similarity and change details

        Verbose output:
            [CHANGE:COMPARE] Comparing structures
              - old_version: 3
              - new_hash: a1b2c3d4...
            [CHANGE:SIMILARITY] Calculated similarity
              - tag_sim: 0.92
              - class_sim: 0.88
              - landmark_sim: 1.00
              - overall: 0.91
            [CHANGE:CLASSIFY] Classification: COSMETIC
            [CHANGE:DETAILS] Found 3 changes
        """
        self.logger.info(
            "CHANGE", "COMPARE",
            "Comparing structures",
            old_version=old.version,
            new_hash=new.content_hash[:16] + "...",
        )

        # Calculate component similarities
        tag_sim = self._tag_similarity(old, new)
        class_sim = self._class_similarity(old, new)
        landmark_sim = self._landmark_similarity(old, new)

        # Overall similarity (weighted average)
        # Tags: 30%, Classes: 50%, Landmarks: 20%
        overall_sim = (tag_sim * 0.30) + (class_sim * 0.50) + (landmark_sim * 0.20)

        self.logger.info(
            "CHANGE", "SIMILARITY",
            f"Calculated similarity: {overall_sim:.3f}",
            tag_sim=f"{tag_sim:.3f}",
            class_sim=f"{class_sim:.3f}",
            landmark_sim=f"{landmark_sim:.3f}",
        )

        # Classify change
        classification = self.classify_similarity(overall_sim)
        breaking = classification == ChangeClassification.BREAKING

        self.logger.info(
            "CHANGE", "CLASSIFY",
            f"Classification: {classification.value.upper()}",
            breaking=breaking,
        )

        # Detect specific changes
        changes = self._detect_specific_changes(old, new)

        self.logger.debug(
            "CHANGE", "DETAILS",
            f"Found {len(changes)} changes",
        )

        # Check if auto-adaptation is possible
        can_adapt = self._can_auto_adapt(changes)
        adapt_confidence = self._adaptation_confidence(changes) if can_adapt else 0.0

        return ChangeAnalysis(
            similarity=overall_sim,
            mode_used=FingerprintMode.RULES,
            classification=classification,
            breaking=breaking,
            changes=changes,
            can_auto_adapt=can_adapt,
            adaptation_confidence=adapt_confidence,
            reason=self._generate_reason(classification, changes),
        )

    def classify_similarity(self, similarity: float) -> ChangeClassification:
        """Classify similarity score into change category."""
        if similarity >= self.thresholds.cosmetic:
            return ChangeClassification.COSMETIC
        elif similarity >= self.thresholds.minor:
            return ChangeClassification.MINOR
        elif similarity >= self.thresholds.moderate:
            return ChangeClassification.MODERATE
        else:
            return ChangeClassification.BREAKING

    def _tag_similarity(self, old: PageStructure, new: PageStructure) -> float:
        """Calculate tag hierarchy similarity."""
        if not old.tag_hierarchy or not new.tag_hierarchy:
            return 0.0

        old_tags = set(old.tag_hierarchy.tag_counts.keys())
        new_tags = set(new.tag_hierarchy.tag_counts.keys())

        if not old_tags:
            return 1.0 if not new_tags else 0.0

        intersection = old_tags & new_tags
        union = old_tags | new_tags

        jaccard = len(intersection) / len(union) if union else 0.0

        # Also consider count similarity for common tags
        count_sim = 0.0
        for tag in intersection:
            old_count = old.tag_hierarchy.tag_counts[tag]
            new_count = new.tag_hierarchy.tag_counts[tag]
            # Use min/max ratio
            count_sim += min(old_count, new_count) / max(old_count, new_count)

        if intersection:
            count_sim /= len(intersection)

        return (jaccard + count_sim) / 2

    def _class_similarity(self, old: PageStructure, new: PageStructure) -> float:
        """Calculate CSS class similarity."""
        old_classes = set(old.css_class_map.keys())
        new_classes = set(new.css_class_map.keys())

        if not old_classes:
            return 1.0 if not new_classes else 0.0

        intersection = old_classes & new_classes
        union = old_classes | new_classes

        return len(intersection) / len(union) if union else 0.0

    def _landmark_similarity(self, old: PageStructure, new: PageStructure) -> float:
        """Calculate landmark similarity."""
        old_landmarks = set(old.semantic_landmarks.keys())
        new_landmarks = set(new.semantic_landmarks.keys())

        if not old_landmarks:
            return 1.0 if not new_landmarks else 0.0

        intersection = old_landmarks & new_landmarks
        union = old_landmarks | new_landmarks

        # Base similarity on presence
        presence_sim = len(intersection) / len(union) if union else 0.0

        # Check if selectors match for common landmarks
        selector_matches = 0
        for landmark in intersection:
            if old.semantic_landmarks[landmark] == new.semantic_landmarks[landmark]:
                selector_matches += 1

        selector_sim = selector_matches / len(intersection) if intersection else 0.0

        return (presence_sim + selector_sim) / 2

    def _detect_specific_changes(
        self,
        old: PageStructure,
        new: PageStructure,
    ) -> list[StructureChange]:
        """Detect specific changes between structures."""
        changes: list[StructureChange] = []

        # Detect class changes
        old_classes = set(old.css_class_map.keys())
        new_classes = set(new.css_class_map.keys())

        added_classes = new_classes - old_classes
        removed_classes = old_classes - new_classes

        if added_classes:
            changes.append(StructureChange(
                change_type=ChangeType.CLASS_ADDED,
                affected_components=list(added_classes)[:10],
                new_value=f"{len(added_classes)} classes added",
                confidence=0.9,
                reason=f"Added CSS classes: {', '.join(list(added_classes)[:5])}",
            ))

        if removed_classes:
            changes.append(StructureChange(
                change_type=ChangeType.CLASS_REMOVED,
                affected_components=list(removed_classes)[:10],
                old_value=f"{len(removed_classes)} classes removed",
                confidence=0.9,
                reason=f"Removed CSS classes: {', '.join(list(removed_classes)[:5])}",
            ))

        # Detect landmark changes
        old_landmarks = set(old.semantic_landmarks.keys())
        new_landmarks = set(new.semantic_landmarks.keys())

        if old_landmarks != new_landmarks:
            changes.append(StructureChange(
                change_type=ChangeType.LANDMARK_CHANGED,
                affected_components=list((old_landmarks - new_landmarks) | (new_landmarks - old_landmarks)),
                breaking=True,
                confidence=0.95,
                reason="Semantic landmarks changed",
            ))

        # Detect navigation changes
        if old.navigation_selectors != new.navigation_selectors:
            changes.append(StructureChange(
                change_type=ChangeType.NAVIGATION_CHANGED,
                old_value=str(old.navigation_selectors),
                new_value=str(new.navigation_selectors),
                confidence=0.8,
                reason="Navigation structure changed",
            ))

        # Detect framework changes
        if old.detected_framework != new.detected_framework:
            changes.append(StructureChange(
                change_type=ChangeType.FRAMEWORK_CHANGED,
                old_value=old.detected_framework,
                new_value=new.detected_framework,
                breaking=True,
                confidence=0.99,
                reason=f"Framework changed from {old.detected_framework} to {new.detected_framework}",
            ))

        # Detect major redesign (significant tag changes)
        if old.tag_hierarchy and new.tag_hierarchy:
            old_depth = old.tag_hierarchy.max_depth
            new_depth = new.tag_hierarchy.max_depth
            depth_change = abs(old_depth - new_depth)

            if depth_change > 3:
                changes.append(StructureChange(
                    change_type=ChangeType.MAJOR_REDESIGN,
                    old_value=f"max_depth={old_depth}",
                    new_value=f"max_depth={new_depth}",
                    breaking=True,
                    confidence=0.85,
                    reason=f"Major structural change: depth changed by {depth_change}",
                ))

        return changes

    def _can_auto_adapt(self, changes: list[StructureChange]) -> bool:
        """Check if changes can be automatically adapted."""
        # Cannot auto-adapt major redesigns or framework changes
        for change in changes:
            if change.change_type in (ChangeType.MAJOR_REDESIGN, ChangeType.FRAMEWORK_CHANGED):
                return False
            if change.breaking:
                return False

        return True

    def _adaptation_confidence(self, changes: list[StructureChange]) -> float:
        """Calculate confidence in auto-adaptation."""
        if not changes:
            return 1.0

        # Average confidence of non-breaking changes
        confidences = [c.confidence for c in changes if not c.breaking]
        if not confidences:
            return 0.0

        return sum(confidences) / len(confidences)

    def _generate_reason(
        self,
        classification: ChangeClassification,
        changes: list[StructureChange],
    ) -> str:
        """Generate human-readable change reason."""
        if classification == ChangeClassification.COSMETIC:
            return "Minor styling adjustments detected"

        if not changes:
            return f"{classification.value.capitalize()} changes detected"

        # Summarize main changes
        change_types = [c.change_type.value for c in changes[:3]]
        return f"{classification.value.capitalize()}: {', '.join(change_types)}"
```

---

## fingerprint/adaptive/strategy_learner.py

```python
"""
CSS selector strategy learning for content extraction.

Infers extraction strategies from page structure:
- Identifies title, content, and metadata regions
- Generates robust CSS selectors with fallbacks
- Adapts strategies when structure changes

Verbose logging pattern:
[STRATEGY:OPERATION] Message
"""

from fingerprint.core.verbose import get_logger
from fingerprint.models import (
    ContentRegion,
    ExtractionStrategy,
    PageStructure,
    SelectorRule,
)


class StrategyLearner:
    """
    Learns extraction strategies from page structures.

    Usage:
        learner = StrategyLearner()
        strategy = learner.infer_strategy(structure)
    """

    def __init__(self):
        self.logger = get_logger()

    def infer_strategy(self, structure: PageStructure) -> ExtractionStrategy:
        """
        Infer extraction strategy from page structure.

        Args:
            structure: PageStructure to learn from

        Returns:
            ExtractionStrategy with selector rules

        Verbose output:
            [STRATEGY:INFER] Learning strategy for example.com/article
            [STRATEGY:TITLE] Found title selector
              - primary: article h1.title
              - fallbacks: 3
              - confidence: 0.92
            [STRATEGY:CONTENT] Found content selector
              - primary: article.post-content
              - confidence: 0.88
        """
        self.logger.info(
            "STRATEGY", "INFER",
            f"Learning strategy for {structure.domain}/{structure.page_type}",
        )

        # Infer title rule
        title_rule = self._infer_title_rule(structure)
        if title_rule:
            self.logger.info(
                "STRATEGY", "TITLE",
                f"Found title selector: {title_rule.primary}",
                fallbacks=len(title_rule.fallbacks),
                confidence=f"{title_rule.confidence:.2f}",
            )

        # Infer content rule
        content_rule = self._infer_content_rule(structure)
        if content_rule:
            self.logger.info(
                "STRATEGY", "CONTENT",
                f"Found content selector: {content_rule.primary}",
                confidence=f"{content_rule.confidence:.2f}",
            )

        # Infer metadata rules
        metadata_rules = self._infer_metadata_rules(structure)

        return ExtractionStrategy(
            domain=structure.domain,
            page_type=structure.page_type,
            title=title_rule,
            content=content_rule,
            metadata=metadata_rules,
            learning_source="inferred",
            confidence_scores={
                "title": title_rule.confidence if title_rule else 0.0,
                "content": content_rule.confidence if content_rule else 0.0,
            },
        )

    def adapt_strategy(
        self,
        old_strategy: ExtractionStrategy,
        new_structure: PageStructure,
    ) -> ExtractionStrategy:
        """
        Adapt existing strategy to new structure.

        Args:
            old_strategy: Previous working strategy
            new_structure: New page structure

        Returns:
            Adapted ExtractionStrategy
        """
        self.logger.info(
            "STRATEGY", "ADAPT",
            f"Adapting strategy for {new_structure.domain}",
            old_version=old_strategy.version,
        )

        # Try to find equivalent selectors
        new_strategy = self.infer_strategy(new_structure)

        # Merge with old fallbacks
        if old_strategy.title and new_strategy.title:
            # Add old primary to new fallbacks if different
            if old_strategy.title.primary != new_strategy.title.primary:
                new_strategy.title.fallbacks.insert(0, old_strategy.title.primary)

        if old_strategy.content and new_strategy.content:
            if old_strategy.content.primary != new_strategy.content.primary:
                new_strategy.content.fallbacks.insert(0, old_strategy.content.primary)

        new_strategy.version = old_strategy.version + 1
        new_strategy.learning_source = "adaptation"

        self.logger.info(
            "STRATEGY", "ADAPTED",
            f"Strategy adapted to version {new_strategy.version}",
        )

        return new_strategy

    def _infer_title_rule(self, structure: PageStructure) -> SelectorRule | None:
        """Infer title extraction rule."""
        # Look in content regions first
        for region in structure.content_regions:
            if region.name == "title":
                return SelectorRule(
                    primary=region.primary_selector,
                    fallbacks=region.fallback_selectors,
                    extraction_method="text",
                    confidence=region.confidence,
                )

        # Fallback: common title patterns
        fallbacks = ["h1", "title", ".title", "#title", "[itemprop='headline']"]

        # Try to find h1 with specific class in css_class_map
        for cls in structure.css_class_map:
            if "title" in cls.lower() or "headline" in cls.lower():
                return SelectorRule(
                    primary=f"h1.{cls}",
                    fallbacks=fallbacks,
                    extraction_method="text",
                    confidence=0.75,
                )

        return SelectorRule(
            primary="h1",
            fallbacks=fallbacks[1:],
            extraction_method="text",
            confidence=0.6,
        )

    def _infer_content_rule(self, structure: PageStructure) -> SelectorRule | None:
        """Infer content extraction rule."""
        # Look in content regions
        for region in structure.content_regions:
            if region.name == "content":
                return SelectorRule(
                    primary=region.primary_selector,
                    fallbacks=region.fallback_selectors,
                    extraction_method="html",
                    confidence=region.confidence,
                )

        # Try landmark-based selection
        if "main" in structure.semantic_landmarks:
            return SelectorRule(
                primary=structure.semantic_landmarks["main"],
                fallbacks=["main", "article", ".content", "#content"],
                extraction_method="html",
                confidence=0.8,
            )

        # Fallback
        return SelectorRule(
            primary="article",
            fallbacks=["main", ".content", "#content", ".post-content"],
            extraction_method="html",
            confidence=0.5,
        )

    def _infer_metadata_rules(
        self,
        structure: PageStructure,
    ) -> dict[str, SelectorRule]:
        """Infer metadata extraction rules."""
        rules: dict[str, SelectorRule] = {}

        # Author
        for cls in structure.css_class_map:
            if "author" in cls.lower():
                rules["author"] = SelectorRule(
                    primary=f".{cls}",
                    fallbacks=["[rel='author']", ".author", "[itemprop='author']"],
                    extraction_method="text",
                    confidence=0.7,
                )
                break

        # Date
        for cls in structure.css_class_map:
            if "date" in cls.lower() or "time" in cls.lower():
                rules["date"] = SelectorRule(
                    primary=f".{cls}",
                    fallbacks=["time", "[datetime]", ".date", "[itemprop='datePublished']"],
                    extraction_method="attribute",
                    attribute_name="datetime",
                    confidence=0.7,
                )
                break

        return rules
```

---

## Verbose Logging

All adaptive module operations use consistent logging:

| Operation | Description |
|-----------|-------------|
| STRUCTURE:PARSE | Starting HTML parsing |
| STRUCTURE:TAGS | Tag hierarchy analysis |
| STRUCTURE:CLASSES | CSS class analysis |
| STRUCTURE:LANDMARKS | Semantic landmark identification |
| STRUCTURE:HASH | Content hash generation |
| CHANGE:COMPARE | Starting change comparison |
| CHANGE:SIMILARITY | Similarity calculation |
| CHANGE:CLASSIFY | Change classification |
| CHANGE:DETAILS | Specific changes found |
| STRATEGY:INFER | Learning new strategy |
| STRATEGY:ADAPT | Adapting existing strategy |

### Example Output

```
[2024-01-15T10:30:00Z] [STRUCTURE:PARSE] Parsing HTML (45230 bytes)

[2024-01-15T10:30:00Z] [STRUCTURE:TAGS] Tag hierarchy analyzed
  - unique_tags: 45
  - max_depth: 12

[2024-01-15T10:30:00Z] [STRUCTURE:CLASSES] CSS classes analyzed
  - unique_classes: 89
  - total_usages: 456

[2024-01-15T10:30:00Z] [STRUCTURE:LANDMARKS] Found 5 landmarks
  - landmarks: ['header', 'nav', 'main', 'footer', 'article']

[2024-01-15T10:30:01Z] [CHANGE:COMPARE] Comparing structures
  - old_version: 3
  - new_hash: a1b2c3d4e5f6...

[2024-01-15T10:30:01Z] [CHANGE:SIMILARITY] Calculated similarity: 0.912
  - tag_sim: 0.920
  - class_sim: 0.880
  - landmark_sim: 1.000

[2024-01-15T10:30:01Z] [CHANGE:CLASSIFY] Classification: COSMETIC
  - breaking: False
```
