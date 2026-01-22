"""
Strategy learner for adaptive extraction.

Learns and adapts extraction strategies based on page structure analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup

from crawler.models import ExtractionStrategy, PageStructure, ContentRegion, SelectorRule
from crawler.utils.logging import CrawlerLogger


@dataclass
class SelectorCandidate:
    """A candidate CSS selector with confidence score."""

    selector: str
    confidence: float
    sample_count: int = 0
    success_rate: float = 1.0
    last_used: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "selector": self.selector,
            "confidence": self.confidence,
            "sample_count": self.sample_count,
            "success_rate": self.success_rate,
            "last_used": self.last_used.isoformat(),
        }


@dataclass
class LearnedStrategy:
    """A learned extraction strategy with metadata."""

    strategy: ExtractionStrategy
    confidence: float
    learned_at: datetime = field(default_factory=datetime.utcnow)
    sample_pages: int = 1
    validated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "domain": self.strategy.domain,
            "page_type": self.strategy.page_type,
            "confidence": self.confidence,
            "learned_at": self.learned_at.isoformat(),
            "sample_pages": self.sample_pages,
            "validated": self.validated,
        }


class StrategyLearner:
    """
    Learns extraction strategies from page structure.

    Analyzes HTML structure to infer CSS selectors for
    content extraction. Adapts existing strategies when
    page structure changes.
    """

    # Common content container patterns
    CONTENT_PATTERNS = [
        ("article", 0.9),
        ("[role='main']", 0.85),
        ("main", 0.85),
        (".article-content", 0.8),
        (".post-content", 0.8),
        (".entry-content", 0.8),
        (".content", 0.7),
        ("#content", 0.7),
        (".main-content", 0.75),
        ("#main-content", 0.75),
    ]

    # Common title patterns
    TITLE_PATTERNS = [
        ("h1.title", 0.9),
        ("h1.entry-title", 0.9),
        ("h1.post-title", 0.9),
        ("article h1", 0.85),
        (".article-title", 0.8),
        ("h1", 0.7),
    ]

    # Common date patterns
    DATE_PATTERNS = [
        ("time[datetime]", 0.95),
        (".published-date", 0.85),
        (".post-date", 0.85),
        (".date", 0.7),
        ("[itemprop='datePublished']", 0.9),
    ]

    # Common author patterns
    AUTHOR_PATTERNS = [
        ("[rel='author']", 0.9),
        (".author", 0.8),
        (".byline", 0.8),
        ("[itemprop='author']", 0.9),
        (".post-author", 0.8),
    ]

    def __init__(
        self,
        min_confidence: float = 0.5,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the strategy learner.

        Args:
            min_confidence: Minimum confidence to accept a selector.
            logger: Logger instance.
        """
        self.min_confidence = min_confidence
        self.logger = logger or CrawlerLogger("strategy_learner")

    def infer(
        self,
        html: str,
        structure: PageStructure | None = None,
    ) -> LearnedStrategy:
        """
        Infer an extraction strategy from HTML.

        Args:
            html: HTML content to analyze.
            structure: Optional pre-analyzed structure.

        Returns:
            LearnedStrategy with inferred selectors.
        """
        soup = BeautifulSoup(html, "lxml")

        # Find content selector
        content_candidate = self._find_best_selector(
            soup, self.CONTENT_PATTERNS, "content"
        )

        # Find title selector
        title_candidate = self._find_best_selector(
            soup, self.TITLE_PATTERNS, "title"
        )

        # Find date selector
        date_candidate = self._find_best_selector(
            soup, self.DATE_PATTERNS, "date"
        )

        # Find author selector
        author_candidate = self._find_best_selector(
            soup, self.AUTHOR_PATTERNS, "author"
        )

        # Build strategy with SelectorRule objects
        confidences = []
        confidence_scores: dict[str, float] = {}

        title_rule = None
        if title_candidate:
            title_rule = SelectorRule(
                primary=title_candidate.selector,
                confidence=title_candidate.confidence,
            )
            confidences.append(title_candidate.confidence)
            confidence_scores["title"] = title_candidate.confidence

        content_rule = None
        if content_candidate:
            content_rule = SelectorRule(
                primary=content_candidate.selector,
                confidence=content_candidate.confidence,
            )
            confidences.append(content_candidate.confidence)
            confidence_scores["content"] = content_candidate.confidence

        # Build metadata rules for date and author
        metadata: dict[str, SelectorRule] = {}
        if date_candidate:
            metadata["date"] = SelectorRule(
                primary=date_candidate.selector,
                confidence=date_candidate.confidence,
            )
            confidences.append(date_candidate.confidence)
            confidence_scores["date"] = date_candidate.confidence

        if author_candidate:
            metadata["author"] = SelectorRule(
                primary=author_candidate.selector,
                confidence=author_candidate.confidence,
            )
            confidences.append(author_candidate.confidence)
            confidence_scores["author"] = author_candidate.confidence

        # Add structure-based selectors if available
        if structure:
            self._enhance_from_structure(
                title_rule, content_rule, metadata, structure, soup
            )

        # Calculate overall confidence
        overall_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        strategy = ExtractionStrategy(
            domain=structure.domain if structure else "",
            page_type=structure.page_type if structure else "unknown",
            title=title_rule,
            content=content_rule,
            metadata=metadata,
            confidence_scores=confidence_scores,
            learning_source="inferred",
        )

        learned = LearnedStrategy(
            strategy=strategy,
            confidence=overall_confidence,
        )

        self.logger.debug(
            "Inferred extraction strategy",
            domain=strategy.domain,
            page_type=strategy.page_type,
            confidence=overall_confidence,
            has_title=title_rule is not None,
            has_content=content_rule is not None,
        )

        return learned

    def adapt(
        self,
        old_strategy: ExtractionStrategy,
        new_structure: PageStructure,
        html: str,
    ) -> LearnedStrategy:
        """
        Adapt an existing strategy to a new page structure.

        Args:
            old_strategy: Previous extraction strategy.
            new_structure: New page structure.
            html: Current HTML content.

        Returns:
            Updated LearnedStrategy.
        """
        soup = BeautifulSoup(html, "lxml")
        confidences = []
        confidence_scores: dict[str, float] = {}

        # Adapt title selector
        title_rule = self._adapt_selector_rule(
            soup, old_strategy.title, self.TITLE_PATTERNS, "title"
        )
        if title_rule:
            confidences.append(title_rule.confidence)
            confidence_scores["title"] = title_rule.confidence

        # Adapt content selector
        content_rule = self._adapt_selector_rule(
            soup, old_strategy.content, self.CONTENT_PATTERNS, "content"
        )
        if content_rule:
            confidences.append(content_rule.confidence)
            confidence_scores["content"] = content_rule.confidence

        # Adapt metadata selectors
        metadata: dict[str, SelectorRule] = {}
        for key, old_rule in (old_strategy.metadata or {}).items():
            patterns = self._get_patterns_for_field(key)
            new_rule = self._adapt_selector_rule(soup, old_rule, patterns, key)
            if new_rule:
                metadata[key] = new_rule
                confidences.append(new_rule.confidence)
                confidence_scores[key] = new_rule.confidence

        overall_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        strategy = ExtractionStrategy(
            domain=new_structure.domain,
            page_type=new_structure.page_type,
            version=old_strategy.version + 1,
            title=title_rule,
            content=content_rule,
            metadata=metadata,
            confidence_scores=confidence_scores,
            learning_source="adaptation",
        )

        return LearnedStrategy(
            strategy=strategy,
            confidence=overall_confidence,
            validated=False,
        )

    def _adapt_selector_rule(
        self,
        soup: BeautifulSoup,
        old_rule: SelectorRule | None,
        patterns: list[tuple[str, float]],
        field: str,
    ) -> SelectorRule | None:
        """Adapt a single selector rule."""
        if old_rule:
            # Test if old selector still works
            try:
                elements = soup.select(old_rule.primary)
                if elements:
                    # Still works - keep it with maintained confidence
                    return SelectorRule(
                        primary=old_rule.primary,
                        fallbacks=old_rule.fallbacks,
                        extraction_method=old_rule.extraction_method,
                        confidence=old_rule.confidence * 0.95,  # Slight confidence decay
                    )
            except Exception:
                pass

        # Need to find new selector
        candidate = self._find_best_selector(soup, patterns, field)
        if candidate:
            self.logger.info(
                "Adapted selector",
                field=field,
                old=old_rule.primary if old_rule else None,
                new=candidate.selector,
            )
            return SelectorRule(
                primary=candidate.selector,
                confidence=candidate.confidence * 0.8,  # Penalty for change
            )

        return None

    def _find_best_selector(
        self,
        soup: BeautifulSoup,
        patterns: list[tuple[str, float]],
        field: str,
    ) -> SelectorCandidate | None:
        """Find the best matching selector from patterns."""
        best: SelectorCandidate | None = None

        for selector, base_confidence in patterns:
            try:
                elements = soup.select(selector)
                if elements:
                    # Adjust confidence based on uniqueness
                    if len(elements) == 1:
                        confidence = base_confidence
                    elif len(elements) <= 3:
                        confidence = base_confidence * 0.9
                    else:
                        confidence = base_confidence * 0.7

                    # Check content quality for content field
                    if field == "content" and elements:
                        text_length = len(elements[0].get_text(strip=True))
                        if text_length < 100:
                            confidence *= 0.5

                    if confidence >= self.min_confidence:
                        if best is None or confidence > best.confidence:
                            best = SelectorCandidate(
                                selector=selector,
                                confidence=confidence,
                            )
            except Exception:
                continue

        return best

    def _get_patterns_for_field(
        self,
        field: str,
    ) -> list[tuple[str, float]]:
        """Get pattern list for a field type."""
        patterns = {
            "content": self.CONTENT_PATTERNS,
            "title": self.TITLE_PATTERNS,
            "date": self.DATE_PATTERNS,
            "author": self.AUTHOR_PATTERNS,
        }
        return patterns.get(field, [])

    def _enhance_from_structure(
        self,
        title_rule: SelectorRule | None,
        content_rule: SelectorRule | None,
        metadata: dict[str, SelectorRule],
        structure: PageStructure,
        soup: BeautifulSoup,
    ) -> None:
        """Enhance selectors using structure analysis."""
        # Use content regions if no content selector found
        if structure.content_regions and content_rule is None:
            for region in structure.content_regions:
                # Use primary_selector from ContentRegion
                if region.primary_selector:
                    try:
                        elements = soup.select(region.primary_selector)
                        if elements:
                            # Can't modify content_rule directly since it's passed by value
                            # This would need refactoring to return the enhanced rules
                            break
                    except Exception:
                        pass

    def validate_strategy(
        self,
        strategy: ExtractionStrategy,
        html: str,
    ) -> tuple[bool, dict[str, bool]]:
        """
        Validate that a strategy works on HTML.

        Args:
            strategy: Strategy to validate.
            html: HTML to test against.

        Returns:
            Tuple of (overall_valid, field_results).
        """
        soup = BeautifulSoup(html, "lxml")
        results: dict[str, bool] = {}

        # Check title
        if strategy.title:
            try:
                elements = soup.select(strategy.title.primary)
                results["title"] = len(elements) > 0
            except Exception:
                results["title"] = False

        # Check content
        if strategy.content:
            try:
                elements = soup.select(strategy.content.primary)
                results["content"] = len(elements) > 0
            except Exception:
                results["content"] = False

        # Check metadata
        for key, rule in (strategy.metadata or {}).items():
            try:
                elements = soup.select(rule.primary)
                results[key] = len(elements) > 0
            except Exception:
                results[key] = False

        overall = all(results.values()) if results else False
        return overall, results
