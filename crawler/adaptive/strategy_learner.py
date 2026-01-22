"""
Strategy learner for adaptive extraction.

Learns and adapts extraction strategies based on page structure analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup

from crawler.models import ExtractionStrategy, PageStructure, ContentRegion
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
            "strategy": self.strategy.to_dict(),
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
        content_selector = self._find_best_selector(
            soup, self.CONTENT_PATTERNS, "content"
        )

        # Find title selector
        title_selector = self._find_best_selector(
            soup, self.TITLE_PATTERNS, "title"
        )

        # Find date selector
        date_selector = self._find_best_selector(
            soup, self.DATE_PATTERNS, "date"
        )

        # Find author selector
        author_selector = self._find_best_selector(
            soup, self.AUTHOR_PATTERNS, "author"
        )

        # Build strategy
        selectors = {}
        confidences = []

        if content_selector:
            selectors["content"] = content_selector.selector
            confidences.append(content_selector.confidence)

        if title_selector:
            selectors["title"] = title_selector.selector
            confidences.append(title_selector.confidence)

        if date_selector:
            selectors["date"] = date_selector.selector
            confidences.append(date_selector.confidence)

        if author_selector:
            selectors["author"] = author_selector.selector
            confidences.append(author_selector.confidence)

        # Add structure-based selectors if available
        if structure:
            self._enhance_from_structure(selectors, structure, soup)

        # Calculate overall confidence
        overall_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        strategy = ExtractionStrategy(
            domain=structure.domain if structure else "",
            page_type=structure.page_type if structure else "unknown",
            selectors=selectors,
            confidence=overall_confidence,
        )

        learned = LearnedStrategy(
            strategy=strategy,
            confidence=overall_confidence,
        )

        self.logger.debug(
            "Inferred extraction strategy",
            selectors=selectors,
            confidence=overall_confidence,
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
        new_selectors = dict(old_strategy.selectors)
        confidences = []

        # Test each existing selector
        for field, selector in old_strategy.selectors.items():
            elements = soup.select(selector)
            if elements:
                # Selector still works
                confidences.append(0.9)
            else:
                # Need to find replacement
                patterns = self._get_patterns_for_field(field)
                replacement = self._find_best_selector(soup, patterns, field)
                if replacement:
                    new_selectors[field] = replacement.selector
                    confidences.append(replacement.confidence * 0.8)  # Penalty for change
                    self.logger.info(
                        "Adapted selector",
                        field=field,
                        old=selector,
                        new=replacement.selector,
                    )
                else:
                    # Remove field if no replacement found
                    del new_selectors[field]
                    self.logger.warning(
                        "Could not find replacement selector",
                        field=field,
                        old=selector,
                    )

        # Check for new fields we might extract
        self._enhance_from_structure(new_selectors, new_structure, soup)

        overall_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        strategy = ExtractionStrategy(
            domain=new_structure.domain,
            page_type=new_structure.page_type,
            selectors=new_selectors,
            confidence=overall_confidence,
            version=old_strategy.version + 1,
        )

        return LearnedStrategy(
            strategy=strategy,
            confidence=overall_confidence,
            validated=False,
        )

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
        selectors: dict[str, str],
        structure: PageStructure,
        soup: BeautifulSoup,
    ) -> None:
        """Enhance selectors using structure analysis."""
        # Use content regions if available
        if structure.content_regions and "content" not in selectors:
            for region in structure.content_regions:
                if region.region_type == "main" and region.selector:
                    elements = soup.select(region.selector)
                    if elements:
                        selectors["content"] = region.selector
                        break

        # Use navigation selectors
        if structure.navigation_selectors and "navigation" not in selectors:
            for nav_selector in structure.navigation_selectors:
                elements = soup.select(nav_selector)
                if elements:
                    selectors["navigation"] = nav_selector
                    break

        # Use pagination info
        if structure.pagination and "next_page" not in selectors:
            if structure.pagination.next_selector:
                selectors["next_page"] = structure.pagination.next_selector

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

        for field, selector in strategy.selectors.items():
            try:
                elements = soup.select(selector)
                results[field] = len(elements) > 0
            except Exception:
                results[field] = False

        overall = all(results.values()) if results else False
        return overall, results
