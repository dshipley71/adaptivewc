"""
DOM structure analyzer for adaptive extraction.

Analyzes page structure to create fingerprints for change detection.
"""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup, Tag

from crawler.models import (
    ContentRegion,
    IframeInfo,
    PageStructure,
    PaginationInfo,
)
from crawler.utils.logging import CrawlerLogger
from crawler.utils.url_utils import extract_url_pattern, get_domain


@dataclass
class AnalysisConfig:
    """Configuration for structure analysis."""

    min_content_length: int = 100
    max_depth: int = 10
    track_classes: bool = True
    track_ids: bool = True
    extract_scripts: bool = True


class StructureAnalyzer:
    """
    Analyzes DOM structure for adaptive extraction.

    Creates fingerprints of page structure that can be used
    to detect changes and adapt extraction strategies.
    """

    # Common semantic elements
    SEMANTIC_ELEMENTS = {
        "header", "nav", "main", "article", "section",
        "aside", "footer", "figure", "figcaption",
    }

    # Content-likely elements
    CONTENT_ELEMENTS = {"article", "main", "div", "section", "p"}

    # Navigation elements
    NAV_ELEMENTS = {"nav", "menu", "ul", "ol"}

    def __init__(
        self,
        config: AnalysisConfig | None = None,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the analyzer.

        Args:
            config: Analysis configuration.
            logger: Logger instance.
        """
        self.config = config or AnalysisConfig()
        self.logger = logger or CrawlerLogger("structure_analyzer")

    def analyze(
        self,
        html: str,
        url: str,
        page_type: str = "unknown",
    ) -> PageStructure:
        """
        Analyze HTML and create a page structure fingerprint.

        Args:
            html: HTML content.
            url: Page URL.
            page_type: Type of page (article, listing, etc.).

        Returns:
            PageStructure fingerprint.
        """
        soup = BeautifulSoup(html, "lxml")
        domain = get_domain(url)
        url_pattern = extract_url_pattern(url)

        structure = PageStructure(
            domain=domain,
            page_type=page_type,
            url_pattern=url_pattern,
        )

        # Build tag hierarchy
        structure.tag_hierarchy = self._build_hierarchy(soup)

        # Extract CSS classes
        if self.config.track_classes:
            structure.css_class_map = self._extract_classes(soup)

        # Extract IDs
        if self.config.track_ids:
            structure.id_attributes = self._extract_ids(soup)

        # Find semantic landmarks
        structure.semantic_landmarks = self._find_landmarks(soup)

        # Extract iframe information
        structure.iframe_locations = self._extract_iframes(soup)

        # Extract script signatures
        if self.config.extract_scripts:
            structure.script_signatures = self._extract_script_signatures(soup)

        # Identify content regions
        structure.content_regions = self._identify_content_regions(soup)

        # Extract navigation selectors
        structure.navigation_selectors = self._find_navigation(soup)

        # Detect pagination
        structure.pagination_pattern = self._detect_pagination(soup, url)

        # Calculate content hash
        structure.content_hash = self._calculate_hash(html)

        return structure

    def _build_hierarchy(self, soup: BeautifulSoup) -> dict[str, Any]:
        """Build a simplified tag hierarchy."""
        hierarchy: dict[str, Any] = {
            "tag_counts": Counter(),
            "depth_distribution": Counter(),
            "parent_child_pairs": Counter(),
        }

        def traverse(elem, depth: int = 0) -> None:
            if depth > self.config.max_depth:
                return

            if isinstance(elem, Tag):
                hierarchy["tag_counts"][elem.name] += 1
                hierarchy["depth_distribution"][depth] += 1

                for child in elem.children:
                    if isinstance(child, Tag):
                        pair = f"{elem.name}>{child.name}"
                        hierarchy["parent_child_pairs"][pair] += 1
                        traverse(child, depth + 1)

        body = soup.find("body")
        if body:
            traverse(body)

        # Convert counters to dicts for serialization
        return {
            "tag_counts": dict(hierarchy["tag_counts"]),
            "depth_distribution": dict(hierarchy["depth_distribution"]),
            "parent_child_pairs": dict(hierarchy["parent_child_pairs"].most_common(50)),
        }

    def _extract_classes(self, soup: BeautifulSoup) -> dict[str, int]:
        """Extract CSS class usage counts."""
        class_counts: Counter = Counter()

        for elem in soup.find_all(class_=True):
            classes = elem.get("class", [])
            for cls in classes:
                class_counts[cls] += 1

        # Return top classes
        return dict(class_counts.most_common(100))

    def _extract_ids(self, soup: BeautifulSoup) -> set[str]:
        """Extract ID attributes."""
        ids = set()

        for elem in soup.find_all(id=True):
            ids.add(elem["id"])

        return ids

    def _find_landmarks(self, soup: BeautifulSoup) -> dict[str, str]:
        """Find semantic landmarks in the page."""
        landmarks: dict[str, str] = {}

        for tag in self.SEMANTIC_ELEMENTS:
            elem = soup.find(tag)
            if elem:
                # Generate a selector for this landmark
                selector = self._generate_selector(elem)
                landmarks[tag] = selector

        # Look for ARIA landmarks
        for elem in soup.find_all(attrs={"role": True}):
            role = elem["role"]
            if role in ("banner", "navigation", "main", "contentinfo"):
                selector = self._generate_selector(elem)
                landmarks[f"aria-{role}"] = selector

        return landmarks

    def _generate_selector(self, elem: Tag) -> str:
        """Generate a CSS selector for an element."""
        parts = [elem.name]

        # Add ID if present
        if elem.get("id"):
            return f"{elem.name}#{elem['id']}"

        # Add classes
        classes = elem.get("class", [])
        if classes:
            # Use first two classes max
            for cls in classes[:2]:
                parts.append(f".{cls}")

        return "".join(parts)

    def _extract_iframes(self, soup: BeautifulSoup) -> list[IframeInfo]:
        """Extract iframe information."""
        iframes = []

        for iframe in soup.find_all("iframe"):
            src = iframe.get("src", "")
            if not src:
                continue

            # Generate selector
            selector = self._generate_selector(iframe)

            # Determine position (rough heuristic)
            position = self._determine_position(iframe)

            # Get dimensions
            width = iframe.get("width")
            height = iframe.get("height")
            dimensions = None
            if width and height:
                try:
                    dimensions = (int(width), int(height))
                except ValueError:
                    pass

            # Check if likely dynamic
            is_dynamic = any(
                indicator in src.lower()
                for indicator in ["embed", "player", "widget", "api"]
            )

            iframes.append(IframeInfo(
                selector=selector,
                src_pattern=self._extract_src_pattern(src),
                position=position,
                dimensions=dimensions,
                is_dynamic=is_dynamic,
            ))

        return iframes

    def _determine_position(self, elem: Tag) -> str:
        """Determine the position of an element in the page."""
        # Look at ancestors
        for parent in elem.parents:
            if parent.name == "header":
                return "header"
            elif parent.name == "footer":
                return "footer"
            elif parent.name == "aside":
                return "sidebar"
            elif parent.name in ("article", "main"):
                return "content"
            elif parent.name == "nav":
                return "navigation"

        return "content"

    def _extract_src_pattern(self, src: str) -> str:
        """Extract a pattern from an iframe src."""
        # Replace specific IDs with placeholders
        pattern = re.sub(r"/\d+", "/{id}", src)
        pattern = re.sub(r"[a-f0-9]{8,}", "{hash}", pattern)
        return pattern

    def _extract_script_signatures(self, soup: BeautifulSoup) -> list[str]:
        """Extract script signatures (src patterns)."""
        signatures = []

        for script in soup.find_all("script", src=True):
            src = script["src"]
            # Create signature from src
            sig = self._create_script_signature(src)
            if sig:
                signatures.append(sig)

        return list(set(signatures))[:20]

    def _create_script_signature(self, src: str) -> str | None:
        """Create a signature from a script src."""
        # Skip inline data URIs
        if src.startswith("data:"):
            return None

        # Extract filename/path pattern
        match = re.search(r"/([^/]+\.js)", src.lower())
        if match:
            return match.group(1)

        # Look for known patterns
        if "jquery" in src.lower():
            return "jquery"
        if "react" in src.lower():
            return "react"
        if "vue" in src.lower():
            return "vue"
        if "angular" in src.lower():
            return "angular"

        return None

    def _identify_content_regions(self, soup: BeautifulSoup) -> list[ContentRegion]:
        """Identify main content regions."""
        regions = []

        # Look for article content
        article = soup.find("article")
        if article:
            regions.append(ContentRegion(
                name="article",
                primary_selector="article",
                fallback_selectors=["main", ".content", "#content"],
                content_type="text",
                confidence=0.9,
            ))

        # Look for main content
        main = soup.find("main")
        if main and not article:
            regions.append(ContentRegion(
                name="main",
                primary_selector="main",
                fallback_selectors=[".main-content", "#main"],
                content_type="text",
                confidence=0.85,
            ))

        # Look for content divs
        for div in soup.find_all("div", class_=True):
            classes = div.get("class", [])
            for cls in classes:
                if any(word in cls.lower() for word in ["content", "article", "post", "entry"]):
                    selector = f"div.{cls}"
                    if not any(r.primary_selector == selector for r in regions):
                        regions.append(ContentRegion(
                            name=cls,
                            primary_selector=selector,
                            content_type="text",
                            confidence=0.7,
                        ))
                    break

        return regions[:5]  # Limit to top 5 regions

    def _find_navigation(self, soup: BeautifulSoup) -> list[str]:
        """Find navigation selectors."""
        selectors = []

        # Look for nav elements
        for nav in soup.find_all("nav"):
            selector = self._generate_selector(nav)
            selectors.append(selector)

        # Look for common navigation classes
        for elem in soup.find_all(class_=re.compile(r"nav|menu|navigation", re.I)):
            selector = self._generate_selector(elem)
            if selector not in selectors:
                selectors.append(selector)

        return selectors[:10]

    def _detect_pagination(self, soup: BeautifulSoup, url: str) -> PaginationInfo | None:
        """Detect pagination patterns."""
        pagination = PaginationInfo()

        # Look for pagination containers
        pag_container = soup.find(class_=re.compile(r"pag|page-nav", re.I))
        if not pag_container:
            pag_container = soup.find("nav", attrs={"aria-label": re.compile(r"pag", re.I)})

        if not pag_container:
            return None

        # Look for next link
        next_link = pag_container.find(
            "a",
            string=re.compile(r"next|→|›|»", re.I)
        )
        if next_link:
            pagination.next_selector = self._generate_selector(next_link)

        # Look for prev link
        prev_link = pag_container.find(
            "a",
            string=re.compile(r"prev|←|‹|«", re.I)
        )
        if prev_link:
            pagination.prev_selector = self._generate_selector(prev_link)

        # Look for page numbers
        page_links = pag_container.find_all("a", string=re.compile(r"^\d+$"))
        if page_links:
            # Try to detect URL pattern
            for link in page_links:
                href = link.get("href", "")
                if "page" in href.lower():
                    pagination.pattern = re.sub(r"\d+", "{n}", href)
                    break

        return pagination if pagination.next_selector else None

    def _calculate_hash(self, html: str) -> str:
        """Calculate a content hash for the structure."""
        # Remove whitespace variation
        normalized = re.sub(r"\s+", " ", html)
        return hashlib.md5(normalized.encode()).hexdigest()

    def compare(
        self,
        old_structure: PageStructure,
        new_structure: PageStructure,
    ) -> float:
        """
        Compare two page structures and return similarity score.

        Args:
            old_structure: Previous structure.
            new_structure: Current structure.

        Returns:
            Similarity score between 0 and 1.
        """
        scores = []

        # Compare tag hierarchy
        old_tags = set(old_structure.tag_hierarchy.get("tag_counts", {}).keys())
        new_tags = set(new_structure.tag_hierarchy.get("tag_counts", {}).keys())
        if old_tags or new_tags:
            tag_sim = len(old_tags & new_tags) / len(old_tags | new_tags)
            scores.append(tag_sim)

        # Compare classes
        old_classes = set(old_structure.css_class_map.keys())
        new_classes = set(new_structure.css_class_map.keys())
        if old_classes or new_classes:
            class_sim = len(old_classes & new_classes) / len(old_classes | new_classes)
            scores.append(class_sim)

        # Compare IDs
        if old_structure.id_attributes or new_structure.id_attributes:
            id_sim = len(old_structure.id_attributes & new_structure.id_attributes) / \
                     len(old_structure.id_attributes | new_structure.id_attributes)
            scores.append(id_sim)

        # Compare landmarks
        old_landmarks = set(old_structure.semantic_landmarks.keys())
        new_landmarks = set(new_structure.semantic_landmarks.keys())
        if old_landmarks or new_landmarks:
            landmark_sim = len(old_landmarks & new_landmarks) / len(old_landmarks | new_landmarks)
            scores.append(landmark_sim)

        return sum(scores) / len(scores) if scores else 0.0
