"""
Link extractor for URL discovery from HTML.

Extracts and validates links from HTML content.
"""

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from crawler.utils.logging import CrawlerLogger
from crawler.utils.url_utils import (
    get_domain,
    is_same_domain,
    is_valid_url,
    normalize_url,
)


@dataclass
class ExtractedLink:
    """An extracted link with metadata."""

    url: str
    text: str
    rel: list[str] = field(default_factory=list)
    is_external: bool = False
    is_resource: bool = False
    link_type: str = "anchor"  # anchor, image, script, stylesheet, etc.


@dataclass
class LinkExtractionResult:
    """Result of link extraction."""

    base_url: str
    links: list[ExtractedLink] = field(default_factory=list)
    internal_links: list[str] = field(default_factory=list)
    external_links: list[str] = field(default_factory=list)
    resource_links: list[str] = field(default_factory=list)
    total_count: int = 0


class LinkExtractor:
    """
    Extracts links from HTML content.

    Features:
    - Extracts anchor, image, script, stylesheet links
    - Classifies links as internal/external
    - Respects nofollow directives
    - Normalizes URLs for deduplication
    """

    # Resource file extensions
    RESOURCE_EXTENSIONS = frozenset([
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
        ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx",
        ".zip", ".tar", ".gz", ".rar",
        ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ])

    # Link types to extract
    LINK_SELECTORS = {
        "anchor": ("a", "href"),
        "image": ("img", "src"),
        "script": ("script", "src"),
        "stylesheet": ("link[rel='stylesheet']", "href"),
        "iframe": ("iframe", "src"),
    }

    def __init__(
        self,
        follow_external: bool = False,
        follow_nofollow: bool = False,
        extract_resources: bool = False,
        allowed_domains: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the link extractor.

        Args:
            follow_external: Whether to include external links.
            follow_nofollow: Whether to follow nofollow links.
            extract_resources: Whether to extract resource links.
            allowed_domains: List of allowed domains (None = same domain only).
            exclude_patterns: URL patterns to exclude.
            logger: Logger instance.
        """
        self.follow_external = follow_external
        self.follow_nofollow = follow_nofollow
        self.extract_resources = extract_resources
        self.allowed_domains = allowed_domains
        self.exclude_patterns = exclude_patterns or []
        self.logger = logger or CrawlerLogger("link_extractor")

        # Compile exclude patterns
        self._exclude_compiled = [
            re.compile(p) for p in self.exclude_patterns
        ]

    def extract(
        self,
        html: str,
        base_url: str,
    ) -> LinkExtractionResult:
        """
        Extract links from HTML content.

        Args:
            html: HTML content.
            base_url: Base URL for resolving relative links.

        Returns:
            LinkExtractionResult with all extracted links.
        """
        result = LinkExtractionResult(base_url=base_url)
        base_domain = get_domain(base_url)
        seen_urls: set[str] = set()

        soup = BeautifulSoup(html, "lxml")

        # Check for base tag
        base_tag = soup.find("base", href=True)
        if base_tag:
            base_url = urljoin(base_url, base_tag["href"])

        # Check for meta robots nofollow
        meta_robots = soup.find("meta", attrs={"name": "robots"})
        page_nofollow = False
        if meta_robots:
            content = meta_robots.get("content", "").lower()
            page_nofollow = "nofollow" in content

        # Extract anchor links
        for link in soup.find_all("a", href=True):
            extracted = self._extract_anchor(link, base_url, base_domain, page_nofollow)
            if extracted and extracted.url not in seen_urls:
                if self._should_include(extracted, base_domain):
                    result.links.append(extracted)
                    seen_urls.add(extracted.url)

                    if extracted.is_external:
                        result.external_links.append(extracted.url)
                    else:
                        result.internal_links.append(extracted.url)

        # Extract resource links if requested
        if self.extract_resources:
            for link_type, (selector, attr) in self.LINK_SELECTORS.items():
                if link_type == "anchor":
                    continue

                for elem in soup.select(selector):
                    url = elem.get(attr)
                    if not url:
                        continue

                    absolute_url = urljoin(base_url, url)
                    if is_valid_url(absolute_url) and absolute_url not in seen_urls:
                        normalized = normalize_url(absolute_url)
                        extracted = ExtractedLink(
                            url=normalized,
                            text="",
                            is_external=not is_same_domain(absolute_url, base_url),
                            is_resource=True,
                            link_type=link_type,
                        )
                        result.links.append(extracted)
                        result.resource_links.append(normalized)
                        seen_urls.add(normalized)

        result.total_count = len(result.links)
        return result

    def _extract_anchor(
        self,
        link,
        base_url: str,
        base_domain: str,
        page_nofollow: bool,
    ) -> ExtractedLink | None:
        """Extract data from an anchor tag."""
        href = link.get("href", "")
        if not href:
            return None

        # Skip javascript: and mailto: links
        if href.startswith(("javascript:", "mailto:", "tel:", "#")):
            return None

        # Resolve relative URL
        absolute_url = urljoin(base_url, href)

        # Validate URL
        if not is_valid_url(absolute_url):
            return None

        # Normalize
        normalized = normalize_url(absolute_url)

        # Get link text
        text = link.get_text(strip=True)

        # Get rel attribute
        rel = link.get("rel", [])
        if isinstance(rel, str):
            rel = rel.split()

        # Check nofollow
        has_nofollow = "nofollow" in rel or page_nofollow

        # Check if external
        is_external = not is_same_domain(absolute_url, base_url)

        # Check if resource
        is_resource = self._is_resource_url(normalized)

        return ExtractedLink(
            url=normalized,
            text=text,
            rel=rel,
            is_external=is_external,
            is_resource=is_resource,
            link_type="anchor",
        )

    def _should_include(self, link: ExtractedLink, base_domain: str) -> bool:
        """Check if a link should be included in results."""
        # Check nofollow
        if "nofollow" in link.rel and not self.follow_nofollow:
            return False

        # Check external
        if link.is_external and not self.follow_external:
            # Check allowed domains
            if self.allowed_domains:
                link_domain = get_domain(link.url)
                if not any(
                    link_domain == d or link_domain.endswith(f".{d}")
                    for d in self.allowed_domains
                ):
                    return False
            else:
                return False

        # Check exclude patterns
        for pattern in self._exclude_compiled:
            if pattern.search(link.url):
                return False

        # Skip resources unless explicitly requested
        if link.is_resource and not self.extract_resources:
            return False

        return True

    def _is_resource_url(self, url: str) -> bool:
        """Check if URL points to a resource file."""
        parsed = urlparse(url)
        path = parsed.path.lower()

        return any(path.endswith(ext) for ext in self.RESOURCE_EXTENSIONS)

    def extract_urls(
        self,
        html: str,
        base_url: str,
    ) -> list[str]:
        """
        Extract just the URLs (convenience method).

        Args:
            html: HTML content.
            base_url: Base URL for resolving.

        Returns:
            List of extracted URLs.
        """
        result = self.extract(html, base_url)
        return result.internal_links + (
            result.external_links if self.follow_external else []
        )


def extract_links_simple(
    html: str,
    base_url: str,
    same_domain_only: bool = True,
) -> list[str]:
    """
    Simple link extraction function.

    Args:
        html: HTML content.
        base_url: Base URL.
        same_domain_only: Only return same-domain links.

    Returns:
        List of extracted URLs.
    """
    extractor = LinkExtractor(
        follow_external=not same_domain_only,
    )
    return extractor.extract_urls(html, base_url)
