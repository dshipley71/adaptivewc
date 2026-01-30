"""
XML Sitemap parser for the adaptive web crawler.

Supports:
- Standard sitemap.xml files (sitemapindex and urlset)
- Gzip compressed sitemaps (.xml.gz)
- Sitemap index files with nested sitemaps
- changefreq, lastmod, priority attributes
- robots.txt sitemap discovery integration

References:
- https://www.sitemaps.org/protocol.html
"""

import gzip
import io
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree as ET

import httpx

from crawler.utils.logging import CrawlerLogger
from crawler.utils.url_utils import get_domain, normalize_url


class ChangeFrequency(str, Enum):
    """Standard sitemap change frequencies."""

    ALWAYS = "always"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    NEVER = "never"


@dataclass
class SitemapURL:
    """A single URL entry from a sitemap."""

    loc: str
    lastmod: datetime | None = None
    changefreq: ChangeFrequency | None = None
    priority: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "loc": self.loc,
            "lastmod": self.lastmod.isoformat() if self.lastmod else None,
            "changefreq": self.changefreq.value if self.changefreq else None,
            "priority": self.priority,
        }


@dataclass
class SitemapIndex:
    """A sitemap index file containing references to other sitemaps."""

    loc: str
    lastmod: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "loc": self.loc,
            "lastmod": self.lastmod.isoformat() if self.lastmod else None,
        }


@dataclass
class Sitemap:
    """Parsed sitemap file."""

    url: str
    domain: str
    is_index: bool = False
    urls: list[SitemapURL] = field(default_factory=list)
    sitemaps: list[SitemapIndex] = field(default_factory=list)  # For sitemap index files
    parsed_at: datetime = field(default_factory=datetime.utcnow)
    fetch_status: int = 200
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "url": self.url,
            "domain": self.domain,
            "is_index": self.is_index,
            "urls": [u.to_dict() for u in self.urls],
            "sitemaps": [s.to_dict() for s in self.sitemaps],
            "parsed_at": self.parsed_at.isoformat(),
            "fetch_status": self.fetch_status,
            "error": self.error,
        }

    @property
    def url_count(self) -> int:
        """Return the number of URLs in this sitemap."""
        return len(self.urls)


# XML namespaces used in sitemaps
SITEMAP_NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "xhtml": "http://www.w3.org/1999/xhtml",
    "image": "http://www.google.com/schemas/sitemap-image/1.1",
    "video": "http://www.google.com/schemas/sitemap-video/1.1",
    "news": "http://www.google.com/schemas/sitemap-news/0.9",
}


class SitemapParser:
    """
    Parser for XML sitemap files.

    Handles both urlset sitemaps and sitemapindex files.
    """

    def __init__(self, logger: CrawlerLogger | None = None):
        """Initialize the parser."""
        self.logger = logger or CrawlerLogger("sitemap_parser")

    def parse(self, content: bytes | str, url: str, status_code: int = 200) -> Sitemap:
        """
        Parse sitemap content.

        Args:
            content: Raw sitemap content (bytes or string).
            url: URL this sitemap was fetched from.
            status_code: HTTP status code when fetching.

        Returns:
            Parsed Sitemap object.
        """
        domain = get_domain(url)
        sitemap = Sitemap(
            url=url,
            domain=domain,
            fetch_status=status_code,
        )

        if status_code != 200:
            sitemap.error = f"HTTP {status_code}"
            return sitemap

        # Handle gzip content
        if isinstance(content, bytes):
            try:
                # Try to decompress if gzipped
                if content[:2] == b"\x1f\x8b":  # Gzip magic number
                    content = gzip.decompress(content)
                content = content.decode("utf-8")
            except (gzip.BadGzipFile, UnicodeDecodeError) as e:
                sitemap.error = f"Decode error: {e}"
                return sitemap

        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            sitemap.error = f"XML parse error: {e}"
            return sitemap

        # Determine sitemap type from root tag
        root_tag = root.tag.lower()

        # Strip namespace prefix if present
        if "}" in root_tag:
            root_tag = root_tag.split("}")[1]

        if root_tag == "sitemapindex":
            sitemap.is_index = True
            self._parse_sitemap_index(root, sitemap, url)
        elif root_tag == "urlset":
            sitemap.is_index = False
            self._parse_urlset(root, sitemap, url)
        else:
            sitemap.error = f"Unknown root element: {root_tag}"

        return sitemap

    def _parse_sitemap_index(self, root: ET.Element, sitemap: Sitemap, base_url: str) -> None:
        """Parse a sitemapindex element."""
        for sm_elem in root.iter():
            tag = sm_elem.tag.lower()
            if "}" in tag:
                tag = tag.split("}")[1]

            if tag == "sitemap":
                sm_index = self._parse_sitemap_entry(sm_elem, base_url)
                if sm_index:
                    sitemap.sitemaps.append(sm_index)

    def _parse_sitemap_entry(self, elem: ET.Element, base_url: str) -> SitemapIndex | None:
        """Parse a single sitemap entry from sitemapindex."""
        loc = None
        lastmod = None

        for child in elem:
            tag = child.tag.lower()
            if "}" in tag:
                tag = tag.split("}")[1]

            if tag == "loc" and child.text:
                loc = self._resolve_url(child.text.strip(), base_url)
            elif tag == "lastmod" and child.text:
                lastmod = self._parse_datetime(child.text.strip())

        if loc:
            return SitemapIndex(loc=loc, lastmod=lastmod)
        return None

    def _parse_urlset(self, root: ET.Element, sitemap: Sitemap, base_url: str) -> None:
        """Parse a urlset element."""
        for url_elem in root.iter():
            tag = url_elem.tag.lower()
            if "}" in tag:
                tag = tag.split("}")[1]

            if tag == "url":
                url_entry = self._parse_url_entry(url_elem, base_url)
                if url_entry:
                    sitemap.urls.append(url_entry)

    def _parse_url_entry(self, elem: ET.Element, base_url: str) -> SitemapURL | None:
        """Parse a single URL entry from urlset."""
        loc = None
        lastmod = None
        changefreq = None
        priority = None

        for child in elem:
            tag = child.tag.lower()
            if "}" in tag:
                tag = tag.split("}")[1]

            if tag == "loc" and child.text:
                loc = self._resolve_url(child.text.strip(), base_url)
            elif tag == "lastmod" and child.text:
                lastmod = self._parse_datetime(child.text.strip())
            elif tag == "changefreq" and child.text:
                try:
                    changefreq = ChangeFrequency(child.text.strip().lower())
                except ValueError:
                    pass
            elif tag == "priority" and child.text:
                try:
                    priority = float(child.text.strip())
                    # Clamp to valid range [0.0, 1.0]
                    priority = max(0.0, min(1.0, priority))
                except ValueError:
                    pass

        if loc:
            return SitemapURL(
                loc=loc,
                lastmod=lastmod,
                changefreq=changefreq,
                priority=priority,
            )
        return None

    def _resolve_url(self, url: str, base_url: str) -> str:
        """Resolve a potentially relative URL against base URL."""
        if url.startswith(("http://", "https://")):
            return normalize_url(url)
        return normalize_url(urljoin(base_url, url))

    def _parse_datetime(self, date_str: str) -> datetime | None:
        """
        Parse sitemap datetime formats.

        Supported formats:
        - YYYY-MM-DD
        - YYYY-MM-DDThh:mm:ss+00:00
        - YYYY-MM-DDThh:mm:ssZ
        """
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S.%fZ",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # Try with timezone offset without colon (e.g., +0000)
        if "+" in date_str or date_str.endswith("Z"):
            try:
                # Remove colon from timezone offset
                clean = date_str.replace("Z", "+0000")
                if "+" in clean:
                    parts = clean.rsplit("+", 1)
                    if len(parts) == 2 and ":" in parts[1]:
                        parts[1] = parts[1].replace(":", "")
                        clean = "+".join(parts)
                return datetime.strptime(clean, "%Y-%m-%dT%H:%M:%S%z")
            except ValueError:
                pass

        return None


class SitemapFetcher:
    """
    Fetches and parses sitemaps from websites.

    Features:
    - Automatic sitemap discovery from robots.txt
    - Recursive sitemap index handling
    - Gzip decompression support
    - Rate limiting integration
    """

    # Common sitemap locations to try if not specified
    DEFAULT_SITEMAP_PATHS = [
        "/sitemap.xml",
        "/sitemap_index.xml",
        "/sitemap-index.xml",
        "/sitemaps.xml",
        "/sitemap1.xml",
    ]

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        user_agent: str = "AdaptiveCrawler/1.0",
        timeout: float = 30.0,
        max_sitemaps: int = 100,
        max_urls_per_sitemap: int = 50000,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the sitemap fetcher.

        Args:
            http_client: HTTP client to use (creates one if None).
            user_agent: User agent for requests.
            timeout: Request timeout in seconds.
            max_sitemaps: Maximum number of sitemaps to process.
            max_urls_per_sitemap: Maximum URLs to extract from a single sitemap.
            logger: Logger instance.
        """
        self._external_client = http_client is not None
        self.http_client = http_client
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_sitemaps = max_sitemaps
        self.max_urls_per_sitemap = max_urls_per_sitemap
        self.logger = logger or CrawlerLogger("sitemap_fetcher")
        self._parser = SitemapParser(logger=self.logger)

    async def __aenter__(self) -> "SitemapFetcher":
        """Async context manager entry."""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if not self._external_client and self.http_client:
            await self.http_client.aclose()

    async def fetch_sitemap(self, url: str) -> Sitemap:
        """
        Fetch and parse a single sitemap.

        Args:
            url: URL of the sitemap to fetch.

        Returns:
            Parsed Sitemap object.
        """
        if not self.http_client:
            raise RuntimeError("HTTP client not initialized. Use async context manager.")

        self.logger.debug("Fetching sitemap", url=url)

        try:
            response = await self.http_client.get(url)
            sitemap = self._parser.parse(response.content, url, response.status_code)

            self.logger.info(
                "Sitemap fetched",
                url=url,
                is_index=sitemap.is_index,
                urls=len(sitemap.urls),
                sitemaps=len(sitemap.sitemaps),
            )

            return sitemap

        except httpx.TimeoutException:
            self.logger.warning("Sitemap fetch timeout", url=url)
            return Sitemap(
                url=url,
                domain=get_domain(url),
                fetch_status=0,
                error="Timeout",
            )
        except httpx.RequestError as e:
            self.logger.warning("Sitemap fetch error", url=url, error=str(e))
            return Sitemap(
                url=url,
                domain=get_domain(url),
                fetch_status=0,
                error=str(e),
            )

    async def fetch_all_sitemaps(
        self,
        sitemap_urls: list[str],
    ) -> AsyncIterator[Sitemap]:
        """
        Fetch multiple sitemaps, recursively processing sitemap indexes.

        Args:
            sitemap_urls: Initial list of sitemap URLs to fetch.

        Yields:
            Parsed Sitemap objects.
        """
        pending = list(sitemap_urls)
        processed: set[str] = set()
        sitemap_count = 0

        while pending and sitemap_count < self.max_sitemaps:
            url = pending.pop(0)

            # Skip if already processed
            if url in processed:
                continue

            processed.add(url)
            sitemap_count += 1

            sitemap = await self.fetch_sitemap(url)
            yield sitemap

            # If it's a sitemap index, add child sitemaps to pending
            if sitemap.is_index:
                for child in sitemap.sitemaps:
                    if child.loc not in processed:
                        pending.append(child.loc)

    async def discover_sitemaps(self, domain: str) -> list[str]:
        """
        Discover sitemap URLs for a domain.

        Tries common sitemap locations if no robots.txt sitemaps found.

        Args:
            domain: Domain to discover sitemaps for.

        Returns:
            List of discovered sitemap URLs.
        """
        discovered = []

        # Try common paths
        base_url = f"https://{domain}"

        for path in self.DEFAULT_SITEMAP_PATHS:
            url = f"{base_url}{path}"
            try:
                response = await self.http_client.head(url)
                if response.status_code == 200:
                    discovered.append(url)
                    self.logger.info("Discovered sitemap", url=url)
                    break  # Found one, stop trying
            except (httpx.RequestError, httpx.TimeoutException):
                continue

        return discovered

    async def get_all_urls(
        self,
        sitemap_urls: list[str],
        filter_domain: str | None = None,
    ) -> AsyncIterator[SitemapURL]:
        """
        Get all URLs from sitemaps, recursively processing indexes.

        Args:
            sitemap_urls: Initial sitemap URLs to process.
            filter_domain: If set, only yield URLs matching this domain.

        Yields:
            SitemapURL objects from all sitemaps.
        """
        url_count = 0

        async for sitemap in self.fetch_all_sitemaps(sitemap_urls):
            for url_entry in sitemap.urls:
                # Apply domain filter if specified
                if filter_domain:
                    url_domain = get_domain(url_entry.loc)
                    if url_domain != filter_domain:
                        continue

                url_count += 1
                if url_count > self.max_urls_per_sitemap * self.max_sitemaps:
                    self.logger.warning(
                        "URL limit reached",
                        limit=self.max_urls_per_sitemap * self.max_sitemaps,
                    )
                    return

                yield url_entry


async def fetch_sitemap_urls(
    domain: str,
    sitemap_urls: list[str] | None = None,
    user_agent: str = "AdaptiveCrawler/1.0",
    timeout: float = 30.0,
) -> list[SitemapURL]:
    """
    Convenience function to fetch all URLs from a domain's sitemaps.

    Args:
        domain: Domain to fetch sitemaps for.
        sitemap_urls: Optional list of sitemap URLs (discovers if None).
        user_agent: User agent for requests.
        timeout: Request timeout.

    Returns:
        List of all SitemapURL objects found.
    """
    async with SitemapFetcher(
        user_agent=user_agent,
        timeout=timeout,
    ) as fetcher:
        if not sitemap_urls:
            sitemap_urls = await fetcher.discover_sitemaps(domain)

        if not sitemap_urls:
            return []

        urls = []
        async for url in fetcher.get_all_urls(sitemap_urls, filter_domain=domain):
            urls.append(url)

        return urls
