#!/usr/bin/env python3
"""
Sitemap-Based Crawling Example

Demonstrates how to use the sitemap processing capabilities to efficiently
discover and crawl URLs from a website's sitemap, with structure fingerprinting
for change detection.

Features demonstrated:
- Fetching and parsing XML sitemaps
- Handling sitemap indexes (nested sitemaps)
- Processing gzip-compressed sitemaps
- Using sitemap metadata (lastmod, changefreq, priority)
- Prioritizing URLs based on sitemap hints
- Structure fingerprinting for each crawled page
- Change detection between crawls (cosmetic, minor, moderate, breaking)

Usage:
    # Basic usage - crawl from sitemap with fingerprinting
    python examples/sitemap_crawl_example.py --domain example.com

    # With custom options
    python examples/sitemap_crawl_example.py \
        --domain example.com \
        --max-urls 100 \
        --priority-threshold 0.5

    # Just list URLs from sitemap (no crawling)
    python examples/sitemap_crawl_example.py --domain example.com --list-only

    # Disable fingerprinting
    python examples/sitemap_crawl_example.py --domain example.com --no-fingerprint

Requirements:
    - pip install -e .
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from crawler.compliance.sitemap_parser import (
    SitemapFetcher,
    SitemapParser,
    SitemapURL,
    ChangeFrequency,
    fetch_sitemap_urls,
)
from crawler.adaptive.structure_analyzer import StructureAnalyzer, AnalysisConfig
from crawler.adaptive.change_detector import ChangeDetector, ChangeAnalysis
from crawler.models import PageStructure
from crawler.utils.logging import CrawlerLogger, setup_logging


class SitemapCrawler:
    """
    Demonstrates sitemap-based URL discovery and prioritization.

    Key concepts:
    - Sitemaps provide a structured index of all pages on a site
    - They include metadata: lastmod, changefreq, priority
    - Sitemap indexes can contain multiple child sitemaps
    - Gzip compression is automatically handled
    """

    def __init__(
        self,
        domain: str,
        user_agent: str = "SitemapCrawler/1.0",
        max_urls: int | None = None,
        priority_threshold: float = 0.0,
        recent_days: int | None = None,
        enable_fingerprinting: bool = True,
    ):
        """
        Initialize the sitemap crawler.

        Args:
            domain: Domain to crawl (e.g., "example.com")
            user_agent: User agent for requests
            max_urls: Maximum URLs to process
            priority_threshold: Only include URLs with priority >= this value
            recent_days: Only include URLs modified within this many days
            enable_fingerprinting: Enable structure fingerprinting for change detection
        """
        self.domain = domain
        self.user_agent = user_agent
        self.max_urls = max_urls
        self.priority_threshold = priority_threshold
        self.recent_days = recent_days
        self.enable_fingerprinting = enable_fingerprinting
        self.logger = CrawlerLogger("sitemap_crawler")

        # Initialize fingerprinting components
        if enable_fingerprinting:
            self.structure_analyzer = StructureAnalyzer(
                config=AnalysisConfig(
                    min_content_length=100,
                    max_depth=10,
                    track_classes=True,
                    track_ids=True,
                    extract_scripts=True,
                )
            )
            self.change_detector = ChangeDetector(breaking_threshold=0.70)
            # Store page structures for change detection
            self._page_structures: dict[str, PageStructure] = {}

    async def discover_sitemaps(self) -> list[str]:
        """
        Discover sitemap URLs for the domain.

        Checks:
        1. robots.txt for Sitemap: directives
        2. Common sitemap paths (/sitemap.xml, /sitemap_index.xml)

        Returns:
            List of sitemap URLs to process
        """
        async with SitemapFetcher(user_agent=self.user_agent) as fetcher:
            sitemap_urls = await fetcher.discover_sitemaps(self.domain)

            if sitemap_urls:
                self.logger.info(
                    "Discovered sitemaps",
                    count=len(sitemap_urls),
                    urls=sitemap_urls,
                )
            else:
                self.logger.warning("No sitemaps found", domain=self.domain)

            return sitemap_urls

    async def fetch_all_urls(self, sitemap_urls: list[str]) -> list[SitemapURL]:
        """
        Fetch all URLs from the discovered sitemaps.

        Handles:
        - Sitemap indexes (recursively fetches child sitemaps)
        - Gzip compression
        - URL metadata extraction

        Args:
            sitemap_urls: List of sitemap URLs to process

        Returns:
            List of SitemapURL objects with metadata
        """
        all_urls: list[SitemapURL] = []

        async with SitemapFetcher(
            user_agent=self.user_agent,
            max_sitemaps=100,
            max_urls_per_sitemap=50000,
        ) as fetcher:
            async for sitemap in fetcher.fetch_all_sitemaps(sitemap_urls):
                self.logger.info(
                    "Processing sitemap",
                    url=sitemap.url,
                    is_index=sitemap.is_index,
                    url_count=len(sitemap.urls),
                    child_sitemaps=len(sitemap.sitemaps),
                )

                # Add URLs from this sitemap
                for url in sitemap.urls:
                    all_urls.append(url)

                    if self.max_urls and len(all_urls) >= self.max_urls:
                        self.logger.info(
                            "Reached max URLs limit",
                            limit=self.max_urls,
                        )
                        return all_urls

        self.logger.info("Fetched all URLs", total=len(all_urls))
        return all_urls

    def filter_urls(self, urls: list[SitemapURL]) -> list[SitemapURL]:
        """
        Filter URLs based on priority and recency.

        Args:
            urls: List of SitemapURL objects

        Returns:
            Filtered list based on configuration
        """
        filtered = []
        cutoff_date = None

        if self.recent_days:
            cutoff_date = datetime.utcnow() - timedelta(days=self.recent_days)

        for url in urls:
            # Check priority threshold
            if url.priority and url.priority < self.priority_threshold:
                continue

            # Check recency
            if cutoff_date and url.lastmod:
                if url.lastmod < cutoff_date:
                    continue

            filtered.append(url)

        self.logger.info(
            "Filtered URLs",
            original=len(urls),
            filtered=len(filtered),
            priority_threshold=self.priority_threshold,
            recent_days=self.recent_days,
        )

        return filtered

    def prioritize_urls(self, urls: list[SitemapURL]) -> list[SitemapURL]:
        """
        Sort URLs by priority (highest first).

        Also considers:
        - Explicit priority value
        - Change frequency (more frequent = higher priority)
        - Recency (more recent = higher priority)

        Args:
            urls: List of SitemapURL objects

        Returns:
            Sorted list with highest priority first
        """
        def priority_score(url: SitemapURL) -> float:
            score = 0.0

            # Explicit priority (0.0 - 1.0)
            if url.priority:
                score += url.priority * 10
            else:
                score += 5  # Default priority

            # Change frequency bonus
            freq_bonus = {
                ChangeFrequency.ALWAYS: 5,
                ChangeFrequency.HOURLY: 4,
                ChangeFrequency.DAILY: 3,
                ChangeFrequency.WEEKLY: 2,
                ChangeFrequency.MONTHLY: 1,
                ChangeFrequency.YEARLY: 0,
                ChangeFrequency.NEVER: -1,
            }
            if url.changefreq:
                score += freq_bonus.get(url.changefreq, 0)

            # Recency bonus
            if url.lastmod:
                days_old = (datetime.utcnow() - url.lastmod).days
                if days_old < 1:
                    score += 5
                elif days_old < 7:
                    score += 3
                elif days_old < 30:
                    score += 1

            return score

        return sorted(urls, key=priority_score, reverse=True)

    def _classify_page_type(self, url: str) -> str:
        """Classify the page type based on URL patterns."""
        url_lower = url.lower()
        if any(x in url_lower for x in ["/article", "/post", "/blog", "/news"]):
            return "article"
        elif any(x in url_lower for x in ["/category", "/tag", "/archive"]):
            return "listing"
        elif any(x in url_lower for x in ["/product", "/item", "/shop"]):
            return "product"
        elif url_lower.endswith("/") or url_lower.count("/") <= 3:
            return "homepage"
        return "content"

    def _analyze_page_structure(
        self,
        html: str,
        url: str,
    ) -> tuple[PageStructure, ChangeAnalysis | None]:
        """
        Analyze page structure and detect changes.

        Args:
            html: Page HTML content
            url: Page URL

        Returns:
            Tuple of (current structure, change analysis if previous exists)
        """
        page_type = self._classify_page_type(url)
        current_structure = self.structure_analyzer.analyze(html, url, page_type)

        # Check for changes if we have a previous structure
        change_analysis = None
        if url in self._page_structures:
            previous_structure = self._page_structures[url]
            change_analysis = self.change_detector.detect_changes(
                previous_structure,
                current_structure,
            )

        # Store current structure for future comparisons
        self._page_structures[url] = current_structure

        return current_structure, change_analysis

    async def crawl_urls(self, urls: list[SitemapURL]) -> dict[str, Any]:
        """
        Crawl the URLs and return results.

        This is a simplified crawl - in production you would:
        - Use the full Crawler class
        - Apply rate limiting
        - Store results properly

        Fingerprinting features:
        - Analyzes page structure for each URL
        - Detects structural changes between crawls
        - Classifies change severity (cosmetic, minor, moderate, breaking)

        Args:
            urls: Prioritized list of URLs to crawl

        Returns:
            Dictionary with crawl statistics
        """
        stats = {
            "total": len(urls),
            "success": 0,
            "failed": 0,
            "results": [],
            "fingerprinting": {
                "pages_analyzed": 0,
                "changes_detected": 0,
                "breaking_changes": 0,
            } if self.enable_fingerprinting else None,
        }

        async with httpx.AsyncClient(
            headers={"User-Agent": self.user_agent},
            timeout=30.0,
            follow_redirects=True,
        ) as client:
            for i, url in enumerate(urls):
                try:
                    self.logger.info(
                        "Crawling",
                        index=i + 1,
                        total=len(urls),
                        url=url.loc,
                        priority=url.priority,
                    )

                    response = await client.get(url.loc)

                    result = {
                        "url": url.loc,
                        "status": response.status_code,
                        "content_length": len(response.content),
                        "priority": url.priority,
                        "lastmod": url.lastmod.isoformat() if url.lastmod else None,
                    }

                    # Apply fingerprinting if enabled and HTML content
                    if (self.enable_fingerprinting and
                        response.status_code == 200 and
                        "text/html" in response.headers.get("content-type", "")):

                        structure, change_analysis = self._analyze_page_structure(
                            response.text,
                            url.loc,
                        )

                        stats["fingerprinting"]["pages_analyzed"] += 1

                        # Add fingerprint info to result
                        result["fingerprint"] = {
                            "content_hash": structure.content_hash,
                            "page_type": structure.page_type,
                            "tag_count": sum(structure.tag_hierarchy.get("tag_counts", {}).values())
                                if structure.tag_hierarchy else 0,
                            "content_regions": len(structure.content_regions),
                            "semantic_landmarks": list(structure.semantic_landmarks.keys())
                                if structure.semantic_landmarks else [],
                        }

                        if change_analysis:
                            stats["fingerprinting"]["changes_detected"] += 1
                            if change_analysis.requires_relearning:
                                stats["fingerprinting"]["breaking_changes"] += 1

                            result["change_analysis"] = {
                                "has_changes": change_analysis.has_changes,
                                "classification": change_analysis.classification.value,
                                "similarity_score": f"{change_analysis.similarity_score:.2%}",
                                "requires_relearning": change_analysis.requires_relearning,
                            }

                            self.logger.info(
                                "Structure change detected",
                                url=url.loc,
                                classification=change_analysis.classification.value,
                                similarity=f"{change_analysis.similarity_score:.2%}",
                            )

                    stats["results"].append(result)

                    if response.status_code == 200:
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1

                    # Polite delay between requests
                    await asyncio.sleep(1.0)

                except Exception as e:
                    self.logger.error("Crawl failed", url=url.loc, error=str(e))
                    stats["failed"] += 1
                    stats["results"].append({
                        "url": url.loc,
                        "status": "error",
                        "error": str(e),
                    })

        return stats

    async def run(self, list_only: bool = False) -> dict[str, Any]:
        """
        Run the complete sitemap crawl workflow.

        Steps:
        1. Discover sitemaps for the domain
        2. Fetch all URLs from sitemaps
        3. Filter based on priority/recency
        4. Prioritize remaining URLs
        5. Crawl (unless list_only)

        Args:
            list_only: If True, just list URLs without crawling

        Returns:
            Results dictionary
        """
        # Step 1: Discover sitemaps
        sitemap_urls = await self.discover_sitemaps()
        if not sitemap_urls:
            return {"error": "No sitemaps found"}

        # Step 2: Fetch all URLs
        all_urls = await self.fetch_all_urls(sitemap_urls)
        if not all_urls:
            return {"error": "No URLs found in sitemaps"}

        # Step 3: Filter
        filtered_urls = self.filter_urls(all_urls)

        # Step 4: Prioritize
        prioritized_urls = self.prioritize_urls(filtered_urls)

        if list_only:
            return {
                "domain": self.domain,
                "sitemap_count": len(sitemap_urls),
                "total_urls": len(all_urls),
                "filtered_urls": len(prioritized_urls),
                "urls": [
                    {
                        "loc": url.loc,
                        "priority": url.priority,
                        "lastmod": url.lastmod.isoformat() if url.lastmod else None,
                        "changefreq": url.changefreq.value if url.changefreq else None,
                    }
                    for url in prioritized_urls[:50]  # First 50 for display
                ],
            }

        # Step 5: Crawl
        stats = await self.crawl_urls(prioritized_urls)
        stats["domain"] = self.domain
        stats["sitemap_count"] = len(sitemap_urls)

        return stats


async def quick_sitemap_fetch(domain: str) -> list[str]:
    """
    Quick utility function to fetch all URLs from a domain's sitemap.

    This demonstrates the simplest possible usage.

    Args:
        domain: Domain to fetch sitemap for

    Returns:
        List of URL strings
    """
    urls = await fetch_sitemap_urls(domain)
    return [url.loc for url in urls]


def print_results(results: dict[str, Any]) -> None:
    """Pretty print the crawl results."""
    print("\n" + "=" * 60)
    print("SITEMAP CRAWL RESULTS")
    print("=" * 60)

    if "error" in results:
        print(f"\nError: {results['error']}")
        return

    print(f"\nDomain: {results.get('domain', 'N/A')}")
    print(f"Sitemaps found: {results.get('sitemap_count', 0)}")
    print(f"Total URLs in sitemaps: {results.get('total_urls', results.get('total', 0))}")

    if "filtered_urls" in results:
        print(f"URLs after filtering: {results['filtered_urls']}")

    if "success" in results:
        print(f"\nCrawl Results:")
        print(f"  Success: {results['success']}")
        print(f"  Failed: {results['failed']}")

    # Print fingerprinting statistics
    if results.get("fingerprinting"):
        fp_stats = results["fingerprinting"]
        print(f"\nFingerprinting Analysis:")
        print(f"  Pages analyzed: {fp_stats['pages_analyzed']}")
        print(f"  Changes detected: {fp_stats['changes_detected']}")
        print(f"  Breaking changes: {fp_stats['breaking_changes']}")

    if "urls" in results:
        print(f"\nTop URLs (by priority):")
        for i, url in enumerate(results["urls"][:10], 1):
            priority = url.get("priority", "N/A")
            lastmod = url.get("lastmod", "N/A")
            print(f"  {i}. {url['loc']}")
            print(f"     Priority: {priority}, Last Modified: {lastmod}")

    # Print fingerprint details for results with fingerprinting
    if "results" in results:
        fingerprinted = [r for r in results["results"] if r.get("fingerprint")]
        if fingerprinted:
            print(f"\nFingerprint Details (first 5):")
            for i, r in enumerate(fingerprinted[:5], 1):
                fp = r["fingerprint"]
                print(f"  {i}. {r['url'][:60]}...")
                print(f"     Type: {fp['page_type']}, Tags: {fp['tag_count']}, "
                      f"Regions: {fp['content_regions']}")
                if r.get("change_analysis"):
                    ca = r["change_analysis"]
                    print(f"     Change: {ca['classification']} "
                          f"(similarity: {ca['similarity_score']})")

    print("=" * 60)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Crawl a website using its sitemap"
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain to crawl (e.g., example.com)",
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=20,
        help="Maximum URLs to crawl (default: 20)",
    )
    parser.add_argument(
        "--priority-threshold",
        type=float,
        default=0.0,
        help="Only include URLs with priority >= this value (0.0-1.0)",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        default=None,
        help="Only include URLs modified within this many days",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Just list URLs from sitemap (no crawling)",
    )
    parser.add_argument(
        "--no-fingerprint",
        action="store_true",
        help="Disable structure fingerprinting",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(
        level="DEBUG" if args.verbose else "INFO",
        format_type="console",
    )

    print("""
    ===============================================
       SITEMAP-BASED CRAWLER EXAMPLE
       Demonstrates sitemap processing features
    ===============================================

    This example shows how to:
    1. Discover sitemaps (robots.txt + common paths)
    2. Parse XML sitemaps and sitemap indexes
    3. Handle gzip-compressed sitemaps
    4. Use sitemap metadata (priority, lastmod, changefreq)
    5. Prioritize URLs for efficient crawling
    6. Fingerprint page structures for change detection
    7. Detect structural changes between crawls

    """)

    crawler = SitemapCrawler(
        domain=args.domain,
        max_urls=args.max_urls,
        priority_threshold=args.priority_threshold,
        recent_days=args.recent_days,
        enable_fingerprinting=not args.no_fingerprint,
    )

    try:
        results = await crawler.run(list_only=args.list_only)
        print_results(results)
    except KeyboardInterrupt:
        print("\n\nCrawl interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
