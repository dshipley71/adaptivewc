#!/usr/bin/env python3
"""
Sitemap-Based Crawling Example

Demonstrates how to use the sitemap processing capabilities to efficiently
discover and crawl URLs from a website's sitemap.

Features demonstrated:
- Fetching and parsing XML sitemaps
- Handling sitemap indexes (nested sitemaps)
- Processing gzip-compressed sitemaps
- Using sitemap metadata (lastmod, changefreq, priority)
- Prioritizing URLs based on sitemap hints

Usage:
    # Basic usage - crawl from sitemap
    python examples/sitemap_crawl_example.py --domain example.com

    # With custom options
    python examples/sitemap_crawl_example.py \
        --domain example.com \
        --max-urls 100 \
        --priority-threshold 0.5

    # Just list URLs from sitemap (no crawling)
    python examples/sitemap_crawl_example.py --domain example.com --list-only

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
    ):
        """
        Initialize the sitemap crawler.

        Args:
            domain: Domain to crawl (e.g., "example.com")
            user_agent: User agent for requests
            max_urls: Maximum URLs to process
            priority_threshold: Only include URLs with priority >= this value
            recent_days: Only include URLs modified within this many days
        """
        self.domain = domain
        self.user_agent = user_agent
        self.max_urls = max_urls
        self.priority_threshold = priority_threshold
        self.recent_days = recent_days
        self.logger = CrawlerLogger("sitemap_crawler")

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

    async def crawl_urls(self, urls: list[SitemapURL]) -> dict[str, Any]:
        """
        Crawl the URLs and return results.

        This is a simplified crawl - in production you would:
        - Use the full Crawler class
        - Apply rate limiting
        - Store results properly

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

                    stats["results"].append({
                        "url": url.loc,
                        "status": response.status_code,
                        "content_length": len(response.content),
                        "priority": url.priority,
                        "lastmod": url.lastmod.isoformat() if url.lastmod else None,
                    })

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

    if "urls" in results:
        print(f"\nTop URLs (by priority):")
        for i, url in enumerate(results["urls"][:10], 1):
            priority = url.get("priority", "N/A")
            lastmod = url.get("lastmod", "N/A")
            print(f"  {i}. {url['loc']}")
            print(f"     Priority: {priority}, Last Modified: {lastmod}")

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

    """)

    crawler = SitemapCrawler(
        domain=args.domain,
        max_urls=args.max_urls,
        priority_threshold=args.priority_threshold,
        recent_days=args.recent_days,
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
