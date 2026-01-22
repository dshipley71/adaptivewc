"""
Diagnostic script to inspect data in the Adaptive Web Crawler.

Provides comprehensive inspection of URL frontier, robots.txt cache,
page structures, extraction strategies, and crawl statistics stored in Redis.

Location: adaptivewc/scripts/inspect_index.py

Usage:
    python scripts/inspect_index.py                          # Show stats and list domains
    python scripts/inspect_index.py --stats                  # Show detailed statistics
    python scripts/inspect_index.py --domains                # List all tracked domains
    python scripts/inspect_index.py --queue example.com      # Show queue for a domain
    python scripts/inspect_index.py --seen example.com       # Show seen URLs for a domain
    python scripts/inspect_index.py --robots example.com     # Show cached robots.txt
    python scripts/inspect_index.py --search "keyword"       # Search queued URLs
    python scripts/inspect_index.py --structures             # List tracked page structures
    python scripts/inspect_index.py --structure example.com  # Show structure for domain
    python scripts/inspect_index.py --strategy example.com   # Show extraction strategy
    python scripts/inspect_index.py --export output.json     # Export full state
    python scripts/inspect_index.py --clear                  # Clear all data (with confirmation)
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import redis.asyncio as redis

# Add the crawler package to path (parent of scripts directory)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from crawler.config import CrawlerSettings, load_config
from crawler.storage.url_store import URLStore, URLEntry
from crawler.storage.robots_cache import RobotsCache
from crawler.storage.structure_store import StructureStore
from crawler.models import PageStructure, ExtractionStrategy


class StructureDescriptionGenerator:
    """Generates human-readable text descriptions of page structures."""

    @staticmethod
    def generate(structure: PageStructure) -> str:
        """
        Generate a semantic text description of a page structure.

        Args:
            structure: PageStructure to describe.

        Returns:
            Human-readable text description.
        """
        lines = []

        # Header
        lines.append(f"Website Structure Analysis for {structure.domain}")
        lines.append(f"Page Type: {structure.page_type}")
        lines.append(f"URL Pattern: {structure.url_pattern or 'N/A'}")
        lines.append(f"Analyzed: {structure.captured_at}")
        lines.append(f"Version: {structure.version}")
        lines.append("")

        # DOM Structure Summary
        tag_counts = structure.tag_hierarchy.get("tag_counts", {}) if structure.tag_hierarchy else {}
        if tag_counts:
            total_elements = sum(tag_counts.values())
            lines.append(f"DOM Structure: {total_elements} total elements, {len(tag_counts)} unique tag types")

            # Identify page characteristics
            characteristics = []
            if tag_counts.get("article", 0) > 0:
                characteristics.append("article-based content")
            if tag_counts.get("nav", 0) > 0:
                characteristics.append(f"{tag_counts['nav']} navigation regions")
            if tag_counts.get("form", 0) > 0:
                characteristics.append(f"{tag_counts['form']} form(s)")
            if tag_counts.get("table", 0) > 0:
                characteristics.append(f"{tag_counts['table']} data table(s)")
            if tag_counts.get("iframe", 0) > 0:
                characteristics.append(f"{tag_counts['iframe']} embedded iframe(s)")
            if tag_counts.get("video", 0) > 0 or tag_counts.get("audio", 0) > 0:
                characteristics.append("multimedia content")

            if characteristics:
                lines.append(f"Page Characteristics: {', '.join(characteristics)}")
            lines.append("")

        # Semantic Landmarks
        if structure.semantic_landmarks:
            lines.append(f"Semantic Landmarks ({len(structure.semantic_landmarks)}):")
            for landmark, selector in structure.semantic_landmarks.items():
                lines.append(f"  - {landmark}: {selector}")
            lines.append("")

        # Content Regions
        if structure.content_regions:
            lines.append(f"Content Regions ({len(structure.content_regions)}):")
            for region in structure.content_regions:
                conf_str = f" (confidence: {region.confidence:.0%})" if region.confidence else ""
                lines.append(f"  - {region.name}: {region.primary_selector}{conf_str}")
                if region.fallback_selectors:
                    lines.append(f"    Fallbacks: {', '.join(region.fallback_selectors)}")
            lines.append("")

        # Navigation
        if structure.navigation_selectors:
            lines.append(f"Navigation Elements ({len(structure.navigation_selectors)}):")
            for sel in structure.navigation_selectors[:5]:
                lines.append(f"  - {sel}")
            if len(structure.navigation_selectors) > 5:
                lines.append(f"  ... and {len(structure.navigation_selectors) - 5} more")
            lines.append("")

        # Pagination
        if structure.pagination_pattern:
            lines.append("Pagination Detected:")
            if structure.pagination_pattern.next_selector:
                lines.append(f"  Next: {structure.pagination_pattern.next_selector}")
            if structure.pagination_pattern.prev_selector:
                lines.append(f"  Previous: {structure.pagination_pattern.prev_selector}")
            if structure.pagination_pattern.pattern:
                lines.append(f"  URL Pattern: {structure.pagination_pattern.pattern}")
            lines.append("")

        # Iframes
        if structure.iframe_locations:
            lines.append(f"Embedded Iframes ({len(structure.iframe_locations)}):")
            for iframe in structure.iframe_locations:
                dynamic = " [dynamic]" if iframe.is_dynamic else ""
                lines.append(f"  - {iframe.selector} ({iframe.position}){dynamic}")
                lines.append(f"    Source: {iframe.src_pattern}")
            lines.append("")

        # CSS Classes Summary
        if structure.css_class_map:
            lines.append(f"CSS Classes: {len(structure.css_class_map)} unique classes detected")
            # Show top classes by usage
            top_classes = sorted(structure.css_class_map.items(), key=lambda x: -x[1])[:5]
            if top_classes:
                lines.append("  Most used: " + ", ".join(f".{cls}({count})" for cls, count in top_classes))
            lines.append("")

        # Script Signatures
        if structure.script_signatures:
            lines.append(f"JavaScript Libraries/Scripts ({len(structure.script_signatures)}):")
            for sig in structure.script_signatures[:10]:
                lines.append(f"  - {sig}")
            lines.append("")

        # Content Hash
        lines.append(f"Content Hash: {structure.content_hash}")

        return "\n".join(lines)


class StrategyDescriptionGenerator:
    """Generates human-readable descriptions of extraction strategies."""

    @staticmethod
    def generate(strategy: ExtractionStrategy) -> str:
        """
        Generate a human-readable description of an extraction strategy.

        Args:
            strategy: ExtractionStrategy to describe.

        Returns:
            Human-readable text description.
        """
        lines = []

        # Header
        lines.append(f"Extraction Strategy for {strategy.domain}")
        lines.append(f"Page Type: {strategy.page_type}")
        lines.append(f"Version: {strategy.version}")
        lines.append(f"Learned: {strategy.learned_at}")
        lines.append(f"Source: {strategy.learning_source}")
        lines.append("")

        # Selector Rules
        lines.append("EXTRACTION SELECTORS:")
        lines.append("-" * 40)

        if strategy.title:
            conf = f" ({strategy.title.confidence:.0%})" if strategy.title.confidence else ""
            lines.append(f"Title{conf}:")
            lines.append(f"  Primary: {strategy.title.primary}")
            if strategy.title.fallbacks:
                lines.append(f"  Fallbacks: {', '.join(strategy.title.fallbacks)}")
            lines.append(f"  Method: {strategy.title.extraction_method}")
            lines.append("")

        if strategy.content:
            conf = f" ({strategy.content.confidence:.0%})" if strategy.content.confidence else ""
            lines.append(f"Content{conf}:")
            lines.append(f"  Primary: {strategy.content.primary}")
            if strategy.content.fallbacks:
                lines.append(f"  Fallbacks: {', '.join(strategy.content.fallbacks)}")
            lines.append(f"  Method: {strategy.content.extraction_method}")
            lines.append("")

        if strategy.images:
            conf = f" ({strategy.images.confidence:.0%})" if strategy.images.confidence else ""
            lines.append(f"Images{conf}:")
            lines.append(f"  Primary: {strategy.images.primary}")
            lines.append("")

        if strategy.links:
            conf = f" ({strategy.links.confidence:.0%})" if strategy.links.confidence else ""
            lines.append(f"Links{conf}:")
            lines.append(f"  Primary: {strategy.links.primary}")
            lines.append("")

        if strategy.metadata:
            lines.append("Metadata Fields:")
            for key, rule in strategy.metadata.items():
                conf = f" ({rule.confidence:.0%})" if rule.confidence else ""
                lines.append(f"  {key}{conf}: {rule.primary}")
            lines.append("")

        # Settings
        lines.append("EXTRACTION SETTINGS:")
        lines.append("-" * 40)
        lines.append(f"Required Fields: {', '.join(strategy.required_fields)}")
        lines.append(f"Min Content Length: {strategy.min_content_length} chars")

        if strategy.wait_for_selectors:
            lines.append(f"Wait-for Selectors: {', '.join(strategy.wait_for_selectors)}")

        if strategy.confidence_scores:
            lines.append("")
            lines.append("Confidence Scores:")
            for field, score in sorted(strategy.confidence_scores.items()):
                bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                lines.append(f"  {field}: {bar} {score:.0%}")

        return "\n".join(lines)


class CrawlerInspector:
    """Inspector for Adaptive Web Crawler Redis data."""

    def __init__(self, redis_client: redis.Redis):
        """Initialize the inspector with Redis client."""
        self.redis = redis_client
        self.url_store = URLStore(redis_client)
        self.robots_cache = RobotsCache(redis_client)
        self.structure_store = StructureStore(redis_client)

    async def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            await self.redis.ping()
            return True
        except redis.RedisError:
            return False

    async def show_stats(self) -> dict:
        """Show comprehensive crawler statistics."""
        stats = await self.url_store.get_stats()
        robots_stats = await self.robots_cache.get_stats()
        structure_stats = await self.structure_store.get_stats()

        print(f"\n{'='*60}")
        print("CRAWLER STATISTICS")
        print(f"{'='*60}\n")

        print("  URL Frontier:")
        print(f"    Total URLs Added:    {stats.get('total_added', 0):,}")
        print(f"    Total URLs Popped:   {stats.get('total_popped', 0):,}")
        print(f"    Total Failed:        {stats.get('total_failed', 0):,}")
        print(f"    Current Queue Size:  {stats.get('queue_size', 0):,}")
        print(f"    Active Domains:      {stats.get('active_domains', 0)}")

        print("\n  Robots.txt Cache:")
        print(f"    Cached Domains:      {robots_stats.get('cached_domains', 'N/A')}")
        print(f"    TTL (seconds):       {robots_stats.get('ttl_seconds', 'N/A')}")

        print("\n  Page Structure Store:")
        print(f"    Tracked Domains:     {structure_stats.get('tracked_domains', 0)}")
        print(f"    Tracked Page Types:  {structure_stats.get('tracked_page_types', 0)}")
        print(f"    Total Structures:    {structure_stats.get('total_structures', 0)}")
        print(f"    Total Strategies:    {structure_stats.get('total_strategies', 0)}")
        print(f"    TTL (seconds):       {structure_stats.get('ttl_seconds', 'N/A')}")

        # Per-domain queue sizes
        if stats.get('domains'):
            print("\n  Queue Sizes by Domain:")
            for domain in stats['domains'][:20]:
                queue_size = await self.url_store.get_queue_size(domain)
                print(f"    - {domain}: {queue_size:,} URLs")
            if len(stats['domains']) > 20:
                print(f"    ... and {len(stats['domains']) - 20} more domains")

        return {**stats, "robots_cache": robots_stats, "structure_store": structure_stats}

    async def list_domains(self, limit: int = 100) -> list[str]:
        """List all tracked domains."""
        domains = await self.redis.smembers(URLStore.DOMAINS_KEY)

        print(f"\n{'='*60}")
        print(f"TRACKED DOMAINS ({len(domains)} total)")
        print(f"{'='*60}\n")

        domain_list = []
        for domain in sorted(domains)[:limit]:
            domain_str = domain.decode() if isinstance(domain, bytes) else domain
            domain_list.append(domain_str)
            queue_size = await self.url_store.get_queue_size(domain_str)
            seen_count = await self._get_seen_count(domain_str)
            status = "+" if queue_size > 0 else "o"
            print(f"  {status} {domain_str}")
            print(f"      Queue: {queue_size:,} | Seen: {seen_count:,}")

        if len(domains) > limit:
            print(f"\n  ... and {len(domains) - limit} more domains")

        return domain_list

    async def _get_seen_count(self, domain: str) -> int:
        """Get count of seen URLs for a domain."""
        seen_key = self.url_store._seen_key(domain)
        return await self.redis.scard(seen_key)

    async def show_queue(self, domain: str, limit: int = 20) -> list[URLEntry]:
        """Show queued URLs for a domain."""
        queue_key = self.url_store._queue_key(domain)
        queue_data = await self.redis.zrange(queue_key, 0, limit - 1, withscores=True)

        print(f"\n{'='*60}")
        print(f"URL QUEUE: {domain}")
        print(f"{'='*60}\n")

        entries = []
        for data_str, score in queue_data:
            data = json.loads(data_str)
            entry = URLEntry.from_dict(data)
            entries.append(entry)

            priority_display = f"{entry.priority:.2f}"
            depth_display = f"D{entry.depth}"
            retries_display = f"R{entry.retries}" if entry.retries > 0 else ""

            print(f"  [URL] {entry.url[:70]}{'...' if len(entry.url) > 70 else ''}")
            print(f"      Priority: {priority_display} | Depth: {depth_display} {retries_display}")
            if entry.parent_url:
                print(f"      Parent: {entry.parent_url[:50]}...")
            if entry.last_error:
                print(f"      ! Error: {entry.last_error}")
            print()

        total = await self.url_store.get_queue_size(domain)
        if total > limit:
            print(f"  ... showing {limit} of {total} queued URLs")

        return entries

    async def show_seen_urls(self, domain: str, limit: int = 50) -> list[str]:
        """Show seen URLs for a domain."""
        seen_key = self.url_store._seen_key(domain)
        seen_urls = await self.redis.srandmember(seen_key, limit)

        print(f"\n{'='*60}")
        print(f"SEEN URLs: {domain}")
        print(f"{'='*60}\n")

        urls = []
        for url in seen_urls or []:
            url_str = url.decode() if isinstance(url, bytes) else url
            urls.append(url_str)
            print(f"  + {url_str[:80]}{'...' if len(url_str) > 80 else ''}")

        total = await self._get_seen_count(domain)
        print(f"\n  Showing {len(urls)} of {total} seen URLs (random sample)")

        return urls

    async def show_robots(self, domain: str) -> dict | None:
        """Show cached robots.txt for a domain."""
        robots = await self.robots_cache.get(domain)

        print(f"\n{'='*60}")
        print(f"ROBOTS.TXT CACHE: {domain}")
        print(f"{'='*60}\n")

        if robots is None:
            print(f"  X No cached robots.txt for {domain}")
            return None

        print(f"  Fetch Status: {robots.fetch_status}")

        if robots.sitemaps:
            print(f"\n  Sitemaps:")
            for sitemap in robots.sitemaps:
                print(f"    - {sitemap}")

        if robots.groups:
            print(f"\n  User-Agent Groups ({len(robots.groups)}):")
            for i, group in enumerate(robots.groups, 1):
                print(f"\n    Group {i}:")
                print(f"      User-Agents: {', '.join(group.user_agents)}")
                if group.crawl_delay is not None:
                    print(f"      Crawl-Delay: {group.crawl_delay}s")
                if group.rules:
                    print(f"      Rules ({len(group.rules)}):")
                    for rule in group.rules[:10]:
                        allow_str = "Allow" if rule.allow else "Disallow"
                        print(f"        {allow_str}: {rule.pattern}")
                    if len(group.rules) > 10:
                        print(f"        ... and {len(group.rules) - 10} more rules")

        return robots.to_dict() if robots else None

    async def search_urls(self, query: str, limit: int = 20) -> list[dict]:
        """Search queued URLs across all domains."""
        query_lower = query.lower()
        matches = []

        print(f"\n{'='*60}")
        print(f"SEARCH: '{query}'")
        print(f"{'='*60}\n")

        domains = await self.redis.smembers(URLStore.DOMAINS_KEY)

        for domain in domains:
            domain_str = domain.decode() if isinstance(domain, bytes) else domain
            queue_key = self.url_store._queue_key(domain_str)
            queue_data = await self.redis.zrange(queue_key, 0, -1)

            for data_str in queue_data:
                data = json.loads(data_str)
                url = data.get("url", "")
                if query_lower in url.lower():
                    matches.append(data)
                    if len(matches) >= limit:
                        break

            if len(matches) >= limit:
                break

        for match in matches:
            url = match.get("url", "")
            domain = match.get("domain", "unknown")
            priority = match.get("priority", 0)

            # Highlight the match
            idx = url.lower().find(query_lower)
            if idx >= 0:
                highlighted = f"{url[:idx]}[{url[idx:idx+len(query)]}]{url[idx+len(query):]}"
            else:
                highlighted = url

            print(f"  [URL] {highlighted[:80]}")
            print(f"      Domain: {domain} | Priority: {priority:.2f}")
            print()

        print(f"  Found {len(matches)} matches (limit: {limit})")

        return matches

    async def export_state(self, output_path: str) -> dict:
        """Export full crawler state to JSON."""
        print(f"\n{'='*60}")
        print("EXPORTING CRAWLER STATE")
        print(f"{'='*60}\n")

        state = await self.url_store.export_state()

        # Add robots cache
        robots_keys = await self.redis.keys(f"{RobotsCache.KEY_PREFIX}*")
        state["robots_cache"] = {}
        for key in robots_keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            domain = key_str.replace(RobotsCache.KEY_PREFIX, "")
            robots = await self.robots_cache.get(domain)
            if robots:
                state["robots_cache"][domain] = robots.to_dict()

        # Add structures and strategies
        state["structures"] = {}
        state["strategies"] = {}
        domains = await self.structure_store.list_domains()
        for domain, page_type in domains:
            structure = await self.structure_store.get_structure(domain, page_type)
            strategy = await self.structure_store.get_strategy(domain, page_type)
            key = f"{domain}:{page_type}"
            if structure:
                state["structures"][key] = self._structure_to_export_dict(structure)
            if strategy:
                state["strategies"][key] = self._strategy_to_export_dict(strategy)

        state["exported_at"] = datetime.utcnow().isoformat()

        with open(output_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

        print(f"  + Exported to: {output_path}")
        print(f"  - Queues: {len(state.get('queues', {}))}")
        print(f"  - Seen sets: {len(state.get('seen', {}))}")
        print(f"  - Robots cache: {len(state.get('robots_cache', {}))}")
        print(f"  - Structures: {len(state.get('structures', {}))}")
        print(f"  - Strategies: {len(state.get('strategies', {}))}")

        return state

    def _structure_to_export_dict(self, structure: PageStructure) -> dict:
        """Convert PageStructure to export dict with all fields."""
        return {
            "domain": structure.domain,
            "page_type": structure.page_type,
            "url_pattern": structure.url_pattern,
            "captured_at": structure.captured_at.isoformat() if structure.captured_at else None,
            "version": structure.version,
            "content_hash": structure.content_hash,
            "tag_hierarchy": structure.tag_hierarchy,
            "css_class_map": structure.css_class_map,
            "id_attributes": list(structure.id_attributes) if structure.id_attributes else [],
            "semantic_landmarks": structure.semantic_landmarks,
            "content_regions": [
                {
                    "name": r.name,
                    "primary_selector": r.primary_selector,
                    "fallback_selectors": r.fallback_selectors,
                    "content_type": r.content_type,
                    "confidence": r.confidence,
                }
                for r in (structure.content_regions or [])
            ],
            "navigation_selectors": structure.navigation_selectors,
            "pagination_pattern": {
                "next_selector": structure.pagination_pattern.next_selector,
                "prev_selector": structure.pagination_pattern.prev_selector,
                "pattern": structure.pagination_pattern.pattern,
            } if structure.pagination_pattern else None,
            "iframe_locations": [
                {
                    "selector": i.selector,
                    "src_pattern": i.src_pattern,
                    "position": i.position,
                    "is_dynamic": i.is_dynamic,
                }
                for i in (structure.iframe_locations or [])
            ],
            "script_signatures": structure.script_signatures,
        }

    def _strategy_to_export_dict(self, strategy: ExtractionStrategy) -> dict:
        """Convert ExtractionStrategy to export dict."""
        def rule_to_dict(rule):
            if rule is None:
                return None
            return {
                "primary": rule.primary,
                "fallbacks": rule.fallbacks,
                "extraction_method": rule.extraction_method,
                "confidence": rule.confidence,
            }

        return {
            "domain": strategy.domain,
            "page_type": strategy.page_type,
            "version": strategy.version,
            "learned_at": strategy.learned_at.isoformat() if strategy.learned_at else None,
            "learning_source": strategy.learning_source,
            "title": rule_to_dict(strategy.title),
            "content": rule_to_dict(strategy.content),
            "images": rule_to_dict(strategy.images),
            "links": rule_to_dict(strategy.links),
            "metadata": {k: rule_to_dict(v) for k, v in (strategy.metadata or {}).items()},
            "required_fields": strategy.required_fields,
            "min_content_length": strategy.min_content_length,
            "wait_for_selectors": strategy.wait_for_selectors,
            "confidence_scores": strategy.confidence_scores,
        }

    async def clear_all(self, confirm: bool = False) -> bool:
        """Clear all crawler data."""
        if not confirm:
            print("\n!!  This will delete ALL crawler data!")
            response = input("Type 'yes' to confirm: ")
            if response.lower() != "yes":
                print("Aborted.")
                return False

        print(f"\n{'='*60}")
        print("CLEARING ALL DATA")
        print(f"{'='*60}\n")

        # Clear URL store
        await self.url_store.clear()
        print("  + URL frontier cleared")

        # Clear robots cache
        cleared = await self.robots_cache.clear()
        print(f"  + Robots cache cleared ({cleared} entries)")

        # Clear structure store
        cleared = await self.structure_store.clear()
        print(f"  + Structure store cleared ({cleared} entries)")

        return True

    async def peek_next(self, domain: str) -> URLEntry | None:
        """Peek at the next URL to be crawled for a domain."""
        entry = await self.url_store.peek(domain)

        print(f"\n{'='*60}")
        print(f"NEXT URL FOR: {domain}")
        print(f"{'='*60}\n")

        if entry is None:
            print(f"  X No URLs in queue for {domain}")
            return None

        print(f"  URL: {entry.url}")
        print(f"  Priority: {entry.priority:.2f}")
        print(f"  Depth: {entry.depth}")
        print(f"  Discovered: {entry.discovered_at}")
        if entry.parent_url:
            print(f"  Parent: {entry.parent_url}")
        if entry.retries > 0:
            print(f"  Retries: {entry.retries}")
            if entry.last_error:
                print(f"  Last Error: {entry.last_error}")

        return entry

    # =========================================================================
    # Structure Store Methods
    # =========================================================================

    async def _get_default_page_type(self, domain: str) -> str | None:
        """Get the first available page_type for a domain."""
        all_domains = await self.structure_store.list_domains()
        for d, pt in all_domains:
            if d == domain:
                return pt
        return None

    async def list_structures(self, limit: int = 50) -> list[tuple[str, str]]:
        """List all tracked page structures."""
        domains = await self.structure_store.list_domains()

        print(f"\n{'='*60}")
        print(f"TRACKED PAGE STRUCTURES ({len(domains)} total)")
        print(f"{'='*60}\n")

        for domain, page_type in domains[:limit]:
            structure = await self.structure_store.get_structure(domain, page_type)
            strategy = await self.structure_store.get_strategy(domain, page_type)
            if structure:
                print(f"  [STRUCT] {domain} [{page_type}]")
                print(f"      Version: {structure.version} | Captured: {structure.captured_at}")
                print(f"      Content Hash: {structure.content_hash[:32]}..." if structure.content_hash else "      Content Hash: N/A")

                # Show structure summary
                tag_counts = structure.tag_hierarchy.get("tag_counts", {}) if structure.tag_hierarchy else {}
                if tag_counts:
                    total_elements = sum(tag_counts.values())
                    print(f"      DOM Elements: {total_elements} | Unique Tags: {len(tag_counts)}")

                if structure.content_regions:
                    print(f"      Content Regions: {len(structure.content_regions)}")

                if strategy:
                    print(f"      Strategy: v{strategy.version} ({strategy.learning_source})")
            else:
                print(f"  [STRUCT] {domain} [{page_type}] (no data)")
            print()

        if len(domains) > limit:
            print(f"  ... and {len(domains) - limit} more")

        return domains[:limit]

    async def show_structure(self, domain: str, page_type: str | None = None) -> dict | None:
        """Show detailed structure for a domain/page_type."""
        # Auto-detect page_type if not specified
        if page_type is None or page_type == "unknown":
            page_type = await self._get_default_page_type(domain)
            if page_type is None:
                print(f"\n{'='*60}")
                print(f"PAGE STRUCTURE: {domain}")
                print(f"{'='*60}\n")
                print(f"  X No structures found for domain: {domain}")
                return None

        structure = await self.structure_store.get_structure(domain, page_type)

        print(f"\n{'='*60}")
        print(f"PAGE STRUCTURE: {domain} [{page_type}]")
        print(f"{'='*60}\n")

        if structure is None:
            # Try to find any page type for this domain
            all_domains = await self.structure_store.list_domains()
            matching = [(d, pt) for d, pt in all_domains if d == domain]

            if matching:
                print(f"  X No structure for page_type '{page_type}'")
                print(f"  Available page types for {domain}:")
                for d, pt in matching:
                    print(f"    - {pt}")
            else:
                print(f"  X No structures found for domain: {domain}")
            return None

        # Generate and print text description
        description = StructureDescriptionGenerator.generate(structure)
        print(description)

        print(f"\n{'='*60}")
        print("RAW STRUCTURE DATA")
        print(f"{'='*60}\n")

        print(f"  Version: {structure.version}")
        print(f"  Captured: {structure.captured_at}")
        print(f"  URL Pattern: {structure.url_pattern}")
        print(f"  Content Hash: {structure.content_hash}")

        # Tag hierarchy summary
        tag_counts = structure.tag_hierarchy.get("tag_counts", {}) if structure.tag_hierarchy else {}
        if tag_counts:
            print(f"\n  Tag Counts ({len(tag_counts)} unique tags):")
            top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:15]
            for tag, count in top_tags:
                print(f"    <{tag}>: {count}")
            if len(tag_counts) > 15:
                print(f"    ... and {len(tag_counts) - 15} more tags")

        # Depth distribution
        depth_dist = structure.tag_hierarchy.get("depth_distribution", {}) if structure.tag_hierarchy else {}
        if depth_dist:
            print(f"\n  DOM Depth Distribution:")
            for depth, count in sorted(depth_dist.items(), key=lambda x: int(x[0]))[:10]:
                bar = "█" * min(count // 10, 30)
                print(f"    Level {depth}: {bar} ({count})")

        # CSS classes
        if structure.css_class_map:
            print(f"\n  CSS Classes ({len(structure.css_class_map)} unique):")
            top_classes = sorted(structure.css_class_map.items(), key=lambda x: -x[1])[:15]
            for cls, count in top_classes:
                print(f"    .{cls}: {count}")
            if len(structure.css_class_map) > 15:
                print(f"    ... and {len(structure.css_class_map) - 15} more classes")

        # IDs
        if structure.id_attributes:
            print(f"\n  ID Attributes ({len(structure.id_attributes)}):")
            for id_attr in list(structure.id_attributes)[:20]:
                print(f"    #{id_attr}")
            if len(structure.id_attributes) > 20:
                print(f"    ... and {len(structure.id_attributes) - 20} more IDs")

        # Semantic landmarks
        if structure.semantic_landmarks:
            print(f"\n  Semantic Landmarks:")
            for landmark, selector in structure.semantic_landmarks.items():
                print(f"    {landmark}: {selector}")

        # Content regions
        if structure.content_regions:
            print(f"\n  Content Regions:")
            for region in structure.content_regions:
                print(f"    {region.name}: {region.primary_selector} (confidence: {region.confidence:.2f})")
                if region.fallback_selectors:
                    print(f"      Fallbacks: {region.fallback_selectors}")

        # Navigation
        if structure.navigation_selectors:
            print(f"\n  Navigation Selectors:")
            for selector in structure.navigation_selectors[:10]:
                print(f"    - {selector}")
            if len(structure.navigation_selectors) > 10:
                print(f"    ... and {len(structure.navigation_selectors) - 10} more")

        # Pagination
        if structure.pagination_pattern:
            print(f"\n  Pagination:")
            if structure.pagination_pattern.next_selector:
                print(f"    Next: {structure.pagination_pattern.next_selector}")
            if structure.pagination_pattern.prev_selector:
                print(f"    Previous: {structure.pagination_pattern.prev_selector}")
            if structure.pagination_pattern.pattern:
                print(f"    Pattern: {structure.pagination_pattern.pattern}")

        # Iframes
        if structure.iframe_locations:
            print(f"\n  Iframes ({len(structure.iframe_locations)}):")
            for iframe in structure.iframe_locations:
                print(f"    - {iframe.selector} [{iframe.position}]")
                print(f"      Pattern: {iframe.src_pattern}")
                if iframe.is_dynamic:
                    print(f"      Dynamic: Yes")

        # Scripts
        if structure.script_signatures:
            print(f"\n  Script Signatures:")
            for sig in structure.script_signatures[:15]:
                print(f"    - {sig}")
            if len(structure.script_signatures) > 15:
                print(f"    ... and {len(structure.script_signatures) - 15} more")

        return self._structure_to_export_dict(structure)

    async def show_strategy(self, domain: str, page_type: str | None = None) -> dict | None:
        """Show extraction strategy for a domain/page_type."""
        # Auto-detect page_type if not specified
        if page_type is None or page_type == "unknown":
            page_type = await self._get_default_page_type(domain)
            if page_type is None:
                print(f"\n{'='*60}")
                print(f"EXTRACTION STRATEGY: {domain}")
                print(f"{'='*60}\n")
                print(f"  X No strategies found for domain: {domain}")
                return None

        # Debug: Check what keys exist for strategies
        strategy_keys = await self.redis.keys(b"crawler:strategy:*")
        print(f"\n  [DEBUG] Strategy keys in Redis: {len(strategy_keys)}")
        for key in strategy_keys[:5]:
            key_str = key.decode() if isinstance(key, bytes) else key
            print(f"    - {key_str}")

        # Check raw data
        strategy_key = f"crawler:strategy:{domain}:{page_type}"
        raw_data = await self.redis.get(strategy_key)
        print(f"  [DEBUG] Looking for key: {strategy_key}")
        print(f"  [DEBUG] Raw data found: {raw_data is not None}")
        if raw_data:
            print(f"  [DEBUG] Raw data type: {type(raw_data)}")
            print(f"  [DEBUG] Raw data preview: {str(raw_data)[:200]}...")

        strategy = await self.structure_store.get_strategy(domain, page_type)

        print(f"\n{'='*60}")
        print(f"EXTRACTION STRATEGY: {domain} [{page_type}]")
        print(f"{'='*60}\n")

        if strategy is None:
            # Try to find any page type for this domain
            all_domains = await self.structure_store.list_domains()
            matching = [(d, pt) for d, pt in all_domains if d == domain]

            if matching:
                print(f"  X No strategy for page_type '{page_type}'")
                print(f"  Available page types for {domain}:")
                for d, pt in matching:
                    print(f"    - {pt}")
            else:
                print(f"  X No strategies found for domain: {domain}")
            return None

        # Generate and print description
        description = StrategyDescriptionGenerator.generate(strategy)
        print(description)

        print(f"\n{'='*60}")
        print("RAW STRATEGY DATA (JSON)")
        print(f"{'='*60}\n")

        export_dict = self._strategy_to_export_dict(strategy)
        print(json.dumps(export_dict, indent=2, default=str))

        return export_dict

    async def show_structure_history(self, domain: str, page_type: str | None = None, limit: int = 10) -> list:
        """Show version history for a structure."""
        # Auto-detect page_type if not specified
        if page_type is None or page_type == "unknown":
            page_type = await self._get_default_page_type(domain)
            if page_type is None:
                print(f"\n  X No structures found for domain: {domain}")
                return []

        history = await self.structure_store.get_history(domain, page_type, limit)

        print(f"\n{'='*60}")
        print(f"STRUCTURE HISTORY: {domain} [{page_type}]")
        print(f"{'='*60}\n")

        if not history:
            print(f"  X No history found for {domain} [{page_type}]")
            return []

        for structure in history:
            tag_counts = structure.tag_hierarchy.get("tag_counts", {}) if structure.tag_hierarchy else {}
            tag_count = len(tag_counts)
            class_count = len(structure.css_class_map) if structure.css_class_map else 0

            print(f"  Version {structure.version}")
            print(f"    Captured: {structure.captured_at}")
            print(f"    Hash: {structure.content_hash[:16]}..." if structure.content_hash else "    Hash: N/A")
            print(f"    Tags: {tag_count} | Classes: {class_count}")
            print()

        return [self._structure_to_export_dict(s) for s in history]

    async def compare_structures(
        self,
        domain: str,
        page_type: str | None,
        version1: int,
        version2: int
    ) -> dict | None:
        """Compare two structure versions."""
        # Auto-detect page_type if not specified
        if page_type is None or page_type == "unknown":
            page_type = await self._get_default_page_type(domain)
            if page_type is None:
                print(f"\n  X No structures found for domain: {domain}")
                return None

        struct1 = await self.structure_store.get_version(domain, page_type, version1)
        struct2 = await self.structure_store.get_version(domain, page_type, version2)

        print(f"\n{'='*60}")
        print(f"STRUCTURE COMPARISON: {domain} [{page_type}]")
        print(f"Version {version1} vs Version {version2}")
        print(f"{'='*60}\n")

        if struct1 is None:
            print(f"  X Version {version1} not found")
            return None
        if struct2 is None:
            print(f"  X Version {version2} not found")
            return None

        # Compare tags
        tags1 = set((struct1.tag_hierarchy or {}).get("tag_counts", {}).keys())
        tags2 = set((struct2.tag_hierarchy or {}).get("tag_counts", {}).keys())
        added_tags = tags2 - tags1
        removed_tags = tags1 - tags2

        print(f"  Tag Changes:")
        print(f"    Common: {len(tags1 & tags2)}")
        if added_tags:
            print(f"    Added: {', '.join(list(added_tags)[:10])}")
        if removed_tags:
            print(f"    Removed: {', '.join(list(removed_tags)[:10])}")

        # Compare classes
        classes1 = set((struct1.css_class_map or {}).keys())
        classes2 = set((struct2.css_class_map or {}).keys())
        added_classes = classes2 - classes1
        removed_classes = classes1 - classes2

        print(f"\n  CSS Class Changes:")
        print(f"    Common: {len(classes1 & classes2)}")
        if added_classes:
            print(f"    Added ({len(added_classes)}): {', '.join(list(added_classes)[:10])}")
        if removed_classes:
            print(f"    Removed ({len(removed_classes)}): {', '.join(list(removed_classes)[:10])}")

        # Compare landmarks
        landmarks1 = set((struct1.semantic_landmarks or {}).keys())
        landmarks2 = set((struct2.semantic_landmarks or {}).keys())

        print(f"\n  Landmark Changes:")
        print(f"    V{version1}: {', '.join(landmarks1) or 'none'}")
        print(f"    V{version2}: {', '.join(landmarks2) or 'none'}")

        # Hash comparison
        print(f"\n  Content Hash:")
        print(f"    V{version1}: {struct1.content_hash}")
        print(f"    V{version2}: {struct2.content_hash}")
        print(f"    Match: {'Yes' if struct1.content_hash == struct2.content_hash else 'No'}")

        # Calculate similarity (simple Jaccard)
        all_tags = tags1 | tags2
        all_classes = classes1 | classes2
        tag_sim = len(tags1 & tags2) / len(all_tags) if all_tags else 1.0
        class_sim = len(classes1 & classes2) / len(all_classes) if all_classes else 1.0
        overall_sim = (tag_sim + class_sim) / 2

        print(f"\n  Similarity Scores:")
        print(f"    Tag similarity: {tag_sim:.2%}")
        print(f"    Class similarity: {class_sim:.2%}")
        print(f"    Overall: {overall_sim:.2%}")

        return {
            "version1": version1,
            "version2": version2,
            "tag_similarity": tag_sim,
            "class_similarity": class_sim,
            "overall_similarity": overall_sim,
            "tags_added": list(added_tags),
            "tags_removed": list(removed_tags),
            "classes_added": list(added_classes),
            "classes_removed": list(removed_classes),
        }


async def main():
    parser = argparse.ArgumentParser(
        description="Inspect Adaptive Web Crawler data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                          Show stats and domains
    %(prog)s --stats                  Show detailed statistics
    %(prog)s --domains                List all tracked domains
    %(prog)s --queue example.com      Show queue for domain
    %(prog)s --seen example.com       Show seen URLs for domain
    %(prog)s --robots example.com     Show cached robots.txt
    %(prog)s --search "api"           Search URLs containing "api"
    %(prog)s --next example.com       Peek at next URL for domain
    %(prog)s --structures             List all tracked page structures
    %(prog)s --structure example.com  Show structure for domain (auto-detects page type)
    %(prog)s --structure example.com --page-type article  Show specific page type
    %(prog)s --strategy example.com   Show extraction strategy for domain
    %(prog)s --structure-history example.com  Show structure versions
    %(prog)s --compare example.com 1 2  Compare structure versions
    %(prog)s --export state.json      Export full state to JSON
    %(prog)s --clear                  Clear all data (with confirm)
        """
    )

    parser.add_argument("--stats", action="store_true", help="Show detailed statistics")
    parser.add_argument("--domains", action="store_true", help="List all tracked domains")
    parser.add_argument("--queue", metavar="DOMAIN", help="Show queue for a domain")
    parser.add_argument("--seen", metavar="DOMAIN", help="Show seen URLs for a domain")
    parser.add_argument("--robots", metavar="DOMAIN", help="Show cached robots.txt for domain")
    parser.add_argument("--search", "-s", metavar="QUERY", help="Search queued URLs")
    parser.add_argument("--next", metavar="DOMAIN", help="Peek at next URL for domain")
    parser.add_argument("--export", metavar="FILE", help="Export full state to JSON file")
    parser.add_argument("--clear", action="store_true", help="Clear all crawler data")

    # Structure inspection commands
    parser.add_argument("--structures", action="store_true", help="List all tracked page structures")
    parser.add_argument("--structure", metavar="DOMAIN", help="Show structure for a domain")
    parser.add_argument("--page-type", metavar="TYPE", default=None, help="Page type for structure commands (auto-detected if not specified)")
    parser.add_argument("--structure-history", metavar="DOMAIN", help="Show structure version history")
    parser.add_argument("--compare", nargs=3, metavar=("DOMAIN", "V1", "V2"), help="Compare two structure versions")
    parser.add_argument("--strategy", metavar="DOMAIN", help="Show extraction strategy for domain")

    parser.add_argument("--limit", type=int, default=20, help="Max results to show (default: 20)")
    parser.add_argument("--redis-url", default=None, help="Redis URL (default: from config or redis://localhost:6379/0)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts")

    args = parser.parse_args()

    # Load config and get Redis URL
    try:
        config = load_config()
        redis_url = args.redis_url or config.redis_url
    except Exception:
        redis_url = args.redis_url or "redis://localhost:6379/0"

    # Connect to Redis
    try:
        client = redis.from_url(redis_url, decode_responses=False)
        inspector = CrawlerInspector(client)

        if not await inspector.ping():
            print(f"X Cannot connect to Redis at {redis_url}")
            return 1
    except Exception as e:
        print(f"X Redis connection error: {e}")
        return 1

    try:
        # Execute requested command
        if args.clear:
            await inspector.clear_all(confirm=args.yes)
        elif args.export:
            await inspector.export_state(args.export)
        elif args.queue:
            await inspector.show_queue(args.queue, args.limit)
        elif args.seen:
            await inspector.show_seen_urls(args.seen, args.limit)
        elif args.robots:
            await inspector.show_robots(args.robots)
        elif args.search:
            await inspector.search_urls(args.search, args.limit)
        elif args.next:
            await inspector.peek_next(args.next)
        elif args.structures:
            await inspector.list_structures(args.limit)
        elif args.structure:
            await inspector.show_structure(args.structure, args.page_type)
        elif args.structure_history:
            await inspector.show_structure_history(args.structure_history, args.page_type, args.limit)
        elif args.compare:
            domain, v1, v2 = args.compare
            await inspector.compare_structures(domain, args.page_type, int(v1), int(v2))
        elif args.strategy:
            await inspector.show_strategy(args.strategy, args.page_type)
        elif args.domains:
            await inspector.list_domains(args.limit)
        elif args.stats:
            await inspector.show_stats()
        else:
            # Default: show stats and domains
            await inspector.show_stats()
            print()
            await inspector.list_domains(10)

    finally:
        await client.aclose()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
