#!/usr/bin/env python3
"""
Sports News Website Monitor

A practical example showing how an admin can use the adaptive web crawler
to monitor a sports news website for changes and automatically pull content
when updates are detected.

Use Case:
    An admin monitors ESPN, BBC Sport, or similar sites for breaking news.
    When content changes are detected, the system extracts the new articles
    and can trigger alerts or store the content for further processing.

Usage:
    # Option 1: Start Redis with Docker
    docker run -d -p 6379:6379 redis:7-alpine

    # Option 2: Install Redis locally (Debian/Ubuntu)
    python examples/sports_news_monitor.py --install-redis

    # Run the monitor
    python examples/sports_news_monitor.py

    # Monitor specific URL
    python examples/sports_news_monitor.py --url https://www.espn.com/nfl/

    # Set check interval (seconds)
    python examples/sports_news_monitor.py --interval 300

Requirements:
    - Redis running on localhost:6379
    - pip install -e .
"""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def install_redis_local() -> bool:
    """
    Install Redis locally on Debian/Ubuntu systems.

    This is an alternative to using Docker for environments where
    Docker is not available (e.g., Google Colab, some CI systems).

    Returns:
        True if installation successful, False otherwise.
    """
    print("Installing Redis locally...")
    print("This requires sudo access on Debian/Ubuntu systems.\n")

    commands = [
        # Add Redis GPG key
        (
            "curl -fsSL https://packages.redis.io/gpg | "
            "sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg --yes"
        ),
        # Add Redis repository
        (
            'echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] '
            'https://packages.redis.io/deb $(lsb_release -cs) main" | '
            "sudo tee /etc/apt/sources.list.d/redis.list"
        ),
        # Update package list
        "sudo apt-get update",
        # Install Redis Stack Server
        "sudo apt-get install -y redis-stack-server",
        # Install Python redis client
        "pip install redis",
    ]

    for cmd in commands:
        print(f"Running: {cmd[:60]}...")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"  Warning: {result.stderr[:100] if result.stderr else 'Command failed'}")
        except Exception as e:
            print(f"  Error: {e}")
            return False

    # Start Redis server
    print("\nStarting Redis server...")
    try:
        subprocess.run(
            "sudo systemctl start redis-stack-server || redis-server --daemonize yes",
            shell=True,
            capture_output=True,
        )
    except Exception:
        # Try alternative start method
        try:
            subprocess.Popen(
                ["redis-server", "--daemonize", "yes"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"Could not start Redis: {e}")
            print("Try starting manually: redis-server --daemonize yes")
            return False

    print("\nRedis installation complete!")
    print("Redis should now be running on localhost:6379")
    return True


def check_redis_installed() -> bool:
    """Check if Redis is available."""
    return shutil.which("redis-server") is not None


import httpx
import redis.asyncio as redis

from crawler.adaptive.change_detector import ChangeDetector, ChangeAnalysis
from crawler.adaptive.strategy_learner import StrategyLearner, LearnedStrategy
from crawler.adaptive.structure_analyzer import StructureAnalyzer
from crawler.compliance.robots_parser import RobotsChecker
from crawler.extraction.content_extractor import ContentExtractor
from crawler.models import ExtractionStrategy, PageStructure, ExtractionResult
from crawler.storage.structure_store import StructureStore
from crawler.utils.logging import CrawlerLogger, setup_logging


@dataclass
class ContentChange:
    """Represents a detected content change."""

    url: str
    detected_at: datetime
    change_type: str  # "new_content", "content_updated", "structure_changed"
    similarity_score: float
    extracted_content: ExtractionResult | None = None
    previous_hash: str | None = None
    current_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "detected_at": self.detected_at.isoformat(),
            "change_type": self.change_type,
            "similarity_score": self.similarity_score,
            "previous_hash": self.previous_hash,
            "current_hash": self.current_hash,
        }


@dataclass
class MonitorConfig:
    """Configuration for the monitor."""

    urls: list[str] = field(default_factory=list)
    check_interval: int = 300  # 5 minutes
    redis_url: str = "redis://localhost:6379/0"
    output_dir: str = "./monitor_output"
    user_agent: str = "SportsNewsMonitor/1.0 (Content Change Detection)"
    respect_robots: bool = True
    max_retries: int = 3
    request_timeout: float = 30.0


class SportsNewsMonitor:
    """
    Monitors sports news websites for content changes.

    Features:
    - Learns page structure on first visit
    - Detects structural changes (site redesigns)
    - Detects content changes (new articles)
    - Adapts extraction strategy when structure changes
    - Stores change history for analysis
    """

    def __init__(
        self,
        config: MonitorConfig,
        on_change: Callable[[ContentChange], None] | None = None,
    ):
        """
        Initialize the monitor.

        Args:
            config: Monitor configuration.
            on_change: Callback function when changes detected.
        """
        self.config = config
        self.on_change = on_change
        self.logger = CrawlerLogger("sports_monitor")

        # Components
        self.structure_analyzer = StructureAnalyzer(logger=self.logger)
        self.change_detector = ChangeDetector(logger=self.logger)
        self.strategy_learner = StrategyLearner(logger=self.logger)
        self.content_extractor = ContentExtractor(logger=self.logger)
        self.robots_checker = RobotsChecker(user_agent=config.user_agent)

        # Storage
        self.redis_client: redis.Redis | None = None
        self.structure_store: StructureStore | None = None

        # HTTP client
        self.http_client: httpx.AsyncClient | None = None

        # State
        self._running = False
        self._content_hashes: dict[str, str] = {}  # URL -> content hash
        self._change_history: list[ContentChange] = []

    async def start(self) -> None:
        """Initialize connections and components."""
        self.logger.info("Starting Sports News Monitor")

        # Connect to Redis
        try:
            self.redis_client = redis.from_url(self.config.redis_url)
            await self.redis_client.ping()
            self.structure_store = StructureStore(
                redis_client=self.redis_client,
                logger=self.logger,
            )
            self.logger.info("Connected to Redis")
        except Exception as e:
            self.logger.error("Failed to connect to Redis", error=str(e))
            raise

        # Initialize HTTP client
        self.http_client = httpx.AsyncClient(
            headers={"User-Agent": self.config.user_agent},
            timeout=self.config.request_timeout,
            follow_redirects=True,
        )

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Load previous content hashes from Redis
        await self._load_content_hashes()

    async def stop(self) -> None:
        """Cleanup resources."""
        self._running = False

        if self.http_client:
            await self.http_client.aclose()

        if self.redis_client:
            await self.redis_client.aclose()

        self.logger.info("Monitor stopped")

    async def _load_content_hashes(self) -> None:
        """Load content hashes from Redis."""
        if not self.redis_client:
            return

        for url in self.config.urls:
            key = f"content_hash:{self._url_to_key(url)}"
            hash_value = await self.redis_client.get(key)
            if hash_value:
                self._content_hashes[url] = hash_value.decode()

    async def _save_content_hash(self, url: str, content_hash: str) -> None:
        """Save content hash to Redis."""
        if not self.redis_client:
            return

        key = f"content_hash:{self._url_to_key(url)}"
        await self.redis_client.set(key, content_hash)
        self._content_hashes[url] = content_hash

    def _url_to_key(self, url: str) -> str:
        """Convert URL to a safe Redis key."""
        return hashlib.md5(url.encode()).hexdigest()

    def _compute_structure_fingerprint(self, structure: PageStructure) -> str:
        """
        Compute a fingerprint hash based on structural elements only.

        This ignores dynamic content like timestamps, scores, and text content.
        Only structural elements are considered:
        - Tag hierarchy (HTML element counts)
        - CSS class names
        - Element IDs
        - Semantic landmarks

        This ensures that dynamic content changes (times, dates, scores)
        don't trigger false positive change detections.
        """
        fingerprint_parts = []

        # Tag hierarchy - stable structural indicator
        if structure.tag_hierarchy:
            tag_counts = structure.tag_hierarchy.get("tag_counts", {})
            # Sort for consistent ordering
            sorted_tags = sorted(tag_counts.items())
            fingerprint_parts.append(f"tags:{sorted_tags}")

        # CSS classes - structural indicator (ignore dynamic class names)
        if structure.css_class_map:
            # Use top 30 most frequent classes for stability
            sorted_classes = sorted(
                structure.css_class_map.items(),
                key=lambda x: x[1],
                reverse=True
            )[:30]
            class_names = [name for name, _ in sorted_classes]
            fingerprint_parts.append(f"classes:{sorted(class_names)}")

        # Element IDs - structural indicator
        if structure.id_attributes:
            sorted_ids = sorted(structure.id_attributes)
            fingerprint_parts.append(f"ids:{sorted_ids}")

        # Semantic landmarks - structural indicator
        if structure.semantic_landmarks:
            sorted_landmarks = sorted(structure.semantic_landmarks.keys())
            fingerprint_parts.append(f"landmarks:{sorted_landmarks}")

        # Navigation selectors - structural indicator
        if structure.navigation_selectors:
            sorted_nav = sorted(structure.navigation_selectors)
            fingerprint_parts.append(f"nav:{sorted_nav}")

        # Combine all parts and hash
        fingerprint_str = "|".join(fingerprint_parts)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc

    def _classify_page_type(self, url: str) -> str:
        """Classify page type based on URL patterns."""
        url_lower = url.lower()

        # Sports-specific patterns
        if any(p in url_lower for p in ["/nfl/", "/football/", "/nba/", "/basketball/"]):
            if "/story/" in url_lower or "/news/" in url_lower:
                return "article"
            return "category"
        elif "/scores" in url_lower or "/scoreboard" in url_lower:
            return "scores"
        elif "/standings" in url_lower:
            return "standings"
        elif "/schedule" in url_lower:
            return "schedule"
        elif "/player/" in url_lower or "/players/" in url_lower:
            return "player"
        elif "/team/" in url_lower or "/teams/" in url_lower:
            return "team"
        elif any(p in url_lower for p in ["/story/", "/news/", "/article/"]):
            return "article"
        else:
            return "homepage"

    async def fetch_page(self, url: str) -> tuple[str, int] | None:
        """
        Fetch a page with retry logic.

        Returns:
            Tuple of (html_content, status_code) or None if failed.
        """
        if not self.http_client:
            return None

        # Check robots.txt if configured
        if self.config.respect_robots:
            try:
                robots_url = f"{url.split('/')[0]}//{self._extract_domain(url)}/robots.txt"
                robots_response = await self.http_client.get(robots_url)
                if robots_response.status_code == 200:
                    allowed = self.robots_checker.is_allowed(
                        url,
                        robots_response.text,
                        self.config.user_agent
                    )
                    if not allowed:
                        self.logger.warning("URL blocked by robots.txt", url=url)
                        return None
            except Exception:
                pass  # If robots.txt check fails, proceed anyway

        # Fetch with retries
        for attempt in range(self.config.max_retries):
            try:
                response = await self.http_client.get(url)
                return response.text, response.status_code
            except Exception as e:
                self.logger.warning(
                    "Fetch failed",
                    url=url,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        return None

    async def check_url(self, url: str) -> ContentChange | None:
        """
        Check a URL for changes.

        This is the main monitoring logic:
        1. Fetch the page
        2. Analyze page structure
        3. Check for structural changes (fingerprint comparison)
        4. Adapt extraction strategy if needed
        5. Extract content

        Change detection is based on structural fingerprint, NOT raw HTML.
        This ignores dynamic content like timestamps, scores, and dates.

        Returns:
            ContentChange if changes detected, None otherwise.
        """
        self.logger.info("Checking URL", url=url)

        # Fetch page
        result = await self.fetch_page(url)
        if not result:
            self.logger.error("Failed to fetch page", url=url)
            return None

        html, status_code = result

        if status_code != 200:
            self.logger.warning("Non-200 status", url=url, status=status_code)
            return None

        # Analyze page structure first (needed for fingerprint)
        domain = self._extract_domain(url)
        page_type = self._classify_page_type(url)
        current_structure = self.structure_analyzer.analyze(html, url, page_type)

        # Compute structural fingerprint (ignores dynamic content like times/dates)
        current_hash = self._compute_structure_fingerprint(current_structure)
        previous_hash = self._content_hashes.get(url)

        # Quick check: if structure fingerprint unchanged, no changes
        if previous_hash and previous_hash == current_hash:
            self.logger.debug("No structural changes", url=url)
            return None

        # Get stored structure and strategy
        assert self.structure_store is not None
        stored_structure = await self.structure_store.get_structure(
            domain, page_type, "default"
        )
        stored_strategy = await self.structure_store.get_strategy(
            domain, page_type, "default"
        )

        change_type = "new_content"
        similarity_score = 1.0
        strategy: ExtractionStrategy

        if stored_structure and stored_strategy:
            # Detect structural changes
            analysis = self.change_detector.detect_changes(
                stored_structure, current_structure
            )
            similarity_score = analysis.similarity_score

            if analysis.requires_relearning:
                # Structure changed significantly - adapt strategy
                self.logger.info(
                    "Structure change detected - adapting",
                    url=url,
                    similarity=f"{similarity_score:.2%}",
                )
                change_type = "structure_changed"

                adapted = self.strategy_learner.adapt(
                    stored_strategy, current_structure, html
                )
                strategy = adapted.strategy
                strategy.version = stored_strategy.version + 1

                # Save new structure and strategy
                current_structure.version = stored_structure.version + 1
                await self.structure_store.save_structure(
                    current_structure, strategy, "default"
                )
            else:
                # Minor or no structural changes
                strategy = stored_strategy
                change_type = "content_updated"
        else:
            # First time seeing this page - learn strategy
            self.logger.info("Learning new page structure", url=url, page_type=page_type)
            change_type = "new_content"

            learned = self.strategy_learner.infer(html, current_structure)
            strategy = learned.strategy

            # Save structure and strategy
            await self.structure_store.save_structure(
                current_structure, strategy, "default"
            )

        # Extract content
        extraction_result = self.content_extractor.extract(url, html, strategy)

        if not extraction_result.success:
            self.logger.warning(
                "Extraction failed",
                url=url,
                errors=extraction_result.errors,
            )

        # Save new content hash
        await self._save_content_hash(url, current_hash)

        # Create change record
        change = ContentChange(
            url=url,
            detected_at=datetime.now(datetime.timezone.utc),
            change_type=change_type,
            similarity_score=similarity_score,
            extracted_content=extraction_result,
            previous_hash=previous_hash,
            current_hash=current_hash,
        )

        self._change_history.append(change)

        return change

    async def check_all_urls(self) -> list[ContentChange]:
        """Check all configured URLs for changes."""
        changes = []

        for url in self.config.urls:
            change = await self.check_url(url)
            if change:
                changes.append(change)

                # Trigger callback if configured
                if self.on_change:
                    self.on_change(change)

            # Be polite - don't hammer the server
            await asyncio.sleep(1)

        return changes

    async def run_monitoring_loop(self) -> None:
        """Run continuous monitoring loop."""
        self._running = True
        self.logger.info(
            "Starting monitoring loop",
            urls=len(self.config.urls),
            interval=self.config.check_interval,
        )

        while self._running:
            try:
                changes = await self.check_all_urls()

                if changes:
                    self.logger.info(
                        "Changes detected",
                        count=len(changes),
                        types=[c.change_type for c in changes],
                    )

                    # Save changes to file
                    await self._save_changes_to_file(changes)
                else:
                    self.logger.debug("No changes detected")

            except Exception as e:
                self.logger.error("Error in monitoring loop", error=str(e))

            # Wait for next check
            await asyncio.sleep(self.config.check_interval)

    async def _save_changes_to_file(self, changes: list[ContentChange]) -> None:
        """Save detected changes to a JSON file."""
        timestamp = datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = Path(self.config.output_dir) / f"changes_{timestamp}.json"

        data = []
        for change in changes:
            record = change.to_dict()

            # Add extracted content if available
            if change.extracted_content and change.extracted_content.content:
                content = change.extracted_content.content
                record["extracted"] = {
                    "title": content.title,
                    "content_preview": content.content[:500] if content.content else None,
                    "content_length": len(content.content) if content.content else 0,
                    "metadata": content.metadata,
                    "images": content.images[:5] if content.images else [],
                }

            data.append(record)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self.logger.info("Saved changes to file", filepath=str(filepath))

    def get_change_history(self) -> list[ContentChange]:
        """Get the change history."""
        return self._change_history.copy()

    async def run_once(self) -> list[ContentChange]:
        """Run a single check cycle (useful for testing)."""
        await self.start()
        try:
            return await self.check_all_urls()
        finally:
            await self.stop()


def print_change(change: ContentChange) -> None:
    """Print a change notification to console."""
    print(f"\n{'='*60}")
    print(f"CHANGE DETECTED: {change.change_type.upper()}")
    print(f"{'='*60}")
    print(f"URL: {change.url}")
    print(f"Time: {change.detected_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Similarity: {change.similarity_score:.1%}")

    if change.extracted_content and change.extracted_content.content:
        content = change.extracted_content.content
        print(f"\nExtracted Content:")
        print(f"  Title: {content.title}")
        if content.content:
            preview = content.content[:200].replace('\n', ' ')
            print(f"  Preview: {preview}...")
        if content.metadata:
            print(f"  Metadata: {content.metadata}")
    print(f"{'='*60}\n")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor sports news websites for content changes"
    )
    parser.add_argument(
        "--url",
        type=str,
        action="append",
        help="URL to monitor (can specify multiple)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./monitor_output",
        help="Output directory for change logs",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (don't loop)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--install-redis",
        action="store_true",
        help="Install Redis locally (Debian/Ubuntu) and exit",
    )

    args = parser.parse_args()

    # Handle Redis installation request
    if args.install_redis:
        success = install_redis_local()
        sys.exit(0 if success else 1)

    # Setup logging
    setup_logging(
        level="DEBUG" if args.verbose else "INFO",
        format_type="console",
    )

    # Default URLs if none specified
    urls = args.url or [
        # Example sports news URLs - replace with actual URLs to monitor
        "https://www.espn.com/",
        "https://www.espn.com/nfl/",
        "https://www.espn.com/nba/",
    ]

    config = MonitorConfig(
        urls=urls,
        check_interval=args.interval,
        output_dir=args.output,
    )

    print("""
    ===============================================
       SPORTS NEWS MONITOR
       Adaptive Web Crawler Example
    ===============================================

    This monitor will:
    1. Learn the structure of each page on first visit
    2. Detect when content or structure changes
    3. Automatically adapt to site redesigns
    4. Extract and save new content

    """)

    print(f"Monitoring {len(urls)} URL(s):")
    for url in urls:
        print(f"  - {url}")
    print(f"\nCheck interval: {args.interval} seconds")
    print(f"Output directory: {args.output}")
    print()

    # Check Redis
    print("Checking prerequisites...")
    try:
        redis_client = redis.from_url(config.redis_url)
        await redis_client.ping()
        await redis_client.aclose()
        print("  [OK] Redis is running")
    except Exception as e:
        print(f"  [FAIL] Redis connection failed: {e}")
        print("\nRedis is required. Choose an option:")
        print("\n  Option 1: Docker (recommended)")
        print("    docker run -d -p 6379:6379 redis:7-alpine")
        print("\n  Option 2: Local install (Debian/Ubuntu)")
        print("    python examples/sports_news_monitor.py --install-redis")
        print("\n  Option 3: Use existing Redis installation")
        print("    redis-server --daemonize yes")
        print("    # Or start via systemctl:")
        print("    sudo systemctl start redis")
        sys.exit(1)

    # Create and run monitor
    monitor = SportsNewsMonitor(
        config=config,
        on_change=print_change,
    )

    try:
        await monitor.start()

        if args.once:
            print("\nRunning single check...")
            changes = await monitor.check_all_urls()
            print(f"\nDetected {len(changes)} change(s)")
        else:
            print("\nStarting continuous monitoring (Ctrl+C to stop)...")
            await monitor.run_monitoring_loop()

    except KeyboardInterrupt:
        print("\n\nStopping monitor...")
    finally:
        await monitor.stop()

    print("Monitor stopped.")


if __name__ == "__main__":
    asyncio.run(main())
