#!/usr/bin/env python3
"""
News Website Monitor

A practical example showing how to use the adaptive web crawler to monitor
news aggregator websites for changes and automatically pull content when
updates are detected.

Use Case:
    An admin monitors news aggregators like Rantingly, Drudge Report, or
    similar sites for breaking news and trending stories. When content
    changes are detected, the system extracts headlines and links.

Usage:
    # Option 1: Start Redis with Docker
    docker run -d -p 6379:6379 redis:7-alpine

    # Option 2: Install Redis locally (Debian/Ubuntu)
    python examples/news_monitor.py --install-redis

    # Run the monitor
    python examples/news_monitor.py

    # Monitor specific URL
    python examples/news_monitor.py --url https://rantingly.com

    # Set check interval (seconds)
    python examples/news_monitor.py --interval 300

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def install_redis_local() -> bool:
    """
    Install Redis locally on Debian/Ubuntu systems.

    Returns:
        True if installation successful, False otherwise.
    """
    print("Installing Redis locally...")
    print("This requires sudo access on Debian/Ubuntu systems.\n")

    commands = [
        (
            "curl -fsSL https://packages.redis.io/gpg | "
            "sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg --yes"
        ),
        (
            'echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] '
            'https://packages.redis.io/deb $(lsb_release -cs) main" | '
            "sudo tee /etc/apt/sources.list.d/redis.list"
        ),
        "sudo apt-get update",
        "sudo apt-get install -y redis-stack-server",
        "pip install redis",
    ]

    for cmd in commands:
        print(f"Running: {cmd[:60]}...")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  Warning: {result.stderr[:100] if result.stderr else 'Command failed'}")
        except Exception as e:
            print(f"  Error: {e}")
            return False

    print("\nStarting Redis server...")
    try:
        subprocess.run(
            "sudo systemctl start redis-stack-server || redis-server --daemonize yes",
            shell=True,
            capture_output=True,
        )
    except Exception:
        try:
            subprocess.Popen(
                ["redis-server", "--daemonize", "yes"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"Could not start Redis: {e}")
            return False

    print("\nRedis installation complete!")
    return True


def check_redis_installed() -> bool:
    """Check if Redis is available."""
    return shutil.which("redis-server") is not None


import httpx
import redis.asyncio as redis

from crawler.adaptive.change_detector import ChangeDetector, ChangeAnalysis
from crawler.adaptive.strategy_learner import StrategyLearner
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
    change_analysis: ChangeAnalysis | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "url": self.url,
            "detected_at": self.detected_at.isoformat(),
            "change_type": self.change_type,
            "similarity_score": self.similarity_score,
            "previous_hash": self.previous_hash,
            "current_hash": self.current_hash,
        }

        if self.change_analysis:
            result["change_analysis"] = self.change_analysis.to_dict()

        return result


@dataclass
class MonitorConfig:
    """Configuration for the monitor."""

    urls: list[str] = field(default_factory=list)
    check_interval: int = 300  # 5 minutes
    redis_url: str = "redis://localhost:6379/0"
    output_dir: str = "./news_monitor_output"
    user_agent: str = "NewsMonitor/1.0 (Content Change Detection)"
    respect_robots: bool = True
    max_retries: int = 3
    request_timeout: float = 30.0


class NewsMonitor:
    """
    Monitors news websites for content changes.

    Features:
    - Learns page structure on first visit
    - Detects structural changes (site redesigns)
    - Detects content changes (new headlines)
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
        self.logger = CrawlerLogger("news_monitor")

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
        self._content_hashes: dict[str, str] = {}
        self._change_history: list[ContentChange] = []

    async def start(self) -> None:
        """Initialize connections and components."""
        self.logger.info("Starting News Monitor")

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
            key = f"news_content_hash:{self._url_to_key(url)}"
            hash_value = await self.redis_client.get(key)
            if hash_value:
                self._content_hashes[url] = hash_value.decode()

    async def _save_content_hash(self, url: str, content_hash: str) -> None:
        """Save content hash to Redis."""
        if not self.redis_client:
            return

        key = f"news_content_hash:{self._url_to_key(url)}"
        await self.redis_client.set(key, content_hash)
        self._content_hashes[url] = content_hash

    def _url_to_key(self, url: str) -> str:
        """Convert URL to a safe Redis key."""
        return hashlib.md5(url.encode()).hexdigest()

    def _compute_structure_fingerprint(self, structure: PageStructure) -> str:
        """
        Compute a fingerprint hash based on structural elements only.

        This ignores dynamic content like timestamps and changing headlines.
        """
        fingerprint_parts = []

        if structure.tag_hierarchy:
            tag_counts = structure.tag_hierarchy.get("tag_counts", {})
            sorted_tags = sorted(tag_counts.items())
            fingerprint_parts.append(f"tags:{sorted_tags}")

        if structure.css_class_map:
            sorted_classes = sorted(
                structure.css_class_map.items(),
                key=lambda x: x[1],
                reverse=True
            )[:30]
            class_names = [name for name, _ in sorted_classes]
            fingerprint_parts.append(f"classes:{sorted(class_names)}")

        if structure.id_attributes:
            sorted_ids = sorted(structure.id_attributes)
            fingerprint_parts.append(f"ids:{sorted_ids}")

        if structure.semantic_landmarks:
            sorted_landmarks = sorted(structure.semantic_landmarks.keys())
            fingerprint_parts.append(f"landmarks:{sorted_landmarks}")

        if structure.navigation_selectors:
            sorted_nav = sorted(structure.navigation_selectors)
            fingerprint_parts.append(f"nav:{sorted_nav}")

        fingerprint_str = "|".join(fingerprint_parts)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc

    def _classify_page_type(self, url: str) -> str:
        """Classify page type based on URL patterns for news sites."""
        url_lower = url.lower()

        # News aggregator patterns
        if any(p in url_lower for p in ["/politics", "/political"]):
            return "politics"
        elif any(p in url_lower for p in ["/business", "/economy", "/finance", "/markets"]):
            return "business"
        elif any(p in url_lower for p in ["/tech", "/technology", "/science"]):
            return "technology"
        elif any(p in url_lower for p in ["/world", "/international", "/global"]):
            return "world"
        elif any(p in url_lower for p in ["/entertainment", "/celebrity", "/culture"]):
            return "entertainment"
        elif any(p in url_lower for p in ["/health", "/medical"]):
            return "health"
        elif any(p in url_lower for p in ["/opinion", "/editorial", "/commentary"]):
            return "opinion"
        elif any(p in url_lower for p in ["/breaking", "/latest", "/trending"]):
            return "breaking"
        elif any(p in url_lower for p in ["/article/", "/story/", "/news/", "/post/"]):
            return "article"
        elif any(p in url_lower for p in ["/category/", "/section/", "/topic/"]):
            return "category"
        elif any(p in url_lower for p in ["/archive", "/search"]):
            return "archive"
        else:
            # Default for news aggregator homepages
            return "homepage"

    async def fetch_page(self, url: str) -> tuple[str, int] | None:
        """Fetch a page with retry logic."""
        if not self.http_client:
            return None

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
                pass

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
                    await asyncio.sleep(2 ** attempt)

        return None

    def _save_html(self, html_content: str, domain: str):
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      filepath = Path(self.config.output_dir) / f"raw_html_change_{timestamp}.html"

      with open(filepath, "w", encoding="utf-8") as f:
          f.write(html_content)
      self.logger.info("Saving HTML to file, changes detected")

    async def check_url(self, url: str, save_html: bool) -> ContentChange | None:
        """
        Check a URL for changes.

        Returns:
            ContentChange if changes detected, None otherwise.
        """
        self.logger.info("Checking URL", url=url)

        result = await self.fetch_page(url)
        if not result:
            self.logger.error("Failed to fetch page", url=url)
            return None

        html, status_code = result

        if status_code != 200:
            self.logger.warning("Non-200 status", url=url, status=status_code)
            return None

        domain = self._extract_domain(url)
        page_type = self._classify_page_type(url)
        current_structure = self.structure_analyzer.analyze(html, url, page_type)

        current_hash = self._compute_structure_fingerprint(current_structure)
        previous_hash = self._content_hashes.get(url)

        if not previous_hash and save_html:
          self.logger.info("First scrape, saving off HTML")
          self._save_html(html, domain)
          
        if previous_hash and previous_hash == current_hash:
            self.logger.debug("No structural changes", url=url)
            return None

        if save_html:
          self.logger.info("change detected, saving off HTML")
          self._save_html(html, domain)

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
        change_analysis: ChangeAnalysis | None = None

        if stored_structure and stored_strategy:
            change_analysis = self.change_detector.detect_changes(
                stored_structure, current_structure
            )
            similarity_score = change_analysis.similarity_score

            if change_analysis.requires_relearning:
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

                current_structure.version = stored_structure.version + 1
                await self.structure_store.save_structure(
                    current_structure, strategy, "default"
                )
            else:
                strategy = stored_strategy
                change_type = "content_updated"
        else:
            self.logger.info("Learning new page structure", url=url, page_type=page_type)
            change_type = "new_content"

            learned = self.strategy_learner.infer(html, current_structure)
            strategy = learned.strategy

            await self.structure_store.save_structure(
                current_structure, strategy, "default"
            )

        extraction_result = self.content_extractor.extract(url, html, strategy)

        if not extraction_result.success:
            self.logger.warning(
                "Extraction failed",
                url=url,
                errors=extraction_result.errors,
            )

        await self._save_content_hash(url, current_hash)

        change = ContentChange(
            url=url,
            detected_at=datetime.now(timezone.utc),
            change_type=change_type,
            similarity_score=similarity_score,
            extracted_content=extraction_result,
            previous_hash=previous_hash,
            current_hash=current_hash,
            change_analysis=change_analysis,
        )

        self._change_history.append(change)

        return change

    async def check_all_urls(self, save_html: bool) -> list[ContentChange]:
        """Check all configured URLs for changes."""
        changes = []

        for url in self.config.urls:
            change = await self.check_url(url, save_html)
            if change:
                changes.append(change)

                if self.on_change:
                    self.on_change(change)

            await asyncio.sleep(1)

        return changes

    async def run_monitoring_loop(self, save_html: bool = False) -> None:
        """Run continuous monitoring loop."""
        self._running = True
        self.logger.info(
            "Starting monitoring loop",
            urls=len(self.config.urls),
            interval=self.config.check_interval,
        )

        while self._running:
            try:
                changes = await self.check_all_urls(save_html)

                if changes:
                    self.logger.info(
                        "Changes detected",
                        count=len(changes),
                        types=[c.change_type for c in changes],
                    )
                    await self._save_changes_to_file(changes)
                else:
                    self.logger.debug("No changes detected")

            except Exception as e:
                self.logger.error("Error in monitoring loop", error=str(e))

            await asyncio.sleep(self.config.check_interval)

    async def _save_changes_to_file(self, changes: list[ContentChange]) -> None:
        """Save detected changes to a JSON file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = Path(self.config.output_dir) / f"news_changes_{timestamp}.json"
        csv_path = Path(self.config.output_dir) / f"changes_tracker.csv"

        data = []
        csv_output = []
        for change in changes:
            record = change.to_dict()

            if change.extracted_content and change.extracted_content.content:
                content = change.extracted_content.content
                record["extracted"] = {
                    "title": content.title,
                    "content": content.content,
                    "content_length": len(content.content) if content.content else 0,
                    "metadata": content.metadata,
                    "links": content.links[:20] if content.links else [],
                }

            data.append(record)

            csv_output.append(",".join(
              [
                change.url, 
                timestamp, 
                filepath,
                change.change_type,
                change.similarity_score,
                change.change_analysis.classification if change.change_analysis else None,
                change.change_analysis.requires_relearning if change.change_analysis else None,
              ]
            ))
            self.logger.info(f"====> to csv will look like: {csv_output})")

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
          
        with open(csv_path, 'a') as f:
          if not os.path.exists(csv_path):
            # Write headers first
            f.write('url,datetime,filepath,change_type,similarity_score,classification,requires_relearning\n')
          for row in csv_output:
            f.write(f'{row}\n')

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
        if content.links:
            print(f"  Links found: {len(content.links)}")
    print(f"{'='*60}\n")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor news websites for content changes"
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
        default="./news_monitor_output",
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
    parser.add_argument(
        "--save_html",
        action="store_true",
        help="If this flag is included, raw HTML will be saved upon change or initial scrape to a 'raw_html' subfolder.",
    )

    args = parser.parse_args()

    if args.install_redis:
        success = install_redis_local()
        sys.exit(0 if success else 1)

    # Setup logging
    setup_logging(
        level="DEBUG" if args.verbose else "INFO",
        format_type="console",
    )

    # Suppress httpx INFO logs (redirect messages, etc.)
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Default URLs for news aggregators
    urls = args.url or [
        "https://rantingly.com",
        "https://www.memeorandum.com/",
        "https://news.ycombinator.com/",
    ]

    config = MonitorConfig(
        urls=urls,
        check_interval=args.interval,
        output_dir=args.output,
    )

    print("""
    ===============================================
       NEWS MONITOR
       Adaptive Web Crawler Example
    ===============================================

    This monitor will:
    1. Learn the structure of each news page on first visit
    2. Detect when content or structure changes
    3. Automatically adapt to site redesigns
    4. Extract and save headlines and links

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
        print("    python examples/news_monitor.py --install-redis")
        print("\n  Option 3: Use existing Redis installation")
        print("    redis-server --daemonize yes")
        sys.exit(1)

    # Create and run monitor
    monitor = NewsMonitor(
        config=config,
        on_change=print_change,
    )

    try:
        await monitor.start()

        if args.once:
            print("\nRunning single check...")
            changes = await monitor.check_all_urls(args.save_html)
            print(f"\nDetected {len(changes)} change(s)")
        else:
            print("\nStarting continuous monitoring (Ctrl+C to stop)...")
            await monitor.run_monitoring_loop(args.save_html)

    except KeyboardInterrupt:
        print("\n\nStopping monitor...")
    finally:
        await monitor.stop()

    print("Monitor stopped.")


if __name__ == "__main__":
    asyncio.run(main())
