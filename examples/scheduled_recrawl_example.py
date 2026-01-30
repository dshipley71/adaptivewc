#!/usr/bin/env python3
"""
Scheduled Recrawling Example

Demonstrates how to use the scheduled recrawling system for periodic
URL monitoring and change detection.

Features demonstrated:
- Cron expression scheduling
- Interval-based scheduling
- Adaptive scheduling (adjusts based on change frequency)
- Sitemap-based scheduling (uses changefreq/lastmod hints)
- Schedule management (add, list, enable/disable, remove)
- Change detection callback

Usage:
    # Start Redis first
    docker run -d -p 6379:6379 redis:7-alpine

    # Add URLs with cron schedule
    python examples/scheduled_recrawl_example.py add \
        --url https://example.com \
        --cron "0 */6 * * *"

    # Add URLs with interval
    python examples/scheduled_recrawl_example.py add \
        --url https://example.com \
        --interval hourly

    # List all schedules
    python examples/scheduled_recrawl_example.py list

    # Run the scheduler
    python examples/scheduled_recrawl_example.py run

    # Run demo with adaptive scheduling
    python examples/scheduled_recrawl_example.py demo

Requirements:
    - Redis running on localhost:6379
    - pip install -e .
"""

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import redis.asyncio as redis

from crawler.core.recrawl_scheduler import (
    RecrawlScheduler,
    RecrawlSchedule,
    CronSchedule,
    ScheduleInterval,
    AdaptiveScheduleConfig,
    SitemapBasedScheduler,
)
from crawler.utils.logging import CrawlerLogger, setup_logging


class ScheduledRecrawlDemo:
    """
    Demonstrates scheduled recrawling capabilities.

    Key concepts:
    - URLs can be scheduled for periodic recrawling
    - Cron expressions provide precise scheduling
    - Adaptive scheduling adjusts frequency based on change rate
    - Sitemap hints (changefreq) inform initial schedules
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize the demo."""
        self.redis_url = redis_url
        self.logger = CrawlerLogger("scheduled_recrawl_demo")
        self.redis_client: redis.Redis | None = None
        self.scheduler: RecrawlScheduler | None = None

        # Track content for change detection
        self._content_hashes: dict[str, str] = {}

    async def connect(self) -> None:
        """Connect to Redis and initialize scheduler."""
        self.redis_client = redis.from_url(self.redis_url)
        await self.redis_client.ping()

        # Initialize scheduler with adaptive config
        adaptive_config = AdaptiveScheduleConfig(
            initial_interval=3600,      # Start with 1 hour
            min_interval=300,           # Never faster than 5 minutes
            max_interval=86400 * 7,     # Never slower than 1 week
            increase_factor=1.5,        # Slow down by 50% when unchanged
            decrease_factor=0.5,        # Speed up by 50% when changed
        )

        self.scheduler = RecrawlScheduler(
            self.redis_client,
            adaptive_config=adaptive_config,
        )

        self.logger.info("Connected to Redis", url=self.redis_url)

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.aclose()

    async def add_schedule(
        self,
        url: str,
        interval: str | int | None = None,
        cron_expr: str | None = None,
        adaptive: bool = False,
    ) -> RecrawlSchedule:
        """
        Add a URL to the recrawl schedule.

        Args:
            url: URL to schedule
            interval: Interval (e.g., "hourly", "daily", or seconds)
            cron_expr: Cron expression (e.g., "0 */6 * * *")
            adaptive: Enable adaptive scheduling

        Returns:
            Created schedule
        """
        if cron_expr:
            # Use cron expression
            schedule = await self.scheduler.add_url_schedule(
                url=url,
                interval=cron_expr,
                adaptive=adaptive,
            )
            self.logger.info(
                "Added cron schedule",
                url=url,
                cron=cron_expr,
                next_run=schedule.next_crawl,
            )
        else:
            # Use interval
            if isinstance(interval, str):
                # Parse named interval
                interval_map = {
                    "15min": ScheduleInterval.MINUTES_15,
                    "hourly": ScheduleInterval.HOURLY,
                    "daily": ScheduleInterval.DAILY,
                    "weekly": ScheduleInterval.WEEKLY,
                    "monthly": ScheduleInterval.MONTHLY,
                }
                interval = interval_map.get(interval.lower(), ScheduleInterval.HOURLY)

            schedule = await self.scheduler.add_url_schedule(
                url=url,
                interval=interval or ScheduleInterval.HOURLY,
                adaptive=adaptive,
            )
            self.logger.info(
                "Added interval schedule",
                url=url,
                interval=schedule.interval_seconds,
                next_run=schedule.next_crawl,
            )

        return schedule

    async def list_schedules(self) -> list[RecrawlSchedule]:
        """List all scheduled URLs."""
        schedules = await self.scheduler.list_schedules()

        print(f"\n{'='*70}")
        print("SCHEDULED URLS")
        print(f"{'='*70}\n")

        if not schedules:
            print("No schedules found.")
            return []

        for schedule in schedules:
            status = "ENABLED" if schedule.enabled else "DISABLED"
            print(f"URL: {schedule.url}")
            print(f"  Status: {status}")
            print(f"  Interval: {schedule.interval_seconds}s "
                  f"({schedule.interval_seconds / 3600:.1f} hours)")
            print(f"  Last crawled: {schedule.last_crawled or 'Never'}")
            print(f"  Next crawl: {schedule.next_crawl}")
            print(f"  Total crawls: {schedule.total_crawls}")
            print(f"  Total changes: {schedule.total_changes}")
            if schedule.adaptive:
                print(f"  Adaptive: YES (consecutive unchanged: "
                      f"{schedule.consecutive_no_change})")
            print()

        return schedules

    async def run_scheduler(
        self,
        check_interval: float = 10.0,
        duration: float | None = None,
    ) -> None:
        """
        Run the scheduler loop.

        Checks for due URLs and processes them.

        Args:
            check_interval: How often to check for due URLs (seconds)
            duration: How long to run (None = forever)
        """
        print(f"\n{'='*60}")
        print("RUNNING SCHEDULER")
        print(f"{'='*60}")
        print(f"Check interval: {check_interval}s")
        print(f"Duration: {duration or 'indefinite'}s")
        print("Press Ctrl+C to stop\n")

        start_time = datetime.now()

        async def on_url_due(schedule: RecrawlSchedule) -> None:
            """Handle a URL that is due for recrawling."""
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Processing: {schedule.url}")

            # Fetch and check for changes
            changed = await self._check_url_for_changes(schedule.url)

            if changed:
                print(f"  -> CHANGED! Content was modified.")
            else:
                print(f"  -> No changes detected.")

            # Record result (updates adaptive interval)
            await self.scheduler.record_crawl_result(
                url=schedule.url,
                changed=changed,
            )

            # Show new interval if adaptive
            if schedule.adaptive:
                updated = await self.scheduler.get_schedule(schedule.url)
                if updated:
                    print(f"  -> Next interval: {updated.interval_seconds}s")

        try:
            while True:
                # Check if we've exceeded duration
                if duration:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed >= duration:
                        print("\nDuration reached, stopping scheduler.")
                        break

                # Get URLs due for recrawling
                due_urls = await self.scheduler.get_due_urls()

                for schedule in due_urls:
                    await on_url_due(schedule)

                if not due_urls:
                    # Show countdown to next due URL
                    next_schedule = await self._get_next_scheduled()
                    if next_schedule:
                        wait_time = (next_schedule.next_crawl - datetime.utcnow()).total_seconds()
                        if wait_time > 0:
                            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                                  f"Next URL due in {wait_time:.0f}s: "
                                  f"{next_schedule.url[:50]}...",
                                  end="", flush=True)

                await asyncio.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n\nScheduler stopped by user.")

    async def _get_next_scheduled(self) -> RecrawlSchedule | None:
        """Get the next URL scheduled to be crawled."""
        schedules = await self.scheduler.list_schedules()
        if not schedules:
            return None

        enabled = [s for s in schedules if s.enabled]
        if not enabled:
            return None

        return min(enabled, key=lambda s: s.next_crawl)

    async def _check_url_for_changes(self, url: str) -> bool:
        """
        Check if a URL's content has changed.

        Uses content hashing for simple change detection.

        Args:
            url: URL to check

        Returns:
            True if content changed, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)

                if response.status_code != 200:
                    return False

                # Compute content hash
                content_hash = hashlib.sha256(response.content).hexdigest()

                # Check for change
                previous_hash = self._content_hashes.get(url)
                self._content_hashes[url] = content_hash

                if previous_hash is None:
                    # First time seeing this URL
                    return True  # Treat as "changed" for initial crawl

                return previous_hash != content_hash

        except Exception as e:
            self.logger.error("Failed to check URL", url=url, error=str(e))
            return False

    async def demo_cron_parsing(self) -> None:
        """Demonstrate cron expression parsing."""
        print(f"\n{'='*60}")
        print("CRON EXPRESSION EXAMPLES")
        print(f"{'='*60}\n")

        examples = [
            ("*/5 * * * *", "Every 5 minutes"),
            ("0 * * * *", "Every hour"),
            ("0 */2 * * *", "Every 2 hours"),
            ("0 9 * * 1-5", "9 AM on weekdays"),
            ("0 0 * * *", "Daily at midnight"),
            ("0 0 * * 0", "Weekly on Sunday"),
            ("0 0 1 * *", "Monthly on the 1st"),
        ]

        for cron_expr, description in examples:
            try:
                cron = CronSchedule.from_string(cron_expr)
                next_run = cron.next_run()
                print(f"'{cron_expr}'")
                print(f"  Description: {description}")
                print(f"  Next run: {next_run}")
                print()
            except Exception as e:
                print(f"'{cron_expr}' - Error: {e}")

    async def demo_adaptive_scheduling(self) -> None:
        """Demonstrate adaptive scheduling behavior."""
        print(f"\n{'='*60}")
        print("ADAPTIVE SCHEDULING DEMO")
        print(f"{'='*60}\n")

        print("Adaptive scheduling adjusts the interval based on how often")
        print("content actually changes:\n")

        # Simulate a sequence of crawls
        url = "https://example.com/adaptive-test"

        # Add with adaptive enabled
        schedule = await self.add_schedule(
            url=url,
            interval="hourly",
            adaptive=True,
        )

        print(f"Initial interval: {schedule.interval_seconds}s")

        # Simulate crawls
        scenarios = [
            (False, "No change - interval increases"),
            (False, "No change - interval increases more"),
            (True, "CHANGE - interval decreases"),
            (False, "No change - interval increases"),
            (True, "CHANGE - interval decreases"),
            (True, "CHANGE - interval decreases more"),
        ]

        for changed, description in scenarios:
            await self.scheduler.record_crawl_result(url, changed=changed)
            updated = await self.scheduler.get_schedule(url)

            if updated:
                print(f"  Crawl: {description}")
                print(f"    New interval: {updated.interval_seconds:.0f}s "
                      f"({updated.interval_seconds / 60:.1f} min)")

        # Cleanup
        await self.scheduler.remove_schedule(url)

    async def run_full_demo(self) -> None:
        """Run a complete demo of scheduled recrawling."""
        print(f"\n{'='*60}")
        print("SCHEDULED RECRAWL DEMO")
        print(f"{'='*60}\n")

        # Demo cron parsing
        await self.demo_cron_parsing()

        # Demo adaptive scheduling
        await self.demo_adaptive_scheduling()

        # Add some test schedules
        print(f"\n{'='*60}")
        print("SETTING UP TEST SCHEDULES")
        print(f"{'='*60}\n")

        test_urls = [
            ("https://httpbin.org/uuid", "15min", True),
            ("https://httpbin.org/html", "hourly", False),
            ("https://example.com", "0 */6 * * *", False),  # cron
        ]

        for url, interval, adaptive in test_urls:
            if "*" in str(interval):
                await self.add_schedule(url, cron_expr=interval, adaptive=adaptive)
            else:
                await self.add_schedule(url, interval=interval, adaptive=adaptive)

        # List schedules
        await self.list_schedules()

        # Run scheduler briefly
        print("Running scheduler for 30 seconds...")
        await self.run_scheduler(check_interval=5.0, duration=30.0)

        # Final status
        await self.list_schedules()


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scheduled recrawling example"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a URL schedule")
    add_parser.add_argument("--url", type=str, required=True)
    add_parser.add_argument("--interval", type=str, help="Interval (15min, hourly, daily)")
    add_parser.add_argument("--cron", type=str, help="Cron expression")
    add_parser.add_argument("--adaptive", action="store_true")

    # List command
    subparsers.add_parser("list", help="List all schedules")

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a schedule")
    remove_parser.add_argument("--url", type=str, required=True)

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the scheduler")
    run_parser.add_argument("--interval", type=float, default=10.0)
    run_parser.add_argument("--duration", type=float, default=None)

    # Demo command
    subparsers.add_parser("demo", help="Run full demo")

    # Cron demo command
    subparsers.add_parser("cron-demo", help="Demo cron parsing")

    # Global options
    parser.add_argument("--redis-url", type=str, default="redis://localhost:6379/0")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\n\nRun 'demo' for a complete example:")
        print("  python examples/scheduled_recrawl_example.py demo")
        sys.exit(1)

    # Setup logging
    setup_logging(
        level="DEBUG" if args.verbose else "INFO",
        format_type="console",
    )

    print("""
    ===============================================
       SCHEDULED RECRAWLING EXAMPLE
       Periodic URL monitoring and change detection
    ===============================================

    This example shows how to:
    1. Schedule URLs for periodic recrawling
    2. Use cron expressions for precise timing
    3. Use adaptive scheduling based on change frequency
    4. Monitor and detect content changes

    Prerequisites:
      docker run -d -p 6379:6379 redis:7-alpine

    """)

    demo = ScheduledRecrawlDemo(redis_url=args.redis_url)

    try:
        await demo.connect()

        if args.command == "add":
            await demo.add_schedule(
                url=args.url,
                interval=args.interval,
                cron_expr=args.cron,
                adaptive=args.adaptive,
            )
            print(f"\nSchedule added for: {args.url}")

        elif args.command == "list":
            await demo.list_schedules()

        elif args.command == "remove":
            await demo.scheduler.remove_schedule(args.url)
            print(f"\nSchedule removed for: {args.url}")

        elif args.command == "run":
            await demo.run_scheduler(
                check_interval=args.interval,
                duration=args.duration,
            )

        elif args.command == "cron-demo":
            await demo.demo_cron_parsing()

        elif args.command == "demo":
            await demo.run_full_demo()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            raise
    finally:
        await demo.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
