#!/usr/bin/env python3
"""
Distributed Crawling Example

Demonstrates how to use the distributed crawling system to coordinate
multiple workers processing URLs in parallel across machines. Includes
structure fingerprinting for tracking page changes across crawls.

Features demonstrated:
- Creating distributed crawl jobs
- Running multiple workers
- Redis-backed URL queues with atomic operations
- Worker heartbeats and health monitoring
- Leader election for coordination tasks
- Job state management
- Progress monitoring
- Structure fingerprinting for each crawled page
- Change detection across crawl iterations

Usage:
    # Start Redis first
    docker run -d -p 6379:6379 redis:7-alpine

    # Create a job
    python examples/distributed_crawl_example.py create \
        --job-id my-crawl-001 \
        --seeds https://example.com https://httpbin.org

    # Run a worker (with fingerprinting)
    python examples/distributed_crawl_example.py worker \
        --job-id my-crawl-001 \
        --worker-id worker-1

    # Monitor job progress
    python examples/distributed_crawl_example.py monitor \
        --job-id my-crawl-001

    # Run complete demo (creates job + workers + monitor)
    python examples/distributed_crawl_example.py demo

    # Disable fingerprinting
    python examples/distributed_crawl_example.py demo --no-fingerprint

Requirements:
    - Redis running on localhost:6379
    - pip install -e .
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import uuid

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import redis.asyncio as redis

from crawler.core.distributed import (
    DistributedQueue,
    WorkerCoordinator,
    CrawlerWorker,
    DistributedCrawlManager,
    CrawlJob,
    URLTask,
    JobState,
    WorkerState,
)
from crawler.adaptive.structure_analyzer import StructureAnalyzer, AnalysisConfig
from crawler.adaptive.change_detector import ChangeDetector, ChangeAnalysis
from crawler.models import PageStructure
from crawler.utils.logging import CrawlerLogger, setup_logging


class DistributedCrawlDemo:
    """
    Demonstrates distributed crawling capabilities with fingerprinting.

    Key concepts:
    - Jobs contain seed URLs and configuration
    - Workers claim URLs from a shared queue
    - Redis ensures atomic operations (no duplicate processing)
    - Leader election handles coordination tasks
    - Heartbeats detect dead workers
    - Structure fingerprinting for page analysis
    - Change detection across crawls
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        enable_fingerprinting: bool = True,
    ):
        """
        Initialize the demo.

        Args:
            redis_url: Redis connection URL
            enable_fingerprinting: Enable structure fingerprinting for change detection
        """
        self.redis_url = redis_url
        self.enable_fingerprinting = enable_fingerprinting
        self.logger = CrawlerLogger("distributed_demo")
        self.redis_client: redis.Redis | None = None

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

    async def connect(self) -> None:
        """Connect to Redis."""
        self.redis_client = redis.from_url(self.redis_url)
        await self.redis_client.ping()
        self.logger.info("Connected to Redis", url=self.redis_url)

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.aclose()

    async def create_job(
        self,
        job_id: str,
        seed_urls: list[str],
        max_urls: int = 100,
        max_depth: int = 3,
    ) -> CrawlJob:
        """
        Create a new distributed crawl job.

        This sets up the job in Redis with:
        - Seed URLs in the pending queue
        - Job configuration and metadata
        - State tracking (pending, running, completed)

        Args:
            job_id: Unique identifier for the job
            seed_urls: Initial URLs to crawl
            max_urls: Maximum URLs to process
            max_depth: Maximum crawl depth

        Returns:
            CrawlJob object
        """
        manager = DistributedCrawlManager(self.redis_client)

        job = await manager.create_job(
            job_id=job_id,
            seed_urls=seed_urls,
            max_urls=max_urls,
            max_depth=max_depth,
        )

        self.logger.info(
            "Job created",
            job_id=job.job_id,
            seed_urls=len(seed_urls),
            max_urls=max_urls,
            max_depth=max_depth,
        )

        return job

    async def run_worker(
        self,
        job_id: str,
        worker_id: str,
        max_urls: int | None = None,
    ) -> dict:
        """
        Run a worker that processes URLs from the job.

        The worker:
        1. Sends heartbeats to indicate it's alive
        2. Claims URLs atomically from the queue
        3. Processes URLs (simulated in this demo)
        4. Reports completion or failure
        5. Discovers new URLs and adds them to queue

        Args:
            job_id: Job to work on
            worker_id: Unique identifier for this worker
            max_urls: Maximum URLs to process (None = until job complete)

        Returns:
            Worker statistics
        """
        worker = CrawlerWorker(
            redis_client=self.redis_client,
            job_id=job_id,
            worker_id=worker_id,
            heartbeat_interval=5.0,
            claim_timeout=300.0,
        )

        stats = {
            "urls_processed": 0,
            "urls_success": 0,
            "urls_failed": 0,
            "new_urls_discovered": 0,
            "fingerprinting": {
                "pages_analyzed": 0,
                "structures_stored": 0,
            } if self.enable_fingerprinting else None,
        }

        # Store page structures for this worker (in production, use Redis)
        page_structures: dict[str, PageStructure] = {}

        self.logger.info("Worker starting", worker_id=worker_id, job_id=job_id)

        async def process_url(task: URLTask) -> tuple[bool, list[str], dict | None]:
            """
            Process a URL and optionally fingerprint its structure.

            Returns:
                Tuple of (success, new_urls, fingerprint_info)
            """
            import httpx

            fingerprint_info = None

            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(task.url)

                    # Extract links (simplified)
                    new_urls = []
                    if response.status_code == 200 and "text/html" in response.headers.get("content-type", ""):
                        # Find links (very simplified)
                        import re
                        links = re.findall(r'href=["\']([^"\']+)["\']', response.text)
                        for link in links[:5]:  # Limit to 5 per page
                            if link.startswith("http"):
                                new_urls.append(link)

                        # Fingerprint the page structure
                        if self.enable_fingerprinting:
                            page_type = self._classify_page_type(task.url)
                            structure = self.structure_analyzer.analyze(
                                response.text, task.url, page_type
                            )

                            fingerprint_info = {
                                "content_hash": structure.content_hash,
                                "page_type": structure.page_type,
                                "tag_count": sum(structure.tag_hierarchy.get("tag_counts", {}).values())
                                    if structure.tag_hierarchy else 0,
                                "content_regions": len(structure.content_regions),
                            }

                            # Check for changes if we've seen this URL before
                            if task.url in page_structures:
                                previous = page_structures[task.url]
                                change = self.change_detector.detect_changes(
                                    previous, structure
                                )
                                fingerprint_info["change_detected"] = change.has_changes
                                fingerprint_info["change_classification"] = change.classification.value
                                fingerprint_info["similarity"] = f"{change.similarity_score:.2%}"

                                if change.has_changes:
                                    self.logger.info(
                                        "Structure change detected",
                                        url=task.url,
                                        classification=change.classification.value,
                                    )

                            # Store for future comparisons
                            page_structures[task.url] = structure
                            stats["fingerprinting"]["pages_analyzed"] += 1
                            stats["fingerprinting"]["structures_stored"] = len(page_structures)

                    return True, new_urls, fingerprint_info

            except Exception as e:
                self.logger.warning("URL processing failed", url=task.url, error=str(e))
                return False, [], None

        def _classify_page_type(url: str) -> str:
            """Classify the page type based on URL patterns."""
            url_lower = url.lower()
            if any(x in url_lower for x in ["/article", "/post", "/blog", "/news"]):
                return "article"
            elif any(x in url_lower for x in ["/category", "/tag", "/archive"]):
                return "listing"
            elif any(x in url_lower for x in ["/product", "/item", "/shop"]):
                return "product"
            return "content"

        # Bind _classify_page_type as instance method
        self._classify_page_type = _classify_page_type

        # Worker loop
        try:
            await worker.register()

            while True:
                # Check if we've hit our limit
                if max_urls and stats["urls_processed"] >= max_urls:
                    self.logger.info("Worker reached URL limit", limit=max_urls)
                    break

                # Claim a URL
                task = await worker.claim_url()

                if task is None:
                    # No more URLs available
                    job_status = await worker.get_job_status()
                    if job_status.get("state") in ["COMPLETED", "FAILED", "CANCELLED"]:
                        self.logger.info("Job complete, worker stopping")
                        break

                    # Wait and try again
                    await asyncio.sleep(1)
                    continue

                self.logger.info(
                    "Processing URL",
                    worker_id=worker_id,
                    url=task.url,
                    depth=task.depth,
                )

                # Process the URL (with optional fingerprinting)
                success, new_urls, fingerprint_info = await process_url(task)

                # Report result
                await worker.complete_url(task.url, success=success)
                stats["urls_processed"] += 1

                if success:
                    stats["urls_success"] += 1

                    # Log fingerprint info if available
                    if fingerprint_info:
                        self.logger.info(
                            "Page fingerprinted",
                            url=task.url,
                            page_type=fingerprint_info.get("page_type"),
                            tag_count=fingerprint_info.get("tag_count"),
                            content_regions=fingerprint_info.get("content_regions"),
                        )

                    # Add discovered URLs
                    for new_url in new_urls:
                        added = await worker.add_url(URLTask(
                            url=new_url,
                            depth=task.depth + 1,
                            priority=5,
                        ))
                        if added:
                            stats["new_urls_discovered"] += 1
                else:
                    stats["urls_failed"] += 1

                # Polite delay
                await asyncio.sleep(0.5)

        finally:
            await worker.deregister()

        self.logger.info("Worker finished", worker_id=worker_id, stats=stats)
        return stats

    async def monitor_job(
        self,
        job_id: str,
        interval: float = 2.0,
        timeout: float | None = None,
    ) -> None:
        """
        Monitor job progress in real-time.

        Displays:
        - Queue status (pending, processing, completed, failed)
        - Active workers
        - Progress percentage

        Args:
            job_id: Job to monitor
            interval: Update interval in seconds
            timeout: Stop after this many seconds (None = until complete)
        """
        manager = DistributedCrawlManager(self.redis_client)
        coordinator = WorkerCoordinator(self.redis_client, job_id)

        start_time = datetime.now()
        last_completed = 0

        print(f"\n{'='*60}")
        print(f"MONITORING JOB: {job_id}")
        print(f"{'='*60}\n")

        try:
            while True:
                status = await manager.get_job_status(job_id)
                workers = await coordinator.get_active_workers()

                # Calculate progress
                total = (
                    status["pending_urls"] +
                    status["processing_urls"] +
                    status["completed_urls"] +
                    status["failed_urls"]
                )
                completed = status["completed_urls"] + status["failed_urls"]
                progress = (completed / total * 100) if total > 0 else 0

                # Calculate rate
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = completed / elapsed if elapsed > 0 else 0

                # Display status
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                      f"State: {status['state']} | "
                      f"Progress: {progress:.1f}% | "
                      f"Pending: {status['pending_urls']} | "
                      f"Processing: {status['processing_urls']} | "
                      f"Completed: {status['completed_urls']} | "
                      f"Failed: {status['failed_urls']} | "
                      f"Workers: {len(workers)} | "
                      f"Rate: {rate:.1f}/s",
                      end="", flush=True)

                # Check for completion
                if status["state"] in ["COMPLETED", "FAILED", "CANCELLED"]:
                    print(f"\n\nJob {status['state']}!")
                    break

                # Check timeout
                if timeout and elapsed > timeout:
                    print(f"\n\nMonitoring timeout reached")
                    break

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n\nMonitoring stopped by user")

    async def demo_queue_operations(self, job_id: str) -> None:
        """
        Demonstrate queue operations.

        Shows:
        - Adding URLs with priority
        - Atomic claiming
        - Timeout recovery
        """
        queue = DistributedQueue(self.redis_client, job_id)

        print(f"\n{'='*60}")
        print("QUEUE OPERATIONS DEMO")
        print(f"{'='*60}\n")

        # Add URLs with different priorities
        print("1. Adding URLs with different priorities:")
        urls = [
            ("https://example.com/high", 10),
            ("https://example.com/medium", 5),
            ("https://example.com/low", 1),
        ]

        for url, priority in urls:
            task = URLTask(url=url, depth=0, priority=priority)
            added = await queue.add_url(task)
            print(f"   Added {url} (priority={priority}): {added}")

        # Check queue size
        size = await queue.size()
        print(f"\n   Queue size: {size}")

        # Claim URLs (should come out in priority order)
        print("\n2. Claiming URLs (highest priority first):")
        worker_id = "demo-worker"

        for i in range(3):
            task = await queue.claim_url(worker_id)
            if task:
                print(f"   Claimed: {task.url} (priority={task.priority})")
                # Complete immediately
                await queue.complete_url(task.url, success=True)

        # Show final status
        stats = await queue.get_stats()
        print(f"\n3. Final queue stats:")
        print(f"   Pending: {stats['pending']}")
        print(f"   Processing: {stats['processing']}")
        print(f"   Completed: {stats['completed']}")
        print(f"   Failed: {stats['failed']}")

    async def run_full_demo(self) -> None:
        """
        Run a complete demo of distributed crawling.

        Creates a job, runs multiple workers concurrently,
        and monitors progress.
        """
        job_id = f"demo-{uuid.uuid4().hex[:8]}"
        seed_urls = [
            "https://httpbin.org/html",
            "https://httpbin.org/robots.txt",
            "https://example.com",
        ]

        print(f"\n{'='*60}")
        print("DISTRIBUTED CRAWL DEMO")
        print(f"{'='*60}\n")

        print(f"Job ID: {job_id}")
        print(f"Seed URLs: {len(seed_urls)}")
        print()

        # Create job
        await self.create_job(
            job_id=job_id,
            seed_urls=seed_urls,
            max_urls=20,
            max_depth=2,
        )

        # Run workers and monitor concurrently
        worker_count = 3

        async def run_workers():
            tasks = []
            for i in range(worker_count):
                worker_id = f"worker-{i}"
                task = asyncio.create_task(
                    self.run_worker(job_id, worker_id, max_urls=10)
                )
                tasks.append(task)

            return await asyncio.gather(*tasks, return_exceptions=True)

        # Start workers and monitor
        worker_task = asyncio.create_task(run_workers())
        monitor_task = asyncio.create_task(
            self.monitor_job(job_id, interval=1.0, timeout=60)
        )

        # Wait for both
        try:
            await asyncio.gather(worker_task, monitor_task)
        except Exception as e:
            print(f"\nError: {e}")

        # Show final results
        manager = DistributedCrawlManager(self.redis_client)
        status = await manager.get_job_status(job_id)

        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Job ID: {job_id}")
        print(f"State: {status['state']}")
        print(f"URLs Completed: {status['completed_urls']}")
        print(f"URLs Failed: {status['failed_urls']}")
        print(f"{'='*60}")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Distributed crawling example"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new job")
    create_parser.add_argument("--job-id", type=str, required=True)
    create_parser.add_argument("--seeds", type=str, nargs="+", required=True)
    create_parser.add_argument("--max-urls", type=int, default=100)
    create_parser.add_argument("--max-depth", type=int, default=3)

    # Worker command
    worker_parser = subparsers.add_parser("worker", help="Run a worker")
    worker_parser.add_argument("--job-id", type=str, required=True)
    worker_parser.add_argument("--worker-id", type=str, required=True)
    worker_parser.add_argument("--max-urls", type=int, default=None)

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor a job")
    monitor_parser.add_argument("--job-id", type=str, required=True)
    monitor_parser.add_argument("--interval", type=float, default=2.0)

    # Queue demo command
    queue_parser = subparsers.add_parser("queue", help="Demo queue operations")
    queue_parser.add_argument("--job-id", type=str, default="queue-demo")

    # Full demo command
    subparsers.add_parser("demo", help="Run full demo")

    # Global options
    parser.add_argument("--redis-url", type=str, default="redis://localhost:6379/0")
    parser.add_argument("--no-fingerprint", action="store_true", help="Disable structure fingerprinting")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\n\nRun 'demo' for a complete example:")
        print("  python examples/distributed_crawl_example.py demo")
        sys.exit(1)

    # Setup logging
    setup_logging(
        level="DEBUG" if args.verbose else "INFO",
        format_type="console",
    )

    print("""
    ===============================================
       DISTRIBUTED CRAWLING EXAMPLE
       Multi-worker URL processing with Redis
    ===============================================

    This example shows how to:
    1. Create distributed crawl jobs
    2. Run multiple workers concurrently
    3. Use atomic queue operations
    4. Monitor job progress
    5. Handle worker coordination
    6. Fingerprint page structures during crawling
    7. Detect structural changes across crawls

    Prerequisites:
      docker run -d -p 6379:6379 redis:7-alpine

    """)

    demo = DistributedCrawlDemo(
        redis_url=args.redis_url,
        enable_fingerprinting=not args.no_fingerprint,
    )

    try:
        await demo.connect()

        if args.command == "create":
            await demo.create_job(
                job_id=args.job_id,
                seed_urls=args.seeds,
                max_urls=args.max_urls,
                max_depth=args.max_depth,
            )
            print(f"\nJob '{args.job_id}' created!")
            print(f"Run workers with:")
            print(f"  python examples/distributed_crawl_example.py worker "
                  f"--job-id {args.job_id} --worker-id worker-1")

        elif args.command == "worker":
            stats = await demo.run_worker(
                job_id=args.job_id,
                worker_id=args.worker_id,
                max_urls=args.max_urls,
            )
            print(f"\nWorker finished: {stats}")

        elif args.command == "monitor":
            await demo.monitor_job(
                job_id=args.job_id,
                interval=args.interval,
            )

        elif args.command == "queue":
            await demo.demo_queue_operations(args.job_id)

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
