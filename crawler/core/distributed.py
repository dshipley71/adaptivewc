"""
Distributed crawling support for the adaptive web crawler.

Provides multi-worker coordination using Redis for:
- Distributed URL queue with priority
- Worker heartbeats and health monitoring
- Job assignment and completion tracking
- Crawl state coordination
- Leader election for coordination tasks

Architecture:
- Workers pull URLs from shared Redis queue
- Coordinator manages job lifecycle
- Heartbeats detect failed workers
- Atomic operations prevent duplicate work
"""

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Callable

import redis.asyncio as redis

from crawler.utils.logging import CrawlerLogger


class WorkerState(str, Enum):
    """Worker lifecycle states."""

    STARTING = "starting"
    IDLE = "idle"
    WORKING = "working"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class JobState(str, Enum):
    """Crawl job states."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class URLState(str, Enum):
    """URL processing states."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkerInfo:
    """Information about a crawler worker."""

    worker_id: str
    hostname: str
    pid: int
    state: WorkerState
    started_at: datetime
    last_heartbeat: datetime
    current_url: str | None = None
    urls_processed: int = 0
    errors: int = 0
    job_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "worker_id": self.worker_id,
            "hostname": self.hostname,
            "pid": self.pid,
            "state": self.state.value,
            "started_at": self.started_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "current_url": self.current_url,
            "urls_processed": self.urls_processed,
            "errors": self.errors,
            "job_id": self.job_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkerInfo":
        """Create from dictionary."""
        return cls(
            worker_id=data["worker_id"],
            hostname=data["hostname"],
            pid=data["pid"],
            state=WorkerState(data["state"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            last_heartbeat=datetime.fromisoformat(data["last_heartbeat"]),
            current_url=data.get("current_url"),
            urls_processed=data.get("urls_processed", 0),
            errors=data.get("errors", 0),
            job_id=data.get("job_id"),
        )


@dataclass
class CrawlJob:
    """A distributed crawl job."""

    job_id: str
    name: str
    seed_urls: list[str]
    config: dict[str, Any]
    state: JobState
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    urls_total: int = 0
    urls_completed: int = 0
    urls_failed: int = 0
    workers_assigned: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "name": self.name,
            "seed_urls": self.seed_urls,
            "config": self.config,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "urls_total": self.urls_total,
            "urls_completed": self.urls_completed,
            "urls_failed": self.urls_failed,
            "workers_assigned": self.workers_assigned,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrawlJob":
        """Create from dictionary."""
        return cls(
            job_id=data["job_id"],
            name=data["name"],
            seed_urls=data["seed_urls"],
            config=data["config"],
            state=JobState(data["state"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=(
                datetime.fromisoformat(data["started_at"])
                if data.get("started_at")
                else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
            urls_total=data.get("urls_total", 0),
            urls_completed=data.get("urls_completed", 0),
            urls_failed=data.get("urls_failed", 0),
            workers_assigned=data.get("workers_assigned", []),
            error=data.get("error"),
        )


@dataclass
class URLTask:
    """A URL task in the distributed queue."""

    url: str
    job_id: str
    priority: float = 0.5
    depth: int = 0
    parent_url: str | None = None
    added_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attempts: int = 0
    max_attempts: int = 3

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "job_id": self.job_id,
            "priority": self.priority,
            "depth": self.depth,
            "parent_url": self.parent_url,
            "added_at": self.added_at.isoformat(),
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "URLTask":
        """Create from dictionary."""
        return cls(
            url=data["url"],
            job_id=data["job_id"],
            priority=data.get("priority", 0.5),
            depth=data.get("depth", 0),
            parent_url=data.get("parent_url"),
            added_at=(
                datetime.fromisoformat(data["added_at"])
                if data.get("added_at")
                else datetime.now(timezone.utc)
            ),
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", 3),
        )


class DistributedQueue:
    """
    Distributed URL queue backed by Redis.

    Features:
    - Priority-based URL ordering
    - Atomic URL claiming (no duplicates)
    - In-progress tracking with timeouts
    - Domain-based rate limiting coordination
    """

    # Redis key prefixes
    PREFIX = "crawler:dist:"
    QUEUE_KEY = PREFIX + "queue:{job_id}"
    PROCESSING_KEY = PREFIX + "processing:{job_id}"
    COMPLETED_KEY = PREFIX + "completed:{job_id}"
    FAILED_KEY = PREFIX + "failed:{job_id}"
    SEEN_KEY = PREFIX + "seen:{job_id}"
    DOMAIN_LOCK_KEY = PREFIX + "domain_lock:{domain}"

    def __init__(
        self,
        redis_client: redis.Redis,
        job_id: str,
        processing_timeout: float = 300.0,  # 5 minutes
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the distributed queue.

        Args:
            redis_client: Redis client.
            job_id: Job identifier.
            processing_timeout: Timeout for in-progress URLs (seconds).
            logger: Logger instance.
        """
        self.redis = redis_client
        self.job_id = job_id
        self.processing_timeout = processing_timeout
        self.logger = logger or CrawlerLogger("dist_queue")

        # Key names for this job
        self.queue_key = self.QUEUE_KEY.format(job_id=job_id)
        self.processing_key = self.PROCESSING_KEY.format(job_id=job_id)
        self.completed_key = self.COMPLETED_KEY.format(job_id=job_id)
        self.failed_key = self.FAILED_KEY.format(job_id=job_id)
        self.seen_key = self.SEEN_KEY.format(job_id=job_id)

    async def add_url(self, task: URLTask) -> bool:
        """
        Add a URL to the queue if not already seen.

        Args:
            task: URL task to add.

        Returns:
            True if added, False if already seen.
        """
        # Check if URL already seen (atomic)
        was_new = await self.redis.sadd(self.seen_key, task.url)
        if not was_new:
            return False

        # Add to priority queue (lower score = higher priority)
        score = (1.0 - task.priority) * 1000000 + task.depth * 1000 + time.time()
        await self.redis.zadd(self.queue_key, {json.dumps(task.to_dict()): score})

        self.logger.debug("URL added to queue", url=task.url, priority=task.priority)
        return True

    async def add_urls(self, tasks: list[URLTask]) -> int:
        """
        Add multiple URLs to the queue.

        Args:
            tasks: List of URL tasks.

        Returns:
            Number of URLs actually added.
        """
        added = 0
        for task in tasks:
            if await self.add_url(task):
                added += 1
        return added

    async def claim_url(self, worker_id: str) -> URLTask | None:
        """
        Claim the highest priority URL for processing.

        Args:
            worker_id: ID of the claiming worker.

        Returns:
            URLTask if available, None otherwise.
        """
        # Get highest priority URL (lowest score)
        results = await self.redis.zpopmin(self.queue_key, 1)
        if not results:
            return None

        task_json, score = results[0]
        task = URLTask.from_dict(json.loads(task_json))
        task.attempts += 1

        # Add to processing set with timestamp
        processing_data = {
            "task": task.to_dict(),
            "worker_id": worker_id,
            "claimed_at": datetime.now(timezone.utc).isoformat(),
        }
        await self.redis.hset(
            self.processing_key, task.url, json.dumps(processing_data)
        )

        self.logger.debug("URL claimed", url=task.url, worker=worker_id)
        return task

    async def complete_url(self, url: str, success: bool, error: str | None = None) -> None:
        """
        Mark a URL as completed or failed.

        Args:
            url: URL that was processed.
            success: Whether processing succeeded.
            error: Error message if failed.
        """
        # Remove from processing
        processing_data = await self.redis.hget(self.processing_key, url)
        await self.redis.hdel(self.processing_key, url)

        if success:
            # Add to completed set
            await self.redis.sadd(self.completed_key, url)
            self.logger.debug("URL completed", url=url)
        else:
            # Check if should retry
            if processing_data:
                data = json.loads(processing_data)
                task = URLTask.from_dict(data["task"])

                if task.attempts < task.max_attempts:
                    # Re-add to queue for retry
                    await self.redis.srem(self.seen_key, url)
                    await self.add_url(task)
                    self.logger.debug(
                        "URL queued for retry",
                        url=url,
                        attempt=task.attempts,
                    )
                else:
                    # Mark as permanently failed
                    await self.redis.hset(self.failed_key, url, error or "Max retries exceeded")
                    self.logger.debug("URL failed permanently", url=url, error=error)
            else:
                await self.redis.hset(self.failed_key, url, error or "Unknown error")

    async def get_stats(self) -> dict[str, int]:
        """
        Get queue statistics.

        Returns:
            Dictionary with queue stats.
        """
        return {
            "pending": await self.redis.zcard(self.queue_key),
            "processing": await self.redis.hlen(self.processing_key),
            "completed": await self.redis.scard(self.completed_key),
            "failed": await self.redis.hlen(self.failed_key),
            "seen": await self.redis.scard(self.seen_key),
        }

    async def recover_stale_urls(self) -> int:
        """
        Recover URLs that have been processing too long.

        Returns:
            Number of URLs recovered.
        """
        recovered = 0
        cutoff = datetime.now(timezone.utc).timestamp() - self.processing_timeout

        processing = await self.redis.hgetall(self.processing_key)
        for url, data_json in processing.items():
            data = json.loads(data_json)
            claimed_at = datetime.fromisoformat(data["claimed_at"]).timestamp()

            if claimed_at < cutoff:
                # URL processing timed out - re-queue
                task = URLTask.from_dict(data["task"])
                await self.redis.hdel(self.processing_key, url)
                await self.redis.srem(self.seen_key, url)
                await self.add_url(task)
                recovered += 1
                self.logger.warning("Recovered stale URL", url=url)

        return recovered

    async def clear(self) -> None:
        """Clear all queue data for this job."""
        await self.redis.delete(
            self.queue_key,
            self.processing_key,
            self.completed_key,
            self.failed_key,
            self.seen_key,
        )


class WorkerCoordinator:
    """
    Coordinates distributed crawler workers.

    Features:
    - Worker registration and heartbeats
    - Job assignment
    - Health monitoring
    - Leader election for coordination tasks
    """

    # Redis key prefixes
    PREFIX = "crawler:coord:"
    WORKERS_KEY = PREFIX + "workers"
    JOBS_KEY = PREFIX + "jobs"
    LEADER_KEY = PREFIX + "leader"

    HEARTBEAT_INTERVAL = 10.0  # seconds
    HEARTBEAT_TIMEOUT = 30.0  # seconds

    def __init__(
        self,
        redis_client: redis.Redis,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the coordinator.

        Args:
            redis_client: Redis client.
            logger: Logger instance.
        """
        self.redis = redis_client
        self.logger = logger or CrawlerLogger("coordinator")

    async def register_worker(self, worker: WorkerInfo) -> None:
        """
        Register a worker.

        Args:
            worker: Worker information.
        """
        await self.redis.hset(self.WORKERS_KEY, worker.worker_id, json.dumps(worker.to_dict()))
        self.logger.info("Worker registered", worker_id=worker.worker_id)

    async def update_worker(self, worker: WorkerInfo) -> None:
        """
        Update worker information (heartbeat).

        Args:
            worker: Updated worker information.
        """
        worker.last_heartbeat = datetime.now(timezone.utc)
        await self.redis.hset(self.WORKERS_KEY, worker.worker_id, json.dumps(worker.to_dict()))

    async def unregister_worker(self, worker_id: str) -> None:
        """
        Unregister a worker.

        Args:
            worker_id: Worker ID to remove.
        """
        await self.redis.hdel(self.WORKERS_KEY, worker_id)
        self.logger.info("Worker unregistered", worker_id=worker_id)

    async def get_worker(self, worker_id: str) -> WorkerInfo | None:
        """
        Get worker information.

        Args:
            worker_id: Worker ID.

        Returns:
            WorkerInfo or None.
        """
        data = await self.redis.hget(self.WORKERS_KEY, worker_id)
        if data:
            return WorkerInfo.from_dict(json.loads(data))
        return None

    async def get_all_workers(self) -> list[WorkerInfo]:
        """
        Get all registered workers.

        Returns:
            List of WorkerInfo objects.
        """
        workers = []
        data = await self.redis.hgetall(self.WORKERS_KEY)
        for worker_data in data.values():
            workers.append(WorkerInfo.from_dict(json.loads(worker_data)))
        return workers

    async def get_active_workers(self) -> list[WorkerInfo]:
        """
        Get workers with recent heartbeats.

        Returns:
            List of active WorkerInfo objects.
        """
        cutoff = datetime.now(timezone.utc).timestamp() - self.HEARTBEAT_TIMEOUT
        active = []

        for worker in await self.get_all_workers():
            if worker.last_heartbeat.timestamp() > cutoff:
                active.append(worker)

        return active

    async def cleanup_dead_workers(self) -> list[str]:
        """
        Remove workers without recent heartbeats.

        Returns:
            List of removed worker IDs.
        """
        removed = []
        cutoff = datetime.now(timezone.utc).timestamp() - self.HEARTBEAT_TIMEOUT

        for worker in await self.get_all_workers():
            if worker.last_heartbeat.timestamp() < cutoff:
                await self.unregister_worker(worker.worker_id)
                removed.append(worker.worker_id)
                self.logger.warning("Dead worker removed", worker_id=worker.worker_id)

        return removed

    # Job management

    async def create_job(self, job: CrawlJob) -> None:
        """
        Create a new crawl job.

        Args:
            job: Crawl job to create.
        """
        await self.redis.hset(self.JOBS_KEY, job.job_id, json.dumps(job.to_dict()))
        self.logger.info("Job created", job_id=job.job_id, name=job.name)

    async def update_job(self, job: CrawlJob) -> None:
        """
        Update job information.

        Args:
            job: Updated job.
        """
        await self.redis.hset(self.JOBS_KEY, job.job_id, json.dumps(job.to_dict()))

    async def get_job(self, job_id: str) -> CrawlJob | None:
        """
        Get job information.

        Args:
            job_id: Job ID.

        Returns:
            CrawlJob or None.
        """
        data = await self.redis.hget(self.JOBS_KEY, job_id)
        if data:
            return CrawlJob.from_dict(json.loads(data))
        return None

    async def get_all_jobs(self) -> list[CrawlJob]:
        """
        Get all jobs.

        Returns:
            List of CrawlJob objects.
        """
        jobs = []
        data = await self.redis.hgetall(self.JOBS_KEY)
        for job_data in data.values():
            jobs.append(CrawlJob.from_dict(json.loads(job_data)))
        return jobs

    async def delete_job(self, job_id: str) -> None:
        """
        Delete a job.

        Args:
            job_id: Job ID to delete.
        """
        await self.redis.hdel(self.JOBS_KEY, job_id)
        self.logger.info("Job deleted", job_id=job_id)

    # Leader election

    async def try_become_leader(self, worker_id: str, ttl: int = 30) -> bool:
        """
        Try to become the leader (coordinator).

        Args:
            worker_id: Worker ID attempting leadership.
            ttl: Leadership lease time in seconds.

        Returns:
            True if became leader.
        """
        # Try to set leader key with NX (only if not exists)
        result = await self.redis.set(
            self.LEADER_KEY, worker_id, nx=True, ex=ttl
        )
        return result is True

    async def renew_leadership(self, worker_id: str, ttl: int = 30) -> bool:
        """
        Renew leadership lease.

        Args:
            worker_id: Current leader's worker ID.
            ttl: New lease time in seconds.

        Returns:
            True if renewed successfully.
        """
        current = await self.redis.get(self.LEADER_KEY)
        if current and current.decode() == worker_id:
            await self.redis.expire(self.LEADER_KEY, ttl)
            return True
        return False

    async def get_leader(self) -> str | None:
        """
        Get current leader worker ID.

        Returns:
            Leader worker ID or None.
        """
        leader = await self.redis.get(self.LEADER_KEY)
        return leader.decode() if leader else None


class CrawlerWorker:
    """
    Distributed crawler worker.

    Pulls URLs from the distributed queue and processes them.
    Sends heartbeats and coordinates with other workers.
    """

    def __init__(
        self,
        redis_url: str,
        process_url: Callable[[str, dict], Any],
        worker_id: str | None = None,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the worker.

        Args:
            redis_url: Redis connection URL.
            process_url: Async function to process a URL.
            worker_id: Optional worker ID (generated if None).
            logger: Logger instance.
        """
        self.redis_url = redis_url
        self.process_url = process_url
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.logger = logger or CrawlerLogger(f"worker:{self.worker_id}")

        self._redis: redis.Redis | None = None
        self._coordinator: WorkerCoordinator | None = None
        self._queue: DistributedQueue | None = None
        self._running = False
        self._current_job_id: str | None = None
        self._info: WorkerInfo | None = None
        self._heartbeat_task: asyncio.Task | None = None

    async def start(self, job_id: str) -> None:
        """
        Start the worker for a specific job.

        Args:
            job_id: Job to work on.
        """
        self._redis = redis.from_url(self.redis_url)
        self._coordinator = WorkerCoordinator(self._redis, logger=self.logger)
        self._queue = DistributedQueue(self._redis, job_id, logger=self.logger)
        self._current_job_id = job_id

        # Register worker
        self._info = WorkerInfo(
            worker_id=self.worker_id,
            hostname=os.uname().nodename,
            pid=os.getpid(),
            state=WorkerState.STARTING,
            started_at=datetime.now(timezone.utc),
            last_heartbeat=datetime.now(timezone.utc),
            job_id=job_id,
        )
        await self._coordinator.register_worker(self._info)

        # Start heartbeat task
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        self.logger.info("Worker started", job_id=job_id)

    async def stop(self) -> None:
        """Stop the worker."""
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._info and self._coordinator:
            self._info.state = WorkerState.STOPPED
            await self._coordinator.update_worker(self._info)

        if self._redis:
            await self._redis.aclose()

        self.logger.info("Worker stopped")

    async def run(self) -> None:
        """Run the worker loop."""
        if not self._queue or not self._info:
            raise RuntimeError("Worker not started")

        self._info.state = WorkerState.IDLE
        await self._coordinator.update_worker(self._info)

        while self._running:
            # Try to claim a URL
            task = await self._queue.claim_url(self.worker_id)

            if task is None:
                # No URLs available, wait a bit
                await asyncio.sleep(1.0)
                continue

            # Process the URL
            self._info.state = WorkerState.WORKING
            self._info.current_url = task.url
            await self._coordinator.update_worker(self._info)

            try:
                # Get job config
                job = await self._coordinator.get_job(task.job_id)
                config = job.config if job else {}

                # Process URL with user-provided function
                await self.process_url(task.url, config)

                # Mark as completed
                await self._queue.complete_url(task.url, success=True)
                self._info.urls_processed += 1

            except Exception as e:
                self.logger.error("URL processing failed", url=task.url, error=str(e))
                await self._queue.complete_url(task.url, success=False, error=str(e))
                self._info.errors += 1

            finally:
                self._info.state = WorkerState.IDLE
                self._info.current_url = None
                await self._coordinator.update_worker(self._info)

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while self._running:
            await asyncio.sleep(WorkerCoordinator.HEARTBEAT_INTERVAL)
            if self._info and self._coordinator:
                await self._coordinator.update_worker(self._info)


class DistributedCrawlManager:
    """
    Manager for distributed crawl jobs.

    Provides high-level API for creating and managing distributed crawls.
    """

    def __init__(
        self,
        redis_url: str,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the manager.

        Args:
            redis_url: Redis connection URL.
            logger: Logger instance.
        """
        self.redis_url = redis_url
        self.logger = logger or CrawlerLogger("dist_manager")
        self._redis: redis.Redis | None = None
        self._coordinator: WorkerCoordinator | None = None

    async def __aenter__(self) -> "DistributedCrawlManager":
        """Async context manager entry."""
        self._redis = redis.from_url(self.redis_url)
        self._coordinator = WorkerCoordinator(self._redis, logger=self.logger)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._redis:
            await self._redis.aclose()

    async def create_job(
        self,
        name: str,
        seed_urls: list[str],
        config: dict[str, Any] | None = None,
    ) -> CrawlJob:
        """
        Create a new distributed crawl job.

        Args:
            name: Job name.
            seed_urls: Initial URLs to crawl.
            config: Crawl configuration.

        Returns:
            Created CrawlJob.
        """
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        job = CrawlJob(
            job_id=job_id,
            name=name,
            seed_urls=seed_urls,
            config=config or {},
            state=JobState.PENDING,
            created_at=datetime.now(timezone.utc),
            urls_total=len(seed_urls),
        )

        await self._coordinator.create_job(job)

        # Create queue and add seed URLs
        queue = DistributedQueue(self._redis, job_id, logger=self.logger)
        for url in seed_urls:
            await queue.add_url(URLTask(url=url, job_id=job_id, priority=1.0))

        return job

    async def start_job(self, job_id: str) -> None:
        """
        Start a pending job.

        Args:
            job_id: Job ID to start.
        """
        job = await self._coordinator.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        job.state = JobState.RUNNING
        job.started_at = datetime.now(timezone.utc)
        await self._coordinator.update_job(job)

        self.logger.info("Job started", job_id=job_id)

    async def pause_job(self, job_id: str) -> None:
        """
        Pause a running job.

        Args:
            job_id: Job ID to pause.
        """
        job = await self._coordinator.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        job.state = JobState.PAUSED
        await self._coordinator.update_job(job)

        self.logger.info("Job paused", job_id=job_id)

    async def cancel_job(self, job_id: str) -> None:
        """
        Cancel a job.

        Args:
            job_id: Job ID to cancel.
        """
        job = await self._coordinator.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        job.state = JobState.CANCELLED
        job.completed_at = datetime.now(timezone.utc)
        await self._coordinator.update_job(job)

        self.logger.info("Job cancelled", job_id=job_id)

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """
        Get detailed job status.

        Args:
            job_id: Job ID.

        Returns:
            Job status dictionary.
        """
        job = await self._coordinator.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        queue = DistributedQueue(self._redis, job_id, logger=self.logger)
        queue_stats = await queue.get_stats()

        workers = await self._coordinator.get_active_workers()
        job_workers = [w for w in workers if w.job_id == job_id]

        return {
            "job": job.to_dict(),
            "queue": queue_stats,
            "workers": {
                "active": len(job_workers),
                "details": [w.to_dict() for w in job_workers],
            },
        }

    async def add_urls_to_job(
        self,
        job_id: str,
        urls: list[str],
        priority: float = 0.5,
        depth: int = 0,
    ) -> int:
        """
        Add URLs to an existing job.

        Args:
            job_id: Job ID.
            urls: URLs to add.
            priority: URL priority (0-1, higher = more important).
            depth: Crawl depth.

        Returns:
            Number of URLs added.
        """
        queue = DistributedQueue(self._redis, job_id, logger=self.logger)
        tasks = [
            URLTask(url=url, job_id=job_id, priority=priority, depth=depth)
            for url in urls
        ]
        return await queue.add_urls(tasks)

    async def get_cluster_status(self) -> dict[str, Any]:
        """
        Get overall cluster status.

        Returns:
            Cluster status dictionary.
        """
        workers = await self._coordinator.get_all_workers()
        active_workers = await self._coordinator.get_active_workers()
        jobs = await self._coordinator.get_all_jobs()
        leader = await self._coordinator.get_leader()

        return {
            "workers": {
                "total": len(workers),
                "active": len(active_workers),
                "by_state": self._count_by_state(workers),
            },
            "jobs": {
                "total": len(jobs),
                "by_state": self._count_jobs_by_state(jobs),
            },
            "leader": leader,
        }

    def _count_by_state(self, workers: list[WorkerInfo]) -> dict[str, int]:
        """Count workers by state."""
        counts = {}
        for worker in workers:
            state = worker.state.value
            counts[state] = counts.get(state, 0) + 1
        return counts

    def _count_jobs_by_state(self, jobs: list[CrawlJob]) -> dict[str, int]:
        """Count jobs by state."""
        counts = {}
        for job in jobs:
            state = job.state.value
            counts[state] = counts.get(state, 0) + 1
        return counts
