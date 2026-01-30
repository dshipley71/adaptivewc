"""
Scheduled recrawling for the adaptive web crawler.

Provides cron-like scheduling for periodic recrawls:
- Cron expression support (minutes, hours, days)
- Per-URL and per-domain schedules
- Adaptive scheduling based on change frequency
- Priority adjustment based on update patterns
- Redis-backed schedule persistence

Features:
- Schedule URLs/domains for periodic recrawling
- Automatic schedule adjustment based on detected changes
- Sitemap-informed scheduling using changefreq/lastmod
- Integration with distributed crawling
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Iterator

import redis.asyncio as redis

from crawler.utils.logging import CrawlerLogger


class ScheduleInterval(str, Enum):
    """Predefined schedule intervals."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


@dataclass
class CronSchedule:
    """
    Cron-like schedule specification.

    Supports:
    - minute: 0-59 or * or */N
    - hour: 0-23 or * or */N
    - day_of_month: 1-31 or * or */N
    - month: 1-12 or * or */N
    - day_of_week: 0-6 (0=Sunday) or * or */N
    """

    minute: str = "*"
    hour: str = "*"
    day_of_month: str = "*"
    month: str = "*"
    day_of_week: str = "*"

    @classmethod
    def from_string(cls, cron_expr: str) -> "CronSchedule":
        """
        Parse a cron expression string.

        Args:
            cron_expr: Cron expression (e.g., "0 */6 * * *").

        Returns:
            CronSchedule object.
        """
        parts = cron_expr.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {cron_expr}")

        return cls(
            minute=parts[0],
            hour=parts[1],
            day_of_month=parts[2],
            month=parts[3],
            day_of_week=parts[4],
        )

    @classmethod
    def from_interval(cls, interval: ScheduleInterval) -> "CronSchedule":
        """
        Create schedule from predefined interval.

        Args:
            interval: Predefined interval.

        Returns:
            CronSchedule object.
        """
        schedules = {
            ScheduleInterval.HOURLY: cls(minute="0"),
            ScheduleInterval.DAILY: cls(minute="0", hour="0"),
            ScheduleInterval.WEEKLY: cls(minute="0", hour="0", day_of_week="0"),
            ScheduleInterval.MONTHLY: cls(minute="0", hour="0", day_of_month="1"),
        }
        return schedules.get(interval, cls())

    @classmethod
    def every_n_hours(cls, n: int) -> "CronSchedule":
        """Create schedule for every N hours."""
        return cls(minute="0", hour=f"*/{n}")

    @classmethod
    def every_n_minutes(cls, n: int) -> "CronSchedule":
        """Create schedule for every N minutes."""
        return cls(minute=f"*/{n}")

    def to_string(self) -> str:
        """Convert to cron expression string."""
        return f"{self.minute} {self.hour} {self.day_of_month} {self.month} {self.day_of_week}"

    def matches(self, dt: datetime) -> bool:
        """
        Check if a datetime matches this schedule.

        Args:
            dt: Datetime to check.

        Returns:
            True if matches.
        """
        return (
            self._matches_field(self.minute, dt.minute, 0, 59)
            and self._matches_field(self.hour, dt.hour, 0, 23)
            and self._matches_field(self.day_of_month, dt.day, 1, 31)
            and self._matches_field(self.month, dt.month, 1, 12)
            and self._matches_field(self.day_of_week, dt.weekday(), 0, 6)
        )

    def _matches_field(self, pattern: str, value: int, min_val: int, max_val: int) -> bool:
        """Check if a value matches a cron field pattern."""
        if pattern == "*":
            return True

        # Handle */N (every N)
        if pattern.startswith("*/"):
            try:
                step = int(pattern[2:])
                return value % step == 0
            except ValueError:
                return False

        # Handle comma-separated values
        if "," in pattern:
            values = [int(v.strip()) for v in pattern.split(",")]
            return value in values

        # Handle range (e.g., 1-5)
        if "-" in pattern:
            try:
                start, end = pattern.split("-")
                return int(start) <= value <= int(end)
            except ValueError:
                return False

        # Handle single value
        try:
            return value == int(pattern)
        except ValueError:
            return False

    def next_run(self, after: datetime | None = None) -> datetime:
        """
        Calculate the next run time after a given datetime.

        Args:
            after: Start time (defaults to now).

        Returns:
            Next scheduled run time.
        """
        if after is None:
            after = datetime.now(timezone.utc)

        # Start from the next minute
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

        # Search for next matching time (limit iterations)
        for _ in range(525600):  # Max 1 year of minutes
            if self.matches(candidate):
                return candidate
            candidate += timedelta(minutes=1)

        # Fallback if no match found
        return after + timedelta(days=1)


@dataclass
class RecrawlSchedule:
    """Schedule for recrawling a URL or domain."""

    target: str  # URL or domain
    target_type: str  # "url" or "domain"
    schedule: CronSchedule
    enabled: bool = True
    priority: float = 0.5
    max_depth: int = 0  # 0 = only this URL, >0 = follow links
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_run: datetime | None = None
    next_run: datetime | None = None
    run_count: int = 0
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate initial next_run if not set."""
        if self.next_run is None and self.enabled:
            self.next_run = self.schedule.next_run()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target": self.target,
            "target_type": self.target_type,
            "schedule": self.schedule.to_string(),
            "enabled": self.enabled,
            "priority": self.priority,
            "max_depth": self.max_depth,
            "created_at": self.created_at.isoformat(),
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "run_count": self.run_count,
            "last_error": self.last_error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecrawlSchedule":
        """Create from dictionary."""
        return cls(
            target=data["target"],
            target_type=data["target_type"],
            schedule=CronSchedule.from_string(data["schedule"]),
            enabled=data.get("enabled", True),
            priority=data.get("priority", 0.5),
            max_depth=data.get("max_depth", 0),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_run=(
                datetime.fromisoformat(data["last_run"]) if data.get("last_run") else None
            ),
            next_run=(
                datetime.fromisoformat(data["next_run"]) if data.get("next_run") else None
            ),
            run_count=data.get("run_count", 0),
            last_error=data.get("last_error"),
            metadata=data.get("metadata", {}),
        )

    def update_after_run(self, success: bool, error: str | None = None) -> None:
        """
        Update schedule after a run.

        Args:
            success: Whether the run succeeded.
            error: Error message if failed.
        """
        self.last_run = datetime.now(timezone.utc)
        self.run_count += 1

        if success:
            self.last_error = None
        else:
            self.last_error = error

        # Calculate next run
        self.next_run = self.schedule.next_run(self.last_run)


@dataclass
class AdaptiveScheduleConfig:
    """Configuration for adaptive scheduling."""

    # Minimum/maximum intervals
    min_interval_minutes: int = 5
    max_interval_minutes: int = 10080  # 1 week

    # Change frequency adjustments
    increase_factor: float = 1.5  # Increase interval when no changes
    decrease_factor: float = 0.5  # Decrease interval when changes detected
    unchanged_threshold: int = 3  # Runs without changes before increasing interval

    # Sitemap-based adjustments
    use_sitemap_hints: bool = True


class RecrawlScheduleStore:
    """
    Redis-backed storage for recrawl schedules.

    Stores schedules and tracks upcoming runs.
    """

    # Redis key prefixes
    PREFIX = "crawler:recrawl:"
    SCHEDULES_KEY = PREFIX + "schedules"
    QUEUE_KEY = PREFIX + "queue"  # Sorted set of upcoming runs
    HISTORY_KEY = PREFIX + "history:{target_hash}"

    def __init__(
        self,
        redis_client: redis.Redis,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the store.

        Args:
            redis_client: Redis client.
            logger: Logger instance.
        """
        self.redis = redis_client
        self.logger = logger or CrawlerLogger("recrawl_store")

    def _target_hash(self, target: str) -> str:
        """Create a hash for a target URL/domain."""
        import hashlib
        return hashlib.md5(target.encode()).hexdigest()[:12]

    async def save_schedule(self, schedule: RecrawlSchedule) -> None:
        """
        Save a schedule.

        Args:
            schedule: Schedule to save.
        """
        target_hash = self._target_hash(schedule.target)
        await self.redis.hset(
            self.SCHEDULES_KEY, target_hash, json.dumps(schedule.to_dict())
        )

        # Add to queue if enabled and has next_run
        if schedule.enabled and schedule.next_run:
            await self.redis.zadd(
                self.QUEUE_KEY,
                {target_hash: schedule.next_run.timestamp()},
            )

        self.logger.debug("Schedule saved", target=schedule.target)

    async def get_schedule(self, target: str) -> RecrawlSchedule | None:
        """
        Get a schedule by target.

        Args:
            target: URL or domain.

        Returns:
            RecrawlSchedule or None.
        """
        target_hash = self._target_hash(target)
        data = await self.redis.hget(self.SCHEDULES_KEY, target_hash)
        if data:
            return RecrawlSchedule.from_dict(json.loads(data))
        return None

    async def delete_schedule(self, target: str) -> None:
        """
        Delete a schedule.

        Args:
            target: URL or domain.
        """
        target_hash = self._target_hash(target)
        await self.redis.hdel(self.SCHEDULES_KEY, target_hash)
        await self.redis.zrem(self.QUEUE_KEY, target_hash)
        self.logger.debug("Schedule deleted", target=target)

    async def get_all_schedules(self) -> list[RecrawlSchedule]:
        """
        Get all schedules.

        Returns:
            List of RecrawlSchedule objects.
        """
        schedules = []
        data = await self.redis.hgetall(self.SCHEDULES_KEY)
        for schedule_data in data.values():
            schedules.append(RecrawlSchedule.from_dict(json.loads(schedule_data)))
        return schedules

    async def get_due_schedules(self, before: datetime | None = None) -> list[RecrawlSchedule]:
        """
        Get schedules that are due for execution.

        Args:
            before: Cutoff time (defaults to now).

        Returns:
            List of due RecrawlSchedule objects.
        """
        if before is None:
            before = datetime.now(timezone.utc)

        # Get target hashes with scores <= before
        due_hashes = await self.redis.zrangebyscore(
            self.QUEUE_KEY, "-inf", before.timestamp()
        )

        schedules = []
        for target_hash in due_hashes:
            if isinstance(target_hash, bytes):
                target_hash = target_hash.decode()
            data = await self.redis.hget(self.SCHEDULES_KEY, target_hash)
            if data:
                schedules.append(RecrawlSchedule.from_dict(json.loads(data)))

        return schedules

    async def mark_running(self, target: str) -> None:
        """
        Mark a schedule as currently running.

        Removes from queue temporarily.

        Args:
            target: URL or domain.
        """
        target_hash = self._target_hash(target)
        await self.redis.zrem(self.QUEUE_KEY, target_hash)

    async def record_run(
        self,
        schedule: RecrawlSchedule,
        success: bool,
        error: str | None = None,
        changes_detected: bool = False,
    ) -> None:
        """
        Record a schedule run and update next run time.

        Args:
            schedule: Schedule that ran.
            success: Whether it succeeded.
            error: Error message if failed.
            changes_detected: Whether content changes were found.
        """
        schedule.update_after_run(success, error)

        # Save updated schedule
        await self.save_schedule(schedule)

        # Record in history
        target_hash = self._target_hash(schedule.target)
        history_key = self.HISTORY_KEY.format(target_hash=target_hash)
        history_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": success,
            "error": error,
            "changes_detected": changes_detected,
        }
        await self.redis.lpush(history_key, json.dumps(history_entry))
        await self.redis.ltrim(history_key, 0, 99)  # Keep last 100 entries

    async def get_run_history(self, target: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get run history for a target.

        Args:
            target: URL or domain.
            limit: Maximum entries to return.

        Returns:
            List of history entries.
        """
        target_hash = self._target_hash(target)
        history_key = self.HISTORY_KEY.format(target_hash=target_hash)
        entries = await self.redis.lrange(history_key, 0, limit - 1)
        return [json.loads(e) for e in entries]


class RecrawlScheduler:
    """
    Scheduler for periodic recrawling.

    Manages schedules and triggers recrawls based on cron expressions.
    """

    def __init__(
        self,
        redis_url: str,
        on_recrawl: Callable[[str, dict[str, Any]], Any] | None = None,
        adaptive_config: AdaptiveScheduleConfig | None = None,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the scheduler.

        Args:
            redis_url: Redis connection URL.
            on_recrawl: Callback function when recrawl is triggered.
            adaptive_config: Adaptive scheduling configuration.
            logger: Logger instance.
        """
        self.redis_url = redis_url
        self.on_recrawl = on_recrawl
        self.adaptive_config = adaptive_config or AdaptiveScheduleConfig()
        self.logger = logger or CrawlerLogger("recrawl_scheduler")

        self._redis: redis.Redis | None = None
        self._store: RecrawlScheduleStore | None = None
        self._running = False
        self._check_interval = 60.0  # Check every minute

    async def __aenter__(self) -> "RecrawlScheduler":
        """Async context manager entry."""
        self._redis = redis.from_url(self.redis_url)
        self._store = RecrawlScheduleStore(self._redis, logger=self.logger)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self._running = False
        if self._redis:
            await self._redis.aclose()

    async def add_schedule(
        self,
        target: str,
        schedule: CronSchedule | str,
        target_type: str = "url",
        priority: float = 0.5,
        max_depth: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> RecrawlSchedule:
        """
        Add a new recrawl schedule.

        Args:
            target: URL or domain to recrawl.
            schedule: Cron schedule or cron expression string.
            target_type: "url" or "domain".
            priority: Recrawl priority (0-1).
            max_depth: Crawl depth (0 = single URL).
            metadata: Additional metadata.

        Returns:
            Created RecrawlSchedule.
        """
        if isinstance(schedule, str):
            schedule = CronSchedule.from_string(schedule)

        recrawl_schedule = RecrawlSchedule(
            target=target,
            target_type=target_type,
            schedule=schedule,
            priority=priority,
            max_depth=max_depth,
            metadata=metadata or {},
        )

        await self._store.save_schedule(recrawl_schedule)
        self.logger.info(
            "Schedule added",
            target=target,
            schedule=schedule.to_string(),
        )

        return recrawl_schedule

    async def add_url_schedule(
        self,
        url: str,
        interval: ScheduleInterval | str = ScheduleInterval.DAILY,
        priority: float = 0.5,
    ) -> RecrawlSchedule:
        """
        Add a URL schedule with predefined interval.

        Args:
            url: URL to schedule.
            interval: Schedule interval or cron expression.
            priority: Recrawl priority.

        Returns:
            Created RecrawlSchedule.
        """
        if isinstance(interval, ScheduleInterval):
            schedule = CronSchedule.from_interval(interval)
        else:
            schedule = CronSchedule.from_string(interval)

        return await self.add_schedule(
            target=url,
            schedule=schedule,
            target_type="url",
            priority=priority,
        )

    async def add_domain_schedule(
        self,
        domain: str,
        interval: ScheduleInterval | str = ScheduleInterval.WEEKLY,
        max_depth: int = 2,
        priority: float = 0.5,
    ) -> RecrawlSchedule:
        """
        Add a domain schedule.

        Args:
            domain: Domain to schedule.
            interval: Schedule interval or cron expression.
            max_depth: Maximum crawl depth.
            priority: Recrawl priority.

        Returns:
            Created RecrawlSchedule.
        """
        if isinstance(interval, ScheduleInterval):
            schedule = CronSchedule.from_interval(interval)
        else:
            schedule = CronSchedule.from_string(interval)

        return await self.add_schedule(
            target=domain,
            schedule=schedule,
            target_type="domain",
            max_depth=max_depth,
            priority=priority,
        )

    async def remove_schedule(self, target: str) -> None:
        """
        Remove a schedule.

        Args:
            target: URL or domain.
        """
        await self._store.delete_schedule(target)
        self.logger.info("Schedule removed", target=target)

    async def get_schedule(self, target: str) -> RecrawlSchedule | None:
        """
        Get a schedule.

        Args:
            target: URL or domain.

        Returns:
            RecrawlSchedule or None.
        """
        return await self._store.get_schedule(target)

    async def list_schedules(self) -> list[RecrawlSchedule]:
        """
        List all schedules.

        Returns:
            List of RecrawlSchedule objects.
        """
        return await self._store.get_all_schedules()

    async def pause_schedule(self, target: str) -> None:
        """
        Pause a schedule.

        Args:
            target: URL or domain.
        """
        schedule = await self._store.get_schedule(target)
        if schedule:
            schedule.enabled = False
            await self._store.save_schedule(schedule)
            self.logger.info("Schedule paused", target=target)

    async def resume_schedule(self, target: str) -> None:
        """
        Resume a paused schedule.

        Args:
            target: URL or domain.
        """
        schedule = await self._store.get_schedule(target)
        if schedule:
            schedule.enabled = True
            schedule.next_run = schedule.schedule.next_run()
            await self._store.save_schedule(schedule)
            self.logger.info("Schedule resumed", target=target)

    async def run(self) -> None:
        """
        Run the scheduler loop.

        Continuously checks for due schedules and triggers recrawls.
        """
        self._running = True
        self.logger.info("Scheduler started")

        while self._running:
            try:
                await self._check_and_run_due()
            except Exception as e:
                self.logger.error("Scheduler error", error=str(e))

            await asyncio.sleep(self._check_interval)

        self.logger.info("Scheduler stopped")

    async def _check_and_run_due(self) -> None:
        """Check for due schedules and trigger recrawls."""
        due_schedules = await self._store.get_due_schedules()

        for schedule in due_schedules:
            if not schedule.enabled:
                continue

            # Mark as running
            await self._store.mark_running(schedule.target)

            # Trigger recrawl
            try:
                self.logger.info(
                    "Triggering recrawl",
                    target=schedule.target,
                    run_count=schedule.run_count + 1,
                )

                changes_detected = False
                if self.on_recrawl:
                    result = await self.on_recrawl(schedule.target, {
                        "target_type": schedule.target_type,
                        "priority": schedule.priority,
                        "max_depth": schedule.max_depth,
                        "metadata": schedule.metadata,
                    })
                    # Assume result indicates changes if truthy
                    changes_detected = bool(result)

                await self._store.record_run(
                    schedule, success=True, changes_detected=changes_detected
                )

            except Exception as e:
                self.logger.error(
                    "Recrawl failed",
                    target=schedule.target,
                    error=str(e),
                )
                await self._store.record_run(schedule, success=False, error=str(e))

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False

    async def trigger_now(self, target: str) -> bool:
        """
        Trigger immediate recrawl of a target.

        Args:
            target: URL or domain.

        Returns:
            True if triggered successfully.
        """
        schedule = await self._store.get_schedule(target)
        if not schedule:
            return False

        try:
            if self.on_recrawl:
                await self.on_recrawl(schedule.target, {
                    "target_type": schedule.target_type,
                    "priority": 1.0,  # High priority for manual trigger
                    "max_depth": schedule.max_depth,
                    "metadata": schedule.metadata,
                })

            await self._store.record_run(schedule, success=True)
            return True

        except Exception as e:
            await self._store.record_run(schedule, success=False, error=str(e))
            return False

    async def get_status(self) -> dict[str, Any]:
        """
        Get scheduler status.

        Returns:
            Status dictionary.
        """
        schedules = await self._store.get_all_schedules()
        now = datetime.now(timezone.utc)

        enabled = [s for s in schedules if s.enabled]
        due = [s for s in enabled if s.next_run and s.next_run <= now]

        return {
            "running": self._running,
            "total_schedules": len(schedules),
            "enabled_schedules": len(enabled),
            "due_now": len(due),
            "next_check": self._check_interval,
        }


class SitemapBasedScheduler:
    """
    Creates schedules based on sitemap changefreq hints.

    Maps sitemap change frequencies to appropriate cron schedules.
    """

    # Sitemap changefreq to cron schedule mapping
    CHANGEFREQ_MAP = {
        "always": CronSchedule.every_n_minutes(5),
        "hourly": CronSchedule.from_interval(ScheduleInterval.HOURLY),
        "daily": CronSchedule.from_interval(ScheduleInterval.DAILY),
        "weekly": CronSchedule.from_interval(ScheduleInterval.WEEKLY),
        "monthly": CronSchedule.from_interval(ScheduleInterval.MONTHLY),
        "yearly": CronSchedule(minute="0", hour="0", day_of_month="1", month="1"),
        "never": None,  # Don't schedule
    }

    def __init__(
        self,
        scheduler: RecrawlScheduler,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the sitemap-based scheduler.

        Args:
            scheduler: Recrawl scheduler to add schedules to.
            logger: Logger instance.
        """
        self.scheduler = scheduler
        self.logger = logger or CrawlerLogger("sitemap_scheduler")

    async def schedule_from_sitemap_urls(
        self,
        sitemap_urls: list[Any],  # List of SitemapURL objects
        default_interval: ScheduleInterval = ScheduleInterval.DAILY,
    ) -> int:
        """
        Create schedules from sitemap URL entries.

        Args:
            sitemap_urls: List of SitemapURL objects.
            default_interval: Default interval if no changefreq.

        Returns:
            Number of schedules created.
        """
        created = 0

        for url_entry in sitemap_urls:
            # Get schedule based on changefreq
            if hasattr(url_entry, "changefreq") and url_entry.changefreq:
                schedule = self.CHANGEFREQ_MAP.get(url_entry.changefreq.value)
                if schedule is None:
                    continue  # "never" - skip
            else:
                schedule = CronSchedule.from_interval(default_interval)

            # Calculate priority based on sitemap priority
            priority = 0.5
            if hasattr(url_entry, "priority") and url_entry.priority is not None:
                priority = url_entry.priority

            await self.scheduler.add_schedule(
                target=url_entry.loc,
                schedule=schedule,
                target_type="url",
                priority=priority,
                metadata={
                    "source": "sitemap",
                    "lastmod": (
                        url_entry.lastmod.isoformat()
                        if hasattr(url_entry, "lastmod") and url_entry.lastmod
                        else None
                    ),
                },
            )
            created += 1

        self.logger.info("Schedules created from sitemap", count=created)
        return created
