"""Core crawler modules."""

from crawler.core.crawler import Crawler, run_crawl, CrawlerStats
from crawler.core.fetcher import Fetcher, FetcherConfig
from crawler.core.scheduler import Scheduler, SchedulerConfig
from crawler.core.renderer import (
    BrowserPool,
    HybridFetcher,
    JSRenderer,
    JSRequirementDetector,
    RenderConfig,
    RenderResult,
    WaitStrategy,
)
from crawler.core.distributed import (
    CrawlerWorker,
    CrawlJob,
    DistributedCrawlManager,
    DistributedQueue,
    JobState,
    URLState,
    URLTask,
    WorkerCoordinator,
    WorkerInfo,
    WorkerState,
)
from crawler.core.recrawl_scheduler import (
    AdaptiveScheduleConfig,
    CronSchedule,
    RecrawlSchedule,
    RecrawlScheduleStore,
    RecrawlScheduler,
    ScheduleInterval,
    SitemapBasedScheduler,
)

__all__ = [
    # Core
    "Crawler",
    "CrawlerStats",
    "Fetcher",
    "FetcherConfig",
    "run_crawl",
    "Scheduler",
    "SchedulerConfig",
    # Rendering
    "BrowserPool",
    "HybridFetcher",
    "JSRenderer",
    "JSRequirementDetector",
    "RenderConfig",
    "RenderResult",
    "WaitStrategy",
    # Distributed
    "CrawlerWorker",
    "CrawlJob",
    "DistributedCrawlManager",
    "DistributedQueue",
    "JobState",
    "URLState",
    "URLTask",
    "WorkerCoordinator",
    "WorkerInfo",
    "WorkerState",
    # Scheduling
    "AdaptiveScheduleConfig",
    "CronSchedule",
    "RecrawlSchedule",
    "RecrawlScheduleStore",
    "RecrawlScheduler",
    "ScheduleInterval",
    "SitemapBasedScheduler",
]
