"""
HTTP fetcher with compliance pipeline for the adaptive web crawler.

Implements the compliance pipeline:
1. CFAA authorization check
2. robots.txt check
3. Rate limiting
4. Fetch
5. GDPR PII check on response
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx

from crawler.compliance.rate_limiter import RateLimiter
from crawler.compliance.robots_parser import RobotsChecker, RobotsTxt
from crawler.config import CrawlerSettings, GDPRConfig, SafetyLimits, SecurityConfig
from crawler.exceptions import (
    ContentTooLargeError,
    FetchError,
    TimeoutError,
    TooManyRedirectsError,
)
from crawler.legal.cfaa_checker import CFAAChecker
from crawler.legal.pii_detector import PIIDetector
from crawler.models import FetchResult, FetchStatus
from crawler.storage.robots_cache import RobotsCache
from crawler.utils.logging import CrawlerLogger
from crawler.utils.url_utils import get_domain, is_private_ip
from crawler.utils import metrics


@dataclass
class FetcherConfig:
    """Configuration for the fetcher."""

    user_agent: str = "AdaptiveCrawler/1.0"
    timeout_seconds: float = 30.0
    max_redirects: int = 10
    max_content_size: int = 10 * 1024 * 1024  # 10MB
    verify_ssl: bool = True
    block_private_ips: bool = True


class Fetcher:
    """
    HTTP fetcher with full compliance pipeline.

    Implements:
    - CFAA authorization checking
    - robots.txt compliance
    - Rate limiting
    - PII detection and handling
    - Safety limits (size, timeout, redirects)
    """

    def __init__(
        self,
        config: FetcherConfig | None = None,
        rate_limiter: RateLimiter | None = None,
        robots_checker: RobotsChecker | None = None,
        robots_cache: RobotsCache | None = None,
        cfaa_checker: CFAAChecker | None = None,
        pii_detector: PIIDetector | None = None,
        gdpr_config: GDPRConfig | None = None,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the fetcher.

        Args:
            config: Fetcher configuration.
            rate_limiter: Rate limiter instance.
            robots_checker: Robots.txt checker instance.
            robots_cache: Robots.txt cache instance.
            cfaa_checker: CFAA authorization checker.
            pii_detector: PII detector for GDPR compliance.
            gdpr_config: GDPR configuration.
            logger: Logger instance.
        """
        self.config = config or FetcherConfig()
        self.rate_limiter = rate_limiter or RateLimiter()
        self.robots_checker = robots_checker or RobotsChecker(
            user_agent=self.config.user_agent
        )
        self.robots_cache = robots_cache
        self.cfaa_checker = cfaa_checker or CFAAChecker(
            user_agent=self.config.user_agent
        )
        self.pii_detector = pii_detector
        self.gdpr_config = gdpr_config
        self.logger = logger or CrawlerLogger("fetcher")

        # HTTP client
        self._client: httpx.AsyncClient | None = None

        # Local robots.txt cache (fallback if no Redis)
        self._robots_local_cache: dict[str, RobotsTxt] = {}

    async def start(self) -> None:
        """Initialize the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout_seconds),
                follow_redirects=True,
                max_redirects=self.config.max_redirects,
                verify=self.config.verify_ssl,
                headers={"User-Agent": self.config.user_agent},
            )

    async def stop(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch(self, url: str) -> FetchResult:
        """
        Fetch a URL with full compliance pipeline.

        Args:
            url: URL to fetch.

        Returns:
            FetchResult with response or error details.
        """
        domain = get_domain(url)
        start_time = time.monotonic()

        self.logger.fetch_start(url=url, domain=domain)

        # 1. Security check - block private IPs
        if self.config.block_private_ips and is_private_ip(url):
            self.logger.fetch_blocked(url=url, reason="private_ip")
            return FetchResult.blocked(
                url=url,
                reason="Private IP addresses are blocked",
                status=FetchStatus.BLOCKED_LEGAL,
            )

        # 2. CFAA authorization check
        robots_txt = await self._get_robots_txt(domain)
        robots_allows = self.robots_checker.is_allowed(url, robots_txt)

        auth_result = await self.cfaa_checker.is_authorized(
            url=url,
            robots_allows=robots_allows,
            requires_auth=False,
        )

        if not auth_result.authorized:
            self.logger.fetch_blocked(url=url, reason=f"cfaa:{auth_result.basis}")
            metrics.record_blocked(domain, "cfaa", {"reason": auth_result.basis})
            return FetchResult.blocked(
                url=url,
                reason=f"Not authorized: {auth_result.basis}",
                status=FetchStatus.BLOCKED_LEGAL,
            )

        # 3. robots.txt check
        if not robots_allows:
            self.logger.fetch_blocked(url=url, reason="robots_txt")
            metrics.record_blocked(domain, "robots")
            self.rate_limiter.record_blocked(domain)
            return FetchResult.blocked(
                url=url,
                reason="Blocked by robots.txt",
                status=FetchStatus.BLOCKED_ROBOTS,
            )

        # Apply crawl-delay from robots.txt
        crawl_delay = self.robots_checker.get_crawl_delay(robots_txt)
        if crawl_delay:
            self.rate_limiter.set_crawl_delay(domain, crawl_delay)

        # 4. Rate limiting
        await self.rate_limiter.acquire(domain)

        # 5. Fetch
        try:
            result = await self._do_fetch(url)
        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.fetch_error(
                url=url,
                error=str(e),
                error_type=type(e).__name__,
            )
            self.rate_limiter.record_error(domain)
            metrics.record_error(domain, type(e).__name__)
            return self._handle_fetch_error(url, e)

        duration_ms = (time.monotonic() - start_time) * 1000
        result.duration_ms = duration_ms

        # 6. GDPR PII check
        if self.gdpr_config and self.gdpr_config.enabled and self.pii_detector:
            result = await self._process_pii(result)

        # Record success
        self.rate_limiter.record_success(domain, duration_ms)
        metrics.record_fetch(
            domain=domain,
            status="success",
            duration_seconds=duration_ms / 1000,
            content_size=len(result.content or b""),
            status_code=result.status_code,
        )

        self.logger.fetch_success(
            url=url,
            status_code=result.status_code or 0,
            duration_ms=duration_ms,
            content_length=len(result.content or b""),
        )

        return result

    async def _do_fetch(self, url: str) -> FetchResult:
        """Perform the actual HTTP fetch."""
        if self._client is None:
            await self.start()

        assert self._client is not None

        # Check content size before downloading
        try:
            response = await self._client.get(url)
        except httpx.TimeoutException:
            raise TimeoutError(url, self.config.timeout_seconds)
        except httpx.TooManyRedirects:
            raise TooManyRedirectsError(url, self.config.max_redirects, self.config.max_redirects)
        except httpx.HTTPError as e:
            raise FetchError(url, str(e))

        # Check content length
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > self.config.max_content_size:
            raise ContentTooLargeError(
                url, int(content_length), self.config.max_content_size
            )

        content = response.content
        if len(content) > self.config.max_content_size:
            raise ContentTooLargeError(
                url, len(content), self.config.max_content_size
            )

        # Build redirect chain
        redirect_chain = [str(r.url) for r in response.history]

        return FetchResult.success(
            url=str(response.url),
            content=content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )

    async def _get_robots_txt(self, domain: str) -> RobotsTxt:
        """Get robots.txt for a domain, using cache."""
        # Try cache first
        if self.robots_cache:
            cached = await self.robots_cache.get(domain)
            if cached:
                return cached

        # Check local cache
        if domain in self._robots_local_cache:
            return self._robots_local_cache[domain]

        # Fetch robots.txt
        robots_url = f"https://{domain}/robots.txt"

        try:
            if self._client is None:
                await self.start()

            assert self._client is not None
            response = await self._client.get(robots_url)
            content = response.text
            status_code = response.status_code
        except Exception:
            content = ""
            status_code = 500

        # Parse and cache
        from crawler.compliance.robots_parser import RobotsParser
        parser = RobotsParser()
        robots_txt = parser.parse(content, domain, status_code)

        # Cache
        if self.robots_cache:
            await self.robots_cache.set(domain, robots_txt)
        self._robots_local_cache[domain] = robots_txt

        return robots_txt

    async def _process_pii(self, result: FetchResult) -> FetchResult:
        """Process PII in response content."""
        if not result.html or not self.pii_detector:
            return result

        processed_text, detection = self.pii_detector.process(
            result.html, result.url
        )

        if self.pii_detector.should_exclude_page(detection):
            self.logger.warning(
                "Page excluded due to sensitive PII",
                url=result.url,
                pii_types=[t.value for t in detection.pii_types_found],
            )
            return FetchResult.blocked(
                url=result.url,
                reason="Page contains sensitive PII",
                status=FetchStatus.BLOCKED_LEGAL,
            )

        # Update result with processed content
        if processed_text != result.html:
            result.html = processed_text
            result.content = processed_text.encode("utf-8")

        return result

    def _handle_fetch_error(self, url: str, error: Exception) -> FetchResult:
        """Convert exception to FetchResult."""
        if isinstance(error, TimeoutError):
            return FetchResult.error(
                url=url,
                message=str(error),
                status=FetchStatus.ERROR_TIMEOUT,
            )
        elif isinstance(error, ContentTooLargeError):
            return FetchResult.error(
                url=url,
                message=str(error),
                status=FetchStatus.ERROR_CONTENT_TOO_LARGE,
            )
        else:
            return FetchResult.error(
                url=url,
                message=str(error),
                status=FetchStatus.ERROR_CONNECTION,
            )

    async def fetch_robots_txt(self, domain: str) -> tuple[str, int]:
        """
        Fetch robots.txt for a domain.

        Args:
            domain: Domain to fetch robots.txt for.

        Returns:
            Tuple of (content, status_code).
        """
        robots_url = f"https://{domain}/robots.txt"

        try:
            if self._client is None:
                await self.start()

            assert self._client is not None
            response = await self._client.get(robots_url)
            return response.text, response.status_code
        except Exception:
            return "", 500

    async def __aenter__(self) -> "Fetcher":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.stop()
