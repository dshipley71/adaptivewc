"""
Rate limiting engine for the adaptive web crawler.

Provides per-domain rate limiting with adaptive delay adjustment.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from crawler.config import RateLimitConfig
from crawler.utils.logging import CrawlerLogger


@dataclass
class DomainState:
    """Rate limiting state for a single domain."""

    domain: str
    last_request_time: float = 0.0
    current_delay: float = 1.0
    consecutive_errors: int = 0
    total_requests: int = 0
    blocked_count: int = 0
    crawl_delay_override: float | None = None

    # Adaptive delay tracking
    response_times: list[float] = field(default_factory=list)
    last_adjustment_time: float = 0.0


class RateLimiter:
    """
    Per-domain rate limiter with adaptive delay adjustment.

    Features:
    - Respects robots.txt crawl-delay
    - Adaptive delay based on server response times
    - Exponential backoff on errors
    - Concurrent request limiting per domain
    """

    def __init__(
        self,
        config: RateLimitConfig | None = None,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the rate limiter.

        Args:
            config: Rate limiting configuration.
            logger: Logger instance.
        """
        self.config = config or RateLimitConfig()
        self.logger = logger or CrawlerLogger("rate_limiter")

        self._domain_states: dict[str, DomainState] = {}
        self._domain_locks: dict[str, asyncio.Lock] = {}
        self._global_semaphore = asyncio.Semaphore(self.config.max_concurrent_global)

    def _get_state(self, domain: str) -> DomainState:
        """Get or create domain state."""
        if domain not in self._domain_states:
            self._domain_states[domain] = DomainState(
                domain=domain,
                current_delay=self.config.default_delay,
            )
        return self._domain_states[domain]

    def _get_lock(self, domain: str) -> asyncio.Lock:
        """Get or create domain lock."""
        if domain not in self._domain_locks:
            self._domain_locks[domain] = asyncio.Lock()
        return self._domain_locks[domain]

    async def acquire(self, domain: str) -> float:
        """
        Acquire permission to make a request to a domain.

        This method will wait until it's safe to make a request,
        respecting rate limits and crawl delays.

        Args:
            domain: The domain to request.

        Returns:
            The number of seconds waited.
        """
        # Acquire global semaphore to limit total concurrent requests
        async with self._global_semaphore:
            lock = self._get_lock(domain)
            state = self._get_state(domain)

            async with lock:
                # Calculate required delay
                delay = self._calculate_delay(state)
                time_since_last = time.monotonic() - state.last_request_time
                wait_time = max(0, delay - time_since_last)

                if wait_time > 0:
                    self.logger.rate_limit_wait(
                        domain=domain,
                        delay_seconds=wait_time,
                        reason="rate_limit",
                    )
                    await asyncio.sleep(wait_time)

                # Update state
                state.last_request_time = time.monotonic()
                state.total_requests += 1

                return wait_time

    def _calculate_delay(self, state: DomainState) -> float:
        """Calculate the required delay for a domain."""
        # Start with crawl-delay override if set
        if state.crawl_delay_override is not None and self.config.respect_crawl_delay:
            base_delay = max(state.crawl_delay_override, self.config.min_delay)
        else:
            base_delay = state.current_delay

        # Apply backoff for consecutive errors (capped at 5 to prevent extreme delays)
        if state.consecutive_errors > 0:
            capped_errors = min(state.consecutive_errors, 5)
            backoff = self.config.backoff_multiplier ** capped_errors
            delay = min(base_delay * backoff, self.config.max_delay)
        else:
            delay = base_delay

        return delay

    def set_crawl_delay(self, domain: str, delay: float) -> None:
        """
        Set the crawl-delay for a domain from robots.txt.

        Args:
            domain: The domain.
            delay: The crawl-delay value in seconds.
        """
        state = self._get_state(domain)
        state.crawl_delay_override = delay

    def record_success(
        self,
        domain: str,
        response_time_ms: float,
    ) -> None:
        """
        Record a successful request.

        Args:
            domain: The domain.
            response_time_ms: Response time in milliseconds.
        """
        state = self._get_state(domain)
        state.consecutive_errors = 0

        # Track response times for adaptive delay
        if self.config.adaptive:
            state.response_times.append(response_time_ms)
            # Keep only recent response times
            if len(state.response_times) > 20:
                state.response_times = state.response_times[-20:]

            # Adjust delay based on response times
            self._maybe_adjust_delay(state)

    def record_error(
        self,
        domain: str,
        is_rate_limit: bool = False,
    ) -> None:
        """
        Record an error for a domain.

        Args:
            domain: The domain.
            is_rate_limit: Whether this was a rate limit error (429).
        """
        state = self._get_state(domain)
        state.consecutive_errors += 1

        if is_rate_limit:
            state.blocked_count += 1
            # Double the delay on rate limit
            state.current_delay = min(
                state.current_delay * 2,
                self.config.max_delay,
            )

    def record_blocked(self, domain: str) -> None:
        """
        Record a blocked request (robots.txt, CFAA, etc).

        Args:
            domain: The domain.
        """
        state = self._get_state(domain)
        state.blocked_count += 1

    def _maybe_adjust_delay(self, state: DomainState) -> None:
        """Adjust delay based on response times."""
        if not state.response_times:
            return

        # Only adjust periodically
        now = time.monotonic()
        if now - state.last_adjustment_time < 60:  # Adjust at most every minute
            return

        state.last_adjustment_time = now

        # Calculate average response time
        avg_response = sum(state.response_times) / len(state.response_times)

        # If responses are very fast, we can be more aggressive
        if avg_response < 100 and state.consecutive_errors == 0:
            new_delay = max(
                state.current_delay * 0.9,
                self.config.min_delay,
            )
        # If responses are slow, back off
        elif avg_response > 2000:
            new_delay = min(
                state.current_delay * 1.2,
                self.config.max_delay,
            )
        else:
            return

        if new_delay != state.current_delay:
            self.logger.debug(
                "Adjusting delay",
                domain=state.domain,
                old_delay=state.current_delay,
                new_delay=new_delay,
                avg_response_ms=avg_response,
            )
            state.current_delay = new_delay

    def get_stats(self, domain: str) -> dict[str, Any]:
        """
        Get statistics for a domain.

        Args:
            domain: The domain.

        Returns:
            Dictionary of statistics.
        """
        state = self._get_state(domain)
        return {
            "domain": domain,
            "current_delay": self._calculate_delay(state),
            "consecutive_errors": state.consecutive_errors,
            "total_requests": state.total_requests,
            "blocked_count": state.blocked_count,
            "crawl_delay_override": state.crawl_delay_override,
        }

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """
        Get statistics for all domains.

        Returns:
            Dictionary mapping domains to their statistics.
        """
        return {
            domain: self.get_stats(domain)
            for domain in self._domain_states
        }

    async def __aenter__(self) -> "RateLimiter":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        pass


class TokenBucket:
    """
    Token bucket rate limiter for request rate control.

    Provides smooth rate limiting with burst capacity.
    """

    def __init__(
        self,
        rate: float,
        capacity: int = 1,
    ):
        """
        Initialize the token bucket.

        Args:
            rate: Token refill rate (tokens per second).
            capacity: Maximum bucket capacity.
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            Time waited in seconds.
        """
        async with self._lock:
            # Refill bucket
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

            # Wait if needed
            if self.tokens < tokens:
                wait_time = (tokens - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
                self.last_update = time.monotonic()
                return wait_time
            else:
                self.tokens -= tokens
                return 0.0
