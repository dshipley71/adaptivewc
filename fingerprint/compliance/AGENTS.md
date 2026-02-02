# AGENTS.md - Compliance Module

Complete specification for ethical web access compliance: robots.txt (RFC 9309), adaptive rate limiting, and anti-bot detection/respect.

---

## Module Purpose

The compliance module ensures ethical web access by:
- Respecting robots.txt directives (RFC 9309 compliant)
- Implementing adaptive rate limiting per domain
- Detecting and respecting anti-bot measures
- Never bypassing compliance checks

---

## Files to Generate

```
fingerprint/compliance/
├── __init__.py
├── robots_parser.py      # robots.txt parser (RFC 9309)
├── rate_limiter.py       # Adaptive rate limiting
└── bot_detector.py       # Anti-bot detection/respect
```

---

## fingerprint/compliance/__init__.py

```python
"""
Compliance module - Ethical web access.

Provides robots.txt parsing, rate limiting, and anti-bot detection.
"""

from fingerprint.compliance.robots_parser import RobotsParser, RobotsChecker
from fingerprint.compliance.rate_limiter import RateLimiter, DomainState
from fingerprint.compliance.bot_detector import BotDetector, BotCheckResult

__all__ = [
    "RobotsParser",
    "RobotsChecker",
    "RateLimiter",
    "DomainState",
    "BotDetector",
    "BotCheckResult",
]
```

---

## fingerprint/compliance/robots_parser.py

```python
"""
robots.txt parser compliant with RFC 9309.

RFC 9309: https://www.rfc-editor.org/rfc/rfc9309.html

Key features:
- Full RFC 9309 compliance
- Crawl-delay support
- Sitemap discovery
- Caching with TTL
- Wildcard pattern matching

Verbose logging pattern:
[ROBOTS:OPERATION] Message
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from fingerprint.config import Config
from fingerprint.core.verbose import get_logger
from fingerprint.exceptions import FetchError
from fingerprint.storage.cache import Cache


@dataclass
class RobotsRule:
    """Single robots.txt rule."""
    path_pattern: str
    allowed: bool
    specificity: int = 0  # Length of pattern for precedence

    def matches(self, path: str) -> bool:
        """Check if path matches this rule pattern."""
        # Convert robots.txt pattern to regex
        # * matches any sequence, $ matches end of URL
        pattern = self.path_pattern
        pattern = re.escape(pattern)
        pattern = pattern.replace(r"\*", ".*")
        pattern = pattern.replace(r"\$", "$")
        if not pattern.endswith("$"):
            pattern = f"^{pattern}"
        else:
            pattern = f"^{pattern[:-1]}$"

        return bool(re.match(pattern, path))


@dataclass
class RobotsData:
    """Parsed robots.txt data for a domain."""
    domain: str
    rules: list[RobotsRule] = field(default_factory=list)
    crawl_delay: float | None = None
    sitemaps: list[str] = field(default_factory=list)
    fetched_at: float = field(default_factory=time.time)

    # RFC 9309 status
    status_code: int = 200
    is_valid: bool = True


class RobotsParser:
    """
    RFC 9309 compliant robots.txt parser.

    Usage:
        parser = RobotsParser()
        robots_data = parser.parse(robots_txt_content, "example.com")
    """

    def __init__(self, user_agent: str = "AdaptiveFingerprint"):
        self.user_agent = user_agent.lower()
        self.logger = get_logger()

    def parse(self, content: str, domain: str) -> RobotsData:
        """
        Parse robots.txt content.

        Args:
            content: Raw robots.txt content
            domain: Domain name for logging

        Returns:
            RobotsData with parsed rules

        Verbose output:
            [ROBOTS:PARSE] Parsing robots.txt for example.com
              - rules_found: 15
              - crawl_delay: 2.0
              - sitemaps: 2
        """
        self.logger.info("ROBOTS", "PARSE", f"Parsing robots.txt for {domain}")

        data = RobotsData(domain=domain)
        current_agents: list[str] = []
        in_matching_group = False

        for line in content.split("\n"):
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Parse directive
            if ":" not in line:
                continue

            directive, value = line.split(":", 1)
            directive = directive.strip().lower()
            value = value.strip()

            if directive == "user-agent":
                agent = value.lower()
                if agent == "*" or agent == self.user_agent or self.user_agent.startswith(agent):
                    in_matching_group = True
                    current_agents.append(agent)
                else:
                    in_matching_group = False
                    current_agents = []

            elif directive == "disallow" and in_matching_group:
                if value:  # Empty disallow means allow all
                    rule = RobotsRule(
                        path_pattern=value,
                        allowed=False,
                        specificity=len(value),
                    )
                    data.rules.append(rule)

            elif directive == "allow" and in_matching_group:
                if value:
                    rule = RobotsRule(
                        path_pattern=value,
                        allowed=True,
                        specificity=len(value),
                    )
                    data.rules.append(rule)

            elif directive == "crawl-delay" and in_matching_group:
                try:
                    data.crawl_delay = float(value)
                except ValueError:
                    pass

            elif directive == "sitemap":
                if value and value not in data.sitemaps:
                    data.sitemaps.append(value)

        # Sort rules by specificity (most specific first)
        data.rules.sort(key=lambda r: r.specificity, reverse=True)

        self.logger.info(
            "ROBOTS", "PARSED",
            f"Parsed robots.txt for {domain}",
            rules_found=len(data.rules),
            crawl_delay=data.crawl_delay,
            sitemaps=len(data.sitemaps),
        )

        return data

    def is_allowed(self, robots_data: RobotsData, path: str) -> bool:
        """
        Check if path is allowed by robots.txt rules.

        Uses RFC 9309 precedence: most specific matching rule wins.
        """
        if not robots_data.is_valid:
            # If robots.txt fetch failed with 4xx, allow all
            # If failed with 5xx, be conservative and block
            return robots_data.status_code < 500

        # Find most specific matching rule
        for rule in robots_data.rules:
            if rule.matches(path):
                return rule.allowed

        # No matching rule = allowed
        return True


class RobotsChecker:
    """
    High-level robots.txt checker with caching.

    Usage:
        checker = RobotsChecker(config)
        allowed = await checker.is_allowed("https://example.com/page")
    """

    def __init__(self, config: Config):
        self.config = config
        self.parser = RobotsParser(config.http.user_agent.split("/")[0])
        self.logger = get_logger()

        # Cache robots.txt data per domain
        self._cache: Cache[RobotsData] = Cache(
            default_ttl=config.compliance.robots_txt.cache_ttl
        )

        self.logger.info(
            "ROBOTS", "INIT",
            "RobotsChecker initialized",
            cache_ttl=config.compliance.robots_txt.cache_ttl,
        )

    async def is_allowed(self, url: str) -> bool:
        """
        Check if URL is allowed by robots.txt.

        Args:
            url: Full URL to check

        Returns:
            True if allowed, False if blocked

        Verbose output:
            [ROBOTS:CHECK] Checking https://example.com/page
            [ROBOTS:ALLOWED] Path /page allowed
        """
        if not self.config.compliance.robots_txt.enabled:
            return True

        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path or "/"

        self.logger.debug("ROBOTS", "CHECK", f"Checking {url}")

        # Get or fetch robots.txt
        robots_data = await self._get_robots_data(domain)

        # Check if path is allowed
        allowed = self.parser.is_allowed(robots_data, path)

        if allowed:
            self.logger.debug("ROBOTS", "ALLOWED", f"Path {path} allowed")
        else:
            self.logger.info("ROBOTS", "BLOCKED", f"Path {path} blocked by robots.txt")

        return allowed

    async def get_crawl_delay(self, domain: str) -> float | None:
        """Get Crawl-delay for domain from robots.txt."""
        robots_data = await self._get_robots_data(domain)
        return robots_data.crawl_delay

    async def get_sitemaps(self, domain: str) -> list[str]:
        """Get sitemaps listed in robots.txt."""
        robots_data = await self._get_robots_data(domain)
        return robots_data.sitemaps

    async def _get_robots_data(self, domain: str) -> RobotsData:
        """Get robots.txt data, fetching if not cached."""
        cached = self._cache.get(domain)
        if cached:
            return cached

        # Fetch robots.txt
        robots_data = await self._fetch_robots(domain)
        self._cache.set(domain, robots_data)
        return robots_data

    async def _fetch_robots(self, domain: str) -> RobotsData:
        """Fetch and parse robots.txt for domain."""
        url = f"https://{domain}/robots.txt"

        self.logger.info("ROBOTS", "FETCH", f"Fetching robots.txt from {domain}")

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(url, follow_redirects=True)

            if response.status_code == 200:
                robots_data = self.parser.parse(response.text, domain)
                robots_data.status_code = 200
                return robots_data

            elif response.status_code in (401, 403):
                # Blocked = assume fully disallowed
                self.logger.warn(
                    "ROBOTS", "BLOCKED_ACCESS",
                    f"robots.txt access blocked ({response.status_code})",
                )
                data = RobotsData(domain=domain, status_code=response.status_code)
                data.rules.append(RobotsRule(path_pattern="/", allowed=False, specificity=1))
                return data

            elif response.status_code == 404:
                # No robots.txt = allow all
                self.logger.debug("ROBOTS", "NOT_FOUND", f"No robots.txt for {domain}")
                return RobotsData(domain=domain, status_code=404)

            else:
                # Server error = be conservative
                self.logger.warn(
                    "ROBOTS", "SERVER_ERROR",
                    f"robots.txt server error ({response.status_code})",
                )
                data = RobotsData(domain=domain, status_code=response.status_code)
                data.is_valid = False
                return data

        except Exception as e:
            self.logger.error("ROBOTS", "FETCH_ERROR", str(e))
            # Network error = be conservative, block
            data = RobotsData(domain=domain, status_code=500)
            data.is_valid = False
            return data
```

---

## fingerprint/compliance/rate_limiter.py

```python
"""
Adaptive rate limiter with per-domain tracking.

Features:
- Per-domain delay tracking
- Crawl-delay respect from robots.txt
- Automatic backoff on errors/429s
- Adaptive delay based on response times
- Token bucket algorithm

Verbose logging pattern:
[RATELIMIT:OPERATION] Message
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from fingerprint.config import Config
from fingerprint.core.verbose import get_logger
from fingerprint.exceptions import RateLimitExceededError


@dataclass
class DomainState:
    """Rate limiting state for a single domain."""
    domain: str

    # Timing
    last_request_at: float = 0.0
    current_delay: float = 1.0

    # Adaptive metrics
    avg_response_time: float = 0.0
    response_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0

    # Backoff state
    backoff_until: float = 0.0
    backoff_count: int = 0

    # From robots.txt
    crawl_delay: float | None = None


class RateLimiter:
    """
    Adaptive rate limiter with per-domain tracking.

    Usage:
        limiter = RateLimiter(config)
        await limiter.acquire("example.com")  # Blocks until slot available
        limiter.report_success("example.com", response_time=1.2)
    """

    def __init__(self, config: Config):
        self.config = config
        self.rate_config = config.compliance.rate_limiting
        self.logger = get_logger()

        # Per-domain state
        self._domains: dict[str, DomainState] = {}

        # Lock for thread safety
        self._lock = asyncio.Lock()

        self.logger.info(
            "RATELIMIT", "INIT",
            "Rate limiter initialized",
            default_delay=self.rate_config.default_delay,
            min_delay=self.rate_config.min_delay,
            max_delay=self.rate_config.max_delay,
        )

    async def acquire(self, domain: str) -> None:
        """
        Acquire a request slot for domain.

        Blocks until the rate limit allows a new request.

        Args:
            domain: Domain to acquire slot for

        Raises:
            RateLimitExceededError: If in extended backoff

        Verbose output:
            [RATELIMIT:ACQUIRE] Acquiring slot for example.com
            [RATELIMIT:WAIT] Waiting 1.5s for example.com
            [RATELIMIT:READY] Slot acquired for example.com
        """
        if not self.rate_config.enabled:
            return

        async with self._lock:
            state = self._get_or_create_state(domain)

            self.logger.debug("RATELIMIT", "ACQUIRE", f"Acquiring slot for {domain}")

            # Check if in backoff
            now = time.time()
            if state.backoff_until > now:
                wait_time = state.backoff_until - now
                if wait_time > self.rate_config.max_delay:
                    # Extended backoff = hard failure
                    raise RateLimitExceededError(domain, retry_after=wait_time)

                self.logger.info(
                    "RATELIMIT", "BACKOFF",
                    f"In backoff for {domain}",
                    wait_seconds=round(wait_time, 2),
                )
                await asyncio.sleep(wait_time)

            # Calculate required delay
            delay = self._calculate_delay(state)

            # Wait if needed
            time_since_last = now - state.last_request_at
            if time_since_last < delay:
                wait_time = delay - time_since_last
                self.logger.info(
                    "RATELIMIT", "WAIT",
                    f"Waiting {wait_time:.2f}s for {domain}",
                    delay=delay,
                )
                await asyncio.sleep(wait_time)

            # Update state
            state.last_request_at = time.time()

            self.logger.debug("RATELIMIT", "READY", f"Slot acquired for {domain}")

    def report_success(self, domain: str, response_time: float) -> None:
        """
        Report successful request for adaptive rate adjustment.

        Args:
            domain: Domain that was accessed
            response_time: Request duration in seconds

        Verbose output:
            [RATELIMIT:SUCCESS] Request succeeded for example.com
              - response_time: 0.85s
              - new_avg: 0.92s
        """
        state = self._get_or_create_state(domain)

        # Update response time average (exponential moving average)
        alpha = 0.3
        if state.response_count == 0:
            state.avg_response_time = response_time
        else:
            state.avg_response_time = alpha * response_time + (1 - alpha) * state.avg_response_time

        state.response_count += 1
        state.consecutive_errors = 0

        # Adapt delay based on response time
        if self.rate_config.adapt_to_response_time:
            self._adapt_delay(state)

        self.logger.debug(
            "RATELIMIT", "SUCCESS",
            f"Request succeeded for {domain}",
            response_time=f"{response_time:.2f}s",
            new_avg=f"{state.avg_response_time:.2f}s",
        )

    def report_error(self, domain: str, status_code: int | None = None) -> None:
        """
        Report request error for backoff adjustment.

        Args:
            domain: Domain that had error
            status_code: HTTP status code if available

        Verbose output:
            [RATELIMIT:ERROR] Request failed for example.com
              - status: 429
              - consecutive_errors: 3
        """
        state = self._get_or_create_state(domain)

        state.error_count += 1
        state.consecutive_errors += 1

        self.logger.info(
            "RATELIMIT", "ERROR",
            f"Request failed for {domain}",
            status=status_code,
            consecutive_errors=state.consecutive_errors,
        )

        # Apply backoff for rate limit errors
        if status_code in (429, 503):
            self.backoff(domain)

    def backoff(self, domain: str, retry_after: float | None = None) -> None:
        """
        Apply backoff for domain.

        Args:
            domain: Domain to backoff
            retry_after: Specific retry-after time (from header)

        Verbose output:
            [RATELIMIT:BACKOFF] Applying backoff for example.com
              - duration: 4.0s
              - backoff_count: 2
        """
        state = self._get_or_create_state(domain)
        state.backoff_count += 1

        if retry_after:
            backoff_time = retry_after
        else:
            # Exponential backoff
            backoff_time = min(
                self.rate_config.default_delay * (self.rate_config.backoff_multiplier ** state.backoff_count),
                self.rate_config.max_delay,
            )

        state.backoff_until = time.time() + backoff_time

        self.logger.info(
            "RATELIMIT", "BACKOFF",
            f"Applying backoff for {domain}",
            duration=f"{backoff_time:.1f}s",
            backoff_count=state.backoff_count,
        )

    def set_crawl_delay(self, domain: str, crawl_delay: float) -> None:
        """
        Set Crawl-delay from robots.txt.

        This takes precedence over default delay.
        """
        state = self._get_or_create_state(domain)
        state.crawl_delay = crawl_delay

        self.logger.info(
            "RATELIMIT", "CRAWL_DELAY",
            f"Set Crawl-delay for {domain}",
            delay=crawl_delay,
        )

    def get_state(self, domain: str) -> DomainState | None:
        """Get current state for domain."""
        return self._domains.get(domain)

    def get_stats(self) -> dict[str, Any]:
        """Get overall rate limiter statistics."""
        return {
            "domains_tracked": len(self._domains),
            "total_requests": sum(s.response_count for s in self._domains.values()),
            "total_errors": sum(s.error_count for s in self._domains.values()),
            "domains_in_backoff": sum(
                1 for s in self._domains.values()
                if s.backoff_until > time.time()
            ),
        }

    def _get_or_create_state(self, domain: str) -> DomainState:
        """Get or create state for domain."""
        if domain not in self._domains:
            self._domains[domain] = DomainState(
                domain=domain,
                current_delay=self.rate_config.default_delay,
            )
        return self._domains[domain]

    def _calculate_delay(self, state: DomainState) -> float:
        """Calculate delay for next request."""
        # Crawl-delay from robots.txt takes precedence
        if state.crawl_delay is not None and self.config.compliance.robots_txt.respect_crawl_delay:
            return max(state.crawl_delay, self.rate_config.min_delay)

        return state.current_delay

    def _adapt_delay(self, state: DomainState) -> None:
        """Adapt delay based on server response times."""
        # If server is slow, increase delay
        if state.avg_response_time > 2.0:
            new_delay = min(
                state.current_delay * 1.5,
                self.rate_config.max_delay,
            )
        # If server is fast and no errors, can decrease
        elif state.avg_response_time < 0.5 and state.consecutive_errors == 0:
            new_delay = max(
                state.current_delay * 0.9,
                self.rate_config.min_delay,
            )
        else:
            return

        if new_delay != state.current_delay:
            self.logger.debug(
                "RATELIMIT", "ADAPT",
                f"Adjusted delay for {state.domain}",
                old_delay=f"{state.current_delay:.2f}s",
                new_delay=f"{new_delay:.2f}s",
            )
            state.current_delay = new_delay
```

---

## fingerprint/compliance/bot_detector.py

```python
"""
Anti-bot detection and respect.

Detects when the site has identified us as a bot and responds appropriately.
We RESPECT anti-bot measures rather than trying to evade them.

Detection types:
- CAPTCHA challenges
- Block pages
- Rate limit responses
- JavaScript challenges
- Browser fingerprinting challenges

Verbose logging pattern:
[ANTIBOT:OPERATION] Message
"""

import re
from dataclasses import dataclass
from typing import Any

from fingerprint.config import Config
from fingerprint.core.verbose import get_logger
from fingerprint.exceptions import BotDetectedError, CaptchaEncounteredError


@dataclass
class BotCheckResult:
    """Result of anti-bot detection check."""
    detected: bool
    detection_type: str = ""
    confidence: float = 0.0
    details: dict[str, Any] | None = None

    @classmethod
    def clear(cls) -> "BotCheckResult":
        """No bot detection found."""
        return cls(detected=False)

    @classmethod
    def found(cls, detection_type: str, confidence: float = 1.0, **details: Any) -> "BotCheckResult":
        """Bot detection found."""
        return cls(
            detected=True,
            detection_type=detection_type,
            confidence=confidence,
            details=details or None,
        )


class BotDetector:
    """
    Detect and respect anti-bot measures.

    We do NOT try to evade detection - we respect it and stop/backoff.

    Usage:
        detector = BotDetector(config)
        result = await detector.check(response)
        if result.detected:
            # Stop or backoff
    """

    # Common CAPTCHA indicators
    CAPTCHA_PATTERNS = [
        r"captcha",
        r"recaptcha",
        r"hcaptcha",
        r"cloudflare.*challenge",
        r"please verify you are (a )?human",
        r"are you a robot",
        r"bot verification",
        r"security check",
        r"prove you.*human",
    ]

    # Common block page indicators
    BLOCK_PATTERNS = [
        r"access denied",
        r"forbidden",
        r"blocked",
        r"your ip (address )?(has been |is )blocked",
        r"too many requests",
        r"rate limit",
        r"please try again later",
        r"temporarily unavailable",
        r"suspicious activity",
        r"automated access",
    ]

    # JavaScript challenge indicators
    JS_CHALLENGE_PATTERNS = [
        r"<noscript>.*enable javascript",
        r"javascript is required",
        r"please enable javascript",
        r"checking your browser",
        r"ddos protection by",
        r"browser verification",
    ]

    def __init__(self, config: Config):
        self.config = config
        self.anti_bot_config = config.compliance.anti_bot
        self.logger = get_logger()

        # Compile patterns
        self._captcha_re = re.compile(
            "|".join(self.CAPTCHA_PATTERNS),
            re.IGNORECASE,
        )
        self._block_re = re.compile(
            "|".join(self.BLOCK_PATTERNS),
            re.IGNORECASE,
        )
        self._js_challenge_re = re.compile(
            "|".join(self.JS_CHALLENGE_PATTERNS),
            re.IGNORECASE,
        )

        self.logger.info("ANTIBOT", "INIT", "Bot detector initialized")

    async def check(self, response: Any) -> BotCheckResult:
        """
        Check response for anti-bot detection.

        Args:
            response: HTTP response object (httpx.Response)

        Returns:
            BotCheckResult indicating if bot detection was found

        Verbose output:
            [ANTIBOT:CHECK] Checking response from example.com
            [ANTIBOT:CLEAR] No bot detection found
            -- or --
            [ANTIBOT:CAPTCHA] CAPTCHA detected
              - type: recaptcha
              - confidence: 0.95
        """
        if not self.anti_bot_config.enabled:
            return BotCheckResult.clear()

        url = str(response.url) if hasattr(response, 'url') else "unknown"
        self.logger.debug("ANTIBOT", "CHECK", f"Checking response from {url}")

        # Check status code first
        status_result = self._check_status_code(response.status_code)
        if status_result.detected:
            return status_result

        # Check Retry-After header
        if self.anti_bot_config.respect_retry_after:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                self.logger.info(
                    "ANTIBOT", "RETRY_AFTER",
                    f"Retry-After header present: {retry_after}",
                )
                return BotCheckResult.found(
                    "rate_limit",
                    confidence=1.0,
                    retry_after=retry_after,
                )

        # Check content for detection patterns
        content = response.text if hasattr(response, 'text') else ""
        if content:
            content_result = self._check_content(content)
            if content_result.detected:
                return content_result

        self.logger.debug("ANTIBOT", "CLEAR", "No bot detection found")
        return BotCheckResult.clear()

    def _check_status_code(self, status_code: int) -> BotCheckResult:
        """Check HTTP status code for bot detection."""
        if status_code == 429:
            self.logger.info("ANTIBOT", "RATE_LIMITED", "HTTP 429 Too Many Requests")
            return BotCheckResult.found("rate_limit", confidence=1.0, status_code=429)

        if status_code == 403:
            self.logger.info("ANTIBOT", "FORBIDDEN", "HTTP 403 Forbidden")
            return BotCheckResult.found("blocked", confidence=0.8, status_code=403)

        if status_code == 503:
            # Could be rate limit or actual server issue
            return BotCheckResult.found("service_unavailable", confidence=0.5, status_code=503)

        return BotCheckResult.clear()

    def _check_content(self, content: str) -> BotCheckResult:
        """Check response content for bot detection patterns."""
        # Check for CAPTCHA
        if self.anti_bot_config.stop_on_captcha:
            captcha_match = self._captcha_re.search(content)
            if captcha_match:
                self.logger.info(
                    "ANTIBOT", "CAPTCHA",
                    f"CAPTCHA detected: {captcha_match.group()}",
                )
                return BotCheckResult.found(
                    "captcha",
                    confidence=0.95,
                    pattern=captcha_match.group(),
                )

        # Check for block pages
        if self.anti_bot_config.stop_on_block_page:
            block_match = self._block_re.search(content)
            if block_match:
                self.logger.info(
                    "ANTIBOT", "BLOCK_PAGE",
                    f"Block page detected: {block_match.group()}",
                )
                return BotCheckResult.found(
                    "block_page",
                    confidence=0.9,
                    pattern=block_match.group(),
                )

        # Check for JS challenges
        js_match = self._js_challenge_re.search(content)
        if js_match:
            self.logger.info(
                "ANTIBOT", "JS_CHALLENGE",
                f"JavaScript challenge detected: {js_match.group()}",
            )
            return BotCheckResult.found(
                "js_challenge",
                confidence=0.85,
                pattern=js_match.group(),
            )

        # Check for suspiciously short response that might be a challenge page
        if len(content) < 1000 and "<html" in content.lower():
            # Very short HTML page - might be a challenge
            if "cloudflare" in content.lower() or "ddos" in content.lower():
                self.logger.info(
                    "ANTIBOT", "SHORT_CHALLENGE",
                    "Suspiciously short page with protection keywords",
                )
                return BotCheckResult.found(
                    "ddos_protection",
                    confidence=0.7,
                    content_length=len(content),
                )

        return BotCheckResult.clear()

    def should_stop(self, result: BotCheckResult) -> bool:
        """Determine if we should stop crawling based on detection."""
        if not result.detected:
            return False

        # Always stop on CAPTCHA
        if result.detection_type == "captcha":
            return self.anti_bot_config.stop_on_captcha

        # Always stop on block page
        if result.detection_type == "block_page":
            return self.anti_bot_config.stop_on_block_page

        # Rate limits = backoff, don't necessarily stop
        if result.detection_type == "rate_limit":
            return False

        # JS challenges = we can't handle, should stop
        if result.detection_type == "js_challenge":
            return True

        # Default: stop if high confidence
        return result.confidence > 0.8
```

---

## Verbose Logging Examples

### robots.txt Parsing

```
[2024-01-15T10:30:00Z] [ROBOTS:INIT] RobotsChecker initialized
  - cache_ttl: 3600

[2024-01-15T10:30:01Z] [ROBOTS:FETCH] Fetching robots.txt from example.com

[2024-01-15T10:30:01Z] [ROBOTS:PARSE] Parsing robots.txt for example.com

[2024-01-15T10:30:01Z] [ROBOTS:PARSED] Parsed robots.txt for example.com
  - rules_found: 15
  - crawl_delay: 2.0
  - sitemaps: 2

[2024-01-15T10:30:02Z] [ROBOTS:CHECK] Checking https://example.com/private/data
[2024-01-15T10:30:02Z] [ROBOTS:BLOCKED] Path /private/data blocked by robots.txt
```

### Rate Limiting

```
[2024-01-15T10:30:00Z] [RATELIMIT:INIT] Rate limiter initialized
  - default_delay: 1.0
  - min_delay: 0.5
  - max_delay: 30.0

[2024-01-15T10:30:01Z] [RATELIMIT:ACQUIRE] Acquiring slot for example.com
[2024-01-15T10:30:01Z] [RATELIMIT:WAIT] Waiting 1.50s for example.com
  - delay: 1.5
[2024-01-15T10:30:02Z] [RATELIMIT:READY] Slot acquired for example.com

[2024-01-15T10:30:03Z] [RATELIMIT:SUCCESS] Request succeeded for example.com
  - response_time: 0.85s
  - new_avg: 0.92s

[2024-01-15T10:30:04Z] [RATELIMIT:ERROR] Request failed for example.com
  - status: 429
  - consecutive_errors: 1

[2024-01-15T10:30:04Z] [RATELIMIT:BACKOFF] Applying backoff for example.com
  - duration: 4.0s
  - backoff_count: 1
```

### Anti-Bot Detection

```
[2024-01-15T10:30:00Z] [ANTIBOT:INIT] Bot detector initialized

[2024-01-15T10:30:01Z] [ANTIBOT:CHECK] Checking response from example.com
[2024-01-15T10:30:01Z] [ANTIBOT:CLEAR] No bot detection found

[2024-01-15T10:30:05Z] [ANTIBOT:CHECK] Checking response from protected-site.com
[2024-01-15T10:30:05Z] [ANTIBOT:CAPTCHA] CAPTCHA detected
  - type: recaptcha
  - confidence: 0.95

[2024-01-15T10:30:10Z] [ANTIBOT:RATE_LIMITED] HTTP 429 Too Many Requests
```

---

## RFC 9309 Compliance Notes

### Key Requirements

1. **User-Agent Matching**: Match specific user-agent first, then `*`
2. **Allow/Disallow Precedence**: Most specific pattern wins
3. **Crawl-delay**: Respect per-agent crawl delay
4. **Status Codes**:
   - 200: Parse and obey
   - 3xx: Follow redirect, parse result
   - 4xx (except 429): Assume allow all
   - 429: Rate limited, must backoff
   - 5xx: Be conservative, retry later

### Pattern Matching

- `*` matches any sequence of characters
- `$` matches end of URL
- Patterns are prefix-matched unless ending with `$`
- Most specific (longest) matching pattern wins

### Example robots.txt

```
User-agent: AdaptiveFingerprint
Disallow: /private/
Disallow: /api/
Allow: /api/public/
Crawl-delay: 2

User-agent: *
Disallow: /admin/
Sitemap: https://example.com/sitemap.xml
```
