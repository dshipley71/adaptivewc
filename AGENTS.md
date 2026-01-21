# AGENTS.md - Adaptive Web Crawler

An intelligent, ethical web crawler with robots.txt compliance, terms of service awareness, adaptive rate limiting, and self-learning structure extraction.

## Project Overview

This adaptive web crawler is designed for responsible, large-scale web data collection. It automatically respects site policies (robots.txt, crawl-delay directives), implements user-configurable rate limiting to prevent denial of service, and adapts its behavior based on server responses.

**Key Differentiator**: The crawler learns and remembers website structures. When a site's DOM changes (iframes move, tags rename, JavaScript updates), the crawler detects these changes, adapts its extraction strategy, logs the reason for adaptation, and persists the updated structure to Redis for future crawls.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v --cov=crawler

# Basic crawl (respects all defaults)
python -m crawler --seed-url https://example.com --output ./data

# With custom rate limit
python -m crawler --seed-url https://example.com --rate-limit 2.0 --output ./data
```

## Architecture Summary

```
crawler/
├── core/
│   ├── crawler.py           # Main crawler orchestrator
│   ├── scheduler.py         # URL frontier and prioritization
│   ├── fetcher.py           # HTTP client with retry logic
│   └── state_manager.py     # Crash recovery and checkpointing
├── compliance/
│   ├── robots_parser.py     # robots.txt parsing and caching
│   ├── robots_meta.py       # Meta robots and X-Robots-Tag handling
│   ├── tos_detector.py      # Terms of service detection
│   ├── bot_detection.py     # Anti-bot measure detection
│   └── rate_limiter.py      # Adaptive rate limiting engine
├── legal/                   # ← Legal compliance (CFAA, GDPR, CCPA)
│   ├── cfaa_checker.py      # Authorization verification
│   ├── gdpr_compliance.py   # GDPR implementation
│   ├── ccpa_compliance.py   # CCPA implementation
│   ├── pii_detector.py      # PII detection and handling
│   ├── data_subject.py      # Data subject request handling
│   ├── retention.py         # Data retention enforcement
│   └── cease_desist.py      # Legal request handling
├── extraction/
│   ├── link_extractor.py    # URL discovery from HTML
│   ├── content_parser.py    # Content extraction (text, metadata)
│   ├── sitemap_parser.py    # XML sitemap processing
│   └── content_validator.py # Extraction quality validation
├── adaptive/                # ← Adaptive structure learning (see adaptive/AGENTS.md)
│   ├── structure_analyzer.py    # DOM structure fingerprinting
│   ├── change_detector.py       # Structure diff and change detection
│   ├── extraction_strategy.py   # Dynamic extraction rule engine
│   ├── strategy_learner.py      # ML-based selector inference
│   ├── change_logger.py         # Change reason documentation
│   ├── js_detector.py           # JavaScript rendering detection
│   └── url_pattern_tracker.py   # URL scheme change detection
├── deduplication/
│   ├── content_hasher.py    # SimHash/MinHash for near-duplicates
│   ├── url_canonicalizer.py # URL normalization and dedup
│   └── dedup_store.py       # Deduplication index
├── alerting/
│   ├── change_alerter.py    # Major change notifications
│   ├── failure_alerter.py   # Extraction failure alerts
│   └── channels.py          # Slack, email, webhook integrations
├── storage/
│   ├── url_store.py         # Visited URL tracking (bloom filter + DB)
│   ├── content_store.py     # Crawled content persistence
│   ├── robots_cache.py      # robots.txt cache with TTL
│   └── structure_store.py   # Redis-backed structure persistence
├── config.py                # Configuration dataclasses
├── exceptions.py            # Custom exception hierarchy
└── utils/
    ├── url_utils.py         # URL normalization and validation
    ├── metrics.py           # Crawl statistics and monitoring
    └── logging.py           # Structured logging setup
```

**Nested Documentation**: 
- See `crawler/adaptive/AGENTS.md` for adaptive extraction patterns
- See `legal/` directory for compliance documentation templates

## Key Patterns

### Compliance-First Architecture

Every URL fetch passes through the compliance pipeline before execution:

```python
async def fetch_url(self, url: str) -> FetchResult:
    # 1. Check robots.txt (cached, auto-refreshed)
    if not await self.robots_checker.is_allowed(url, self.user_agent):
        return FetchResult.blocked(url, reason="robots.txt")
    
    # 2. Apply rate limiting (per-domain)
    await self.rate_limiter.acquire(get_domain(url))
    
    # 3. Perform fetch with timeout and retry
    response = await self.http_client.get(url)
    
    # 4. Adapt rate based on response
    self.rate_limiter.adapt(get_domain(url), response)
    
    return FetchResult.success(url, response)
```

### Adaptive Extraction Pipeline

After fetching, content passes through the adaptive extraction system:

```python
async def extract_content(self, url: str, html: str) -> ExtractionResult:
    domain = get_domain(url)
    page_type = self.classifier.classify(url, html)
    
    # 1. Load stored structure from Redis (if exists)
    stored = await self.structure_store.get(domain, page_type)
    
    # 2. Analyze current page structure
    current = self.structure_analyzer.analyze(html)
    
    # 3. Detect changes and determine extraction strategy
    if stored:
        changes = self.change_detector.diff(stored.structure, current)
        if changes.has_breaking_changes:
            # Re-learn extraction strategy
            strategy = await self.strategy_learner.infer(html, stored.strategy)
            reason = self.change_logger.document(changes, stored, current)
            await self.structure_store.update(domain, page_type, current, strategy, reason)
        else:
            strategy = stored.strategy
    else:
        # First time seeing this page type - learn from scratch
        strategy = await self.strategy_learner.infer(html)
        reason = "Initial structure capture"
        await self.structure_store.save(domain, page_type, current, strategy, reason)
    
    # 4. Extract using current strategy
    return self.extraction_engine.extract(html, strategy)
```

See `crawler/adaptive/AGENTS.md` for detailed structure learning patterns.

### Rate Limiter Configuration

The rate limiter is user-configurable with sensible defaults:

```python
@dataclass
class RateLimitConfig:
    # Base delay between requests to same domain (seconds)
    default_delay: float = 1.0
    
    # Minimum delay (floor, even if site allows faster)
    min_delay: float = 0.5
    
    # Maximum delay (ceiling for backoff)
    max_delay: float = 60.0
    
    # Respect Crawl-delay directive in robots.txt
    respect_crawl_delay: bool = True
    
    # Adaptive rate limiting based on response codes
    adaptive: bool = True
    
    # Backoff multiplier on 429/503 responses
    backoff_multiplier: float = 2.0
    
    # Concurrent requests per domain
    max_concurrent_per_domain: int = 1
    
    # Global concurrent requests across all domains
    max_concurrent_global: int = 10
```

### robots.txt Compliance

```python
class RobotsChecker:
    """
    Full robots.txt compliance with caching.
    
    Supports:
    - User-agent matching (specific and wildcard)
    - Allow/Disallow directives
    - Crawl-delay directive
    - Sitemap discovery
    - Request-rate directive (non-standard but common)
    
    Cache TTL: 24 hours (configurable)
    """
    
    async def is_allowed(self, url: str, user_agent: str) -> bool:
        """Check if URL is allowed for given user-agent."""
        
    async def get_crawl_delay(self, domain: str, user_agent: str) -> float | None:
        """Get Crawl-delay directive value if specified."""
        
    async def get_sitemaps(self, domain: str) -> list[str]:
        """Extract Sitemap URLs from robots.txt."""
```

### Adaptive Rate Limiting

The crawler automatically adjusts request rates based on server feedback:

| Response Code | Behavior |
|---------------|----------|
| 200 OK | Maintain current rate |
| 429 Too Many Requests | Backoff × 2, respect Retry-After header |
| 503 Service Unavailable | Backoff × 2, respect Retry-After header |
| 5xx Server Error | Backoff × 1.5, retry with exponential delay |
| Connection Timeout | Backoff × 1.5, reduce concurrency |

```python
def adapt(self, domain: str, response: Response) -> None:
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After", self.config.max_delay)
        self.set_domain_delay(domain, max(float(retry_after), self.current_delay * 2))
        logger.warning(f"Rate limited by {domain}, backing off to {self.get_delay(domain)}s")
    elif response.status_code >= 500:
        self.set_domain_delay(domain, self.current_delay * 1.5)
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CRAWLER_USER_AGENT` | `AdaptiveCrawler/1.0 (+https://example.com/bot)` | User-agent string (include contact info) |
| `CRAWLER_DEFAULT_DELAY` | `1.0` | Base delay between requests (seconds) |
| `CRAWLER_MIN_DELAY` | `0.5` | Minimum delay floor |
| `CRAWLER_MAX_DELAY` | `60.0` | Maximum delay ceiling |
| `CRAWLER_MAX_CONCURRENT` | `10` | Global concurrent request limit |
| `CRAWLER_ROBOTS_CACHE_TTL` | `86400` | robots.txt cache TTL (seconds) |
| `CRAWLER_RESPECT_CRAWL_DELAY` | `true` | Honor Crawl-delay directives |
| `CRAWLER_RESPECT_TOS` | `true` | Check for ToS restrictions |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection for structure storage |
| `CRAWLER_STRUCTURE_TTL` | `604800` | Structure cache TTL in seconds (default: 7 days) |
| `CRAWLER_CHANGE_THRESHOLD` | `0.3` | Similarity threshold for triggering re-learning (0-1) |
| `CRAWLER_ENABLE_ADAPTIVE` | `true` | Enable adaptive structure learning |
| `GDPR_ENABLED` | `true` | Enable GDPR compliance features |
| `GDPR_RETENTION_DAYS` | `365` | Data retention period (auto-delete after) |
| `GDPR_LAWFUL_BASIS` | `legitimate_interest` | Legal basis for processing |
| `PII_HANDLING` | `redact` | PII handling: redact, pseudonymize, exclude_page |
| `CCPA_ENABLED` | `true` | Enable CCPA compliance features |
| `CCPA_HONOR_GPC` | `true` | Respect Global Privacy Control header |
| `DPO_EMAIL` | `` | Data Protection Officer contact email |
| `LEGAL_BLOCKLIST_PATH` | `/etc/crawler/legal_blocklist.txt` | Domains blocked for legal reasons |

### Command Line Arguments

```bash
python -m crawler \
    --seed-url URL              # Starting URL(s), can specify multiple
    --output DIR                # Output directory for crawled content
    --rate-limit FLOAT          # Requests per second (default: 1.0)
    --max-depth INT             # Maximum crawl depth (default: 10)
    --max-pages INT             # Maximum pages to crawl (default: unlimited)
    --user-agent STRING         # Custom user-agent
    --respect-robots BOOL       # Respect robots.txt (default: true)
    --allowed-domains LIST      # Restrict to specific domains
    --exclude-patterns LIST     # URL patterns to skip (regex)
    --timeout FLOAT             # Request timeout in seconds (default: 30)
    --retries INT               # Max retries per URL (default: 3)
```

## Compliance Requirements

### robots.txt Handling

The crawler MUST:

1. **Fetch robots.txt first** before crawling any URL on a domain
2. **Cache robots.txt** with configurable TTL (default 24h)
3. **Handle missing robots.txt** as "allow all" (per RFC 9309)
4. **Respect Crawl-delay** when `respect_crawl_delay=True` (default)
5. **Identify correctly** using a descriptive User-Agent with contact info
6. **Handle robots.txt errors gracefully**:
   - 4xx errors: Treat as "allow all"
   - 5xx errors: Retry with backoff, assume "disallow all" if persistent

```python
# Example robots.txt parsing
User-agent: AdaptiveCrawler
Disallow: /private/
Disallow: /api/
Crawl-delay: 2
Allow: /public/

User-agent: *
Disallow: /admin/
```

### Terms of Service Detection

The crawler includes optional ToS detection to identify sites that prohibit automated access:

```python
class ToSDetector:
    """
    Heuristic detection of Terms of Service restrictions.
    
    Checks:
    - /terms, /tos, /terms-of-service pages
    - Meta tags indicating bot restrictions
    - Common anti-scraping phrases in ToS content
    
    Note: This is advisory. Legal compliance is the user's responsibility.
    """
    
    RESTRICTION_PATTERNS = [
        r"prohibit.*(?:automated|bot|crawler|scraping)",
        r"may not.*(?:scrape|crawl|harvest)",
        r"automated access.*(?:forbidden|prohibited)",
    ]
```

### Anti-Bot Detection Handling

**Philosophy**: The crawler respects anti-bot measures as implicit "no crawling" signals. We do not attempt to evade detection.

```python
class BotDetectionHandler:
    """
    Detects and respects anti-bot measures.
    
    When detected:
    - Cloudflare challenge pages → Stop crawling domain, log reason
    - CAPTCHA pages → Stop crawling domain, log reason
    - 403 with bot-detection signatures → Back off, retry once, then stop
    - JavaScript challenges → Do NOT attempt to solve
    
    We treat these as the site saying "we don't want bots" and respect that.
    """
    
    DETECTION_SIGNATURES = [
        "cloudflare",
        "captcha",
        "challenge-platform",
        "access denied",
        "bot detected",
        "unusual traffic",
    ]
    
    def is_blocked(self, response: Response) -> tuple[bool, str]:
        """Check if response indicates bot detection."""
```

**What we DON'T do:**
- Solve CAPTCHAs (automated or via services)
- Rotate user agents to evade fingerprinting
- Use residential proxies to appear as regular users
- Execute JavaScript challenges designed to detect bots
- Spoof headers to bypass detection

**What we DO:**
- Log blocked domains for operator review
- Respect the implicit "no bots" signal
- Provide metrics on block rates by domain
- Allow operators to manually allowlist domains after obtaining permission

### Meta Robots and HTTP Headers

Beyond robots.txt, the crawler respects in-page directives:

```python
class RobotsDirectiveChecker:
    """
    Checks all robots directives, not just robots.txt.
    
    Sources (in priority order):
    1. X-Robots-Tag HTTP header
    2. <meta name="robots"> tag
    3. <meta name="googlebot"> (if we match this user-agent pattern)
    4. robots.txt rules
    """
    
    def check_response(self, response: Response, html: str) -> RobotsDirective:
        """
        Parse all directive sources and return combined policy.
        
        Directives honored:
        - noindex: Don't store content (but can follow links)
        - nofollow: Don't follow links on this page
        - none: Equivalent to noindex, nofollow
        - noarchive: Don't cache content
        - nosnippet: Don't use content in snippets
        """
```

### Denial of Service Prevention

The crawler implements multiple safeguards:

1. **Per-domain rate limiting**: Never exceeds configured requests/second per domain
2. **Global concurrency cap**: Limits total simultaneous connections
3. **Adaptive backoff**: Automatically reduces rate on server stress signals
4. **Circuit breaker**: Temporarily stops crawling domains returning persistent errors
5. **Resource limits**: Configurable max page size, timeout, and retry limits

```python
@dataclass
class SafetyLimits:
    max_page_size_mb: float = 10.0          # Skip pages larger than this
    request_timeout_seconds: float = 30.0    # Per-request timeout
    max_retries: int = 3                     # Retries before giving up
    circuit_breaker_threshold: int = 5       # Errors before circuit opens
    circuit_breaker_timeout: float = 300.0   # Seconds before retry after circuit opens
```

### Politeness Beyond Rate Limits

```python
@dataclass
class PolitenessConfig:
    """Advanced politeness settings for considerate crawling."""
    
    # Time-based politeness
    prefer_off_peak: bool = True              # Prefer crawling during off-peak hours
    off_peak_hours: tuple[int, int] = (1, 6)  # 1 AM - 6 AM local time
    off_peak_rate_multiplier: float = 1.5     # Can go faster during off-peak
    
    # Load-based politeness
    respect_server_timing: bool = True        # Back off if Server-Timing shows high load
    max_response_time_ms: int = 5000          # Slow responses trigger backoff
    
    # Retry politeness
    retry_respectful: bool = True             # Exponential backoff on retries
    max_retry_delay: float = 3600.0           # Cap retry delay at 1 hour
```

### Crash Recovery and State Persistence

The crawler survives crashes and restarts without losing progress:

```python
class CrawlStateManager:
    """
    Persists crawler state for recovery.
    
    State stored in Redis:
    - URL frontier (pending URLs with priorities)
    - In-progress URLs (being fetched)
    - Visited URLs (bloom filter + overflow to disk)
    - Domain states (rate limit delays, circuit breaker status)
    - Extraction progress (partial results)
    
    Recovery behavior:
    - On startup, load state from Redis
    - In-progress URLs are re-queued (may have failed mid-fetch)
    - Visited URLs prevent re-crawling
    - Domain states restore rate limit adaptations
    """
    
    async def checkpoint(self) -> None:
        """Periodic state snapshot (every 60s by default)."""
    
    async def recover(self) -> CrawlState:
        """Load state from last checkpoint."""
    
    async def mark_in_progress(self, url: str) -> None:
        """Track URL being fetched (for crash recovery)."""
    
    async def complete_url(self, url: str, result: FetchResult) -> None:
        """Move URL from in-progress to visited."""
```

**Redis keys for state:**

```
# URL frontier (sorted set by priority)
frontier:{crawl_id}                    → ZSET[(url, priority)]

# In-progress tracking
in_progress:{crawl_id}                 → SET[url]
in_progress:{crawl_id}:{url}:started   → timestamp

# Domain state
domain_state:{domain}                  → JSON(rate_delay, circuit_state, last_fetch)

# Crawl metadata
crawl:{crawl_id}:config                → JSON(CrawlConfig)
crawl:{crawl_id}:stats                 → JSON(CrawlStats)
crawl:{crawl_id}:checkpoint            → timestamp
```

## Content Deduplication

The crawler detects and handles duplicate content across URLs:

```python
class ContentDeduplicator:
    """
    Identifies duplicate and near-duplicate content.
    
    Techniques:
    - Exact hash: SHA-256 of normalized content
    - Near-duplicate: SimHash with hamming distance threshold
    - URL-based: Canonical URL detection, www/non-www, trailing slashes
    
    Dedup decisions:
    - Exact duplicate: Skip extraction, link to original
    - Near-duplicate (>95% similar): Flag for review, extract anyway
    - URL variant: Canonicalize and check against canonical
    """
    
    def compute_simhash(self, text: str) -> int:
        """64-bit SimHash for near-duplicate detection."""
    
    def is_duplicate(self, content: str, url: str) -> DedupResult:
        """Check content against dedup index."""
    
    def canonicalize_url(self, url: str) -> str:
        """
        Normalize URL for dedup comparison.
        
        - Lowercase scheme and host
        - Remove default ports
        - Sort query parameters
        - Remove tracking parameters (utm_*, fbclid, etc.)
        - Resolve www/non-www based on site preference
        - Remove trailing slashes (configurable)
        """
```

## Content Validation

Extraction results are validated before storage:

```python
class ContentValidator:
    """
    Validates extraction quality to catch failures early.
    
    Checks:
    - Required fields present (title, content at minimum)
    - Content length thresholds (not too short, not too long)
    - Language detection (matches expected language)
    - Boilerplate ratio (too much nav/footer text)
    - Entity density (proper nouns, dates suggest real content)
    """
    
    @dataclass
    class ValidationResult:
        valid: bool
        confidence: float           # 0-1, how confident in extraction quality
        issues: list[str]           # What failed validation
        suggestions: list[str]      # How to fix
    
    def validate(
        self, 
        result: ExtractionResult, 
        strategy: ExtractionStrategy
    ) -> ValidationResult:
        """Validate extraction against quality thresholds."""
    
    # Thresholds (configurable)
    MIN_TITLE_LENGTH = 10
    MAX_TITLE_LENGTH = 200
    MIN_CONTENT_LENGTH = 100
    MAX_BOILERPLATE_RATIO = 0.3     # Max 30% boilerplate
    MIN_CONFIDENCE = 0.6             # Below this, flag for review
```

## Alerting System

Operators are notified of significant events:

```python
class AlertManager:
    """
    Sends alerts for events requiring human attention.
    
    Alert channels:
    - Slack webhook
    - Email (via SMTP or SendGrid)
    - PagerDuty (for critical alerts)
    - Generic webhook (custom integrations)
    
    Alert types:
    - CRITICAL: Extraction completely failing for domain
    - WARNING: Major structure change detected
    - INFO: New page type discovered, strategy learned
    """
    
    @dataclass
    class AlertConfig:
        channels: list[AlertChannel]
        throttle_minutes: int = 60      # Don't repeat same alert within window
        aggregate: bool = True          # Batch similar alerts
        severity_filter: Severity = Severity.WARNING  # Minimum severity to alert
    
    async def alert(
        self, 
        severity: Severity,
        domain: str,
        event_type: str,
        message: str,
        context: dict
    ) -> None:
        """Send alert to configured channels."""

# Alert triggers
ALERT_TRIGGERS = {
    "extraction_failure_rate": {
        "threshold": 0.5,              # >50% failures
        "window_minutes": 30,
        "severity": Severity.CRITICAL
    },
    "major_structure_change": {
        "similarity_threshold": 0.5,   # <50% similar to previous
        "severity": Severity.WARNING
    },
    "bot_detection_blocked": {
        "severity": Severity.WARNING
    },
    "new_page_type_discovered": {
        "severity": Severity.INFO
    }
}
```

**Example alert (Slack):**

```json
{
    "severity": "WARNING",
    "domain": "example.com",
    "event": "major_structure_change",
    "message": "Major redesign detected on example.com/article pages",
    "context": {
        "similarity_score": 0.42,
        "changes_detected": 15,
        "breaking_changes": 8,
        "affected_fields": ["content", "title", "author"],
        "auto_adapted": true,
        "confidence": 0.65
    },
    "action_required": "Review extraction quality for recent articles",
    "dashboard_link": "https://crawler.internal/domains/example.com"
}
```

## Testing Requirements

### Before Committing

```bash
# Run full test suite
pytest tests/ -v --cov=crawler --cov-report=term-missing

# Type checking
mypy crawler/ --strict

# Lint
ruff check crawler/

# Integration tests (uses mock server)
pytest tests/integration/ -v
```

### Test Categories

| Category | Location | Description |
|----------|----------|-------------|
| Unit tests | `tests/unit/` | Individual component tests |
| Integration | `tests/integration/` | End-to-end crawl scenarios |
| Compliance | `tests/compliance/` | robots.txt and rate limit verification |
| Performance | `tests/performance/` | Benchmark and stress tests |

### Required Test Coverage

- `compliance/robots_parser.py`: 100% (critical path)
- `compliance/rate_limiter.py`: 100% (critical path)
- `core/fetcher.py`: ≥95%
- Overall: ≥90%

## Common Tasks

### Adding a New Extraction Plugin

1. Create extractor in `crawler/extraction/`
2. Implement the `Extractor` protocol:

```python
from typing import Protocol
from crawler.models import FetchResult, ExtractedData

class Extractor(Protocol):
    def can_handle(self, content_type: str) -> bool:
        """Return True if this extractor handles the content type."""
        ...
    
    def extract(self, result: FetchResult) -> ExtractedData:
        """Extract structured data from fetch result."""
        ...
```

3. Register in `crawler/extraction/__init__.py`
4. Add tests to `tests/unit/test_extraction/`

### Modifying Rate Limiting Behavior

1. Rate limit logic lives in `crawler/compliance/rate_limiter.py`
2. Configuration in `crawler/config.py` → `RateLimitConfig`
3. Key methods:
   - `acquire(domain)`: Wait for rate limit slot
   - `adapt(domain, response)`: Adjust rate based on response
   - `set_domain_delay(domain, delay)`: Override delay for domain

### Adding Domain-Specific Rules

```python
# In config.yaml or via API
domain_overrides:
  "api.example.com":
    delay: 5.0              # Slower rate for API endpoints
    max_concurrent: 1       # Single connection only
  "static.example.com":
    delay: 0.1              # Faster for static assets
    max_concurrent: 5       # Allow parallel downloads
```

## Monitoring and Metrics

### Exported Metrics (Prometheus Format)

```
# Request metrics
crawler_requests_total{domain, status_code}
crawler_request_duration_seconds{domain}
crawler_bytes_downloaded_total{domain}

# Compliance metrics
crawler_robots_blocked_total{domain}
crawler_rate_limited_total{domain}
crawler_crawl_delay_seconds{domain}
crawler_bot_detection_blocked_total{domain}
crawler_meta_robots_noindex_total{domain}

# Adaptive extraction metrics
crawler_structure_changes_total{domain, page_type, change_type}
crawler_strategy_relearns_total{domain, page_type}
crawler_extraction_success_rate{domain, page_type}
crawler_structure_cache_hits{domain}
crawler_structure_cache_misses{domain}
crawler_js_rendering_required_total{domain, framework}

# Content quality metrics
crawler_validation_failures_total{domain, reason}
crawler_duplicates_detected_total{domain, dedup_type}
crawler_content_length_histogram{domain, page_type}

# Legal compliance metrics
crawler_cfaa_authorization_checks_total{domain, result}
crawler_pii_detected_total{domain, pii_type, action}
crawler_gdpr_requests_total{request_type, status}
crawler_data_retention_deletions_total
crawler_cease_desist_blocks_total{domain}
crawler_legal_blocklist_size

# Health metrics
crawler_circuit_breaker_state{domain}  # 0=closed, 1=open
crawler_queue_size{priority}
crawler_active_connections
crawler_checkpoint_age_seconds
crawler_alerts_sent_total{severity, event_type}
```

### Logging

Structured JSON logging with configurable levels:

```python
# Example log entry
{
    "timestamp": "2025-01-20T10:30:00Z",
    "level": "INFO",
    "event": "fetch_complete",
    "url": "https://example.com/page",
    "domain": "example.com",
    "status_code": 200,
    "duration_ms": 245,
    "content_length": 15234,
    "rate_delay_ms": 1000
}
```

## Error Handling

### Exception Hierarchy

```python
class CrawlerError(Exception):
    """Base exception for crawler errors."""

class ComplianceError(CrawlerError):
    """Raised when compliance check fails."""

class RobotsBlockedError(ComplianceError):
    """URL blocked by robots.txt."""

class RateLimitExceededError(ComplianceError):
    """Rate limit would be exceeded."""

class FetchError(CrawlerError):
    """HTTP fetch failed."""

class TimeoutError(FetchError):
    """Request timed out."""

class CircuitOpenError(FetchError):
    """Circuit breaker is open for domain."""

class ExtractionError(CrawlerError):
    """Content extraction failed."""

class StructureChangeError(ExtractionError):
    """Page structure changed beyond adaptation capability."""

class StrategyInferenceError(ExtractionError):
    """Failed to infer extraction strategy for new structure."""

class StructureStoreError(CrawlerError):
    """Redis structure storage operation failed."""

class LegalComplianceError(CrawlerError):
    """Legal compliance requirement not met."""

class UnauthorizedAccessError(LegalComplianceError):
    """Access not authorized under CFAA analysis."""

class GDPRViolationError(LegalComplianceError):
    """Operation would violate GDPR requirements."""

class DataSubjectRequestError(LegalComplianceError):
    """Failed to process data subject request."""

class CeaseAndDesistError(LegalComplianceError):
    """Domain blocked due to legal request."""

class PIIExposureError(LegalComplianceError):
    """Sensitive PII detected, handling failed."""
```

### Retry Strategy

```python
RETRY_CONFIG = {
    "max_attempts": 3,
    "backoff_base": 1.0,
    "backoff_multiplier": 2.0,
    "backoff_max": 60.0,
    "retryable_status_codes": [408, 429, 500, 502, 503, 504],
    "retryable_exceptions": [TimeoutError, ConnectionError],
}
```

## Security Considerations

1. **URL validation**: Sanitize and validate all URLs before fetching
2. **Content limits**: Enforce max page size to prevent memory exhaustion
3. **Redirect limits**: Cap redirect chains (default: 10)
4. **Private network blocking**: Optionally block requests to private IP ranges
5. **SSL verification**: Enabled by default, configurable per-domain

```python
@dataclass  
class SecurityConfig:
    verify_ssl: bool = True
    block_private_ips: bool = True
    max_redirects: int = 10
    allowed_schemes: list[str] = field(default_factory=lambda: ["http", "https"])
```

## Proxy Configuration

For large-scale crawling, proxy support is available but governed by ethics:

```python
@dataclass
class ProxyConfig:
    """
    Proxy configuration with ethical constraints.
    
    IMPORTANT: Proxies are for legitimate purposes only:
    - Geographic distribution of requests
    - Redundancy and failover
    - Respecting per-IP rate limits at scale
    
    NOT for:
    - Evading bot detection
    - Bypassing blocks after being banned
    - Circumventing rate limits
    """
    
    enabled: bool = False
    proxy_urls: list[str] = field(default_factory=list)
    rotation_strategy: str = "round_robin"  # or "random", "least_used"
    
    # Ethical constraints (enforced)
    respect_rate_limits_per_proxy: bool = True   # Each proxy has own rate limit
    aggregate_rate_limit: bool = True            # Total rate across all proxies capped
    max_aggregate_rps: float = 10.0              # Never exceed this total
    
    # Proxy health
    health_check_interval: int = 300             # Seconds between checks
    failure_threshold: int = 3                   # Failures before removing proxy
```

**Rate limiting with proxies:**

```python
# Even with 10 proxies, we respect the site's intended limits
# If a site wants 1 req/sec, we do 1 req/sec total, not 10 req/sec

class ProxyAwareRateLimiter:
    def __init__(self, config: ProxyConfig, rate_config: RateLimitConfig):
        self.per_proxy_limiters = {
            proxy: RateLimiter(rate_config) 
            for proxy in config.proxy_urls
        }
        self.aggregate_limiter = RateLimiter(
            RateLimitConfig(default_delay=1/config.max_aggregate_rps)
        )
    
    async def acquire(self, domain: str, proxy: str) -> None:
        # Must pass BOTH limiters
        await self.per_proxy_limiters[proxy].acquire(domain)
        await self.aggregate_limiter.acquire(domain)
```

## Dependencies

### Required

- `httpx>=0.27.0` - Async HTTP client
- `beautifulsoup4>=4.12.0` - HTML parsing
- `lxml>=5.0.0` - Fast XML/HTML parser
- `pydantic>=2.0.0` - Configuration validation
- `aiosqlite>=0.20.0` - Async SQLite for URL store
- `bloom-filter2>=2.0.0` - Probabilistic URL deduplication
- `redis>=5.0.0` - Structure storage and caching (required for adaptive extraction)

### Adaptive Extraction (Required for structure learning)

- `lightgbm>=4.0.0` - Selector scoring models (CPU, no GPU needed)
- `sentence-transformers>=2.2.0` - Semantic similarity for selector adaptation
- `numpy>=1.26.0` - Numerical operations for feature extraction and similarity

### Optional

- `prometheus-client>=0.20.0` - Metrics export
- `playwright>=1.40.0` - JavaScript rendering
- `scikit-learn>=1.4.0` - Additional ML utilities (train/test split, metrics)

## Legal Compliance

### CFAA (Computer Fraud and Abuse Act) Compliance

The crawler is designed to operate within CFAA boundaries by ensuring all access is authorized:

```python
class CFAAComplianceChecker:
    """
    Ensures crawling activities comply with CFAA requirements.
    
    CFAA prohibits:
    - Accessing computers without authorization
    - Exceeding authorized access
    - Circumventing technical access controls
    
    Our approach:
    - robots.txt = explicit authorization scope
    - ToS = contractual authorization limits
    - Anti-bot measures = technical access controls (respected)
    - Rate limits = preventing "damage" through overload
    """
    
    def is_access_authorized(self, url: str, domain_context: DomainContext) -> AuthorizationResult:
        """
        Determine if accessing URL is legally authorized.
        
        Authorization sources (ALL must pass):
        1. robots.txt allows access
        2. No ToS prohibition detected (or ToS explicitly permits)
        3. No technical access controls blocking us
        4. No prior cease & desist for this domain
        5. Domain not on manual blocklist
        """

@dataclass
class AuthorizationResult:
    authorized: bool
    basis: str                      # "robots_txt", "tos_permitted", "public_access"
    restrictions: list[str]         # Any limitations on use
    documentation: str              # Human-readable authorization reasoning
    logged_at: datetime             # For audit trail
```

**CFAA-Specific Safeguards:**

| Safeguard | Implementation | CFAA Relevance |
|-----------|----------------|----------------|
| Robots.txt as authorization | Strict compliance, cached | Establishes access scope |
| ToS detection & respect | Heuristic + manual review | Contractual authorization |
| Anti-bot = access control | Never circumvent | Technical barrier respect |
| Cease & desist handling | Immediate block + legal review | Explicit revocation |
| Rate limiting | Per-domain + global | Prevents "damage" claims |
| Comprehensive logging | Every request logged | Audit trail for defense |

**Cease & Desist Protocol:**

```python
class CeaseAndDesistHandler:
    """
    Handles legal requests to stop crawling.
    
    Process:
    1. Immediately stop crawling domain
    2. Log all details for legal review
    3. Preserve evidence of prior authorization (robots.txt snapshots)
    4. Notify legal team
    5. Domain remains blocked until legal clearance
    """
    
    async def process_c_and_d(self, domain: str, notice: CeaseAndDesistNotice) -> None:
        # Immediate block
        await self.blocklist.add(domain, reason="cease_and_desist", permanent=True)
        
        # Preserve authorization evidence
        await self.evidence_store.snapshot(domain, include=[
            "robots_txt_history",
            "tos_snapshots", 
            "crawl_logs",
            "rate_limit_compliance"
        ])
        
        # Alert legal
        await self.alert_manager.alert(
            severity=Severity.CRITICAL,
            event_type="cease_and_desist",
            requires_legal_review=True
        )
```

### GDPR Compliance

The crawler implements GDPR requirements when processing data from EU sources or about EU residents:

```python
@dataclass
class GDPRConfig:
    """
    GDPR compliance configuration.
    
    Applies when:
    - Crawling EU-based websites
    - Processing data about EU residents
    - Operating from EU infrastructure
    """
    
    enabled: bool = True
    
    # Lawful basis (Article 6)
    lawful_basis: str = "legitimate_interest"  # or "consent", "contract", "legal_obligation"
    legitimate_interest_assessment: str = ""    # Document your LIA
    
    # Data minimization (Article 5(1)(c))
    collect_only: list[str] = field(default_factory=lambda: [
        "url", "title", "content", "metadata", "published_date"
    ])
    exclude_pii_patterns: bool = True           # Strip emails, phones, etc.
    
    # Storage limitation (Article 5(1)(e))
    retention_days: int = 365                   # Auto-delete after this
    retention_policy: str = "delete"            # or "anonymize"
    
    # Geographic scope
    eu_domains_only: bool = False               # Limit to EU TLDs
    process_in_eu: bool = True                  # Keep data in EU infrastructure
```

**GDPR Implementation:**

```python
class GDPRCompliance:
    """
    Implements GDPR requirements throughout the crawl pipeline.
    """
    
    # Article 5: Principles
    def apply_data_minimization(self, content: ExtractedContent) -> ExtractedContent:
        """Remove any data not strictly necessary for stated purpose."""
        
    def enforce_storage_limitation(self) -> None:
        """Scheduled job to delete/anonymize data past retention period."""
    
    # Article 12-23: Data Subject Rights
    async def handle_access_request(self, subject_id: str) -> DataAccessResponse:
        """Return all data held about a data subject."""
        
    async def handle_erasure_request(self, subject_id: str) -> ErasureConfirmation:
        """Delete all data about a data subject (Right to be Forgotten)."""
        
    async def handle_rectification_request(self, subject_id: str, corrections: dict) -> None:
        """Correct inaccurate data about a data subject."""
        
    async def handle_portability_request(self, subject_id: str) -> PortableDataPackage:
        """Export data in machine-readable format."""
    
    # Article 25: Privacy by Design
    def pseudonymize_pii(self, content: str) -> tuple[str, PseudonymMap]:
        """Replace PII with pseudonyms, maintain mapping securely."""
        
    # Article 30: Records of Processing
    def log_processing_activity(self, activity: ProcessingActivity) -> None:
        """Maintain required records of all processing activities."""
    
    # Article 33-34: Breach Notification
    async def handle_data_breach(self, breach: DataBreachInfo) -> None:
        """72-hour notification to supervisory authority if applicable."""
```

**PII Detection and Handling:**

```python
class PIIDetector:
    """
    Detects and handles personally identifiable information.
    
    Detected PII types:
    - Email addresses
    - Phone numbers
    - Physical addresses
    - Names (via NER)
    - National ID numbers (SSN, etc.)
    - Financial information
    - Health information
    - Biometric data references
    """
    
    PII_PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        # ... additional patterns
    }
    
    def detect(self, content: str) -> list[PIIMatch]:
        """Find all PII in content."""
        
    def redact(self, content: str, matches: list[PIIMatch]) -> str:
        """Replace PII with redaction markers."""
        
    def pseudonymize(self, content: str, matches: list[PIIMatch]) -> tuple[str, dict]:
        """Replace PII with consistent pseudonyms."""

@dataclass
class PIIHandlingConfig:
    action: str = "redact"          # "redact", "pseudonymize", "exclude_page", "flag_for_review"
    log_detections: bool = True     # Audit trail (without actual PII)
    alert_on_sensitive: bool = True # Alert on health/financial/biometric
```

**Data Subject Request Workflow:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Subject Request                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Request received (email to dpo@company.com)                 │
│  2. Identity verification (within 72 hours)                     │
│  3. Request type classification:                                │
│     - Access (Article 15): Search all stores, compile report    │
│     - Erasure (Article 17): Delete from all stores + backups    │
│     - Rectification (Article 16): Update incorrect data         │
│     - Portability (Article 20): Export in JSON/CSV              │
│  4. Execute request (within 30 days)                            │
│  5. Confirm completion to data subject                          │
│  6. Log for compliance records                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Additional Legal Frameworks

**CCPA (California Consumer Privacy Act):**

```python
@dataclass
class CCPAConfig:
    """California Consumer Privacy Act compliance."""
    enabled: bool = True
    
    # Right to know
    disclosure_categories: list[str] = field(default_factory=lambda: [
        "identifiers", "internet_activity", "geolocation"
    ])
    
    # Right to delete
    deletion_verification: bool = True
    
    # Right to opt-out
    honor_gpc_header: bool = True    # Global Privacy Control
    do_not_sell: bool = True         # Never sell personal info
```

**Copyright Considerations:**

```python
class CopyrightCompliance:
    """
    Respects copyright in crawled content.
    
    Approach:
    - Store for indexing/search only (transformative use)
    - Don't republish full content
    - Respect meta tags: <meta name="robots" content="noarchive">
    - Honor DMCA takedown requests
    """
    
    async def handle_dmca_takedown(self, notice: DMCANotice) -> None:
        """Process DMCA takedown request."""
        await self.content_store.remove(notice.urls)
        await self.log_dmca_action(notice)
        await self.notify_legal(notice)
```

### Compliance Documentation Requirements

The crawler maintains documentation for legal compliance:

```
legal/
├── data_processing_agreement.md      # Template DPA for clients
├── legitimate_interest_assessment.md # LIA for GDPR basis
├── privacy_impact_assessment.md      # PIA/DPIA documentation
├── data_inventory.md                 # What data we collect and why
├── retention_schedule.md             # How long we keep what
├── subprocessor_list.md              # Third parties with data access
└── incident_response_plan.md         # Breach response procedures
```

### Compliance Checklist

Before deploying the crawler, ensure:

- [ ] **Legal basis documented** (GDPR Article 6)
- [ ] **Legitimate Interest Assessment** completed if using LI basis
- [ ] **Privacy policy** updated to reflect crawling activities
- [ ] **Data Processing Agreement** in place with clients
- [ ] **Retention periods** defined and automated
- [ ] **Data subject request process** operational
- [ ] **Breach notification process** documented
- [ ] **DPO contact** published if required
- [ ] **Records of processing** maintained (Article 30)
- [ ] **Cross-border transfer mechanisms** in place (SCCs, adequacy)

## License and Ethical Use

This crawler is designed for ethical, legal web data collection. Users are responsible for:

1. Complying with applicable laws and regulations (CFAA, GDPR, CCPA, local laws)
2. Respecting website terms of service
3. Obtaining necessary legal advice for their jurisdiction
4. Using collected data appropriately
5. Maintaining reasonable crawl rates
6. Responding to abuse reports and legal requests promptly
7. Implementing appropriate data protection measures

**Disclaimer**: This documentation provides technical implementation guidance, not legal advice. Consult qualified legal counsel for your specific use case and jurisdiction.

The default configuration prioritizes compliance and politeness over speed. Adjust settings responsibly and with legal guidance.
