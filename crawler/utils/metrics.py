"""
Prometheus metrics for the adaptive web crawler.

Provides instrumentation for monitoring crawler health and performance.
"""

from prometheus_client import Counter, Gauge, Histogram, Info

# =============================================================================
# Crawler Info
# =============================================================================

CRAWLER_INFO = Info(
    "crawler",
    "Crawler metadata",
)

# =============================================================================
# Fetch Metrics
# =============================================================================

FETCH_TOTAL = Counter(
    "crawler_fetch_total",
    "Total number of fetch operations",
    ["status", "domain"],
)

FETCH_DURATION = Histogram(
    "crawler_fetch_duration_seconds",
    "Fetch operation duration",
    ["domain"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

FETCH_CONTENT_SIZE = Histogram(
    "crawler_fetch_content_size_bytes",
    "Size of fetched content",
    ["domain"],
    buckets=[1024, 10240, 102400, 1048576, 10485760],  # 1KB to 10MB
)

FETCH_STATUS_CODES = Counter(
    "crawler_fetch_status_code_total",
    "HTTP status codes received",
    ["status_code", "domain"],
)

# =============================================================================
# Compliance Metrics
# =============================================================================

ROBOTS_BLOCKED = Counter(
    "crawler_robots_blocked_total",
    "Number of URLs blocked by robots.txt",
    ["domain"],
)

RATE_LIMIT_WAITS = Counter(
    "crawler_rate_limit_wait_total",
    "Number of rate limit wait events",
    ["domain"],
)

RATE_LIMIT_DELAY = Histogram(
    "crawler_rate_limit_delay_seconds",
    "Rate limit delay duration",
    ["domain"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

BOT_DETECTION_BLOCKED = Counter(
    "crawler_bot_detection_blocked_total",
    "Number of requests blocked by bot detection",
    ["domain", "detection_type"],
)

# =============================================================================
# Legal Compliance Metrics
# =============================================================================

CFAA_BLOCKED = Counter(
    "crawler_cfaa_blocked_total",
    "URLs blocked due to CFAA authorization concerns",
    ["domain", "reason"],
)

PII_DETECTED = Counter(
    "crawler_pii_detected_total",
    "PII detections by type",
    ["pii_type", "action"],
)

GDPR_REDACTIONS = Counter(
    "crawler_gdpr_redactions_total",
    "Content redacted for GDPR compliance",
    ["domain"],
)

# =============================================================================
# Adaptive Extraction Metrics
# =============================================================================

STRUCTURE_CHANGES = Counter(
    "crawler_structure_change_total",
    "Number of structure changes detected",
    ["domain", "change_type", "breaking"],
)

STRATEGY_ADAPTATIONS = Counter(
    "crawler_strategy_adaptation_total",
    "Number of extraction strategy adaptations",
    ["domain", "page_type"],
)

EXTRACTION_SUCCESS = Counter(
    "crawler_extraction_total",
    "Extraction operations by result",
    ["domain", "status"],
)

EXTRACTION_CONFIDENCE = Histogram(
    "crawler_extraction_confidence",
    "Extraction confidence scores",
    ["domain", "page_type"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

EXTRACTION_DURATION = Histogram(
    "crawler_extraction_duration_ms",
    "Extraction operation duration in milliseconds",
    ["domain", "page_type"],
    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500],
)

EXTRACTION_CONTENT_LENGTH = Histogram(
    "crawler_extraction_content_length",
    "Length of extracted content",
    ["domain", "page_type", "field"],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000],
)

# =============================================================================
# Queue/Scheduler Metrics
# =============================================================================

QUEUE_SIZE = Gauge(
    "crawler_queue_size",
    "Number of URLs in the queue",
    ["domain"],
)

QUEUE_TOTAL = Gauge(
    "crawler_queue_total",
    "Total URLs in all queues",
)

DOMAINS_ACTIVE = Gauge(
    "crawler_domains_active",
    "Number of domains currently being crawled",
)

CRAWL_DEPTH = Histogram(
    "crawler_depth",
    "Crawl depth of discovered URLs",
    buckets=[1, 2, 3, 4, 5, 10, 20, 50],
)

# =============================================================================
# Storage Metrics
# =============================================================================

REDIS_OPERATIONS = Counter(
    "crawler_redis_operations_total",
    "Redis operations by type",
    ["operation", "status"],
)

REDIS_LATENCY = Histogram(
    "crawler_redis_latency_seconds",
    "Redis operation latency",
    ["operation"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)

CONTENT_STORED = Counter(
    "crawler_content_stored_total",
    "Content stored by type",
    ["content_type"],
)

# =============================================================================
# Error Metrics
# =============================================================================

ERRORS = Counter(
    "crawler_errors_total",
    "Errors by type",
    ["error_type", "domain"],
)

CIRCUIT_BREAKER_TRIPS = Counter(
    "crawler_circuit_breaker_trips_total",
    "Circuit breaker trip events",
    ["domain"],
)

RETRIES = Counter(
    "crawler_retries_total",
    "Retry attempts",
    ["domain", "reason"],
)


# =============================================================================
# Helper Functions
# =============================================================================


def record_fetch(
    domain: str,
    status: str,
    duration_seconds: float,
    content_size: int,
    status_code: int | None = None,
) -> None:
    """Record metrics for a fetch operation."""
    FETCH_TOTAL.labels(status=status, domain=domain).inc()
    FETCH_DURATION.labels(domain=domain).observe(duration_seconds)
    FETCH_CONTENT_SIZE.labels(domain=domain).observe(content_size)
    if status_code:
        FETCH_STATUS_CODES.labels(
            status_code=str(status_code), domain=domain
        ).inc()


def record_blocked(
    domain: str,
    block_type: str,
    details: dict | None = None,
) -> None:
    """Record a blocked request."""
    if block_type == "robots":
        ROBOTS_BLOCKED.labels(domain=domain).inc()
    elif block_type == "rate_limit":
        RATE_LIMIT_WAITS.labels(domain=domain).inc()
    elif block_type == "bot_detection":
        detection_type = details.get("type", "unknown") if details else "unknown"
        BOT_DETECTION_BLOCKED.labels(
            domain=domain, detection_type=detection_type
        ).inc()
    elif block_type == "cfaa":
        reason = details.get("reason", "unknown") if details else "unknown"
        CFAA_BLOCKED.labels(domain=domain, reason=reason).inc()


def record_pii_detection(
    pii_type: str,
    action: str,
) -> None:
    """Record a PII detection event."""
    PII_DETECTED.labels(pii_type=pii_type, action=action).inc()


def record_structure_change(
    domain: str,
    change_type: str,
    breaking: bool,
) -> None:
    """Record a structure change detection."""
    STRUCTURE_CHANGES.labels(
        domain=domain,
        change_type=change_type,
        breaking=str(breaking).lower(),
    ).inc()


def record_extraction(
    domain: str,
    page_type: str,
    success: bool,
    confidence: float,
    duration_ms: float = 0.0,
    title_length: int = 0,
    content_length: int = 0,
) -> None:
    """Record an extraction result."""
    status = "success" if success else "failure"
    EXTRACTION_SUCCESS.labels(domain=domain, status=status).inc()
    EXTRACTION_CONFIDENCE.labels(domain=domain, page_type=page_type).observe(
        confidence
    )

    if duration_ms > 0:
        EXTRACTION_DURATION.labels(domain=domain, page_type=page_type).observe(
            duration_ms
        )

    if title_length > 0:
        EXTRACTION_CONTENT_LENGTH.labels(
            domain=domain, page_type=page_type, field="title"
        ).observe(title_length)

    if content_length > 0:
        EXTRACTION_CONTENT_LENGTH.labels(
            domain=domain, page_type=page_type, field="content"
        ).observe(content_length)


def record_error(
    domain: str,
    error_type: str,
) -> None:
    """Record an error."""
    ERRORS.labels(error_type=error_type, domain=domain).inc()


def update_queue_metrics(
    domain: str,
    queue_size: int,
    total_size: int,
    active_domains: int,
) -> None:
    """Update queue-related gauges."""
    QUEUE_SIZE.labels(domain=domain).set(queue_size)
    QUEUE_TOTAL.set(total_size)
    DOMAINS_ACTIVE.set(active_domains)
