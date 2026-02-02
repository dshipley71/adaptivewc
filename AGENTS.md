# AGENTS.md - Adaptive Web Crawler

An intelligent, ethical web crawler with legal compliance (CFAA/GDPR/CCPA), robots.txt respect, adaptive rate limiting, and dual fingerprinting (rules-based + ML-based) for self-learning structure extraction.

## Project Overview

This adaptive web crawler is designed for responsible, large-scale web data collection. It automatically respects site policies (robots.txt, crawl-delay directives), implements user-configurable rate limiting to prevent denial of service, and adapts its behavior based on server responses.

**Key Differentiators:**
1. **Independent Fingerprinting Modes**: Three modes (Rules, ML, Adaptive) that operate independently - not combined
2. **Ollama Cloud LLM Integration**: Uses Ollama Cloud API for intelligent structure descriptions and change analysis
3. **Comprehensive Verbose Logging**: Every operation is logged with detailed context for debugging and monitoring
4. **Adaptive Learning**: When a site's DOM changes, the crawler detects changes, adapts extraction strategies, and persists to Redis

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Start Redis (required for structure storage)
docker run -d -p 6379:6379 redis:7-alpine

# Set Ollama Cloud API key
export OLLAMA_CLOUD_API_KEY="your-api-key"

# Basic crawl with verbose logging
python -m crawler --seed-url https://example.com --output ./data --verbose

# Run example with ML fingerprinting
python examples/news_monitor_ml.py --url https://news.ycombinator.com --verbose
```

## Architecture Overview

```
crawler/
├── core/                           # Orchestration & HTTP
│   ├── crawler.py                  # Main crawler orchestrator
│   ├── fetcher.py                  # HTTP client with compliance pipeline
│   ├── scheduler.py                # URL frontier and prioritization
│   ├── recrawl_scheduler.py        # Cron-based periodic recrawling
│   ├── renderer.py                 # Playwright JS rendering
│   └── distributed.py              # Multi-worker coordination
│
├── compliance/                     # Ethical crawling constraints
│   ├── robots_parser.py            # robots.txt parsing and caching
│   ├── rate_limiter.py             # Per-domain adaptive rate limiting
│   └── sitemap_parser.py           # XML sitemap processing
│
├── legal/                          # CFAA/GDPR/CCPA compliance
│   ├── cfaa_checker.py             # Computer Fraud & Abuse Act authorization
│   └── pii_detector.py             # PII detection & GDPR/CCPA handling
│
├── extraction/                     # Content parsing & learning
│   ├── content_extractor.py        # CSS selector-based extraction
│   └── link_extractor.py           # URL discovery from HTML
│
├── adaptive/                       # Structure learning (see adaptive/AGENTS.md)
│   ├── structure_analyzer.py       # Rules-based DOM fingerprinting
│   ├── change_detector.py          # Structure diff and change classification
│   └── strategy_learner.py         # CSS selector inference
│
├── ml/                             # ML & Ollama Cloud integration (see ml/AGENTS.md)
│   └── embeddings.py               # Embeddings, classifiers, LLM descriptions
│
├── storage/                        # Redis persistence layer
│   ├── structure_store.py          # Basic structure fingerprint storage
│   ├── structure_llm_store.py      # LLM-enhanced storage with embeddings
│   ├── url_store.py                # URL frontier & visited tracking
│   ├── robots_cache.py             # Cached robots.txt with TTL
│   └── factory.py                  # Storage provider factory
│
├── deduplication/
│   └── content_hasher.py           # Content deduplication (SimHash)
│
├── alerting/
│   └── alerter.py                  # Change & failure notifications
│
├── utils/
│   ├── url_utils.py                # URL normalization and validation
│   ├── metrics.py                  # Prometheus metrics
│   └── logging.py                  # Structured verbose logging
│
├── config.py                       # Configuration dataclasses
├── models.py                       # Core data models (30+ dataclasses)
├── exceptions.py                   # Custom exception hierarchy
└── __main__.py                     # CLI entry point

examples/                           # Reference implementations
├── news_monitor_ml.py              # ML fingerprinting with Ollama Cloud
├── sports_news_monitor_ml.py       # Sports monitoring with embeddings
├── demo_adaptive_extraction.py     # Strategy learning demonstration
├── scheduled_recrawl_example.py    # Cron-based recrawling
├── js_rendering_example.py         # Playwright integration
├── sitemap_crawl_example.py        # Sitemap-based crawling
└── distributed_crawl_example.py    # Multi-worker coordination
```

---

## Module Documentation

### 1. Core Module (`crawler/core/`)

#### 1.1 Crawler Orchestrator (`crawler.py`)

The main entry point that coordinates all crawling operations.

```python
class Crawler:
    """
    Main crawler orchestrator.

    Responsibilities:
    - Initialize all subsystems (fetcher, analyzer, extractor)
    - Manage URL frontier via scheduler
    - Coordinate compliance pipeline
    - Handle graceful shutdown and checkpointing

    Verbose Logging:
    - [CRAWLER:INIT] Configuration loaded, subsystems initialized
    - [CRAWLER:START] Crawl session started with seed URLs
    - [CRAWLER:URL] Processing URL with priority and depth
    - [CRAWLER:FETCH] Fetch result with status, timing, size
    - [CRAWLER:EXTRACT] Extraction result with field counts
    - [CRAWLER:CHANGE] Structure change detected with similarity score
    - [CRAWLER:CHECKPOINT] State checkpointed to Redis
    - [CRAWLER:COMPLETE] Crawl statistics summary
    """

    async def crawl(self, seed_urls: list[str]) -> CrawlResult:
        """Execute crawl from seed URLs."""

    async def process_url(self, url: str) -> ProcessResult:
        """Process single URL through full pipeline."""
```

**Verbose Output Example:**
```
[CRAWLER:INIT] Initializing crawler with config:
  - Rate limit: 1.0 req/sec per domain
  - Max depth: 10
  - Fingerprinting: rules-based + ML (Ollama Cloud)
  - Redis: redis://localhost:6379/0

[CRAWLER:URL] Processing: https://example.com/article/123
  - Priority: 0.8
  - Depth: 2
  - Domain queue size: 15

[CRAWLER:FETCH] Fetch complete:
  - Status: 200 OK
  - Content-Type: text/html
  - Size: 45,230 bytes
  - Duration: 245ms
  - Rate delay applied: 1,000ms

[CRAWLER:CHANGE] Structure change detected:
  - Domain: example.com
  - Page type: article
  - Similarity: 0.72 (MODERATE change)
  - Breaking: Yes
  - Action: Re-learning extraction strategy
```

#### 1.2 Fetcher (`fetcher.py`)

HTTP client with full compliance pipeline.

```python
class Fetcher:
    """
    HTTP fetcher with compliance-first architecture.

    Pipeline Order (ALWAYS this sequence):
    1. CFAA authorization check
    2. robots.txt check
    3. Rate limiting acquire
    4. HTTP fetch with timeout/retry
    5. GDPR PII processing
    6. Rate adaptation based on response

    Verbose Logging:
    - [FETCHER:CFAA] Authorization check result
    - [FETCHER:ROBOTS] robots.txt check result
    - [FETCHER:RATE] Rate limit acquired, delay applied
    - [FETCHER:HTTP] Request sent, response received
    - [FETCHER:PII] PII detection and handling
    - [FETCHER:ADAPT] Rate adaptation from response
    """
```

**Compliance Pipeline:**
```python
async def fetch_url(self, url: str) -> FetchResult:
    domain = get_domain(url)

    # 1. CFAA authorization check
    self._log_verbose(f"[FETCHER:CFAA] Checking authorization for {url}")
    auth = await self.cfaa_checker.is_authorized(url)
    if not auth.authorized:
        self._log_verbose(f"[FETCHER:CFAA] BLOCKED: {auth.reason}")
        return FetchResult.blocked(url, reason=auth.reason)
    self._log_verbose(f"[FETCHER:CFAA] AUTHORIZED: {auth.basis}")

    # 2. robots.txt check
    self._log_verbose(f"[FETCHER:ROBOTS] Checking robots.txt for {url}")
    if not await self.robots_checker.is_allowed(url, self.user_agent):
        self._log_verbose(f"[FETCHER:ROBOTS] BLOCKED by robots.txt")
        return FetchResult.blocked(url, reason="robots.txt")
    self._log_verbose(f"[FETCHER:ROBOTS] ALLOWED")

    # 3. Rate limiting
    delay = await self.rate_limiter.get_delay(domain)
    self._log_verbose(f"[FETCHER:RATE] Acquiring rate limit slot, delay={delay}s")
    await self.rate_limiter.acquire(domain)
    self._log_verbose(f"[FETCHER:RATE] Slot acquired")

    # 4. HTTP fetch
    self._log_verbose(f"[FETCHER:HTTP] Sending GET request to {url}")
    start_time = time.monotonic()
    response = await self.http_client.get(url, timeout=self.config.timeout)
    duration = time.monotonic() - start_time
    self._log_verbose(f"[FETCHER:HTTP] Response: {response.status_code}, "
                      f"size={len(response.content)} bytes, duration={duration*1000:.0f}ms")

    # 5. GDPR PII check
    if self.gdpr_config.enabled:
        self._log_verbose(f"[FETCHER:PII] Scanning for PII...")
        response = await self.pii_handler.process(response)
        self._log_verbose(f"[FETCHER:PII] Scan complete, action={self.pii_handler.action}")

    # 6. Rate adaptation
    self._log_verbose(f"[FETCHER:ADAPT] Adapting rate based on response")
    self.rate_limiter.adapt(domain, response)

    return FetchResult.success(url, response)
```

#### 1.3 Scheduler (`scheduler.py`)

URL frontier management with priority queuing.

```python
class Scheduler:
    """
    URL frontier with priority-based scheduling.

    Features:
    - Priority queue (higher = more important)
    - Per-domain queuing to respect rate limits
    - Depth tracking for crawl limits
    - Deduplication via visited set

    Verbose Logging:
    - [SCHEDULER:ADD] URL added with priority and depth
    - [SCHEDULER:SKIP] URL skipped (visited, blocked, depth exceeded)
    - [SCHEDULER:NEXT] Next URL selected from frontier
    - [SCHEDULER:STATS] Queue statistics
    """
```

#### 1.4 Recrawl Scheduler (`recrawl_scheduler.py`)

Cron-based periodic recrawling.

```python
class RecrawlScheduler:
    """
    Scheduled recrawling with cron expressions.

    Features:
    - Cron expression parsing (e.g., "0 */6 * * *" = every 6 hours)
    - Adaptive intervals based on change frequency
    - Priority boosting for frequently changing pages

    Verbose Logging:
    - [RECRAWL:SCHEDULE] URL scheduled with cron expression
    - [RECRAWL:DUE] URL due for recrawl
    - [RECRAWL:ADAPTIVE] Interval adjusted based on change history
    """
```

#### 1.5 Renderer (`renderer.py`)

Playwright-based JavaScript rendering.

```python
class JSRenderer:
    """
    JavaScript rendering via Playwright.

    Used when:
    - JS detection score > 0.8
    - Static extraction fails
    - Framework detected (React, Vue, Angular, etc.)

    Verbose Logging:
    - [RENDERER:DETECT] JS rendering needed, score={score}, framework={framework}
    - [RENDERER:LAUNCH] Browser launched
    - [RENDERER:NAVIGATE] Navigating to URL
    - [RENDERER:WAIT] Waiting for selectors/network idle
    - [RENDERER:CAPTURE] DOM captured after JS execution
    - [RENDERER:COMPARE] Static vs JS content size comparison
    """
```

#### 1.6 Distributed Coordinator (`distributed.py`)

Multi-worker coordination via Redis.

```python
class DistributedCoordinator:
    """
    Coordinates multiple crawler workers.

    Features:
    - Worker registration and heartbeat
    - Domain assignment to prevent conflicts
    - Load balancing across workers
    - Failure detection and reassignment

    Verbose Logging:
    - [DISTRIBUTED:REGISTER] Worker registered with ID
    - [DISTRIBUTED:HEARTBEAT] Heartbeat sent/received
    - [DISTRIBUTED:ASSIGN] Domain assigned to worker
    - [DISTRIBUTED:REBALANCE] Load rebalanced across workers
    """
```

---

### 2. Compliance Module (`crawler/compliance/`)

#### 2.1 Robots Parser (`robots_parser.py`)

Full robots.txt compliance with caching.

```python
class RobotsChecker:
    """
    robots.txt parsing and compliance checking.

    Supports:
    - User-agent matching (specific and wildcard)
    - Allow/Disallow directives
    - Crawl-delay directive
    - Sitemap discovery
    - Request-rate directive (non-standard)

    Verbose Logging:
    - [ROBOTS:FETCH] Fetching robots.txt for domain
    - [ROBOTS:CACHE] Cache hit/miss for domain
    - [ROBOTS:PARSE] Parsing robots.txt content
    - [ROBOTS:MATCH] User-agent matching result
    - [ROBOTS:CHECK] URL allowed/disallowed with rule matched
    - [ROBOTS:DELAY] Crawl-delay extracted
    """

    async def is_allowed(self, url: str, user_agent: str) -> bool:
        """Check if URL is allowed for given user-agent."""

    async def get_crawl_delay(self, domain: str, user_agent: str) -> float | None:
        """Get Crawl-delay directive value if specified."""

    async def get_sitemaps(self, domain: str) -> list[str]:
        """Extract Sitemap URLs from robots.txt."""
```

**Verbose Output Example:**
```
[ROBOTS:FETCH] Fetching robots.txt for example.com
[ROBOTS:CACHE] Cache MISS, fetching fresh
[ROBOTS:PARSE] Parsing robots.txt (1,245 bytes)
  - Found 3 user-agent groups
  - Crawl-delay: 2s for AdaptiveCrawler
  - Sitemaps: 2 found
[ROBOTS:MATCH] Matching user-agent 'AdaptiveCrawler'
  - Matched group: 'AdaptiveCrawler' (exact)
[ROBOTS:CHECK] URL /article/123
  - Rule matched: Allow /article/
  - Result: ALLOWED
```

#### 2.2 Rate Limiter (`rate_limiter.py`)

Adaptive per-domain rate limiting.

```python
class RateLimiter:
    """
    Per-domain rate limiting with adaptive backoff.

    Features:
    - Configurable base delay
    - Respects Crawl-delay from robots.txt
    - Adaptive backoff on 429/5xx responses
    - Per-domain state tracking

    Verbose Logging:
    - [RATE:INIT] Rate limiter initialized with config
    - [RATE:STATE] Domain state: current_delay, last_request, backoff_count
    - [RATE:ACQUIRE] Acquiring slot, waiting {delay}s
    - [RATE:ADAPT] Adapting rate: {old_delay} -> {new_delay}, reason={reason}
    - [RATE:BACKOFF] Backoff triggered: {status_code}, multiplier={mult}
    - [RATE:RECOVER] Rate recovered to baseline
    """

    Adaptation Rules:
    | Response Code | Action |
    |---------------|--------|
    | 200 OK        | Maintain rate, gradual recovery |
    | 429 Too Many  | Backoff x2, respect Retry-After |
    | 503 Unavailable | Backoff x2, respect Retry-After |
    | 5xx Error     | Backoff x1.5 |
    | Timeout       | Backoff x1.5, reduce concurrency |
```

**Verbose Output Example:**
```
[RATE:STATE] Domain: api.example.com
  - Current delay: 2.0s
  - Last request: 1.5s ago
  - Backoff count: 1
  - Circuit: CLOSED

[RATE:ACQUIRE] Acquiring slot for api.example.com
  - Need to wait: 0.5s (2.0s delay - 1.5s elapsed)
  - Waiting...

[RATE:ADAPT] Response 429 from api.example.com
  - Retry-After header: 30s
  - Old delay: 2.0s
  - New delay: 30.0s (from Retry-After)
  - Backoff count: 1 -> 2
```

#### 2.3 Sitemap Parser (`sitemap_parser.py`)

XML sitemap processing.

```python
class SitemapParser:
    """
    XML sitemap parsing and URL extraction.

    Supports:
    - Standard sitemaps (urlset)
    - Sitemap index files (sitemapindex)
    - Gzip compressed sitemaps
    - lastmod, changefreq, priority attributes

    Verbose Logging:
    - [SITEMAP:FETCH] Fetching sitemap from URL
    - [SITEMAP:TYPE] Sitemap type detected (urlset/index)
    - [SITEMAP:PARSE] Parsing sitemap, found {count} URLs
    - [SITEMAP:INDEX] Processing sitemap index, found {count} child sitemaps
    - [SITEMAP:URL] URL extracted with metadata
    """
```

---

### 3. Legal Module (`crawler/legal/`)

#### 3.1 CFAA Checker (`cfaa_checker.py`)

Computer Fraud and Abuse Act authorization verification.

```python
class CFAAChecker:
    """
    CFAA compliance checker.

    Authorization Sources (ALL must pass):
    1. robots.txt allows access
    2. No ToS prohibition detected
    3. No technical access controls blocking
    4. No prior cease & desist
    5. Domain not on manual blocklist

    Verbose Logging:
    - [CFAA:CHECK] Checking authorization for URL
    - [CFAA:ROBOTS] robots.txt authorization status
    - [CFAA:TOS] Terms of Service analysis result
    - [CFAA:BLOCK] Access blocked, reason={reason}
    - [CFAA:AUTH] Access authorized, basis={basis}
    """

    async def is_authorized(self, url: str) -> AuthorizationResult:
        """Check if accessing URL is legally authorized."""

    async def analyze_tos(self, domain: str) -> ToSAnalysisResult:
        """Analyze Terms of Service for crawling restrictions."""
```

**ToS Analysis Output:**
```
[CFAA:TOS] Analyzing Terms of Service for example.com
  - ToS page found: /terms
  - Analyzing content (5,230 words)
  - Restriction patterns found: 0
  - Bot-specific clauses: None
  - Result: PERMITTED (no restrictions detected)
  - Confidence: 0.85
```

#### 3.2 PII Detector (`pii_detector.py`)

GDPR/CCPA compliant PII detection and handling.

```python
class PIIDetector:
    """
    Personally Identifiable Information detection.

    Detected PII Types:
    - Email addresses
    - Phone numbers
    - Physical addresses
    - National ID numbers (SSN, etc.)
    - Financial information
    - Health information

    Actions:
    - redact: Replace with [REDACTED]
    - pseudonymize: Replace with consistent pseudonyms
    - exclude_page: Skip entire page
    - flag_for_review: Store but flag

    Verbose Logging:
    - [PII:SCAN] Scanning content for PII
    - [PII:DETECT] PII detected: type={type}, count={count}
    - [PII:ACTION] Action taken: {action}
    - [PII:REDACT] Redacted {count} occurrences
    - [PII:ALERT] Sensitive PII alert sent
    """
```

---

### 4. Extraction Module (`crawler/extraction/`)

#### 4.1 Content Extractor (`content_extractor.py`)

CSS selector-based content extraction.

```python
class ContentExtractor:
    """
    Extracts content using learned extraction strategies.

    Fields Extracted:
    - title: Page title
    - content: Main content body
    - metadata: author, date, tags, etc.
    - images: Image URLs with alt text
    - links: Outbound links

    Verbose Logging:
    - [EXTRACT:STRATEGY] Loading strategy for domain/page_type
    - [EXTRACT:SELECTOR] Applying selector: {selector}
    - [EXTRACT:MATCH] Selector matched {count} elements
    - [EXTRACT:FALLBACK] Primary failed, trying fallback {n}
    - [EXTRACT:FIELD] Field extracted: {field}={value[:50]}
    - [EXTRACT:VALIDATE] Validation result: {valid}, confidence={conf}
    """
```

#### 4.2 Link Extractor (`link_extractor.py`)

URL discovery from HTML.

```python
class LinkExtractor:
    """
    Extracts and normalizes URLs from HTML.

    Features:
    - Absolute URL resolution
    - URL normalization and canonicalization
    - Domain filtering (allow/block lists)
    - Duplicate detection

    Verbose Logging:
    - [LINKS:EXTRACT] Extracting links from page
    - [LINKS:FOUND] Found {count} raw links
    - [LINKS:NORMALIZE] Normalizing URL: {raw} -> {normalized}
    - [LINKS:FILTER] Filtered {count} links (reason: {reason})
    - [LINKS:DEDUP] Removed {count} duplicates
    - [LINKS:RESULT] Yielding {count} unique links
    """
```

---

### 5. Adaptive Module (`crawler/adaptive/`)

See `crawler/adaptive/AGENTS.md` for detailed documentation.

#### 5.1 Structure Analyzer (`structure_analyzer.py`)

Rules-based DOM fingerprinting.

```python
class StructureAnalyzer:
    """
    Creates DOM fingerprints using rules-based analysis.

    Fingerprint Components:
    - tag_hierarchy: Nested tag structure with counts
    - css_class_map: Class names and frequencies
    - id_attributes: All ID attributes
    - semantic_landmarks: HTML5 semantics and ARIA roles
    - iframe_locations: iframe details
    - script_signatures: JS framework detection
    - content_regions: Main content boundaries
    - navigation_selectors: Nav element selectors
    - pagination_pattern: Pagination detection

    Verbose Logging:
    - [STRUCTURE:ANALYZE] Analyzing HTML ({size} bytes)
    - [STRUCTURE:TAGS] Tag distribution: {top_5_tags}
    - [STRUCTURE:CLASSES] Found {count} unique CSS classes
    - [STRUCTURE:LANDMARKS] Semantic landmarks: {landmarks}
    - [STRUCTURE:IFRAMES] Found {count} iframes
    - [STRUCTURE:SCRIPTS] Framework detected: {framework}
    - [STRUCTURE:REGIONS] Content regions identified: {regions}
    - [STRUCTURE:HASH] Structure hash: {hash}
    """
```

#### 5.2 Change Detector (`change_detector.py`)

Structure comparison and change classification.

```python
class ChangeDetector:
    """
    Detects and classifies structure changes.

    Change Classification:
    - COSMETIC (>95% similar): Minor style changes
    - MINOR (85-95%): Non-breaking layout adjustments
    - MODERATE (70-85%): Significant changes, may need adaptation
    - BREAKING (<70%): Major redesign, requires re-learning

    Change Types (14 types):
    - iframe_added, iframe_removed, iframe_relocated, iframe_resized
    - tag_renamed, class_renamed, id_changed
    - structure_reorganized, content_relocated
    - script_added, script_removed, script_modified
    - url_pattern_changed, pagination_changed
    - minor_layout_shift, major_redesign

    Verbose Logging:
    - [CHANGE:COMPARE] Comparing structures (v{old} vs v{new})
    - [CHANGE:SIMILARITY] Overall similarity: {score}
    - [CHANGE:CLASSIFY] Classification: {type}
    - [CHANGE:DETAILS] Changes detected:
        - {change_type}: {description}
    - [CHANGE:BREAKING] Breaking change: {yes/no}
    - [CHANGE:FIELDS] Affected fields: {fields}
    """
```

#### 5.3 Strategy Learner (`strategy_learner.py`)

CSS selector inference and adaptation.

```python
class StrategyLearner:
    """
    Learns extraction strategies from page structure.

    Learning Priority:
    1. Semantic HTML: <article>, <main>, <h1>
    2. ARIA landmarks: role="main", aria-label
    3. Schema.org: itemtype, itemprop
    4. Common patterns: .content, .article, .post
    5. Structural heuristics: largest text block

    Verbose Logging:
    - [LEARN:START] Learning strategy for {domain}/{page_type}
    - [LEARN:SEMANTIC] Checking semantic HTML elements
    - [LEARN:ARIA] Checking ARIA landmarks
    - [LEARN:SCHEMA] Checking Schema.org markup
    - [LEARN:HEURISTIC] Applying heuristics
    - [LEARN:CANDIDATE] Candidate selector: {selector}, confidence={conf}
    - [LEARN:SELECT] Selected: {field} -> {selector} (confidence={conf})
    - [LEARN:FALLBACK] Generated {count} fallback selectors
    - [LEARN:COMPLETE] Strategy learned with {field_count} fields
    """
```

---

### 6. ML Module (`crawler/ml/`)

See `crawler/ml/AGENTS.md` for detailed documentation.

#### 6.1 Embeddings (`embeddings.py`)

ML models and Ollama Cloud LLM integration.

```python
class StructureEmbeddingModel:
    """
    Generates semantic embeddings for page structures.

    Model: all-MiniLM-L6-v2 (384 dimensions)

    Verbose Logging:
    - [EMBED:INIT] Model loaded: {model_name}
    - [EMBED:DESCRIBE] Generating description for structure
    - [EMBED:ENCODE] Encoding description to embedding
    - [EMBED:DIMS] Embedding shape: {dims}
    - [EMBED:NORM] Embedding L2 norm: {norm}
    """

class LLMDescriptionGenerator:
    """
    Generates structure descriptions using Ollama Cloud.

    Provider: Ollama Cloud only
    Model: Configurable (default: gemma3:12b)
    Endpoint: https://ollama.com/api/chat

    Verbose Logging:
    - [LLM:INIT] Ollama Cloud initialized, model={model}
    - [LLM:PROMPT] Generating prompt for structure
    - [LLM:REQUEST] Sending request to Ollama Cloud API
    - [LLM:RESPONSE] Response received ({length} chars)
    - [LLM:DESCRIPTION] Generated description: {description}
    """

class MLChangeDetector:
    """
    ML-based change detection using embeddings.

    Features:
    - Cosine similarity between embeddings
    - Site baseline drift detection
    - Change impact classification

    Verbose Logging:
    - [ML:CHANGE] Comparing structures via embeddings
    - [ML:SIMILARITY] Cosine similarity: {score}
    - [ML:THRESHOLD] Breaking threshold: {threshold}
    - [ML:BREAKING] Is breaking: {yes/no}
    - [ML:IMPACT] Predicted impact: {impact}
    """

class StructureClassifier:
    """
    Page type classification using ML models.

    Backends: LogisticRegression, XGBoost, LightGBM

    Verbose Logging:
    - [CLASSIFY:INIT] Classifier loaded: {backend}
    - [CLASSIFY:FEATURES] Extracting features from structure
    - [CLASSIFY:PREDICT] Predicting page type
    - [CLASSIFY:RESULT] Prediction: {type}, confidence={conf}
    """
```

---

### 7. Storage Module (`crawler/storage/`)

#### 7.1 Structure Store (`structure_store.py`)

Basic Redis-backed structure storage.

```python
class StructureStore:
    """
    Redis storage for page structures and strategies.

    Redis Key Patterns:
    - crawler:structure:{domain}:{page_type}:{variant_id}
    - crawler:strategy:{domain}:{page_type}:{variant_id}
    - crawler:variants:{domain}:{page_type}
    - crawler:changes:{domain}:{page_type}

    Verbose Logging:
    - [STORE:GET] Getting structure for {domain}/{page_type}
    - [STORE:HIT] Cache HIT, version={version}
    - [STORE:MISS] Cache MISS
    - [STORE:SAVE] Saving structure, version={version}
    - [STORE:UPDATE] Updating structure, {old_version} -> {new_version}
    - [STORE:VARIANT] Variant tracking: {variant_id}
    - [STORE:HISTORY] Storing in history, keeping {max_versions} versions
    """
```

#### 7.2 LLM Structure Store (`structure_llm_store.py`)

Enhanced storage with Ollama Cloud descriptions and embeddings.

```python
class LLMStructureStore:
    """
    Structure storage enhanced with Ollama Cloud LLM descriptions.

    Additional Data Stored:
    - LLM-generated description
    - Semantic embedding vector
    - Description generation timestamp

    Redis Key Patterns:
    - crawler:structure_llm:{domain}:{page_type}:{variant_id}
    - crawler:embedding:{domain}:{page_type}:{variant_id}

    Verbose Logging:
    - [LLM_STORE:SAVE] Saving structure with LLM description
    - [LLM_STORE:DESCRIBE] Generating LLM description via Ollama Cloud
    - [LLM_STORE:EMBED] Generating embedding from description
    - [LLM_STORE:GET] Getting structure with description
    - [LLM_STORE:COMPARE] Comparing via embeddings, similarity={score}
    """
```

#### 7.3 URL Store (`url_store.py`)

URL frontier and visited tracking.

```python
class URLStore:
    """
    Redis-backed URL frontier management.

    Features:
    - Priority queue (sorted set)
    - Visited URL tracking (set)
    - In-progress tracking for crash recovery

    Redis Key Patterns:
    - crawler:url_frontier:{priority}
    - crawler:url_visited
    - crawler:url_in_progress

    Verbose Logging:
    - [URL:ADD] Adding URL to frontier, priority={priority}
    - [URL:SKIP] Skipping URL (visited/duplicate)
    - [URL:NEXT] Popping next URL from frontier
    - [URL:PROGRESS] Marking URL in-progress
    - [URL:COMPLETE] Marking URL complete
    - [URL:STATS] Frontier size={size}, visited={visited}
    """
```

#### 7.4 Robots Cache (`robots_cache.py`)

Cached robots.txt with TTL.

```python
class RobotsCache:
    """
    Redis-cached robots.txt files.

    Features:
    - Configurable TTL (default: 24 hours)
    - Automatic refresh on expiry
    - Fallback handling for fetch failures

    Redis Key Pattern:
    - crawler:robots:{domain}

    Verbose Logging:
    - [ROBOTS_CACHE:GET] Getting robots.txt for {domain}
    - [ROBOTS_CACHE:HIT] Cache HIT, age={age}s
    - [ROBOTS_CACHE:MISS] Cache MISS, fetching fresh
    - [ROBOTS_CACHE:STORE] Storing robots.txt, TTL={ttl}s
    - [ROBOTS_CACHE:EXPIRE] Cache expired for {domain}
    """
```

---

### 8. Alerting Module (`crawler/alerting/`)

#### 8.1 Alerter (`alerter.py`)

Change and failure notifications.

```python
class Alerter:
    """
    Alert management for significant events.

    Alert Types:
    - CRITICAL: Extraction completely failing
    - WARNING: Major structure change detected
    - INFO: New page type discovered

    Channels:
    - Slack webhook
    - Email (SMTP)
    - Generic webhook

    Verbose Logging:
    - [ALERT:TRIGGER] Alert triggered: {type}, severity={severity}
    - [ALERT:THROTTLE] Alert throttled (sent {ago}s ago)
    - [ALERT:SEND] Sending alert to {channel}
    - [ALERT:SUCCESS] Alert sent successfully
    - [ALERT:FAIL] Alert send failed: {error}
    """
```

---

### 9. Configuration (`crawler/config.py`)

Comprehensive configuration dataclasses.

```python
# Main configuration structure
CrawlerSettings          # Environment variable loading (CRAWLER_* prefix)
CrawlConfig              # Main YAML configuration
├── RateLimitConfig      # Rate limiting settings
├── GDPRConfig           # GDPR compliance settings
├── CFAAConfig           # CFAA compliance settings
├── CCPAConfig           # CCPA compliance settings
├── PIIHandlingConfig    # PII detection settings
├── StructureStoreConfig # Storage settings
│   └── LLMConfig        # Ollama Cloud settings
├── SecurityConfig       # Security constraints
└── PolitenessConfig     # Advanced politeness

# Verbose logging config
@dataclass
class VerboseConfig:
    """
    Verbose logging configuration.

    Levels:
    - 0: Errors only
    - 1: Warnings + Errors
    - 2: Info + Warnings + Errors
    - 3: Debug (all verbose logging)

    Module-specific overrides available.
    """
    level: int = 2
    modules: dict[str, int] = field(default_factory=dict)
    format: str = "structured"  # or "plain"
    include_timestamp: bool = True
    include_context: bool = True
```

---

## Ollama Cloud Integration

### Configuration

```yaml
# config.yaml
structure_store:
  store_type: llm

  llm:
    provider: ollama-cloud
    model: gemma3:12b  # or llama3.2, mistral, etc.
    api_key: ${OLLAMA_CLOUD_API_KEY}

    # Request settings
    timeout: 30
    max_retries: 3

    # Generation settings
    temperature: 0.3
    max_tokens: 500
```

### Environment Variables

```bash
# Required for Ollama Cloud
export OLLAMA_CLOUD_API_KEY="your-api-key"

# Optional overrides
export OLLAMA_CLOUD_MODEL="gemma3:12b"
export OLLAMA_CLOUD_TIMEOUT="30"
```

### API Integration

```python
class OllamaCloudClient:
    """
    Ollama Cloud API client.

    Endpoint: https://ollama.com/api/chat
    Authentication: Bearer token in Authorization header

    Request Format:
    {
        "model": "gemma3:12b",
        "messages": [{"role": "user", "content": "..."}],
        "stream": false,
        "options": {
            "num_predict": 500,
            "temperature": 0.3
        }
    }

    Response Format:
    {
        "message": {
            "role": "assistant",
            "content": "Generated description..."
        }
    }

    Verbose Logging:
    - [OLLAMA:INIT] Client initialized, model={model}
    - [OLLAMA:REQUEST] Sending request, prompt_length={length}
    - [OLLAMA:RESPONSE] Response received, content_length={length}
    - [OLLAMA:ERROR] API error: {status_code}, {error}
    - [OLLAMA:RETRY] Retrying request, attempt={n}/{max}
    """
```

### Usage Example

```python
from crawler.ml.embeddings import LLMDescriptionGenerator

# Initialize with Ollama Cloud
generator = LLMDescriptionGenerator(
    provider="ollama-cloud",
    model="gemma3:12b",
    api_key=os.environ["OLLAMA_CLOUD_API_KEY"],
    verbose=True
)

# Generate description for a structure
description = await generator.generate(page_structure)
# Output:
# [LLM:INIT] Ollama Cloud initialized, model=gemma3:12b
# [LLM:PROMPT] Generating prompt for structure (domain=example.com)
# [LLM:REQUEST] Sending request to Ollama Cloud API
# [LLM:RESPONSE] Response received (245 chars)
# [LLM:DESCRIPTION] Generated description: "News article page with semantic
#   HTML structure. Main content in <article> element with clear heading
#   hierarchy. Sidebar navigation and footer present."
```

---

## Fingerprinting Modes

The crawler provides three independent fingerprinting modes. Each mode operates independently - they are **not combined** into a weighted score. You select one mode based on your use case.

### Mode Overview

| Mode | Latency | Best For | Trade-offs |
|------|---------|----------|------------|
| **Rules** | ~15ms | Stable sites, high throughput | Sensitive to class renames |
| **ML** | ~200ms | Sites with frequent CSS changes | Requires embedding model, slower |
| **Adaptive** | ~15-200ms | Unknown sites, mixed environments | Slightly more complex logic |

### Configuration

```yaml
# config.yaml
fingerprinting:
  mode: adaptive  # "rules", "ml", or "adaptive"

  # Adaptive mode settings
  adaptive:
    class_change_threshold: 0.15    # Trigger ML if >15% classes changed
    rules_uncertainty_threshold: 0.80  # Trigger ML if rules similarity < 0.80
    cache_ml_results: true          # Cache embeddings for reuse
```

```bash
# Environment variable
export CRAWLER_FINGERPRINT_MODE="adaptive"  # or "rules" or "ml"
```

---

### Mode 1: Rules-Based (Default)

Fast, deterministic fingerprinting using DOM structure analysis.

**When to Use:**
- High-throughput crawling where speed matters
- Sites with stable CSS class names
- When you need deterministic, reproducible results
- Offline environments without embedding model access

**Advantages:**
- Fast (~15ms per comparison)
- Deterministic (same input = same output)
- No external dependencies
- Interpretable results (can see exactly what changed)

**Limitations:**
- Sensitive to CSS class renames (site refactors break detection)
- Cannot detect semantic similarity (different classes, same structure)

**Components Analyzed:**
```
1. Tag Hierarchy
   - Tag counts: {"div": 450, "p": 120, "article": 1}
   - Depth distribution: {1: 5, 2: 15, 3: 45, ...}
   - Parent-child pairs: {"body>div": 10, "article>p": 8, ...}

2. CSS Classes
   - Class frequencies: {"container": 5, "article": 1, ...}
   - Semantic class detection (content, article, nav, etc.)

3. Semantic Landmarks
   - HTML5: <header>, <nav>, <main>, <article>, <aside>, <footer>
   - ARIA: role="main", role="navigation", etc.

4. Iframes
   - Selector, src pattern, position, dimensions

5. Scripts
   - Framework detection: React, Vue, Angular, Next.js, etc.
```

**Verbose Output:**
```
[FINGERPRINT:MODE] Using RULES mode
[COMPARE:RULES] Computing rules-based similarity
  - Tag similarity: 0.92
  - Class similarity: 0.68 (significant class changes detected)
  - Landmark similarity: 0.95
  - Structure similarity: 0.88
[COMPARE:RULES:RESULT] Similarity: 0.78
[COMPARE:CLASSIFY] Classification: MODERATE (0.70 < 0.78 < 0.85)
[COMPARE:RESULT] Breaking: NO
```

---

### Mode 2: ML-Based

Semantic fingerprinting using sentence transformer embeddings.

**When to Use:**
- Sites known to frequently rename CSS classes
- When semantic similarity matters more than exact structure
- Sites that undergo regular CSS refactoring
- When you need rich, human-readable change descriptions

**Advantages:**
- Robust to class renames (understands semantic meaning)
- Better at detecting "same structure, different names"
- Rich descriptions via Ollama Cloud LLM
- Handles superficial changes gracefully

**Limitations:**
- Slower (~200ms per comparison)
- Requires embedding model (90MB for MiniLM)
- Non-deterministic (minor floating-point variations)
- Ollama Cloud adds network latency for descriptions

**Pipeline:**
```
PageStructure
    │
    ▼
DescriptionGenerator (Rules or LLM)
    │ (creates semantic text: "Article page with sidebar...")
    ▼
SentenceTransformer (all-MiniLM-L6-v2)
    │ (encodes to 384-dim vector)
    ▼
StructureEmbedding
    │
    ▼
Cosine Similarity (0.0 - 1.0)
```

**Verbose Output:**
```
[FINGERPRINT:MODE] Using ML mode
[COMPARE:ML] Computing ML-based similarity
[COMPARE:ML:DESCRIBE] Generating description for stored structure
  - Using: RulesBasedDescriptionGenerator
  - Description: "Article page with semantic HTML5..."
[COMPARE:ML:EMBED] Generating embedding
  - Model: all-MiniLM-L6-v2
  - Dimensions: 384
[COMPARE:ML:DESCRIBE] Generating description for current structure
  - Description: "Article page with refactored CSS..."
[COMPARE:ML:EMBED] Generating embedding
  - Dimensions: 384
[COMPARE:ML:SIMILARITY] Computing cosine similarity
  - Dot product: 0.912
  - Similarity: 0.912
[COMPARE:ML:RESULT] Similarity: 0.91
[COMPARE:CLASSIFY] Classification: MINOR (0.85 < 0.91 < 0.95)
[COMPARE:RESULT] Breaking: NO
```

---

### Mode 3: Adaptive (Recommended for Production)

Intelligent mode selection that starts with fast rules-based comparison and escalates to ML only when needed.

**When to Use:**
- Production environments with mixed site types
- When you don't know site characteristics in advance
- First-time visits to new domains
- When you want optimal speed without sacrificing accuracy

**How It Works:**

```
┌─────────────────────────────────────────────────────────────┐
│                    ADAPTIVE MODE FLOW                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  1. Run Rules-Based     │
              │     Comparison (~15ms)  │
              └─────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  2. Analyze Results     │
              │     Check triggers      │
              └─────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
    ┌───────────────┐               ┌───────────────┐
    │ NO TRIGGERS   │               │ TRIGGERS MET  │
    │               │               │               │
    │ Return rules  │               │ Run ML-based  │
    │ result        │               │ comparison    │
    │ (~15ms total) │               │ (~200ms total)│
    └───────────────┘               └───────────────┘
            │                               │
            └───────────────┬───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Return Result │
                    └───────────────┘
```

**Escalation Triggers (any one triggers ML):**

| Trigger | Condition | Rationale |
|---------|-----------|-----------|
| **Class Volatility** | >15% of classes changed | Likely CSS refactor, ML handles renames |
| **Rules Uncertainty** | Rules similarity < 0.80 | Rules result is ambiguous, need ML clarity |
| **Known Volatile Site** | Domain flagged as volatile | Historical data shows frequent changes |
| **Explicit Class Renames** | Detected rename patterns | e.g., `post-*` classes all disappeared |

**Decision Logic:**

```python
async def compare_adaptive(
    self,
    stored: PageStructure,
    current: PageStructure,
    verbose: bool = True
) -> ChangeAnalysis:
    """
    Adaptive comparison: Rules first, ML if needed.

    Verbose Logging:
    - [ADAPTIVE:START] Starting adaptive comparison
    - [ADAPTIVE:RULES] Running rules-based comparison
    - [ADAPTIVE:ANALYZE] Analyzing rules result for triggers
    - [ADAPTIVE:TRIGGER] Trigger detected: {reason}
    - [ADAPTIVE:ESCALATE] Escalating to ML comparison
    - [ADAPTIVE:RESULT] Final result from {mode}
    """

    if verbose:
        self._log("[ADAPTIVE:START] Starting adaptive comparison")
        self._log(f"  - Domain: {stored.domain}")
        self._log(f"  - Page type: {stored.page_type}")

    # Step 1: Always run rules-based first (fast)
    if verbose:
        self._log("[ADAPTIVE:RULES] Running rules-based comparison")

    rules_result = self.rules_compare(stored, current)

    if verbose:
        self._log(f"[ADAPTIVE:RULES:RESULT] Similarity: {rules_result.similarity:.3f}")

    # Step 2: Check escalation triggers
    if verbose:
        self._log("[ADAPTIVE:ANALYZE] Checking escalation triggers")

    triggers = self._check_triggers(stored, current, rules_result)

    if not triggers:
        # No triggers - return rules result
        if verbose:
            self._log("[ADAPTIVE:RESULT] No triggers, using rules result")
            self._log(f"  - Mode used: RULES")
            self._log(f"  - Total time: ~15ms")
        return rules_result

    # Step 3: Triggers detected - escalate to ML
    if verbose:
        self._log(f"[ADAPTIVE:TRIGGER] Escalation triggered")
        for trigger in triggers:
            self._log(f"  - {trigger.name}: {trigger.reason}")
        self._log("[ADAPTIVE:ESCALATE] Running ML comparison")

    ml_result = await self.ml_compare(stored, current)

    if verbose:
        self._log(f"[ADAPTIVE:ML:RESULT] Similarity: {ml_result.similarity:.3f}")
        self._log("[ADAPTIVE:RESULT] Using ML result")
        self._log(f"  - Mode used: ML (escalated)")
        self._log(f"  - Total time: ~200ms")
        self._log(f"  - Escalation reason: {triggers[0].name}")

    # Return ML result (not combined with rules)
    return ml_result


def _check_triggers(
    self,
    stored: PageStructure,
    current: PageStructure,
    rules_result: ChangeAnalysis
) -> list[EscalationTrigger]:
    """
    Check if any escalation triggers are met.

    Verbose Logging:
    - [ADAPTIVE:CHECK:CLASSES] Checking class volatility
    - [ADAPTIVE:CHECK:UNCERTAINTY] Checking rules uncertainty
    - [ADAPTIVE:CHECK:VOLATILE] Checking known volatile sites
    - [ADAPTIVE:CHECK:RENAMES] Checking for rename patterns
    """
    triggers = []

    # Trigger 1: Class volatility
    class_change_ratio = self._compute_class_change_ratio(stored, current)
    if class_change_ratio > self.config.class_change_threshold:
        triggers.append(EscalationTrigger(
            name="CLASS_VOLATILITY",
            reason=f"{class_change_ratio:.0%} of classes changed (threshold: {self.config.class_change_threshold:.0%})"
        ))

    # Trigger 2: Rules uncertainty
    if rules_result.similarity < self.config.rules_uncertainty_threshold:
        triggers.append(EscalationTrigger(
            name="RULES_UNCERTAINTY",
            reason=f"Rules similarity {rules_result.similarity:.2f} below threshold {self.config.rules_uncertainty_threshold}"
        ))

    # Trigger 3: Known volatile site
    if self._is_known_volatile(stored.domain):
        triggers.append(EscalationTrigger(
            name="KNOWN_VOLATILE",
            reason=f"Domain {stored.domain} flagged as volatile based on history"
        ))

    # Trigger 4: Explicit rename pattern detected
    rename_pattern = self._detect_rename_pattern(stored, current)
    if rename_pattern:
        triggers.append(EscalationTrigger(
            name="RENAME_PATTERN",
            reason=f"Detected rename pattern: {rename_pattern}"
        ))

    return triggers
```

**Verbose Output (No Escalation):**
```
[ADAPTIVE:START] Starting adaptive comparison
  - Domain: stable-site.com
  - Page type: article

[ADAPTIVE:RULES] Running rules-based comparison
[COMPARE:RULES] Computing similarity...
  - Tag similarity: 0.95
  - Class similarity: 0.92
  - Landmark similarity: 0.98
[ADAPTIVE:RULES:RESULT] Similarity: 0.94

[ADAPTIVE:ANALYZE] Checking escalation triggers
[ADAPTIVE:CHECK:CLASSES] Class change ratio: 8% (threshold: 15%) - NO TRIGGER
[ADAPTIVE:CHECK:UNCERTAINTY] Similarity 0.94 >= 0.80 - NO TRIGGER
[ADAPTIVE:CHECK:VOLATILE] Domain not in volatile list - NO TRIGGER
[ADAPTIVE:CHECK:RENAMES] No rename patterns detected - NO TRIGGER

[ADAPTIVE:RESULT] No triggers, using rules result
  - Mode used: RULES
  - Similarity: 0.94
  - Classification: MINOR
  - Total time: 18ms
```

**Verbose Output (With Escalation):**
```
[ADAPTIVE:START] Starting adaptive comparison
  - Domain: frequently-refactored.com
  - Page type: article

[ADAPTIVE:RULES] Running rules-based comparison
[COMPARE:RULES] Computing similarity...
  - Tag similarity: 0.91
  - Class similarity: 0.52 (significant changes!)
  - Landmark similarity: 0.95
[ADAPTIVE:RULES:RESULT] Similarity: 0.72

[ADAPTIVE:ANALYZE] Checking escalation triggers
[ADAPTIVE:CHECK:CLASSES] Class change ratio: 38% (threshold: 15%) - TRIGGERED
[ADAPTIVE:CHECK:UNCERTAINTY] Similarity 0.72 < 0.80 - TRIGGERED
[ADAPTIVE:CHECK:RENAMES] Pattern detected: "post-*" prefix removed - TRIGGERED

[ADAPTIVE:TRIGGER] Escalation triggered
  - CLASS_VOLATILITY: 38% of classes changed (threshold: 15%)
  - RULES_UNCERTAINTY: Rules similarity 0.72 below threshold 0.80
  - RENAME_PATTERN: Detected rename pattern: post-* prefix removal

[ADAPTIVE:ESCALATE] Running ML comparison
[COMPARE:ML:DESCRIBE] Generating descriptions...
[COMPARE:ML:EMBED] Computing embeddings...
[COMPARE:ML:SIMILARITY] Cosine similarity: 0.89
[ADAPTIVE:ML:RESULT] Similarity: 0.89

[ADAPTIVE:RESULT] Using ML result
  - Mode used: ML (escalated)
  - Similarity: 0.89
  - Classification: MINOR
  - Total time: 215ms
  - Escalation reason: CLASS_VOLATILITY

[ADAPTIVE:INSIGHT] ML detected semantic similarity despite class renames
  - Rules saw: 0.72 (MODERATE, potentially breaking)
  - ML saw: 0.89 (MINOR, safe to continue)
  - Action: Adapt selectors, no re-learning needed
```

---

### Mode Comparison Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DECISION GUIDE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  "I need maximum speed"                                              │
│      └─→ Use RULES mode                                             │
│                                                                      │
│  "Site frequently renames CSS classes"                               │
│      └─→ Use ML mode                                                │
│                                                                      │
│  "I'm crawling many different sites"                                 │
│      └─→ Use ADAPTIVE mode (recommended)                            │
│                                                                      │
│  "I don't know the site characteristics"                             │
│      └─→ Use ADAPTIVE mode                                          │
│                                                                      │
│  "I need deterministic results for testing"                          │
│      └─→ Use RULES mode                                             │
│                                                                      │
│  "I want rich change descriptions"                                   │
│      └─→ Use ML mode with Ollama Cloud                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Important: Modes are Independent

The three modes are **mutually exclusive** - you choose one:

- **No weighted combination**: We do not combine rules and ML scores
- **No parallel execution**: Adaptive runs rules first, then ML only if triggered
- **Single result**: Each comparison returns one similarity score from one mode
- **Clear provenance**: Result always indicates which mode produced it

This design ensures:
1. Predictable performance (you know latency based on mode)
2. Clear debugging (one method to investigate, not two)
3. Appropriate tool for the job (fast when possible, accurate when needed)

---

## Redis Storage Schema

### Key Patterns

```
# Basic structure storage
crawler:structure:{domain}:{page_type}:{variant_id}     → JSON(PageStructure)
crawler:structure:{domain}:{page_type}:{variant_id}:v{n} → JSON(PageStructure history)

# Extraction strategies
crawler:strategy:{domain}:{page_type}:{variant_id}      → JSON(ExtractionStrategy)

# Variant tracking
crawler:variants:{domain}:{page_type}                   → SET[variant_id]

# Change history
crawler:changes:{domain}:{page_type}                    → LIST[JSON(StructureChange)]

# LLM-enhanced storage
crawler:structure_llm:{domain}:{page_type}:{variant_id} → JSON(PageStructure + description)
crawler:embedding:{domain}:{page_type}:{variant_id}     → BLOB(embedding vector)

# URL management
crawler:url_frontier                                     → ZSET[(url, priority)]
crawler:url_visited                                      → SET[url]
crawler:url_in_progress                                  → SET[url]

# Robots cache
crawler:robots:{domain}                                  → JSON(robots.txt + metadata)

# Rate limiting
crawler:rate_state:{domain}                             → JSON(delay, last_request, backoff)

# Distributed coordination
crawler:workers                                          → SET[worker_id]
crawler:worker:{worker_id}                              → JSON(state, heartbeat)
crawler:domain_assignment:{domain}                      → worker_id
```

### Example Data

```json
// Key: crawler:structure:example.com:article:default
{
    "domain": "example.com",
    "page_type": "article",
    "variant_id": "default",
    "version": 3,
    "captured_at": "2025-01-20T10:30:00Z",

    "tag_hierarchy": {
        "tag_counts": {"div": 45, "p": 12, "article": 1, "h1": 1},
        "depth_distribution": {"1": 3, "2": 8, "3": 25, "4": 15},
        "parent_child_pairs": {"article>h1": 1, "article>p": 8}
    },

    "css_class_map": {
        "container": 3,
        "article-content": 1,
        "title": 1
    },

    "semantic_landmarks": {
        "header": "header.site-header",
        "nav": "nav.main-nav",
        "main": "main.content",
        "article": "article.post",
        "footer": "footer.site-footer"
    },

    "content_regions": [
        {
            "name": "main_content",
            "primary_selector": "article.post",
            "fallback_selectors": ["main.content", "div.article"],
            "confidence": 0.92
        }
    ],

    "content_hash": "a1b2c3d4e5f6"
}
```

---

## Verbose Logging System

### Enabling Verbose Mode

```python
# Via environment variable
export CRAWLER_VERBOSE=3  # 0=errors, 1=warn, 2=info, 3=debug

# Via command line
python -m crawler --seed-url https://example.com --verbose

# Via config
verbose:
  level: 3
  modules:
    fetcher: 3
    structure_analyzer: 3
    ml: 2

# Via code
crawler = Crawler(config, verbose=True)
```

### Log Format

```
[{TIMESTAMP}] [{MODULE}:{OPERATION}] {MESSAGE}
  - {detail_1}
  - {detail_2}
  ...
```

### Module Prefixes

| Module | Prefix | Operations |
|--------|--------|------------|
| Crawler | `CRAWLER` | INIT, START, URL, FETCH, EXTRACT, CHANGE, CHECKPOINT, COMPLETE |
| Fetcher | `FETCHER` | CFAA, ROBOTS, RATE, HTTP, PII, ADAPT |
| Scheduler | `SCHEDULER` | ADD, SKIP, NEXT, STATS |
| Robots | `ROBOTS` | FETCH, CACHE, PARSE, MATCH, CHECK, DELAY |
| Rate | `RATE` | INIT, STATE, ACQUIRE, ADAPT, BACKOFF, RECOVER |
| Structure | `STRUCTURE` | ANALYZE, TAGS, CLASSES, LANDMARKS, IFRAMES, SCRIPTS, REGIONS, HASH |
| Change | `CHANGE` | COMPARE, SIMILARITY, CLASSIFY, DETAILS, BREAKING, FIELDS |
| Learn | `LEARN` | START, SEMANTIC, ARIA, SCHEMA, HEURISTIC, CANDIDATE, SELECT, FALLBACK, COMPLETE |
| Extract | `EXTRACT` | STRATEGY, SELECTOR, MATCH, FALLBACK, FIELD, VALIDATE |
| ML | `ML` | CHANGE, SIMILARITY, THRESHOLD, BREAKING, IMPACT |
| Embed | `EMBED` | INIT, DESCRIBE, ENCODE, DIMS, NORM |
| LLM | `LLM` | INIT, PROMPT, REQUEST, RESPONSE, DESCRIPTION |
| Classify | `CLASSIFY` | INIT, FEATURES, PREDICT, RESULT |
| Store | `STORE` | GET, HIT, MISS, SAVE, UPDATE, VARIANT, HISTORY |
| Alert | `ALERT` | TRIGGER, THROTTLE, SEND, SUCCESS, FAIL |

### Full Verbose Output Example

```
[2025-01-20T10:30:00Z] [CRAWLER:START] Starting crawl session
  - Seed URLs: 1
  - Max depth: 10
  - Rate limit: 1.0 req/sec

[2025-01-20T10:30:00Z] [CRAWLER:URL] Processing: https://example.com/
  - Priority: 1.0
  - Depth: 0
  - Queue size: 0

[2025-01-20T10:30:00Z] [FETCHER:CFAA] Checking authorization
  - Domain: example.com
  - Checking robots.txt...
  - Checking ToS...
  - Result: AUTHORIZED (basis: public_access)

[2025-01-20T10:30:00Z] [FETCHER:ROBOTS] Checking robots.txt
  - URL: https://example.com/
  - Cache: HIT (age: 3600s)
  - Rule matched: Allow /
  - Result: ALLOWED

[2025-01-20T10:30:00Z] [FETCHER:RATE] Acquiring rate limit slot
  - Domain: example.com
  - Current delay: 1.0s
  - Waiting: 0.0s (first request)
  - Slot acquired

[2025-01-20T10:30:01Z] [FETCHER:HTTP] Request complete
  - URL: https://example.com/
  - Status: 200 OK
  - Size: 45,230 bytes
  - Duration: 245ms
  - Content-Type: text/html

[2025-01-20T10:30:01Z] [STRUCTURE:ANALYZE] Analyzing page structure
  - HTML size: 45,230 bytes
  - Parsing...

[2025-01-20T10:30:01Z] [STRUCTURE:TAGS] Tag distribution
  - div: 145
  - p: 42
  - a: 38
  - span: 25
  - article: 1

[2025-01-20T10:30:01Z] [STRUCTURE:CLASSES] CSS classes
  - Found 67 unique classes
  - Semantic classes: container, article, content, nav

[2025-01-20T10:30:01Z] [STRUCTURE:LANDMARKS] Semantic landmarks
  - header: header.site-header
  - nav: nav.main-navigation
  - main: main#content
  - footer: footer.site-footer

[2025-01-20T10:30:01Z] [STRUCTURE:REGIONS] Content regions
  - main_content: article.post (confidence: 0.92)
  - sidebar: aside.sidebar (confidence: 0.85)

[2025-01-20T10:30:01Z] [STORE:GET] Getting stored structure
  - Domain: example.com
  - Page type: homepage
  - Result: MISS (first visit)

[2025-01-20T10:30:01Z] [LEARN:START] Learning extraction strategy
  - Domain: example.com
  - Page type: homepage

[2025-01-20T10:30:01Z] [LEARN:SEMANTIC] Checking semantic HTML
  - Found: <main>, <article>, <h1>
  - Title candidate: article h1 (confidence: 0.95)
  - Content candidate: article (confidence: 0.92)

[2025-01-20T10:30:01Z] [LLM:INIT] Generating LLM description
  - Provider: ollama-cloud
  - Model: gemma3:12b

[2025-01-20T10:30:02Z] [LLM:REQUEST] Sending to Ollama Cloud API
  - Prompt length: 850 chars
  - Waiting for response...

[2025-01-20T10:30:03Z] [LLM:RESPONSE] Response received
  - Content length: 234 chars
  - Description: "Modern homepage with semantic HTML5 structure.
    Features hero section, article grid, and sidebar navigation.
    Clean separation between header, main content, and footer."

[2025-01-20T10:30:03Z] [EMBED:ENCODE] Generating embedding
  - Model: all-MiniLM-L6-v2
  - Input: LLM description (234 chars)
  - Output: 384 dimensions
  - L2 norm: 1.0

[2025-01-20T10:30:03Z] [STORE:SAVE] Saving structure
  - Domain: example.com
  - Page type: homepage
  - Version: 1
  - With LLM description: Yes
  - With embedding: Yes

[2025-01-20T10:30:03Z] [EXTRACT:STRATEGY] Applying extraction strategy
  - Fields: title, content, links
  - Selectors loaded

[2025-01-20T10:30:03Z] [EXTRACT:FIELD] Extracting title
  - Selector: article h1
  - Matched: 1 element
  - Value: "Welcome to Example.com"

[2025-01-20T10:30:03Z] [CRAWLER:COMPLETE] URL processing complete
  - URL: https://example.com/
  - Status: SUCCESS
  - Extracted fields: 3
  - New URLs discovered: 15
  - Duration: 3.2s
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| **Core** |
| `CRAWLER_USER_AGENT` | `AdaptiveCrawler/1.0` | User-agent string |
| `CRAWLER_VERBOSE` | `2` | Verbose logging level (0-3) |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| **Rate Limiting** |
| `CRAWLER_DEFAULT_DELAY` | `1.0` | Base delay (seconds) |
| `CRAWLER_MIN_DELAY` | `0.5` | Minimum delay floor |
| `CRAWLER_MAX_DELAY` | `60.0` | Maximum delay ceiling |
| `CRAWLER_RESPECT_CRAWL_DELAY` | `true` | Honor Crawl-delay |
| **Ollama Cloud** |
| `OLLAMA_CLOUD_API_KEY` | `` | **Required** API key |
| `OLLAMA_CLOUD_MODEL` | `gemma3:12b` | Model to use |
| `OLLAMA_CLOUD_TIMEOUT` | `30` | Request timeout |
| **ML** |
| `CRAWLER_ENABLE_ML` | `true` | Enable ML fingerprinting |
| `CRAWLER_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `CRAWLER_BREAKING_THRESHOLD` | `0.70` | Breaking change threshold |
| **Legal** |
| `GDPR_ENABLED` | `true` | Enable GDPR compliance |
| `GDPR_RETENTION_DAYS` | `365` | Data retention period |
| `PII_HANDLING` | `redact` | PII action |
| `CCPA_ENABLED` | `true` | Enable CCPA compliance |
| **Storage** |
| `CRAWLER_STRUCTURE_TTL` | `604800` | Structure TTL (7 days) |
| `CRAWLER_MAX_VERSIONS` | `10` | Max structure versions |

---

## Exception Hierarchy

```python
CrawlerError                    # Base exception
├── ComplianceError             # Compliance check failed
│   ├── RobotsBlockedError      # Blocked by robots.txt
│   ├── RateLimitExceededError  # Rate limit exceeded
│   └── CFAABlockedError        # CFAA authorization denied
├── FetchError                  # HTTP fetch failed
│   ├── TimeoutError            # Request timeout
│   ├── ConnectionError         # Connection failed
│   └── CircuitOpenError        # Circuit breaker open
├── ExtractionError             # Content extraction failed
│   ├── StructureChangeError    # Breaking structure change
│   └── StrategyInferenceError  # Failed to learn strategy
├── StorageError                # Storage operation failed
│   ├── RedisConnectionError    # Redis connection failed
│   └── SerializationError      # JSON serialization failed
├── LLMError                    # Ollama Cloud error
│   ├── OllamaAuthError         # Authentication failed
│   ├── OllamaTimeoutError      # Request timeout
│   └── OllamaRateLimitError    # API rate limited
└── LegalComplianceError        # Legal requirement not met
    ├── GDPRViolationError      # GDPR violation
    ├── PIIExposureError        # PII handling failed
    └── CeaseAndDesistError     # Domain blocked
```

---

## Dependencies

### Required

```toml
[project]
dependencies = [
    "httpx>=0.27.0",              # Async HTTP client
    "beautifulsoup4>=4.12.0",     # HTML parsing
    "lxml>=5.0.0",                # Fast XML parser
    "redis>=5.0.0",               # Redis client
    "pydantic>=2.0.0",            # Configuration
    "pydantic-settings>=2.0.0",   # Env loading
    "sentence-transformers>=2.3.0", # Embeddings
    "numpy>=1.24.0",              # Numerical ops
]
```

### Optional

```toml
[project.optional-dependencies]
ml = [
    "xgboost>=2.0.0",             # Gradient boosting
    "lightgbm>=4.0.0",            # Fast gradient boosting
    "scikit-learn>=1.3.0",        # ML utilities
]
js = [
    "playwright>=1.43.0",         # Browser automation
]
metrics = [
    "prometheus-client>=0.20.0",  # Metrics export
]
```

---

## Common Tasks

### Adding a New Change Type

1. Add enum to `ChangeType` in `crawler/models.py`
2. Implement detection in `crawler/adaptive/change_detector.py`
3. Add reason template in detection
4. Update verbose logging

### Adding a New Extraction Field

1. Add field to `ExtractionStrategy` in `crawler/models.py`
2. Add inference logic in `crawler/adaptive/strategy_learner.py`
3. Add extraction logic in `crawler/extraction/content_extractor.py`
4. Update verbose logging

### Modifying Rate Limiting

1. Configuration: `crawler/config.py` → `RateLimitConfig`
2. Logic: `crawler/compliance/rate_limiter.py`
3. Key methods: `acquire()`, `adapt()`, `set_domain_delay()`

### Customizing LLM Prompts

1. Edit prompts in `crawler/ml/embeddings.py` → `LLMDescriptionGenerator`
2. Modify `_create_structure_prompt()` method
3. Adjust `max_tokens` and `temperature` as needed

---

## License and Ethical Use

This crawler is designed for ethical, legal web data collection. Users are responsible for:

1. Complying with applicable laws (CFAA, GDPR, CCPA)
2. Respecting website terms of service
3. Obtaining necessary legal advice
4. Using collected data appropriately
5. Maintaining reasonable crawl rates
6. Responding to abuse reports promptly

**Disclaimer**: This documentation provides technical guidance, not legal advice. Consult qualified legal counsel for your specific use case.
