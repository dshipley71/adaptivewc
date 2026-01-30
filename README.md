# Adaptive Web Crawler

An intelligent, compliance-first web crawler with ML-based structure learning that automatically adapts to website changes. Built for ethical, legal web data collection with full CFAA/GDPR/CCPA compliance.

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Crawl Workflow](#crawl-workflow)
- [Adaptive Learning System](#adaptive-learning-system)
- [Sitemap Processing](#sitemap-processing)
- [JavaScript Rendering](#javascript-rendering)
- [Distributed Crawling](#distributed-crawling)
- [Scheduled Recrawling](#scheduled-recrawling)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Sports News Monitor Example](#sports-news-monitor-example)
- [API Reference](#api-reference)
- [Machine Learning Features](#machine-learning-features)
- [Development](#development)
- [Legal Notice](#legal-notice)

---

## Features

### Compliance & Legal

The crawler is designed with legal compliance as a first-class priority, ensuring your web scraping operations remain within legal boundaries.

#### CFAA Compliance (Computer Fraud and Abuse Act)

The CFAA is a U.S. federal law that prohibits unauthorized access to computer systems. The crawler implements authorization checks before every request:

```python
from crawler.legal import CFAAChecker

checker = CFAAChecker()

# Check if crawling is authorized
result = await checker.is_authorized("https://example.com/page")
if result.authorized:
    print("Crawling is authorized")
else:
    print(f"Blocked: {result.reason}")
    # Possible reasons:
    # - "Terms of service explicitly prohibit crawling"
    # - "Login-required content without authorization"
    # - "Previously received cease-and-desist"
```

**Authorization indicators the crawler checks:**
- Public accessibility (no authentication required)
- Presence of robots.txt (indicates expectation of bots)
- Meta tags allowing/disallowing indexing
- **Terms of Service analysis (enabled by default)** - automatically analyzes ToS for crawling restrictions
- Previous crawl history and any blocks received

**Terms of Service Analysis:**

The crawler **automatically analyzes Terms of Service** pages to detect crawling restrictions. This feature is **enabled by default** to ensure maximum legal compliance.

```python
from crawler.config import CFAAConfig
from crawler.legal import CFAAChecker

# ToS analysis is enabled by default
cfaa_config = CFAAConfig(
    enabled=True,
    tos_analysis_enabled=True,        # Enabled by default
    block_on_restrictive_tos=True,    # Block crawling if ToS prohibits it
    tos_cache_ttl=86400,              # Cache ToS analysis for 24 hours
    common_tos_paths=[                # Paths to check for ToS
        "/terms",
        "/terms-of-service",
        "/tos",
        "/legal/terms",
        "/terms-and-conditions",
        "/terms-of-use",
    ],
)

# The checker automatically analyzes ToS
checker = CFAAChecker(config=cfaa_config)

# When checking authorization, ToS is analyzed automatically
result = await checker.is_authorized(url)
if not result.authorized and result.basis == "terms_of_service":
    print(f"ToS prohibits crawling: {result.documentation}")

# You can also analyze ToS text directly
tos_analysis = checker.analyze_tos(tos_text, domain)
print(f"Restrictive: {tos_analysis['is_restrictive']}")
print(f"Restrictions: {tos_analysis['restrictions']}")
```

**What ToS analysis detects:**
- Explicit prohibition of scraping/crawling
- Prohibition of automated access
- Requirements to use official APIs only
- Rate limit mentions
- Bot/spider restrictions

#### GDPR/CCPA Support

The General Data Protection Regulation (GDPR) and California Consumer Privacy Act (CCPA) require special handling of personal data. The crawler provides:

```python
from crawler.legal import PIIDetector, PIIHandler
from crawler.config import GDPRConfig, PIIHandlingConfig

# Configure GDPR compliance
gdpr_config = GDPRConfig(
    enabled=True,
    retention_days=365,           # Auto-delete data after 1 year
    collect_only=["url", "title", "content"],  # Whitelist fields
    exclude_countries=["EU"],     # Optional: skip EU-based sites
)

# Configure PII handling
pii_config = PIIHandlingConfig(
    action="redact",              # Options: "redact", "pseudonymize", "exclude_page"
    patterns=[
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\d{16}\b",             # Credit card
    ],
    log_detections=True,          # Audit trail for compliance
)

# Detect PII in content
detector = PIIDetector()
findings = detector.scan(html_content)
for finding in findings:
    print(f"Found {finding.pii_type} at position {finding.start}-{finding.end}")
    # Output: Found EMAIL at position 1234-1256

# Handle PII according to policy
handler = PIIHandler(pii_config)
clean_content = handler.process(html_content)
```

#### robots.txt Respect (RFC 9309)

The crawler fully implements the robots.txt standard including the latest RFC 9309 specification:

```python
from crawler.compliance import RobotsChecker

checker = RobotsChecker(
    user_agent="MyCrawler/1.0",
    cache_ttl=3600,  # Cache robots.txt for 1 hour
)

# Check if path is allowed
allowed = await checker.is_allowed("https://example.com/private/page")
print(f"Allowed: {allowed}")

# Get crawl delay
delay = await checker.get_crawl_delay("https://example.com")
print(f"Crawl-delay: {delay} seconds")

# Get sitemap URLs from robots.txt
sitemaps = await checker.get_sitemaps("https://example.com")
print(f"Sitemaps: {sitemaps}")
```

**Supported robots.txt directives:**
- `User-agent` - Matches crawler identity
- `Allow` / `Disallow` - Path-based access control
- `Crawl-delay` - Per-domain rate limiting
- `Sitemap` - Sitemap discovery
- Wildcard patterns (`*`, `$`)

#### Anti-Bot Respect

Unlike aggressive scrapers, this crawler treats bot detection as "access denied" and never attempts evasion:

```python
# The crawler automatically detects and respects:
# - CAPTCHA challenges → marks URL as blocked
# - JavaScript challenges (Cloudflare, etc.) → marks as blocked
# - Rate limit responses (429) → backs off exponentially
# - IP blocks → stops crawling that domain

# You can check if a domain has blocked the crawler:
from crawler.compliance import BlockedDomainTracker

tracker = BlockedDomainTracker(redis_client)
if await tracker.is_blocked("example.com"):
    print("Domain has blocked our crawler")
    print(f"Blocked since: {await tracker.get_block_time('example.com')}")
    print(f"Reason: {await tracker.get_block_reason('example.com')}")
```

### Intelligent Crawling

#### Adaptive Rate Limiting

The crawler automatically adjusts request rates based on server responses:

```python
from crawler.compliance import AdaptiveRateLimiter

limiter = AdaptiveRateLimiter(
    default_delay=1.0,    # Start with 1 second between requests
    min_delay=0.5,        # Never go faster than 0.5 seconds
    max_delay=60.0,       # Never wait more than 60 seconds
    backoff_factor=2.0,   # Double delay on rate limit
    recovery_factor=0.9,  # Slowly recover after success
)

# The limiter automatically tracks per-domain delays
async with limiter.acquire("example.com"):
    # Make request here
    response = await fetch(url)

    # Report response for adaptive adjustment
    if response.status_code == 429:
        limiter.report_rate_limited("example.com")
        # Delay automatically increases
    elif response.status_code == 503:
        limiter.report_server_overload("example.com")
        # Delay automatically increases
    else:
        limiter.report_success("example.com")
        # Delay slowly decreases toward default
```

#### Structure Learning

The ML-based DOM analysis learns page layouts automatically:

```python
from crawler.adaptive import StructureAnalyzer, StructureLearner

analyzer = StructureAnalyzer()

# Analyze page structure
structure = analyzer.analyze(
    html=html_content,
    url="https://example.com/article/123",
    page_type="article",
)

# The structure contains:
print(f"Domain: {structure.domain}")
print(f"Tag hierarchy: {structure.tag_hierarchy}")
print(f"CSS classes: {structure.css_class_map}")
print(f"Element IDs: {structure.id_attributes}")
print(f"Semantic landmarks: {structure.semantic_landmarks}")
print(f"Content regions: {structure.content_regions}")
print(f"Navigation selectors: {structure.navigation_selectors}")

# Learn extraction strategy
learner = StructureLearner()
strategy = learner.infer(html_content, structure)

print(f"Title selector: {strategy.title.selector} (confidence: {strategy.title.confidence})")
print(f"Content selector: {strategy.content.selector} (confidence: {strategy.content.confidence})")
```

#### Change Detection

Automatically detects when websites change their structure:

```python
from crawler.adaptive import ChangeDetector, ChangeClassification

detector = ChangeDetector()

# Compare old and new structures
analysis = detector.detect_changes(old_structure, new_structure)

print(f"Has changes: {analysis.has_changes}")
print(f"Similarity: {analysis.similarity_score:.2%}")
print(f"Classification: {analysis.classification.name}")

# Classification levels:
# COSMETIC (≥95%): CSS-only changes, no action needed
# MINOR (85-95%): Small tweaks, keep strategy
# MODERATE (70-85%): Significant changes, consider adapting
# BREAKING (<70%): Major redesign, must re-learn strategy

# Get detailed change information
for change in analysis.changes:
    print(f"  - {change.change_type}: {change.description}")
    # Examples:
    # - TAG_COUNT_CHANGED: div count changed from 45 to 52
    # - CLASS_RENAMED: .article-content → .post-body
    # - ELEMENT_MOVED: #sidebar moved from right to left
    # - LANDMARK_ADDED: New <aside> element detected
```

### Production Ready

#### Redis-Backed Persistence

All learned structures and strategies are stored in Redis for persistence:

```python
from crawler.storage import StructureStore

store = StructureStore(redis_url="redis://localhost:6379/0")

# Save structure
await store.save_structure("example.com", "article", structure)

# Load structure
stored = await store.get_structure("example.com", "article")

# Get structure history (for rollback)
history = await store.get_history("example.com", "article", limit=10)
for version in history:
    print(f"Version {version.version} at {version.timestamp}")

# Rollback to previous version
await store.rollback("example.com", "article", version=3)

# Structure TTL and expiration
await store.set_ttl("example.com", "article", seconds=604800)  # 7 days
```

#### Structured Logging

Complete audit trail of all operations:

```python
import structlog
from crawler.utils import configure_logging

# Configure structured logging
configure_logging(
    level="INFO",
    format="json",  # or "console" for development
    output="crawler.log",
)

log = structlog.get_logger()

# All crawler operations are logged with context
log.info("page_crawled",
    url="https://example.com/page",
    status_code=200,
    content_length=15234,
    extraction_success=True,
    selectors_used=["h1.title", "article.content"],
)

# Compliance events are logged for audit
log.info("robots_check",
    url="https://example.com/private",
    allowed=False,
    reason="Disallow: /private",
)

log.info("pii_detected",
    url="https://example.com/page",
    pii_type="EMAIL",
    action="redacted",
    count=3,
)
```

#### Circuit Breakers

Automatic failure isolation per domain prevents cascading failures:

```python
from crawler.utils import CircuitBreaker

breaker = CircuitBreaker(
    failure_threshold=5,     # Open after 5 failures
    recovery_timeout=60,     # Try again after 60 seconds
    half_open_requests=3,    # Allow 3 test requests when half-open
)

async def fetch_with_circuit_breaker(url: str):
    domain = get_domain(url)

    if breaker.is_open(domain):
        raise CircuitOpenError(f"Circuit open for {domain}")

    try:
        response = await fetch(url)
        breaker.record_success(domain)
        return response
    except Exception as e:
        breaker.record_failure(domain)
        raise

# Check circuit status
status = breaker.get_status("example.com")
print(f"State: {status.state}")  # CLOSED, OPEN, or HALF_OPEN
print(f"Failures: {status.failure_count}")
print(f"Last failure: {status.last_failure_time}")
```

#### Parallel Crawling

Configurable concurrency with domain politeness:

```python
from crawler.core import ConcurrencyManager

manager = ConcurrencyManager(
    global_limit=50,          # Max 50 concurrent requests total
    per_domain_limit=5,       # Max 5 concurrent requests per domain
    per_ip_limit=10,          # Max 10 concurrent requests per IP
)

async def crawl_with_limits(urls: list[str]):
    async with manager:
        tasks = []
        for url in urls:
            # This automatically respects all limits
            task = manager.submit(fetch, url)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

    return results

# Monitor concurrency
stats = manager.get_stats()
print(f"Active requests: {stats.active_count}")
print(f"Queued requests: {stats.queued_count}")
print(f"Domains active: {stats.domains_active}")
```

### Advanced Crawling
- **Sitemap Processing**: Full XML sitemap support with recursive index handling and gzip decompression
- **JavaScript Rendering**: Playwright integration for SPAs with automatic JS requirement detection
- **Distributed Crawling**: Multi-worker coordination with Redis queues, heartbeats, and leader election
- **Scheduled Recrawling**: Cron-like scheduling with adaptive intervals based on content change frequency

---

## How It Works

### High-Level Flow

```
                                    ADAPTIVE WEB CRAWLER
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                                                                             │
    │   SEED URLs ──► SCHEDULER ──► FETCHER ──► ANALYZER ──► EXTRACTOR ──► STORAGE│
    │                     │            │           │             │                │
    │                     │            │           │             │                │
    │                     ▼            ▼           ▼             ▼                │
    │               ┌─────────┐  ┌──────────┐ ┌─────────┐  ┌──────────┐          │
    │               │  URL    │  │Compliance│ │Structure│  │ Learned  │          │
    │               │Frontier │  │ Pipeline │ │Learning │  │ Strategy │          │
    │               └─────────┘  └──────────┘ └─────────┘  └──────────┘          │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
```

### The Compliance Pipeline

Every URL request passes through a strict compliance pipeline:

```
    URL Request
         │
         ▼
    ┌────────────────────┐
    │ 1. CFAA Check      │ ◄── Is crawling this URL legally authorized?
    │    (Authorization) │
    └────────┬───────────┘
             │ Authorized
             ▼
    ┌────────────────────┐
    │ 2. robots.txt      │ ◄── Does the site allow crawling this path?
    │    Check           │
    └────────┬───────────┘
             │ Allowed
             ▼
    ┌────────────────────┐
    │ 3. Rate Limiter    │ ◄── Wait for appropriate delay (respects Crawl-delay)
    │    (Per-Domain)    │
    └────────┬───────────┘
             │ Ready
             ▼
    ┌────────────────────┐
    │ 4. HTTP Fetch      │ ◄── Actual request with timeout & retries
    │                    │
    └────────┬───────────┘
             │ Response
             ▼
    ┌────────────────────┐
    │ 5. GDPR/PII Check  │ ◄── Detect and handle personal data
    │    (if enabled)    │
    └────────┬───────────┘
             │
             ▼
       FetchResult
```

---

## Quick Start

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.11+ | Runtime environment |
| Redis | 7.0+ | Structure persistence, rate limiting, distributed features |
| Docker (optional) | 20.0+ | Easiest way to run Redis |

**Verify Python version:**
```bash
python --version
# Should output: Python 3.11.x or higher
```

### Installation

#### Step 1: Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/adaptive-crawler.git
cd adaptive-crawler

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (Command Prompt):
.venv\Scripts\activate.bat
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Verify activation (should show path to .venv)
which python  # Linux/macOS
where python  # Windows
```

#### Step 2: Install Dependencies

```bash
# Basic installation
pip install -e .

# With development dependencies (testing, linting)
pip install -e ".[dev]"

# With ML features (embeddings, classification)
pip install -e ".[ml]"

# With LLM support (OpenAI, Anthropic, Ollama)
pip install -e ".[llm]"

# With JavaScript rendering (Playwright)
pip install -e ".[js-rendering]"
playwright install chromium

# Everything
pip install -e ".[dev,ml,llm,js-rendering]"
```

#### Step 3: Verify Installation

```bash
# Check the crawler is installed
python -c "import crawler; print(f'Crawler version: {crawler.__version__}')"

# Run module check
python -m crawler --help
```

**Expected output:**
```
usage: crawler [-h] --seed-url URL [--output DIR] [--max-depth N]
               [--max-pages N] [--rate-limit SECONDS] ...

Adaptive Web Crawler - Intelligent, compliance-first web crawling
```

### Start Redis

Redis is required for the adaptive features (structure learning, change detection, rate limiting).

#### Option 1: Docker (Recommended)

```bash
# Start Redis container
docker run -d --name redis-crawler -p 6379:6379 redis:7-alpine

# Verify it's running
docker ps | grep redis-crawler

# Check Redis is responding
docker exec redis-crawler redis-cli ping
# Should output: PONG

# View logs if needed
docker logs redis-crawler
```

#### Option 2: Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  redis_data:
```

```bash
docker-compose up -d
docker-compose ps  # Verify status
```

#### Option 3: Local Installation

**Debian/Ubuntu:**
```bash
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt-get update
sudo apt-get install -y redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Verify
redis-cli ping
```

**macOS (Homebrew):**
```bash
brew install redis
brew services start redis
redis-cli ping
```

**Windows:**
```powershell
# Using Windows Subsystem for Linux (WSL) is recommended
# Or use Docker Desktop for Windows
```

#### Verify Redis Connection

```bash
# Test connection from Python
python -c "
import redis
r = redis.Redis(host='localhost', port=6379, db=0)
print(f'Redis connected: {r.ping()}')
print(f'Redis version: {r.info()[\"redis_version\"]}')
"
```

### Run Your First Crawl

#### Basic Crawl

```bash
# Minimal crawl
python -m crawler --seed-url https://example.com --output ./data

# Expected output:
# [INFO] Starting crawl with 1 seed URL(s)
# [INFO] Crawling: https://example.com
# [INFO] Fetched: https://example.com (200 OK, 1.2KB)
# [INFO] Structure learned for example.com/homepage
# [INFO] Extracted: title, content
# [INFO] Crawl complete: 1 pages, 0 errors
```

#### Crawl with Options

```bash
python -m crawler \
    --seed-url https://example.com \
    --output ./data \
    --max-depth 5 \
    --max-pages 100 \
    --rate-limit 0.5 \
    --user-agent "MyCrawler/1.0 (+https://mysite.com/bot)" \
    --respect-robots \
    --verbose
```

#### Verify Output

```bash
# Check output directory structure
ls -la ./data/
# Expected:
# data/
# ├── raw/                    # Raw HTML files
# │   └── example.com/
# │       └── index.html
# ├── extracted/              # Extracted JSON data
# │   └── example.com/
# │       └── index.json
# ├── metadata/               # Crawl metadata
# │   └── crawl_stats.json
# └── logs/                   # Crawl logs
#     └── crawler.log

# View extracted content
cat ./data/extracted/example.com/index.json
```

### Troubleshooting

#### Common Issues

**1. "Redis connection refused"**
```bash
# Check if Redis is running
redis-cli ping

# If not, start it
docker start redis-crawler  # If using Docker
sudo systemctl start redis-server  # If local install

# Check Redis logs
docker logs redis-crawler
# or
sudo journalctl -u redis-server
```

**2. "Module not found: crawler"**
```bash
# Make sure you're in the virtual environment
source .venv/bin/activate

# Reinstall the package
pip install -e .
```

**3. "Permission denied" when writing output**
```bash
# Check directory permissions
ls -la ./data/

# Create directory with proper permissions
mkdir -p ./data && chmod 755 ./data
```

**4. "SSL certificate verify failed"**
```bash
# Update certificates
pip install --upgrade certifi

# Or disable SSL verification (not recommended for production)
python -m crawler --seed-url https://example.com --no-verify-ssl
```

**5. "Rate limited (429)" errors**
```bash
# Increase delay between requests
python -m crawler --seed-url https://example.com --rate-limit 2.0

# The crawler will automatically back off, but you can start slower
```

**6. "Playwright not found" for JS rendering**
```bash
# Install Playwright and browsers
pip install playwright
playwright install chromium

# Verify installation
python -c "from playwright.sync_api import sync_playwright; print('Playwright OK')"
```

#### Debug Mode

```bash
# Run with maximum verbosity
python -m crawler \
    --seed-url https://example.com \
    --output ./data \
    --log-level DEBUG \
    --log-file ./crawler-debug.log

# View real-time logs
tail -f ./crawler-debug.log
```

#### Health Check Script

Create `check_setup.py`:
```python
#!/usr/bin/env python
"""Verify crawler setup is complete."""
import sys

def check_python():
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print(f"❌ Python 3.11+ required (found {version.major}.{version.minor})")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_redis():
    try:
        import redis
        r = redis.Redis()
        r.ping()
        print("✅ Redis connected")
        return True
    except Exception as e:
        print(f"❌ Redis: {e}")
        return False

def check_crawler():
    try:
        import crawler
        print(f"✅ Crawler installed (v{crawler.__version__})")
        return True
    except ImportError as e:
        print(f"❌ Crawler: {e}")
        return False

def check_optional():
    results = []

    # Check Playwright
    try:
        from playwright.sync_api import sync_playwright
        results.append("✅ Playwright (JS rendering)")
    except ImportError:
        results.append("⚠️  Playwright not installed (optional)")

    # Check ML dependencies
    try:
        import sentence_transformers
        results.append("✅ sentence-transformers (ML features)")
    except ImportError:
        results.append("⚠️  sentence-transformers not installed (optional)")

    for r in results:
        print(r)

if __name__ == "__main__":
    print("Checking Adaptive Crawler setup...\n")

    all_ok = all([
        check_python(),
        check_redis(),
        check_crawler(),
    ])

    print()
    check_optional()

    print()
    if all_ok:
        print("🎉 All required components are ready!")
        sys.exit(0)
    else:
        print("❌ Some required components are missing.")
        sys.exit(1)
```

Run the health check:
```bash
python check_setup.py
```

---

## Architecture

### Directory Structure

```
crawler/
├── core/                    # Core orchestration
│   ├── crawler.py          # Main crawler orchestrator
│   ├── fetcher.py          # HTTP client + compliance pipeline
│   ├── scheduler.py        # URL frontier management
│   ├── renderer.py         # Playwright JS rendering
│   ├── distributed.py      # Multi-worker coordination
│   └── recrawl_scheduler.py # Scheduled recrawling
│
├── compliance/             # Legal compliance
│   ├── robots_parser.py    # RFC 9309 robots.txt parsing
│   ├── rate_limiter.py     # Adaptive per-domain rate limiting
│   └── sitemap_parser.py   # XML sitemap parsing
│
├── legal/                  # Legal frameworks
│   ├── cfaa_checker.py     # CFAA authorization checks
│   └── pii_detector.py     # GDPR/CCPA PII handling
│
├── extraction/             # Content extraction
│   ├── link_extractor.py   # URL discovery
│   └── content_extractor.py # CSS selector-based extraction
│
├── adaptive/               # ML-based adaptation
│   ├── structure_analyzer.py   # DOM fingerprinting
│   ├── change_detector.py      # Structure comparison
│   └── strategy_learner.py     # CSS selector inference
│
├── storage/                # Persistence
│   ├── url_store.py        # Visited URL tracking
│   ├── robots_cache.py     # robots.txt caching
│   └── structure_store.py  # Learned structures (Redis)
│
└── utils/                  # Utilities
    ├── logging.py          # Structured logging
    └── metrics.py          # Statistics tracking
```

### Component Interaction

```
┌─────────────────────────────────────────────────────────────────────┐
│                           CRAWLER                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │  Scheduler  │◄──►│   Fetcher   │◄──►│  Extractor  │             │
│  │             │    │             │    │             │             │
│  │ • URL Queue │    │ • Compliance│    │ • Links     │             │
│  │ • Priorities│    │ • HTTP      │    │ • Content   │             │
│  │ • Dedup     │    │ • Retries   │    │ • Metadata  │             │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘             │
│         │                  │                  │                     │
│         │                  ▼                  │                     │
│         │         ┌─────────────────┐         │                     │
│         │         │    ADAPTIVE     │         │                     │
│         │         │    SYSTEM       │◄────────┘                     │
│         │         │                 │                               │
│         │         │ • Analyzer      │                               │
│         │         │ • Detector      │                               │
│         │         │ • Learner       │                               │
│         │         └────────┬────────┘                               │
│         │                  │                                        │
│         ▼                  ▼                                        │
│  ┌─────────────────────────────────────────────────┐               │
│  │                    STORAGE                       │               │
│  │                                                  │               │
│  │  Redis: Structures, Strategies, URLs, Robots    │               │
│  │  Disk:  HTML, JSON, Extracted Content           │               │
│  └──────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Crawl Workflow

### Complete Crawl Cycle

```
START
  │
  ▼
┌─────────────────────────────────────┐
│ 1. INITIALIZE                       │
│    • Connect to Redis               │
│    • Load configuration             │
│    • Create output directory        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2. ADD SEED URLs TO FRONTIER        │
│    • Validate URLs                  │
│    • Check allowed domains          │
│    • Initialize depth = 0           │
└──────────────┬──────────────────────┘
               │
               ▼
        ┌──────────────┐
        │  MAIN LOOP   │◄─────────────────────────────────┐
        └──────┬───────┘                                  │
               │                                          │
               ▼                                          │
┌─────────────────────────────────────┐                  │
│ 3. GET NEXT URL FROM SCHEDULER      │                  │
│    • Priority: breadth-first        │                  │
│    • Respect domain politeness      │                  │
│    • Check max_depth, max_pages     │                  │
└──────────────┬──────────────────────┘                  │
               │                                          │
               ▼                                          │
┌─────────────────────────────────────┐                  │
│ 4. FETCH URL (Compliance Pipeline)  │                  │
│    • CFAA check                     │                  │
│    • robots.txt check               │                  │
│    • Rate limit wait                │                  │
│    • HTTP GET with timeout          │                  │
│    • GDPR/PII processing            │                  │
└──────────────┬──────────────────────┘                  │
               │                                          │
               ▼                                          │
        ┌──────────────┐                                  │
        │   SUCCESS?   │                                  │
        └──────┬───────┘                                  │
               │                                          │
      ┌────────┴────────┐                                 │
      │                 │                                 │
      ▼                 ▼                                 │
┌──────────┐     ┌──────────────┐                        │
│ BLOCKED/ │     │   SUCCESS    │                        │
│ ERROR    │     └──────┬───────┘                        │
└────┬─────┘            │                                │
     │                  ▼                                │
     │     ┌─────────────────────────────────────┐      │
     │     │ 5. SAVE RAW CONTENT                 │      │
     │     │    • HTML to disk                   │      │
     │     │    • Metadata (headers, status)     │      │
     │     └──────────────┬──────────────────────┘      │
     │                    │                              │
     │                    ▼                              │
     │     ┌─────────────────────────────────────┐      │
     │     │ 6. EXTRACT & QUEUE LINKS            │      │
     │     │    • Parse <a href>                 │      │
     │     │    • Normalize URLs                 │      │
     │     │    • Add to scheduler               │      │
     │     └──────────────┬──────────────────────┘      │
     │                    │                              │
     │                    ▼                              │
     │     ┌─────────────────────────────────────┐      │
     │     │ 7. ADAPTIVE ANALYSIS                │      │
     │     │    • Analyze current structure      │      │
     │     │    • Compare with stored            │      │
     │     │    • Detect changes                 │      │
     │     │    • Adapt strategy if needed       │      │
     │     └──────────────┬──────────────────────┘      │
     │                    │                              │
     │                    ▼                              │
     │     ┌─────────────────────────────────────┐      │
     │     │ 8. EXTRACT CONTENT                  │      │
     │     │    • Apply learned CSS selectors    │      │
     │     │    • Extract title, content, meta   │      │
     │     │    • Save extracted JSON            │      │
     │     └──────────────┬──────────────────────┘      │
     │                    │                              │
     ▼                    ▼                              │
┌─────────────────────────────────────┐                 │
│ 9. UPDATE STATISTICS                │                 │
│    • Increment counters             │                 │
│    • Log progress                   │                 │
└──────────────┬──────────────────────┘                 │
               │                                         │
               ▼                                         │
        ┌──────────────┐                                │
        │ MORE URLs?   │────── YES ─────────────────────┘
        └──────┬───────┘
               │ NO
               ▼
┌─────────────────────────────────────┐
│ 10. FINALIZE                        │
│     • Close connections             │
│     • Return CrawlerStats           │
└─────────────────────────────────────┘
               │
               ▼
             END
```

---

## Adaptive Learning System

The adaptive system learns how to extract content from websites and automatically adjusts when sites change.

### Structure Analysis

The `StructureAnalyzer` creates a fingerprint of each page's DOM:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PAGE STRUCTURE FINGERPRINT                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Tag Hierarchy          CSS Classes           Element IDs        │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │ div: 45      │      │ .article: 12 │      │ #header      │  │
│  │ span: 23     │      │ .nav-item: 8 │      │ #content     │  │
│  │ a: 67        │      │ .btn: 15     │      │ #footer      │  │
│  │ p: 12        │      │ .card: 6     │      │ #sidebar     │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│                                                                  │
│  Semantic Landmarks     Navigation            Content Regions    │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │ <article>    │      │ nav.main-nav │      │ .post-body   │  │
│  │ <nav>        │      │ ul.menu      │      │ article      │  │
│  │ <header>     │      │ .breadcrumb  │      │ .content     │  │
│  │ <footer>     │      └──────────────┘      └──────────────┘  │
│  └──────────────┘                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Change Detection

The `ChangeDetector` compares structures using weighted similarity:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIMILARITY CALCULATION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Component              Weight    Example Similarity             │
│  ─────────────────────────────────────────────────────          │
│  Tag Hierarchy          30%       0.95 (minor changes)          │
│  Content Regions        25%       0.90 (same regions)           │
│  Navigation             15%       1.00 (unchanged)              │
│  Semantic Landmarks     15%       0.85 (added footer)           │
│  CSS Classes            10%       0.75 (renamed some)           │
│  Element IDs             5%       1.00 (unchanged)              │
│  ─────────────────────────────────────────────────────          │
│  WEIGHTED TOTAL                   0.92 (MINOR change)           │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                    CLASSIFICATION THRESHOLDS                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ≥ 95%  COSMETIC   ──►  CSS-only changes, keep strategy         │
│  85-95% MINOR      ──►  Small tweaks, keep strategy             │
│  70-85% MODERATE   ──►  Significant changes, may adapt          │
│  < 70%  BREAKING   ──►  Major redesign, re-learn strategy       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Strategy Learning

The `StrategyLearner` infers CSS selectors for content extraction:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SELECTOR INFERENCE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  For each field (title, content, date, author):                 │
│                                                                  │
│  1. Try patterns in order of confidence:                        │
│                                                                  │
│     TITLE PATTERNS                 CONTENT PATTERNS             │
│     ───────────────                ─────────────────             │
│     h1.title        (0.90)         article        (0.90)        │
│     h1.entry-title  (0.90)         main           (0.85)        │
│     h1.post-title   (0.90)         .article-content (0.80)      │
│     article h1      (0.85)         .post-content  (0.80)        │
│     .article-title  (0.80)         .content       (0.70)        │
│     h1              (0.70)         body           (0.75) ◄─ fallback
│     title           (0.75) ◄─ fallback                          │
│                                                                  │
│  2. Adjust confidence based on matches:                         │
│     • 1 element:   keep base confidence                         │
│     • 2-3 elements: × 0.9                                       │
│     • 4+ elements:  × 0.7                                       │
│                                                                  │
│  3. Accept if confidence ≥ min_confidence (0.5)                 │
│                                                                  │
│  4. Build ExtractionStrategy with selected rules                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Adaptation Flow

```
                    FIRST VISIT                    SUBSEQUENT VISITS
                    ───────────                    ─────────────────
                         │                               │
                         ▼                               ▼
              ┌─────────────────────┐      ┌─────────────────────┐
              │ Analyze Structure   │      │ Analyze Structure   │
              └──────────┬──────────┘      └──────────┬──────────┘
                         │                            │
                         ▼                            ▼
              ┌─────────────────────┐      ┌─────────────────────┐
              │ Infer Strategy      │      │ Load Stored         │
              │ (pattern matching)  │      │ Structure + Strategy│
              └──────────┬──────────┘      └──────────┬──────────┘
                         │                            │
                         ▼                            ▼
              ┌─────────────────────┐      ┌─────────────────────┐
              │ Save to Redis       │      │ Compare Structures  │
              │ • Structure         │      │ (similarity score)  │
              │ • Strategy          │      └──────────┬──────────┘
              └──────────┬──────────┘                 │
                         │                  ┌────────┴────────┐
                         │                  │                 │
                         │           ≥ 70% similar    < 70% similar
                         │                  │                 │
                         │                  ▼                 ▼
                         │       ┌─────────────────┐ ┌─────────────────┐
                         │       │ Use Existing    │ │ Adapt Strategy  │
                         │       │ Strategy        │ │ • Re-infer      │
                         │       └────────┬────────┘ │ • Save new      │
                         │                │          │ • Log change    │
                         │                │          └────────┬────────┘
                         ▼                ▼                   │
              ┌──────────────────────────────────────────────────┐
              │              EXTRACT CONTENT                      │
              │  Apply CSS selectors to get title, content, etc.  │
              └───────────────────────────────────────────────────┘
```

---

## Sitemap Processing

The crawler includes comprehensive XML sitemap support for efficient URL discovery. Instead of crawling an entire site link-by-link, sitemaps provide a structured index of all pages a site wants indexed.

### What is a Sitemap?

XML sitemaps are files that list URLs for a site along with metadata about each URL (when it was last updated, how often it changes, how important it is relative to other URLs). Search engines use sitemaps to crawl sites more efficiently.

### Sitemap Types Supported

```
┌─────────────────────────────────────────────────────────────────┐
│                    SITEMAP FORMATS                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. URLSET (Standard Sitemap)                                   │
│     └──► Contains individual URLs with metadata                 │
│                                                                  │
│     <?xml version="1.0" encoding="UTF-8"?>                      │
│     <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">│
│       <url>                                                      │
│         <loc>https://example.com/page1</loc>                    │
│         <lastmod>2025-01-15</lastmod>                           │
│         <changefreq>weekly</changefreq>                         │
│         <priority>0.8</priority>                                │
│       </url>                                                     │
│     </urlset>                                                    │
│                                                                  │
│  2. SITEMAPINDEX (Sitemap Index)                                │
│     └──► Points to multiple child sitemaps                      │
│     └──► Used by large sites (50,000+ URLs)                     │
│                                                                  │
│     <sitemapindex xmlns="...">                                  │
│       <sitemap>                                                  │
│         <loc>https://example.com/sitemap-articles.xml</loc>     │
│         <lastmod>2025-01-20</lastmod>                           │
│       </sitemap>                                                 │
│       <sitemap>                                                  │
│         <loc>https://example.com/sitemap-products.xml</loc>     │
│       </sitemap>                                                 │
│     </sitemapindex>                                              │
│                                                                  │
│  3. GZIP COMPRESSED (.xml.gz)                                   │
│     └──► Automatically detected and decompressed                │
│     └──► Common for large sitemaps                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### URL Metadata Fields

| Field | Description | Example |
|-------|-------------|---------|
| `loc` | The URL (required) | `https://example.com/page` |
| `lastmod` | Last modification date | `2025-01-15` or `2025-01-15T10:30:00Z` |
| `changefreq` | How often the page changes | `always`, `hourly`, `daily`, `weekly`, `monthly`, `yearly`, `never` |
| `priority` | Relative importance (0.0-1.0) | `0.8` (higher = more important) |

### How Sitemap Processing Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    SITEMAP PROCESSING FLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. DISCOVERY                                                    │
│     ├──► Check robots.txt for Sitemap: directives               │
│     └──► Try common paths: /sitemap.xml, /sitemap_index.xml     │
│                                                                  │
│  2. FETCH                                                        │
│     ├──► HTTP GET with User-Agent                               │
│     ├──► Handle gzip compression                                │
│     └──► Follow redirects                                        │
│                                                                  │
│  3. PARSE                                                        │
│     ├──► Detect type (urlset vs sitemapindex)                   │
│     ├──► Extract URLs and metadata                              │
│     └──► Validate against sitemap protocol                      │
│                                                                  │
│  4. RECURSE (for sitemap indexes)                               │
│     ├──► Queue child sitemaps                                   │
│     ├──► Track processed sitemaps (avoid duplicates)            │
│     └──► Respect max_sitemaps limit                             │
│                                                                  │
│  5. YIELD URLS                                                   │
│     ├──► Stream URLs as discovered                              │
│     ├──► Include metadata (lastmod, changefreq, priority)       │
│     └──► Filter by domain if specified                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Python API

```python
from crawler.compliance import (
    SitemapFetcher,
    SitemapParser,
    fetch_sitemap_urls,
    ChangeFrequency,
)

# Quick fetch all URLs from a domain's sitemaps
urls = await fetch_sitemap_urls(
    domain="example.com",
    user_agent="MyCrawler/1.0",
    timeout=30.0,
)
for url in urls:
    print(f"{url.loc} - last modified: {url.lastmod}")

# Full control with SitemapFetcher
async with SitemapFetcher(
    user_agent="MyCrawler/1.0",
    max_sitemaps=100,           # Max sitemap files to process
    max_urls_per_sitemap=50000, # Max URLs per sitemap
) as fetcher:
    # Discover sitemaps for a domain
    sitemap_urls = await fetcher.discover_sitemaps("example.com")

    # Fetch and parse all sitemaps (handles indexes recursively)
    async for sitemap in fetcher.fetch_all_sitemaps(sitemap_urls):
        print(f"Sitemap: {sitemap.url}")
        print(f"  Is index: {sitemap.is_index}")
        print(f"  URLs: {len(sitemap.urls)}")
        print(f"  Child sitemaps: {len(sitemap.sitemaps)}")

        # Access individual URLs
        for url in sitemap.urls:
            print(f"  - {url.loc}")
            if url.changefreq == ChangeFrequency.DAILY:
                print("    (updates daily)")

# Parse sitemap content directly
parser = SitemapParser()
sitemap = parser.parse(
    content=xml_bytes,
    url="https://example.com/sitemap.xml",
    status_code=200,
)
```

### Integration with Crawler

```python
from crawler.core import Crawler
from crawler.compliance import SitemapFetcher

async def crawl_from_sitemap():
    # Fetch URLs from sitemap first
    async with SitemapFetcher() as fetcher:
        sitemap_urls = await fetcher.discover_sitemaps("example.com")
        seed_urls = []
        async for url in fetcher.get_all_urls(sitemap_urls):
            seed_urls.append(url.loc)

    # Use sitemap URLs as seeds
    config = CrawlConfig(
        seed_urls=seed_urls[:1000],  # Limit initial seeds
        output_dir="./data",
        max_pages=5000,
    )

    async with Crawler(config) as crawler:
        stats = await crawler.crawl()
```

---

## JavaScript Rendering

Modern websites often rely heavily on JavaScript to render content. Single Page Applications (SPAs) built with React, Vue, Angular, or similar frameworks may show only a loading spinner when fetched with a simple HTTP request. The JavaScript Rendering module uses Playwright to execute JavaScript and capture the fully-rendered DOM.

### When JS Rendering is Needed

```
┌─────────────────────────────────────────────────────────────────┐
│                 JS RENDERING DETECTION                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  The crawler automatically detects when JS rendering is needed  │
│  by looking for common SPA framework patterns:                  │
│                                                                  │
│  REACT                          VUE                             │
│  ─────                          ───                             │
│  • <div id="root"></div>        • <div id="app"></div>         │
│  • data-reactroot               • data-v- attributes            │
│  • __NEXT_DATA__ (Next.js)      • __NUXT__ (Nuxt.js)           │
│                                                                  │
│  ANGULAR                        SVELTE                          │
│  ───────                        ──────                          │
│  • ng-app attribute             • svelte- classes               │
│  • _nghost attributes           • __svelte_                     │
│  • ng-version                                                    │
│                                                                  │
│  GENERIC SPA INDICATORS                                         │
│  ─────────────────────                                          │
│  • Empty body with JS includes                                  │
│  • "Loading..." placeholder text                                │
│  • Minimal HTML with large JS bundles                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 JS RENDERING ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                      ┌─────────────────┐                        │
│                      │  HybridFetcher  │                        │
│                      │  (Entry Point)  │                        │
│                      └────────┬────────┘                        │
│                               │                                  │
│            ┌──────────────────┼──────────────────┐              │
│            │                  │                  │              │
│            ▼                  ▼                  ▼              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │   HTTP Fetch    │ │ JS Requirement  │ │   JSRenderer    │  │
│  │   (Fast Path)   │ │    Detector     │ │  (Slow Path)    │  │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘  │
│           │                   │                   │            │
│           │          ┌───────┴───────┐           │            │
│           │          │ Needs JS?     │           │            │
│           │          └───────┬───────┘           │            │
│           │                  │                   │            │
│           │        ┌────────┴────────┐          │            │
│           │        │                 │          │            │
│           │       NO                YES         │            │
│           │        │                 │          │            │
│           ▼        ▼                 ▼          │            │
│      Return HTML directly      Use JSRenderer ◄─┘            │
│                                      │                        │
│                                      ▼                        │
│                              ┌─────────────────┐              │
│                              │  BrowserPool    │              │
│                              │                 │              │
│                              │ • Chromium      │              │
│                              │ • Firefox       │              │
│                              │ • WebKit        │              │
│                              └────────┬────────┘              │
│                                       │                        │
│                                       ▼                        │
│                              ┌─────────────────┐              │
│                              │ Rendered HTML   │              │
│                              │ + Screenshots   │              │
│                              │ + Console Logs  │              │
│                              └─────────────────┘              │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

### Wait Strategies

The renderer supports multiple strategies for determining when a page is "ready":

| Strategy | Description | Best For |
|----------|-------------|----------|
| `load` | Wait for window.onload | Simple pages |
| `domcontentloaded` | Wait for DOMContentLoaded | Static content |
| `networkidle` | Wait until no network requests for 500ms | API-heavy SPAs |
| `selector` | Wait for specific CSS selector to appear | Known content markers |
| `function` | Wait for custom JS function to return true | Complex conditions |

### Python API

```python
from crawler.core import (
    JSRenderer,
    BrowserPool,
    HybridFetcher,
    JSRequirementDetector,
    RenderConfig,
    WaitStrategy,
)

# Basic rendering
async with JSRenderer() as renderer:
    result = await renderer.render("https://spa-example.com")
    print(f"Status: {result.status_code}")
    print(f"HTML length: {len(result.html)}")
    print(f"Render time: {result.render_time_ms}ms")
    print(f"Console logs: {result.console_logs}")

# With custom configuration
config = RenderConfig(
    wait_strategy=WaitStrategy.NETWORKIDLE,
    timeout_ms=30000,
    viewport_width=1920,
    viewport_height=1080,
    user_agent="MyBot/1.0",
    block_resources=["image", "media", "font"],  # Speed up rendering
    capture_screenshot=True,
)

async with JSRenderer() as renderer:
    result = await renderer.render("https://example.com", config)
    if result.screenshot:
        with open("screenshot.png", "wb") as f:
            f.write(result.screenshot)

# Smart rendering (only when needed)
async with JSRenderer() as renderer:
    # First fetch with HTTP
    http_html = "<html>..."  # From regular HTTP fetch

    # Check if JS rendering is needed
    result = await renderer.render_if_needed(
        url="https://example.com",
        initial_html=http_html,
    )

    if result.was_rendered:
        print("Page required JS rendering")
    else:
        print("HTTP fetch was sufficient")

# Browser pool for high-volume rendering
async with BrowserPool(
    max_browsers=3,
    max_contexts_per_browser=5,
    browser_type="chromium",  # or "firefox", "webkit"
) as pool:
    # Acquire browser context
    async with pool.acquire() as context:
        page = await context.new_page()
        await page.goto("https://example.com")
        content = await page.content()

# Hybrid fetcher (combines HTTP + JS rendering)
async with HybridFetcher(
    js_renderer=renderer,
    http_timeout=10.0,
) as fetcher:
    # Automatically uses JS rendering when needed
    result = await fetcher.fetch("https://spa-example.com")
    print(f"Used JS: {result.used_js_rendering}")
    print(f"HTML: {result.html[:500]}...")
```

### Installation

```bash
# Install Playwright
pip install playwright

# Install browser binaries
playwright install chromium  # or: playwright install firefox webkit
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `browser_type` | `chromium` | Browser engine: `chromium`, `firefox`, `webkit` |
| `headless` | `True` | Run browser without GUI |
| `timeout_ms` | `30000` | Page load timeout |
| `wait_strategy` | `networkidle` | When to consider page loaded |
| `viewport_width` | `1280` | Browser viewport width |
| `viewport_height` | `720` | Browser viewport height |
| `block_resources` | `[]` | Resource types to block for speed |
| `capture_screenshot` | `False` | Take screenshot after render |
| `capture_console` | `True` | Capture browser console logs |

---

## Distributed Crawling

For large-scale crawling operations, the distributed crawling system enables multiple workers to coordinate and process URLs in parallel across machines.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                 DISTRIBUTED CRAWLING SYSTEM                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    ┌─────────────────────┐                      │
│                    │ DistributedCrawl    │                      │
│                    │     Manager         │                      │
│                    │                     │                      │
│                    │ • Create jobs       │                      │
│                    │ • Monitor progress  │                      │
│                    │ • Collect results   │                      │
│                    └──────────┬──────────┘                      │
│                               │                                  │
│                               ▼                                  │
│                    ┌─────────────────────┐                      │
│                    │       REDIS         │                      │
│                    │                     │                      │
│                    │ • URL Queue         │                      │
│                    │ • Worker Registry   │                      │
│                    │ • Job State         │                      │
│                    │ • Leader Lock       │                      │
│                    └──────────┬──────────┘                      │
│                               │                                  │
│          ┌────────────────────┼────────────────────┐            │
│          │                    │                    │            │
│          ▼                    ▼                    ▼            │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐    │
│  │   Worker 1    │   │   Worker 2    │   │   Worker 3    │    │
│  │   (Leader)    │   │               │   │               │    │
│  │               │   │               │   │               │    │
│  │ • Claim URLs  │   │ • Claim URLs  │   │ • Claim URLs  │    │
│  │ • Fetch pages │   │ • Fetch pages │   │ • Fetch pages │    │
│  │ • Heartbeat   │   │ • Heartbeat   │   │ • Heartbeat   │    │
│  │ • Coordinate  │   │               │   │               │    │
│  └───────────────┘   └───────────────┘   └───────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. DistributedQueue

The URL queue manages URL distribution across workers with atomic operations:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED QUEUE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  OPERATIONS (all atomic)                                        │
│  ───────────────────────                                        │
│                                                                  │
│  add_url(task)                                                   │
│    └──► Add URL if not already queued/visited                   │
│    └──► Sets priority for ordering                              │
│                                                                  │
│  claim_url(worker_id)                                           │
│    └──► Atomically pop highest priority URL                     │
│    └──► Mark as processing by this worker                       │
│    └──► Set claim timestamp for timeout detection               │
│                                                                  │
│  complete_url(url, success)                                     │
│    └──► Mark URL as completed/failed                            │
│    └──► Release from processing state                           │
│                                                                  │
│  recover_stale_urls()                                           │
│    └──► Find URLs claimed but not completed (timeout)           │
│    └──► Re-queue for another worker to process                  │
│                                                                  │
│  REDIS KEYS                                                      │
│  ──────────                                                      │
│  job:{id}:pending     - Sorted set (priority queue)             │
│  job:{id}:processing  - Hash (url -> worker_id)                 │
│  job:{id}:completed   - Set (finished URLs)                     │
│  job:{id}:failed      - Set (failed URLs)                       │
│  job:{id}:seen        - Set (all URLs ever added)               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 2. WorkerCoordinator

Manages worker registration, heartbeats, and leader election:

```
┌─────────────────────────────────────────────────────────────────┐
│                    WORKER COORDINATION                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  HEARTBEATS                                                      │
│  ──────────                                                      │
│  • Workers send heartbeat every N seconds                       │
│  • Heartbeat includes: URLs processed, errors, last activity    │
│  • Missing heartbeats = worker presumed dead                    │
│                                                                  │
│  LEADER ELECTION                                                 │
│  ───────────────                                                 │
│  • Redis SETNX for distributed lock                             │
│  • Leader performs coordination tasks:                          │
│    - Cleanup dead workers                                       │
│    - Recover stale URLs                                         │
│    - Publish global stats                                       │
│  • Lock auto-expires (TTL) if leader dies                       │
│                                                                  │
│  WORKER STATES                                                   │
│  ─────────────                                                   │
│  IDLE       → Waiting for work                                  │
│  ACTIVE     → Processing URLs                                   │
│  PAUSED     → Temporarily stopped                               │
│  STOPPING   → Graceful shutdown in progress                     │
│  DEAD       → No heartbeat, presumed failed                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Python API

```python
from crawler.core import (
    DistributedQueue,
    DistributedCrawlManager,
    CrawlerWorker,
    WorkerCoordinator,
    CrawlJob,
    URLTask,
    JobState,
)
import redis.asyncio as redis

# Create a distributed crawl job
async def create_job():
    redis_client = redis.from_url("redis://localhost:6379/0")

    manager = DistributedCrawlManager(redis_client)

    # Create job with seed URLs
    job = await manager.create_job(
        seed_urls=["https://example.com", "https://example.com/about"],
        job_id="my-crawl-001",
        max_urls=10000,
        max_depth=5,
    )

    print(f"Job created: {job.job_id}")
    print(f"State: {job.state}")

    return job

# Run a worker
async def run_worker(job_id: str):
    redis_client = redis.from_url("redis://localhost:6379/0")

    worker = CrawlerWorker(
        redis_client=redis_client,
        job_id=job_id,
        worker_id="worker-1",  # Unique per worker
        heartbeat_interval=5.0,
        claim_timeout=300.0,   # 5 minutes to process a URL
    )

    # Start worker (runs until job complete or stopped)
    await worker.start()

# Monitor job progress
async def monitor_job(job_id: str):
    redis_client = redis.from_url("redis://localhost:6379/0")
    manager = DistributedCrawlManager(redis_client)

    while True:
        status = await manager.get_job_status(job_id)

        print(f"Pending: {status['pending_urls']}")
        print(f"Processing: {status['processing_urls']}")
        print(f"Completed: {status['completed_urls']}")
        print(f"Failed: {status['failed_urls']}")
        print(f"Workers: {status['active_workers']}")

        if status['state'] == JobState.COMPLETED:
            break

        await asyncio.sleep(5)

# Direct queue operations
async def queue_operations():
    redis_client = redis.from_url("redis://localhost:6379/0")
    queue = DistributedQueue(redis_client, job_id="my-crawl-001")

    # Add URLs with priority
    await queue.add_url(URLTask(
        url="https://example.com/important",
        depth=0,
        priority=10,  # Higher = processed first
    ))

    # Claim a URL for processing
    task = await queue.claim_url(worker_id="worker-1")
    if task:
        print(f"Claimed: {task.url}")

        # Process the URL...

        # Mark as complete
        await queue.complete_url(task.url, success=True)

    # Recover timed-out URLs
    recovered = await queue.recover_stale_urls()
    print(f"Recovered {recovered} stale URLs")
```

### Running Multiple Workers

```bash
# Terminal 1: Create job
python -c "
import asyncio
from my_crawler import create_job
asyncio.run(create_job())
"

# Terminal 2: Worker 1
WORKER_ID=worker-1 python -m crawler.worker --job-id my-crawl-001

# Terminal 3: Worker 2
WORKER_ID=worker-2 python -m crawler.worker --job-id my-crawl-001

# Terminal 4: Worker 3 (on different machine)
WORKER_ID=worker-3 REDIS_URL=redis://192.168.1.100:6379 \
    python -m crawler.worker --job-id my-crawl-001
```

### Job States

| State | Description |
|-------|-------------|
| `PENDING` | Job created, not started |
| `RUNNING` | Workers actively processing |
| `PAUSED` | Temporarily stopped |
| `COMPLETED` | All URLs processed |
| `FAILED` | Job failed (too many errors) |
| `CANCELLED` | Manually cancelled |

---

## Scheduled Recrawling

The scheduled recrawling system enables periodic re-crawling of URLs to detect content changes. It supports cron-like scheduling, adaptive intervals based on change frequency, and sitemap-based scheduling hints.

### Why Scheduled Recrawling?

```
┌─────────────────────────────────────────────────────────────────┐
│                    USE CASES                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. NEWS MONITORING                                              │
│     └──► Check news sites every 15 minutes for new articles     │
│                                                                  │
│  2. PRICE TRACKING                                               │
│     └──► Monitor e-commerce prices daily                        │
│                                                                  │
│  3. COMPLIANCE CHECKING                                          │
│     └──► Verify terms of service weekly                         │
│                                                                  │
│  4. SEO MONITORING                                               │
│     └──► Track competitor content changes                       │
│                                                                  │
│  5. ARCHIVAL                                                     │
│     └──► Capture snapshots of pages over time                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Scheduling Options

#### 1. Cron Expressions

Standard cron syntax for precise scheduling:

```
┌───────────── minute (0 - 59)
│ ┌───────────── hour (0 - 23)
│ │ ┌───────────── day of month (1 - 31)
│ │ │ ┌───────────── month (1 - 12)
│ │ │ │ ┌───────────── day of week (0 - 6, 0 = Sunday)
│ │ │ │ │
│ │ │ │ │
* * * * *

Examples:
─────────
0 * * * *       Every hour at minute 0
*/15 * * * *    Every 15 minutes
0 9 * * 1-5     9 AM on weekdays
0 0 1 * *       First day of each month at midnight
0 */6 * * *     Every 6 hours
```

#### 2. Interval-Based

Simple time intervals:

```python
from crawler.core import ScheduleInterval

# Predefined intervals
ScheduleInterval.MINUTES_15    # Every 15 minutes
ScheduleInterval.HOURLY        # Every hour
ScheduleInterval.DAILY         # Once a day
ScheduleInterval.WEEKLY        # Once a week
ScheduleInterval.MONTHLY       # Once a month
```

#### 3. Adaptive Scheduling

Automatically adjusts recrawl frequency based on how often content actually changes:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ADAPTIVE SCHEDULING                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Initial interval: 1 hour                                       │
│                                                                  │
│  Crawl 1: No change   → Interval × 1.5 = 1.5 hours             │
│  Crawl 2: No change   → Interval × 1.5 = 2.25 hours            │
│  Crawl 3: CHANGED!    → Interval × 0.5 = 1.125 hours           │
│  Crawl 4: No change   → Interval × 1.5 = 1.69 hours            │
│  ...                                                             │
│                                                                  │
│  Bounds:                                                         │
│  • Min interval: 15 minutes (never faster)                      │
│  • Max interval: 7 days (never slower)                          │
│                                                                  │
│  Benefits:                                                       │
│  • Frequently-changing pages crawled more often                 │
│  • Stable pages crawled less often                              │
│  • Automatically optimizes crawl resources                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 RECRAWL SCHEDULER SYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│              ┌───────────────────────────┐                      │
│              │     RecrawlScheduler      │                      │
│              │                           │                      │
│              │ • Manage URL schedules    │                      │
│              │ • Check for due URLs      │                      │
│              │ • Trigger recrawls        │                      │
│              │ • Update intervals        │                      │
│              └─────────────┬─────────────┘                      │
│                            │                                     │
│            ┌───────────────┼───────────────┐                    │
│            │               │               │                    │
│            ▼               ▼               ▼                    │
│  ┌─────────────────┐ ┌───────────┐ ┌─────────────────┐        │
│  │  CronSchedule   │ │  Redis    │ │ SitemapBased    │        │
│  │                 │ │  Store    │ │   Scheduler     │        │
│  │ • Parse cron    │ │           │ │                 │        │
│  │ • Next run time │ │ Schedules │ │ • Use lastmod   │        │
│  │ • Validate      │ │ History   │ │ • Use changefreq│        │
│  └─────────────────┘ │ Metrics   │ │ • Batch schedule│        │
│                      └───────────┘ └─────────────────┘        │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  SCHEDULE RECORD                         │   │
│  │                                                          │   │
│  │  url: https://example.com/page                          │   │
│  │  schedule_type: cron | interval | adaptive              │   │
│  │  cron_expr: "0 */6 * * *"                               │   │
│  │  interval_seconds: 21600                                │   │
│  │  last_crawled: 2025-01-30T10:00:00Z                     │   │
│  │  next_crawl: 2025-01-30T16:00:00Z                       │   │
│  │  consecutive_no_change: 3                               │   │
│  │  total_crawls: 47                                       │   │
│  │  total_changes: 12                                      │   │
│  │  enabled: true                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Python API

```python
from crawler.core import (
    RecrawlScheduler,
    RecrawlSchedule,
    CronSchedule,
    ScheduleInterval,
    AdaptiveScheduleConfig,
    SitemapBasedScheduler,
)
import redis.asyncio as redis

# Basic cron scheduling
async def schedule_with_cron():
    redis_client = redis.from_url("redis://localhost:6379/0")
    scheduler = RecrawlScheduler(redis_client)

    # Schedule URL with cron expression
    schedule = await scheduler.add_url_schedule(
        url="https://news.example.com",
        interval="0 */2 * * *",  # Every 2 hours
    )
    print(f"Next crawl: {schedule.next_crawl}")

# Interval-based scheduling
async def schedule_with_interval():
    redis_client = redis.from_url("redis://localhost:6379/0")
    scheduler = RecrawlScheduler(redis_client)

    # Schedule with predefined interval
    await scheduler.add_url_schedule(
        url="https://blog.example.com",
        interval=ScheduleInterval.DAILY,
    )

    # Or with custom seconds
    await scheduler.add_url_schedule(
        url="https://prices.example.com",
        interval=3600,  # Every hour
    )

# Adaptive scheduling
async def schedule_adaptive():
    redis_client = redis.from_url("redis://localhost:6379/0")

    adaptive_config = AdaptiveScheduleConfig(
        initial_interval=3600,      # Start with 1 hour
        min_interval=900,           # Never faster than 15 minutes
        max_interval=604800,        # Never slower than 1 week
        increase_factor=1.5,        # Slow down by 50% when unchanged
        decrease_factor=0.5,        # Speed up by 50% when changed
    )

    scheduler = RecrawlScheduler(
        redis_client,
        adaptive_config=adaptive_config,
    )

    await scheduler.add_url_schedule(
        url="https://example.com",
        interval=ScheduleInterval.HOURLY,
        adaptive=True,  # Enable adaptive adjustment
    )

# Sitemap-based scheduling
async def schedule_from_sitemap():
    redis_client = redis.from_url("redis://localhost:6379/0")

    sitemap_scheduler = SitemapBasedScheduler(redis_client)

    # Import schedules from sitemap
    await sitemap_scheduler.import_from_sitemap(
        sitemap_url="https://example.com/sitemap.xml",
        default_interval=ScheduleInterval.DAILY,
    )
    # Uses sitemap's changefreq and lastmod to set intelligent intervals:
    # - changefreq: "hourly" → 1 hour interval
    # - changefreq: "daily" → 24 hour interval
    # - lastmod: recent → shorter interval

# Run the scheduler
async def run_scheduler():
    redis_client = redis.from_url("redis://localhost:6379/0")
    scheduler = RecrawlScheduler(redis_client)

    # Define what happens when a URL is due
    async def on_url_due(schedule: RecrawlSchedule):
        print(f"Time to recrawl: {schedule.url}")

        # Perform the crawl...
        content_changed = await crawl_and_check(schedule.url)

        # Report result (updates adaptive interval)
        await scheduler.record_crawl_result(
            url=schedule.url,
            changed=content_changed,
        )

    # Start scheduler loop
    await scheduler.run(
        callback=on_url_due,
        check_interval=60,  # Check for due URLs every minute
    )

# Query schedules
async def query_schedules():
    redis_client = redis.from_url("redis://localhost:6379/0")
    scheduler = RecrawlScheduler(redis_client)

    # Get all schedules
    all_schedules = await scheduler.list_schedules()

    # Get schedules due now
    due_now = await scheduler.get_due_urls()

    # Get schedule for specific URL
    schedule = await scheduler.get_schedule("https://example.com")
    print(f"URL: {schedule.url}")
    print(f"Last crawled: {schedule.last_crawled}")
    print(f"Next crawl: {schedule.next_crawl}")
    print(f"Change rate: {schedule.total_changes}/{schedule.total_crawls}")

    # Disable/enable schedule
    await scheduler.disable_schedule("https://example.com")
    await scheduler.enable_schedule("https://example.com")

    # Remove schedule
    await scheduler.remove_schedule("https://example.com")
```

### Cron Expression Reference

| Expression | Description |
|------------|-------------|
| `* * * * *` | Every minute |
| `*/5 * * * *` | Every 5 minutes |
| `0 * * * *` | Every hour |
| `0 */2 * * *` | Every 2 hours |
| `0 0 * * *` | Daily at midnight |
| `0 9 * * 1-5` | Weekdays at 9 AM |
| `0 0 * * 0` | Weekly on Sunday |
| `0 0 1 * *` | Monthly on the 1st |
| `0 0 1 1 *` | Yearly on Jan 1st |

### Change Frequency to Interval Mapping

When using sitemap-based scheduling, `changefreq` values are mapped to intervals:

| changefreq | Default Interval |
|------------|------------------|
| `always` | 1 hour |
| `hourly` | 1 hour |
| `daily` | 24 hours |
| `weekly` | 7 days |
| `monthly` | 30 days |
| `yearly` | 365 days |
| `never` | Not scheduled |

---

## Configuration

### Environment Variables

```bash
# .env file

# Required
REDIS_URL=redis://localhost:6379/0

# Crawler identity
CRAWLER_USER_AGENT=MyCrawler/1.0 (+https://mysite.com/bot; bot@mysite.com)

# Rate limiting
CRAWLER_DEFAULT_DELAY=1.0      # Seconds between requests per domain
CRAWLER_MAX_CONCURRENT=10      # Global concurrent connections

# GDPR/Privacy
GDPR_ENABLED=true
GDPR_RETENTION_DAYS=365
PII_HANDLING=redact            # redact, pseudonymize, or exclude_page

# Adaptive features
ENABLE_EMBEDDINGS=false
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Python Configuration

```python
from crawler.config import (
    CrawlConfig,
    RateLimitConfig,
    SafetyLimits,
    GDPRConfig,
    PIIHandlingConfig,
)

config = CrawlConfig(
    # Required
    seed_urls=["https://example.com"],
    output_dir="./data",

    # Crawl limits
    max_depth=10,                    # How deep to crawl
    max_pages=1000,                  # Total page limit
    max_pages_per_domain=100,        # Per-domain limit
    allowed_domains=["example.com"], # Restrict to domains
    exclude_patterns=["/admin/"],    # Skip these paths

    # Rate limiting
    rate_limit=RateLimitConfig(
        default_delay=1.0,           # Base delay (seconds)
        min_delay=0.5,               # Minimum delay
        max_delay=60.0,              # Maximum backoff
        adaptive=True,               # Auto-adjust on 429/503
        respect_crawl_delay=True,    # Honor robots.txt
    ),

    # Safety
    safety=SafetyLimits(
        max_page_size_mb=10.0,       # Skip large pages
        request_timeout_seconds=30,  # Per-request timeout
        max_retries=3,               # Retry failed requests
    ),

    # GDPR compliance
    gdpr=GDPRConfig(
        enabled=True,
        retention_days=365,
        collect_only=["url", "title", "content"],
    ),

    # PII handling
    pii=PIIHandlingConfig(
        action="redact",             # What to do with PII
        log_detections=True,         # Audit trail
    ),
)
```

### Configuration Reference

| Category | Option | Default | Description |
|----------|--------|---------|-------------|
| **Crawl** | `max_depth` | 10 | Maximum link depth from seed |
| | `max_pages` | None | Total pages to crawl |
| | `max_pages_per_domain` | None | Per-domain page limit |
| | `allowed_domains` | [] | Restrict to these domains |
| | `exclude_patterns` | [] | URL patterns to skip |
| **Rate Limit** | `default_delay` | 1.0 | Seconds between requests |
| | `min_delay` | 0.5 | Minimum delay floor |
| | `max_delay` | 60.0 | Maximum backoff ceiling |
| | `adaptive` | True | Auto-adjust on rate limits |
| | `respect_crawl_delay` | True | Honor robots.txt delay |
| **Safety** | `max_page_size_mb` | 10.0 | Skip pages larger than |
| | `request_timeout_seconds` | 30 | Request timeout |
| | `max_retries` | 3 | Retry attempts |
| | `verify_ssl` | True | Verify SSL certificates |
| | `block_private_ips` | True | Block 192.168.x, etc. |
| **GDPR** | `enabled` | False | Enable GDPR compliance |
| | `retention_days` | 365 | Data retention period |
| | `exclude_pii_patterns` | True | Strip PII from content |
| **Storage** | `ttl_seconds` | 604800 | Structure cache TTL (7 days) |
| | `max_versions` | 10 | Keep structure history |

---

## Usage Examples

This section provides comprehensive examples for common crawling scenarios, from simple single-page crawls to complex multi-site monitoring.

### Command Line Usage

#### Basic Crawl

```bash
# Crawl a single site
python -m crawler \
    --seed-url https://example.com \
    --output ./data

# Multiple seed URLs
python -m crawler \
    --seed-url https://site1.com \
    --seed-url https://site2.com \
    --output ./data

# Read seed URLs from a file
cat seeds.txt
# https://example1.com
# https://example2.com
# https://example3.com

python -m crawler \
    --seed-file seeds.txt \
    --output ./data
```

#### Crawl Depth and Limits

```bash
# Limit crawl depth (how many links to follow from seed)
python -m crawler \
    --seed-url https://example.com \
    --output ./data \
    --max-depth 3

# Limit total pages
python -m crawler \
    --seed-url https://example.com \
    --output ./data \
    --max-pages 100

# Limit pages per domain (useful for multi-domain crawls)
python -m crawler \
    --seed-url https://example.com \
    --output ./data \
    --max-pages-per-domain 50

# Combined limits
python -m crawler \
    --seed-url https://example.com \
    --output ./data \
    --max-depth 5 \
    --max-pages 1000 \
    --max-pages-per-domain 200
```

#### Domain Restrictions

```bash
# Stay within specific domains
python -m crawler \
    --seed-url https://docs.example.com \
    --output ./data \
    --allowed-domains docs.example.com api.example.com

# Exclude specific paths
python -m crawler \
    --seed-url https://example.com \
    --output ./data \
    --exclude-patterns "/admin/*" "/private/*" "/api/*"

# Combine domain and path restrictions
python -m crawler \
    --seed-url https://example.com \
    --output ./data \
    --allowed-domains example.com \
    --exclude-patterns "/cdn/*" "*.pdf" "*.zip"
```

#### Rate Limiting and Politeness

```bash
# Set delay between requests (seconds)
python -m crawler \
    --seed-url https://example.com \
    --output ./data \
    --rate-limit 2.0

# Respect robots.txt (enabled by default)
python -m crawler \
    --seed-url https://example.com \
    --output ./data \
    --respect-robots

# Set custom User-Agent
python -m crawler \
    --seed-url https://example.com \
    --output ./data \
    --user-agent "MyCompanyBot/1.0 (+https://company.com/bot; contact@company.com)"

# Very polite crawl for sensitive sites
python -m crawler \
    --seed-url https://example.com \
    --output ./data \
    --rate-limit 5.0 \
    --max-concurrent 2 \
    --respect-robots \
    --user-agent "ResearchBot/1.0 (Academic research; contact@university.edu)"
```

### Python API Examples

#### Example 1: Simple Crawl

```python
import asyncio
from crawler.core.crawler import Crawler
from crawler.config import CrawlConfig

async def simple_crawl():
    """Basic crawl with minimal configuration."""
    config = CrawlConfig(
        seed_urls=["https://example.com"],
        output_dir="./data",
    )

    async with Crawler(config) as crawler:
        stats = await crawler.crawl()

    print(f"Crawled {stats.pages_crawled} pages")
    print(f"Found {stats.links_discovered} links")
    print(f"Errors: {stats.errors}")

asyncio.run(simple_crawl())
```

#### Example 2: Crawl with Callbacks

```python
import asyncio
from crawler.core.crawler import Crawler
from crawler.config import CrawlConfig

async def crawl_with_callbacks():
    """Crawl with real-time progress callbacks."""
    config = CrawlConfig(
        seed_urls=["https://example.com"],
        output_dir="./data",
        max_pages=50,
    )

    # Track progress
    pages_crawled = []
    errors = []

    async with Crawler(config) as crawler:
        # Register callbacks
        @crawler.on_page_crawled
        def handle_page(url: str, result):
            pages_crawled.append(url)
            print(f"✓ [{len(pages_crawled)}] {url}")
            print(f"  Title: {result.extracted.get('title', 'N/A')}")
            print(f"  Size: {result.content_length} bytes")

        @crawler.on_structure_learned
        def handle_structure(domain: str, page_type: str, structure):
            print(f"📊 Learned structure for {domain}/{page_type}")
            print(f"   Tags: {len(structure.tag_hierarchy)} unique")
            print(f"   Classes: {len(structure.css_class_map)}")

        @crawler.on_error
        def handle_error(url: str, error: Exception):
            errors.append((url, error))
            print(f"✗ Error: {url} - {error}")

        @crawler.on_rate_limited
        def handle_rate_limit(domain: str, delay: float):
            print(f"⏳ Rate limited on {domain}, waiting {delay}s")

        stats = await crawler.crawl()

    print(f"\n=== Crawl Complete ===")
    print(f"Pages: {len(pages_crawled)}")
    print(f"Errors: {len(errors)}")

asyncio.run(crawl_with_callbacks())
```

#### Example 3: E-commerce Product Scraping

```python
import asyncio
import json
from crawler.core.crawler import Crawler
from crawler.config import CrawlConfig, RateLimitConfig
from crawler.extraction import ContentExtractor

async def scrape_products():
    """Scrape product pages from an e-commerce site."""
    config = CrawlConfig(
        seed_urls=["https://shop.example.com/products"],
        output_dir="./products",
        max_depth=3,
        max_pages=500,
        allowed_domains=["shop.example.com"],
        # Only crawl product pages
        include_patterns=["/products/*", "/product/*"],
        exclude_patterns=["/cart", "/checkout", "/account/*"],

        rate_limit=RateLimitConfig(
            default_delay=1.5,  # Be polite to the server
            adaptive=True,
        ),
    )

    products = []

    async with Crawler(config) as crawler:
        @crawler.on_page_crawled
        def extract_product(url: str, result):
            if "/product/" in url:
                # Custom product extraction
                product = {
                    "url": url,
                    "title": result.extracted.get("title"),
                    "price": extract_price(result.html),
                    "description": result.extracted.get("content"),
                    "images": result.extracted.get("images", []),
                    "sku": extract_sku(result.html),
                }
                products.append(product)
                print(f"Found product: {product['title']} - ${product['price']}")

        await crawler.crawl()

    # Save products
    with open("./products/products.json", "w") as f:
        json.dump(products, f, indent=2)

    print(f"Scraped {len(products)} products")

def extract_price(html: str) -> float:
    """Extract price from HTML (simplified example)."""
    import re
    match = re.search(r'\$(\d+(?:\.\d{2})?)', html)
    return float(match.group(1)) if match else 0.0

def extract_sku(html: str) -> str:
    """Extract SKU from HTML (simplified example)."""
    import re
    match = re.search(r'SKU:\s*(\w+)', html)
    return match.group(1) if match else ""

asyncio.run(scrape_products())
```

#### Example 4: News Article Monitoring

```python
import asyncio
from datetime import datetime
from crawler.core.crawler import Crawler
from crawler.config import CrawlConfig
from crawler.adaptive import ChangeDetector

async def monitor_news_site():
    """Monitor a news site for new articles."""
    config = CrawlConfig(
        seed_urls=["https://news.example.com"],
        output_dir="./news",
        max_depth=2,
        include_patterns=["/article/*", "/news/*", "/story/*"],
        exclude_patterns=["/archive/*", "/author/*"],
    )

    detector = ChangeDetector()
    new_articles = []

    async with Crawler(config) as crawler:
        @crawler.on_page_crawled
        def check_article(url: str, result):
            # Check if this is a new or updated article
            stored = crawler.structure_store.get(url)

            if stored is None:
                # New article
                new_articles.append({
                    "url": url,
                    "title": result.extracted.get("title"),
                    "published": datetime.now().isoformat(),
                    "status": "new",
                })
                print(f"📰 NEW: {result.extracted.get('title')}")
            else:
                # Check for updates
                analysis = detector.detect_changes(
                    stored.structure,
                    result.structure
                )
                if analysis.has_changes:
                    new_articles.append({
                        "url": url,
                        "title": result.extracted.get("title"),
                        "updated": datetime.now().isoformat(),
                        "status": "updated",
                        "change_type": analysis.classification.name,
                    })
                    print(f"📝 UPDATED: {result.extracted.get('title')}")

        await crawler.crawl()

    print(f"\nFound {len(new_articles)} new/updated articles")
    return new_articles

asyncio.run(monitor_news_site())
```

#### Example 5: Multi-Site Comparison

```python
import asyncio
from crawler.core.crawler import Crawler
from crawler.config import CrawlConfig

async def compare_sites():
    """Compare structure across multiple competitor sites."""
    sites = [
        "https://competitor1.com",
        "https://competitor2.com",
        "https://competitor3.com",
    ]

    results = {}

    for site in sites:
        config = CrawlConfig(
            seed_urls=[site],
            output_dir=f"./comparison/{site.split('//')[1]}",
            max_pages=20,
            max_depth=2,
        )

        async with Crawler(config) as crawler:
            stats = await crawler.crawl()

            # Collect structure data
            results[site] = {
                "pages": stats.pages_crawled,
                "structures": {},
            }

            for domain, structures in crawler.structure_store.get_all().items():
                for page_type, structure in structures.items():
                    results[site]["structures"][page_type] = {
                        "tag_count": len(structure.tag_hierarchy),
                        "css_classes": len(structure.css_class_map),
                        "has_article": "article" in structure.semantic_landmarks,
                        "has_nav": "nav" in structure.semantic_landmarks,
                    }

    # Compare results
    print("\n=== Site Comparison ===")
    for site, data in results.items():
        print(f"\n{site}:")
        print(f"  Pages crawled: {data['pages']}")
        for page_type, info in data["structures"].items():
            print(f"  {page_type}: {info['tag_count']} tags, {info['css_classes']} classes")

asyncio.run(compare_sites())
```

#### Example 6: Sitemap-Based Crawl

```python
import asyncio
from crawler.core.crawler import Crawler
from crawler.config import CrawlConfig
from crawler.compliance import SitemapFetcher

async def crawl_from_sitemap():
    """Use sitemap to discover and prioritize URLs."""
    # First, fetch URLs from sitemap
    async with SitemapFetcher(user_agent="MyCrawler/1.0") as fetcher:
        sitemap_urls = await fetcher.discover_sitemaps("example.com")

        all_urls = []
        async for sitemap in fetcher.fetch_all_sitemaps(sitemap_urls):
            for url in sitemap.urls:
                all_urls.append({
                    "url": url.loc,
                    "priority": url.priority or 0.5,
                    "lastmod": url.lastmod,
                })

        print(f"Found {len(all_urls)} URLs in sitemap")

    # Sort by priority (highest first)
    all_urls.sort(key=lambda x: x["priority"], reverse=True)

    # Crawl top priority URLs first
    seed_urls = [u["url"] for u in all_urls[:100]]

    config = CrawlConfig(
        seed_urls=seed_urls,
        output_dir="./sitemap_crawl",
        max_pages=500,
    )

    async with Crawler(config) as crawler:
        stats = await crawler.crawl()

    print(f"Crawled {stats.pages_crawled} pages from sitemap")

asyncio.run(crawl_from_sitemap())
```

#### Example 7: JavaScript-Heavy Site

```python
import asyncio
from crawler.core.crawler import Crawler
from crawler.core.renderer import JSRenderer, HybridFetcher
from crawler.config import CrawlConfig

async def crawl_spa():
    """Crawl a JavaScript-heavy Single Page Application."""
    config = CrawlConfig(
        seed_urls=["https://spa.example.com"],
        output_dir="./spa_data",
        max_pages=50,
    )

    # Configure JS rendering
    async with JSRenderer(
        browser_type="chromium",
        headless=True,
    ) as renderer:
        # Use hybrid fetcher that automatically detects JS need
        async with HybridFetcher(js_renderer=renderer) as fetcher:
            async with Crawler(config, fetcher=fetcher) as crawler:
                @crawler.on_page_crawled
                def log_render(url: str, result):
                    if result.used_js_rendering:
                        print(f"🌐 JS rendered: {url}")
                    else:
                        print(f"📄 HTTP only: {url}")

                stats = await crawler.crawl()

    print(f"Crawled {stats.pages_crawled} SPA pages")

asyncio.run(crawl_spa())
```

#### Example 8: Distributed Crawl

```python
import asyncio
from crawler.core.distributed import (
    DistributedCrawlManager,
    CrawlerWorker,
)
import redis.asyncio as redis

async def run_distributed_crawl():
    """Run a distributed crawl across multiple workers."""
    redis_client = redis.from_url("redis://localhost:6379/0")

    # Manager creates the job
    manager = DistributedCrawlManager(redis_client)

    job = await manager.create_job(
        job_id="large-crawl-001",
        seed_urls=[
            "https://example1.com",
            "https://example2.com",
            "https://example3.com",
        ],
        max_urls=10000,
        max_depth=5,
    )

    print(f"Created job: {job.job_id}")
    print(f"Seed URLs: {len(job.seed_urls)}")

    # In production, run workers on different machines
    # Here we simulate with multiple async workers
    workers = []
    for i in range(3):
        worker = CrawlerWorker(
            redis_client=redis_client,
            job_id=job.job_id,
            worker_id=f"worker-{i}",
        )
        workers.append(worker.start())

    # Monitor progress
    async def monitor():
        while True:
            status = await manager.get_job_status(job.job_id)
            print(f"\rPending: {status['pending']} | "
                  f"Processing: {status['processing']} | "
                  f"Completed: {status['completed']}", end="")

            if status['state'] == 'COMPLETED':
                break
            await asyncio.sleep(2)

    # Run workers and monitor concurrently
    await asyncio.gather(
        *workers,
        monitor(),
    )

    print(f"\nDistributed crawl complete!")

asyncio.run(run_distributed_crawl())
```

### Output Format

The crawler produces structured output in the following format:

#### Directory Structure
```
output/
├── raw/                          # Raw HTML files
│   └── example.com/
│       ├── index.html
│       └── about.html
├── extracted/                    # Extracted JSON data
│   └── example.com/
│       ├── index.json
│       └── about.json
├── structures/                   # Learned page structures
│   └── example.com/
│       ├── homepage.json
│       └── article.json
├── metadata/
│   ├── crawl_stats.json         # Overall statistics
│   ├── url_map.json             # URL to file mapping
│   └── errors.json              # Error log
└── logs/
    └── crawler.log              # Full crawl log
```

#### Extracted JSON Format
```json
{
    "url": "https://example.com/article/123",
    "crawled_at": "2025-01-30T10:30:00Z",
    "status_code": 200,
    "content_type": "text/html",
    "content_length": 15234,
    "extracted": {
        "title": "Article Title Here",
        "content": "Full article text content...",
        "description": "Meta description if available",
        "author": "John Doe",
        "published_date": "2025-01-29",
        "images": [
            "https://example.com/img/hero.jpg"
        ],
        "links": [
            {"href": "/related/456", "text": "Related Article"}
        ]
    },
    "structure": {
        "page_type": "article",
        "similarity_to_stored": 0.97,
        "version": 3
    },
    "compliance": {
        "robots_allowed": true,
        "crawl_delay_respected": true,
        "pii_detected": false
    }
}
```

---

## Sports News Monitor Example

The `examples/sports_news_monitor.py` demonstrates real-world usage for monitoring websites for changes.

### Use Case

Monitor sports news sites (ESPN, BBC Sport, etc.) for content updates:
- Learn page structure on first visit
- Detect structural changes (site redesigns)
- Extract content when changes occur
- Ignore dynamic content (timestamps, scores)

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    MONITORING WORKFLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. FETCH PAGE                                                   │
│     └──► HTTP GET with robots.txt respect                       │
│                                                                  │
│  2. ANALYZE STRUCTURE                                            │
│     └──► Create DOM fingerprint (tags, classes, IDs)            │
│                                                                  │
│  3. COMPUTE STRUCTURAL FINGERPRINT                               │
│     └──► Hash of structural elements only                       │
│     └──► Ignores: timestamps, scores, text content              │
│                                                                  │
│  4. COMPARE FINGERPRINTS                                         │
│     ├──► Same fingerprint ──► No changes, skip extraction       │
│     └──► Different fingerprint ──► Continue to step 5           │
│                                                                  │
│  5. DETECT CHANGE TYPE                                           │
│     ├──► First visit ──► "new_content"                          │
│     ├──► < 70% similar ──► "structure_changed" (adapt)          │
│     └──► ≥ 70% similar ──► "content_updated"                    │
│                                                                  │
│  6. EXTRACT CONTENT                                              │
│     └──► Apply learned CSS selectors                            │
│     └──► Get title, content, metadata                           │
│                                                                  │
│  7. SAVE & NOTIFY                                                │
│     └──► Save to JSON file                                      │
│     └──► Trigger callback                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Structural Fingerprinting

The monitor uses structural fingerprints to avoid false positives:

```python
# What's INCLUDED in fingerprint (structural):
- Tag counts (div: 45, span: 23, etc.)
- CSS class names (top 30 most frequent)
- Element IDs
- Semantic landmarks
- Navigation selectors

# What's EXCLUDED (dynamic content):
- Text content
- Timestamps and dates
- Scores and statistics
- Image URLs
- Ad content
```

### Usage

```bash
# Install Redis first (if not running)
python examples/sports_news_monitor.py --install-redis

# Monitor a URL (one-time check)
python examples/sports_news_monitor.py \
    --url https://news.ycombinator.com \
    --once

# Continuous monitoring
python examples/sports_news_monitor.py \
    --url https://www.bbc.com/sport \
    --interval 300 \
    --output ./sports_output

# Multiple URLs
python examples/sports_news_monitor.py \
    --url https://www.espn.com \
    --url https://www.espn.com/nfl \
    --interval 600
```

### Output

Changes are saved to JSON:

```json
{
    "url": "https://www.espn.com/nfl/",
    "detected_at": "2025-01-29T10:30:00Z",
    "change_type": "content_updated",
    "similarity_score": 0.97,
    "previous_hash": "a1b2c3d4e5f6g7h8",
    "current_hash": "h8g7f6e5d4c3b2a1",
    "extracted": {
        "title": "NFL News - Latest Headlines",
        "content_preview": "Breaking: Team announces...",
        "content_length": 4523,
        "metadata": {},
        "images": ["https://..."]
    }
}
```

---

## API Reference

### Core Classes

#### `Crawler`

Main orchestrator for crawling operations.

```python
class Crawler:
    def __init__(
        self,
        config: CrawlConfig,
        redis_url: str = "redis://localhost:6379/0",
        user_agent: str = "AdaptiveCrawler/1.0",
    ): ...

    async def start(self) -> None:
        """Initialize all components."""

    async def stop(self) -> None:
        """Cleanup and close connections."""

    async def crawl(self) -> CrawlerStats:
        """Run the crawl and return statistics."""

    def on_page_crawled(self, callback: Callable) -> None:
        """Register callback for successful crawls."""

    def on_error(self, callback: Callable) -> None:
        """Register callback for errors."""
```

#### `StructureAnalyzer`

Analyzes HTML to create page structure fingerprints.

```python
class StructureAnalyzer:
    def analyze(
        self,
        html: str,
        url: str,
        page_type: str = "unknown",
    ) -> PageStructure:
        """Analyze HTML and return structure fingerprint."""
```

#### `ChangeDetector`

Detects and classifies changes between structures.

```python
class ChangeDetector:
    def detect_changes(
        self,
        old_structure: PageStructure,
        new_structure: PageStructure,
    ) -> ChangeAnalysis:
        """Compare structures and return analysis."""

    def has_breaking_changes(
        self,
        old_structure: PageStructure,
        new_structure: PageStructure,
    ) -> bool:
        """Quick check for breaking changes."""
```

#### `StrategyLearner`

Learns CSS selectors for content extraction.

```python
class StrategyLearner:
    def infer(
        self,
        html: str,
        structure: PageStructure | None = None,
    ) -> LearnedStrategy:
        """Infer extraction strategy from HTML."""

    def adapt(
        self,
        old_strategy: ExtractionStrategy,
        new_structure: PageStructure,
        html: str,
    ) -> LearnedStrategy:
        """Adapt existing strategy to new structure."""
```

#### `ContentExtractor`

Extracts content using learned strategies.

```python
class ContentExtractor:
    def extract(
        self,
        url: str,
        html: str,
        strategy: ExtractionStrategy,
    ) -> ExtractionResult:
        """Extract content using strategy."""
```

### Data Models

#### `PageStructure`

```python
@dataclass
class PageStructure:
    domain: str
    page_type: str
    url_pattern: str
    tag_hierarchy: dict[str, Any]
    css_class_map: dict[str, int]
    id_attributes: set[str]
    semantic_landmarks: dict[str, list[str]]
    content_regions: list[ContentRegion]
    navigation_selectors: list[str]
    content_hash: str
    version: int = 1
```

#### `ExtractionStrategy`

```python
@dataclass
class ExtractionStrategy:
    domain: str
    page_type: str
    title: SelectorRule | None
    content: SelectorRule | None
    metadata: dict[str, SelectorRule]
    confidence_scores: dict[str, float]
    required_fields: list[str] = ["title", "content"]
    version: int = 1
```

#### `ChangeAnalysis`

```python
@dataclass
class ChangeAnalysis:
    has_changes: bool
    classification: ChangeClassification  # COSMETIC, MINOR, MODERATE, BREAKING
    similarity_score: float
    changes: list[StructureChange]
    requires_relearning: bool
```

---

## Machine Learning Features

The crawler includes advanced ML capabilities for semantic change detection, content classification, and LLM-powered descriptions.

### ML Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML-ENHANCED PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PAGE STRUCTURE                                                  │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              EMBEDDING MODEL                             │    │
│  │         (sentence-transformers)                          │    │
│  │                                                          │    │
│  │  • all-MiniLM-L6-v2 (default, 384 dims, fast)           │    │
│  │  • all-mpnet-base-v2 (768 dims, best quality)           │    │
│  │  • paraphrase-MiniLM-L6-v2 (paraphrase optimized)       │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           ML CAPABILITIES                                │    │
│  │                                                          │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │    │
│  │  │  Semantic   │  │    Page     │  │    LLM      │     │    │
│  │  │  Similarity │  │   Type      │  │ Description │     │    │
│  │  │  Detection  │  │ Classifier  │  │  Generator  │     │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │    │
│  │        │                │                │              │    │
│  │        ▼                ▼                ▼              │    │
│  │   cosine sim       LR/XGB/LGBM    OpenAI/Anthropic     │    │
│  │   threshold        prediction      /Ollama             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Enabling ML Features

#### 1. Install ML Dependencies

```bash
# Core ML (embeddings + classification)
pip install sentence-transformers scikit-learn

# Gradient boosting classifiers
pip install xgboost lightgbm

# LLM providers
pip install openai anthropic

# All features
pip install -e ".[ml,llm]"
```

#### 2. Configure ML in Python

```python
from crawler.config import (
    CrawlConfig,
    StructureStoreConfig,
    StructureStoreType,
    LLMProviderType,
)

config = CrawlConfig(
    seed_urls=["https://example.com"],
    output_dir="./data",

    structure_store=StructureStoreConfig(
        # Enable LLM-powered descriptions
        store_type=StructureStoreType.LLM,

        # Enable embeddings for semantic similarity
        enable_embeddings=True,
        embedding_model="all-MiniLM-L6-v2",

        # LLM provider settings
        llm_provider=LLMProviderType.ANTHROPIC,
        llm_model="claude-sonnet-4-20250514",
        # API key from environment: ANTHROPIC_API_KEY
    ),
)
```

#### 3. Environment Variables

```bash
# LLM API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OLLAMA_API_KEY="your-key"  # For Ollama Cloud

# Local Ollama (no key needed)
# Just run: ollama serve
```

### Embedding-Based Change Detection

Use semantic embeddings instead of rule-based comparison for more intelligent change detection.

```
┌─────────────────────────────────────────────────────────────────┐
│                EMBEDDING SIMILARITY DETECTION                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  OLD STRUCTURE                    NEW STRUCTURE                  │
│       │                                │                         │
│       ▼                                ▼                         │
│  ┌─────────────┐                ┌─────────────┐                 │
│  │  Embedding  │                │  Embedding  │                 │
│  │  [384 dims] │                │  [384 dims] │                 │
│  └──────┬──────┘                └──────┬──────┘                 │
│         │                              │                         │
│         └──────────────┬───────────────┘                         │
│                        ▼                                         │
│               ┌─────────────────┐                                │
│               │ Cosine Similarity│                               │
│               │   (0.0 - 1.0)   │                                │
│               └────────┬────────┘                                │
│                        │                                         │
│         ┌──────────────┼──────────────┐                         │
│         │              │              │                          │
│         ▼              ▼              ▼                          │
│     ≥ 0.95         0.7-0.95        < 0.70                       │
│    COSMETIC       MODERATE        BREAKING                       │
│   (no action)   (log change)    (re-learn)                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Python API

```python
from crawler.ml.embeddings import StructureEmbeddingModel, MLChangeDetector

# Create embedding model
model = StructureEmbeddingModel(model_name="all-MiniLM-L6-v2")

# Embed structures
old_embedding = model.embed_structure(old_structure)
new_embedding = model.embed_structure(new_structure)

# Compute similarity
similarity = model.compute_similarity(
    old_embedding.embedding,
    new_embedding.embedding
)
print(f"Semantic similarity: {similarity:.2%}")

# Find similar structures
similar = model.find_similar(
    query_embedding.embedding,
    all_embeddings,
    top_k=5
)

# ML-based change detection
detector = MLChangeDetector(embedding_model=model)
result = detector.detect_change(old_structure, new_structure)
print(f"Is breaking: {result['is_breaking']}")
print(f"Similarity: {result['similarity']:.2%}")
```

### Page Type Classification

Train ML classifiers to automatically categorize pages.

```
┌─────────────────────────────────────────────────────────────────┐
│                 PAGE TYPE CLASSIFICATION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT: Page Structure                                           │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              FEATURE EXTRACTION                          │    │
│  │                                                          │    │
│  │  • Tag counts (div, article, nav, etc.)                 │    │
│  │  • CSS class patterns                                   │    │
│  │  • Semantic landmarks                                   │    │
│  │  • Content region characteristics                       │    │
│  │  • Navigation patterns                                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              CLASSIFIER                                  │    │
│  │                                                          │    │
│  │  Choose one:                                            │    │
│  │  • LogisticRegression (fast, interpretable)             │    │
│  │  • XGBoost (high accuracy, feature importance)          │    │
│  │  • LightGBM (fast training, large datasets)             │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  OUTPUT: Page Type + Confidence                                  │
│                                                                  │
│  Examples:                                                       │
│  • "article" (92% confidence)                                   │
│  • "homepage" (87% confidence)                                  │
│  • "product_listing" (78% confidence)                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Training a Classifier

```python
from crawler.ml.embeddings import StructureClassifier, ClassifierType

# Create classifier
classifier = StructureClassifier(
    classifier_type=ClassifierType.XGBOOST  # or LIGHTGBM, LOGISTIC_REGRESSION
)

# Prepare training data
structures = [structure1, structure2, ...]  # PageStructure objects
labels = ["article", "homepage", ...]        # Page type labels

# Train
metrics = classifier.train(structures, labels)
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"F1 Score: {metrics['f1_score']:.2%}")

# Predict
label, confidence = classifier.predict(new_structure)
print(f"Predicted: {label} ({confidence:.2%} confidence)")

# Get feature importance (XGBoost/LightGBM)
importance = classifier.get_feature_importance()
for feature, score in importance[:10]:
    print(f"  {feature}: {score:.4f}")

# Save/load model
classifier.save("page_classifier.pkl")
classifier.load("page_classifier.pkl")
```

### LLM-Powered Descriptions

Generate rich, semantic descriptions of page structures using LLMs.

```
┌─────────────────────────────────────────────────────────────────┐
│                 LLM DESCRIPTION GENERATION                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PAGE STRUCTURE                                                  │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Structure Summary                                       │    │
│  │  • 45 div, 23 span, 67 anchor tags                      │    │
│  │  • Semantic: article, nav, header, footer               │    │
│  │  • Classes: .article-content, .nav-item, .btn           │    │
│  │  • Content regions: main content, sidebar               │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              LLM PROVIDER                                │    │
│  │                                                          │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │    │
│  │  │ OpenAI  │  │Anthropic│  │ Ollama  │  │ Ollama  │   │    │
│  │  │ GPT-4o  │  │ Claude  │  │ (Local) │  │ (Cloud) │   │    │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  RICH DESCRIPTION:                                               │
│  "This is a news article page with a prominent header            │
│   containing navigation. The main content area uses an           │
│   <article> tag with structured sections. The page follows       │
│   a standard blog layout with sidebar widgets and a footer       │
│   containing social links and copyright information."            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Using LLM Descriptions

```python
from crawler.ml.embeddings import get_description_generator

# Rules-based (no API, fast, deterministic)
rules_gen = get_description_generator("rules")
desc = rules_gen.generate(structure)

# OpenAI
openai_gen = get_description_generator(
    "llm",
    provider="openai",
    model="gpt-4o-mini"
)
desc = openai_gen.generate(structure)

# Anthropic (Claude)
claude_gen = get_description_generator(
    "llm",
    provider="anthropic",
    model="claude-sonnet-4-20250514"
)
desc = claude_gen.generate(structure)

# Local Ollama (free, private)
ollama_gen = get_description_generator(
    "llm",
    provider="ollama",
    model="llama3.2"
)
desc = ollama_gen.generate(structure)

# Generate change description
change_desc = generator.generate_for_change_detection(
    old_structure,
    new_structure
)
print(change_desc)
# "The page structure changed significantly: the navigation
#  moved from sidebar to header, article content wrapper
#  changed from .post-content to .article-body, and new
#  advertisement slots were added between paragraphs."
```

### ML Training Script

Use the built-in script for ML operations:

```bash
# Export training data from Redis
python scripts/train_embeddings.py export -o training_data.jsonl

# Create embeddings for all structures
python scripts/train_embeddings.py embed -o embeddings.json

# Find similar structures
python scripts/train_embeddings.py similar example.com --top-k 10

# Train classifiers
python scripts/train_embeddings.py train -o classifier.pkl
python scripts/train_embeddings.py train --classifier-type xgboost
python scripts/train_embeddings.py train --classifier-type lightgbm

# Predict page type
python scripts/train_embeddings.py predict example.com --classifier classifier.pkl

# Generate descriptions
python scripts/train_embeddings.py describe example.com --mode rules
python scripts/train_embeddings.py describe example.com --mode llm --provider anthropic

# Baseline drift detection
python scripts/train_embeddings.py set-baseline example.com
python scripts/train_embeddings.py detect-drift example.com
python scripts/train_embeddings.py set-all-baselines
python scripts/train_embeddings.py check-all-drift
```

### Baseline Drift Detection

Monitor sites for gradual structural drift over time.

```python
from crawler.ml.embeddings import MLChangeDetector

detector = MLChangeDetector()

# Set baseline from current structure
detector.set_site_baseline("example.com", current_structure)

# Later: check for drift
drift = detector.detect_drift_from_baseline(new_structure)
print(f"Drift detected: {drift['has_drift']}")
print(f"Drift amount: {drift['drift_score']:.2%}")
print(f"Recommendation: {drift['recommendation']}")
```

### ML Configuration Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `store_type` | enum | `basic` | `basic` (rules) or `llm` (LLM-powered) |
| `enable_embeddings` | bool | `false` | Enable semantic embeddings |
| `embedding_model` | str | `all-MiniLM-L6-v2` | HuggingFace model name |
| `llm_provider` | enum | `anthropic` | `anthropic`, `openai`, `ollama`, `ollama-cloud` |
| `llm_model` | str | `""` | Model name (empty = provider default) |
| `llm_api_key` | str | `""` | API key (empty = use env var) |
| `ollama_base_url` | str | `http://localhost:11434` | Ollama server URL |

### LLM Provider Comparison

| Provider | Models | Latency | Cost | Privacy |
|----------|--------|---------|------|---------|
| **OpenAI** | gpt-4o-mini, gpt-4 | Low | $$ | Cloud |
| **Anthropic** | claude-sonnet, claude-opus | Low | $$ | Cloud |
| **Ollama (Local)** | llama3.2, mistral, codellama | Medium | Free | Full |
| **Ollama Cloud** | Same as local | Low | Varies | Depends |

### Example: Full ML Pipeline

```python
import asyncio
from crawler.core.crawler import Crawler
from crawler.config import CrawlConfig, StructureStoreConfig, StructureStoreType
from crawler.ml.embeddings import (
    StructureEmbeddingModel,
    StructureClassifier,
    MLChangeDetector,
    get_description_generator,
)

async def ml_enhanced_crawl():
    # Configure with ML features
    config = CrawlConfig(
        seed_urls=["https://example.com"],
        output_dir="./data",
        structure_store=StructureStoreConfig(
            store_type=StructureStoreType.LLM,
            enable_embeddings=True,
            embedding_model="all-MiniLM-L6-v2",
            llm_provider="anthropic",
        ),
    )

    # Initialize ML components
    embedding_model = StructureEmbeddingModel()
    classifier = StructureClassifier(classifier_type="xgboost")
    change_detector = MLChangeDetector(embedding_model=embedding_model)
    description_gen = get_description_generator("llm", provider="anthropic")

    # Run crawler
    async with Crawler(config) as crawler:
        stats = await crawler.crawl()

    # Post-process with ML
    for structure in collected_structures:
        # Classify page type
        page_type, confidence = classifier.predict(structure)
        print(f"Page type: {page_type} ({confidence:.0%})")

        # Generate description
        desc = description_gen.generate(structure)
        print(f"Description: {desc}")

        # Check similarity to baseline
        drift = change_detector.detect_drift_from_baseline(structure)
        if drift['has_drift']:
            print(f"Warning: Site structure drifted {drift['drift_score']:.0%}")

asyncio.run(ml_enhanced_crawl())
```

---

## Development

### Running Tests

```bash
# All tests with coverage
pytest tests/ -v --cov=crawler

# Specific test file
pytest tests/unit/test_change_detector.py -v

# With parallel execution
pytest tests/ -v -n auto
```

### Type Checking

```bash
mypy crawler/ --strict
```

### Linting & Formatting

```bash
# Check and fix
ruff check crawler/ --fix

# Format
ruff format crawler/
```

### Coverage Requirements

| Module | Minimum |
|--------|---------|
| `compliance/*` | 100% |
| `legal/*` | 100% |
| `adaptive/*` | 95% |
| `core/*` | 90% |
| **Overall** | **90%** |

---

## Documentation

- [AGENTS.md](AGENTS.md) - Comprehensive project documentation
- [crawler/adaptive/AGENTS.md](crawler/adaptive/AGENTS.md) - Adaptive subsystem details
- [CLAUDE.md](CLAUDE.md) - Claude Code development guidance

---

## Legal Notice

This crawler is designed for **ethical, legal web data collection**. This section provides guidance on legal compliance, but **this is technical documentation, not legal advice**. Always consult with a qualified attorney for your specific use case.

### Your Legal Responsibilities

As a user of this crawler, you are responsible for:

| Responsibility | Description |
|----------------|-------------|
| **Legal Compliance** | Comply with all applicable laws including CFAA, GDPR, CCPA, and local regulations |
| **Terms of Service** | Respect website terms of service and acceptable use policies |
| **Authorization** | Ensure you have proper authorization before crawling any site |
| **Rate Limiting** | Configure appropriate rate limits to avoid service disruption |
| **Data Handling** | Use collected data responsibly and in accordance with privacy laws |
| **Legal Counsel** | Obtain legal advice for your specific jurisdiction and use case |

### Legal Frameworks

#### Computer Fraud and Abuse Act (CFAA) - United States

The CFAA prohibits unauthorized access to computer systems. Key considerations:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CFAA COMPLIANCE CHECKLIST                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ✓ Public Content Only                                          │
│    Only crawl publicly accessible pages (no login required)     │
│                                                                  │
│  ✓ Respect Access Controls                                      │
│    Stop if you encounter authentication prompts                 │
│                                                                  │
│  ✓ Honor robots.txt                                             │
│    Respect crawl restrictions and rate limits                   │
│                                                                  │
│  ✓ Identify Your Bot                                            │
│    Use a clear User-Agent with contact information              │
│                                                                  │
│  ✓ Stop on Request                                              │
│    Immediately cease crawling if asked by site owner            │
│                                                                  │
│  ✓ No Circumvention                                             │
│    Never bypass security measures, CAPTCHAs, or IP blocks       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Crawler implementation:**
```python
from crawler.legal import CFAAChecker

# The crawler automatically checks authorization
checker = CFAAChecker()
result = await checker.is_authorized(url)

# Blocks crawling if:
# - Authentication is required
# - Site has sent cease-and-desist
# - robots.txt explicitly blocks crawlers
# - Previous access was denied
```

#### General Data Protection Regulation (GDPR) - European Union

GDPR applies when collecting data from EU residents. Requirements:

| Principle | Implementation |
|-----------|----------------|
| **Lawful Basis** | Ensure legitimate interest or consent for data collection |
| **Data Minimization** | Only collect data necessary for your stated purpose |
| **Purpose Limitation** | Don't use collected data for incompatible purposes |
| **Storage Limitation** | Delete data when no longer needed (configure retention) |
| **Accuracy** | Keep collected data accurate and up-to-date |
| **Security** | Implement appropriate security measures |
| **Rights** | Support data subject rights (access, deletion, etc.) |

**Crawler implementation:**
```python
from crawler.config import GDPRConfig, PIIHandlingConfig

config = CrawlConfig(
    gdpr=GDPRConfig(
        enabled=True,
        retention_days=365,           # Auto-delete after 1 year
        collect_only=["url", "title", "content"],  # Data minimization
    ),
    pii=PIIHandlingConfig(
        action="redact",              # Remove PII from collected data
        log_detections=True,          # Audit trail
    ),
)
```

#### California Consumer Privacy Act (CCPA)

CCPA provides California residents with privacy rights. Key considerations:

- Right to know what data is collected
- Right to delete personal information
- Right to opt-out of data sale
- Non-discrimination for exercising rights

**Best practices:**
- Document what data you collect and why
- Implement data deletion capabilities
- Never sell collected personal data without consent
- Maintain records of data collection activities

### Ethical Crawling Guidelines

Beyond legal requirements, follow these ethical guidelines:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ETHICAL CRAWLING PRINCIPLES                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. TRANSPARENCY                                                 │
│     • Identify your crawler clearly in User-Agent               │
│     • Provide contact information                               │
│     • Explain your purpose if asked                             │
│                                                                  │
│  2. RESPECT                                                      │
│     • Honor robots.txt directives                               │
│     • Respect Crawl-delay specifications                        │
│     • Stop crawling if asked                                    │
│                                                                  │
│  3. MINIMAL IMPACT                                               │
│     • Use appropriate rate limiting                             │
│     • Avoid peak traffic hours for high-volume crawls           │
│     • Don't overwhelm small servers                             │
│                                                                  │
│  4. DATA RESPONSIBILITY                                          │
│     • Only collect what you need                                │
│     • Store data securely                                       │
│     • Delete data when no longer needed                         │
│     • Never collect or store PII unnecessarily                  │
│                                                                  │
│  5. GOOD CITIZENSHIP                                             │
│     • Don't crawl content behind paywalls                       │
│     • Respect copyright and intellectual property               │
│     • Don't redistribute collected content without permission   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### User-Agent Best Practices

Always identify your crawler with a proper User-Agent:

```python
# Good User-Agent examples:
user_agent = "CompanyBot/1.0 (+https://company.com/bot; bot@company.com)"
user_agent = "ResearchCrawler/2.0 (+https://university.edu/research-bot; researcher@university.edu)"
user_agent = "NewsAggregator/1.5 (https://news-site.com/about; contact@news-site.com)"

# Include:
# - Bot name and version
# - URL with more information
# - Contact email for issues

# Bad User-Agent examples (don't do this):
# - "" (empty)
# - "Mozilla/5.0" (pretending to be a browser)
# - "curl/7.68.0" (anonymous)
```

### Handling Blocks and Restrictions

When a site blocks or restricts your crawler:

1. **Respect the block immediately** - Don't try to circumvent
2. **Review your crawling behavior** - Were you too aggressive?
3. **Contact the site owner** - Explain your purpose, ask for permission
4. **Wait before retrying** - Give adequate time before checking again
5. **Document the block** - Keep records for compliance purposes

```python
# The crawler automatically handles blocks
from crawler.compliance import BlockedDomainTracker

tracker = BlockedDomainTracker(redis_client)

# Check before crawling
if await tracker.is_blocked("example.com"):
    # Don't crawl - the site has blocked us
    reason = await tracker.get_block_reason("example.com")
    blocked_since = await tracker.get_block_time("example.com")
    # Consider contacting site owner

# Blocks are recorded automatically when:
# - Receiving 403 Forbidden responses
# - Encountering CAPTCHA challenges
# - Getting IP-blocked
# - Receiving cease-and-desist requests
```

### Disclaimer

**IMPORTANT NOTICES:**

1. **Not Legal Advice**: This documentation provides technical guidance only. It is not legal advice and should not be relied upon as such.

2. **User Responsibility**: Users are solely responsible for ensuring their use of this crawler complies with all applicable laws, regulations, and third-party terms of service.

3. **No Warranties**: This software is provided "as is" without warranties of any kind. The authors are not liable for any damages or legal issues arising from its use.

4. **Jurisdiction Varies**: Laws regarding web scraping vary significantly by jurisdiction. What's legal in one country may be illegal in another.

5. **Consult an Attorney**: Before using this crawler for any commercial purpose or on any scale, consult with a qualified attorney familiar with technology law in your jurisdiction.

**By using this crawler, you acknowledge that you have read and understood these disclaimers and accept full responsibility for your use of the software.**

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure all tests pass (`pytest tests/ -v`)
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.
