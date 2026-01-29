# Adaptive Web Crawler

An intelligent, compliance-first web crawler with ML-based structure learning that automatically adapts to website changes. Built for ethical, legal web data collection with full CFAA/GDPR/CCPA compliance.

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Crawl Workflow](#crawl-workflow)
- [Adaptive Learning System](#adaptive-learning-system)
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
- **CFAA Compliance**: Authorization checks before every request
- **GDPR/CCPA Support**: PII detection, redaction, and configurable retention policies
- **robots.txt Respect**: Full RFC 9309 compliance with Crawl-delay support
- **Anti-Bot Respect**: Treats bot detection as "access denied" (never attempts evasion)

### Intelligent Crawling
- **Adaptive Rate Limiting**: Per-domain limits with automatic backoff on 429/503
- **Structure Learning**: ML-based DOM analysis that learns page layouts
- **Change Detection**: Detects site redesigns, class renames, element relocations
- **Auto-Adaptation**: Automatically re-learns extraction strategies when sites change

### Production Ready
- **Redis-Backed Persistence**: Learned structures survive restarts
- **Structured Logging**: Full audit trail of all operations
- **Circuit Breakers**: Automatic failure isolation per domain
- **Parallel Crawling**: Configurable concurrency with domain politeness

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

- Python 3.11+
- Redis (for adaptive features)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/adaptive-crawler.git
cd adaptive-crawler

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"
```

### Start Redis

Choose one option:

```bash
# Option 1: Docker (recommended)
docker run -d -p 6379:6379 redis:7-alpine

# Option 2: Local install (Debian/Ubuntu)
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt-get update && sudo apt-get install -y redis-stack-server

# Option 3: Start existing installation
redis-server --daemonize yes
```

### Run Your First Crawl

```bash
# Basic crawl
python -m crawler --seed-url https://example.com --output ./data

# With options
python -m crawler \
    --seed-url https://example.com \
    --output ./data \
    --max-depth 5 \
    --max-pages 100 \
    --rate-limit 0.5
```

---

## Architecture

### Directory Structure

```
crawler/
├── core/                    # Core orchestration
│   ├── crawler.py          # Main crawler orchestrator
│   ├── fetcher.py          # HTTP client + compliance pipeline
│   └── scheduler.py        # URL frontier management
│
├── compliance/             # Legal compliance
│   ├── robots_parser.py    # RFC 9309 robots.txt parsing
│   └── rate_limiter.py     # Adaptive per-domain rate limiting
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

### Basic Crawl

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
```

### Limited Crawl

```bash
# Limit depth and pages
python -m crawler \
    --seed-url https://example.com \
    --output ./data \
    --max-depth 3 \
    --max-pages 50 \
    --max-pages-per-domain 25
```

### Domain-Restricted Crawl

```bash
# Stay within specific domains
python -m crawler \
    --seed-url https://docs.example.com \
    --output ./data \
    --allowed-domains docs.example.com example.com
```

### Polite Crawl

```bash
# Slower, more polite crawling
python -m crawler \
    --seed-url https://example.com \
    --output ./data \
    --rate-limit 0.2 \
    --respect-robots
```

### Python API

```python
import asyncio
from crawler.core.crawler import Crawler
from crawler.config import CrawlConfig

async def main():
    config = CrawlConfig(
        seed_urls=["https://example.com"],
        output_dir="./data",
        max_depth=5,
    )

    async with Crawler(config, redis_url="redis://localhost:6379/0") as crawler:
        # Optional: Add callbacks
        crawler.on_page_crawled(lambda url, result: print(f"Crawled: {url}"))
        crawler.on_error(lambda url, error: print(f"Error: {url} - {error}"))

        # Run crawl
        stats = await crawler.crawl()

        print(f"Pages crawled: {stats.pages_crawled}")
        print(f"Links discovered: {stats.links_discovered}")
        print(f"Structures learned: {stats.structures_learned}")

asyncio.run(main())
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

This crawler is designed for **ethical, legal web data collection**. Users are responsible for:

1. **Complying with applicable laws** (CFAA, GDPR, CCPA, local regulations)
2. **Respecting website terms of service**
3. **Obtaining necessary legal advice** for their jurisdiction
4. **Configuring appropriate rate limits** to avoid service disruption
5. **Using collected data responsibly**

The crawler includes compliance features, but **proper configuration and legal review are the user's responsibility**.

**This documentation is technical guidance, not legal advice.**

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
