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
