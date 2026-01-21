# CLAUDE.md - Adaptive Web Crawler

This file provides guidance for Claude Code when working on this project.

## Project Overview

An adaptive web crawler with legal compliance (CFAA/GDPR/CCPA), robots.txt respect, rate limiting, and ML-based structure learning that adapts to website changes.

## Quick Reference

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=crawler

# Type checking
mypy crawler/ --strict

# Lint
ruff check crawler/ --fix

# Format
ruff format crawler/

# Start Redis (required for adaptive features)
docker run -d -p 6379:6379 redis:7-alpine

# Run crawler
python -m crawler --seed-url https://example.com --output ./data
```

## Architecture

```
crawler/
├── core/           # Orchestrator, scheduler, fetcher, state management
├── compliance/     # robots.txt, rate limiting, bot detection
├── legal/          # CFAA, GDPR, CCPA, PII handling
├── extraction/     # Content parsing, validation
├── adaptive/       # Structure learning, change detection, ML models
├── deduplication/  # Content hashing, URL canonicalization
├── alerting/       # Notifications for changes/failures
├── storage/        # Redis, SQLite, content persistence
└── utils/          # URL utils, metrics, logging
```

## Key Patterns

### 1. Compliance Pipeline (ALWAYS follows this order)

```python
async def fetch_url(self, url: str) -> FetchResult:
    # 1. CFAA authorization check
    auth = await self.cfaa_checker.is_authorized(url)
    if not auth.authorized:
        return FetchResult.blocked(url, reason=auth.reason)
    
    # 2. robots.txt check
    if not await self.robots_checker.is_allowed(url):
        return FetchResult.blocked(url, reason="robots.txt")
    
    # 3. Rate limiting
    await self.rate_limiter.acquire(get_domain(url))
    
    # 4. Fetch
    response = await self.http_client.get(url)
    
    # 5. GDPR PII check on response
    if self.gdpr_config.enabled:
        response = await self.pii_handler.process(response)
    
    return response
```

### 2. Adaptive Extraction Pattern

```python
async def extract(self, url: str, html: str) -> ExtractionResult:
    domain = get_domain(url)
    page_type = self.classifier.classify(url)
    
    # Load or learn strategy
    stored = await self.structure_store.get(domain, page_type)
    current = self.analyzer.analyze(html)
    
    if stored and self.change_detector.has_breaking_changes(stored, current):
        strategy = await self.learner.adapt(stored.strategy, current)
        reason = self.change_logger.document(stored, current)
        await self.structure_store.update(domain, page_type, current, strategy, reason)
    elif not stored:
        strategy = await self.learner.infer(html)
        await self.structure_store.save(domain, page_type, current, strategy)
    else:
        strategy = stored.strategy
    
    return self.extractor.extract(html, strategy)
```

### 3. All Public Methods Return Result Types

```python
# Good
async def fetch(self, url: str) -> FetchResult:
    ...

# Bad - don't raise for expected failures
async def fetch(self, url: str) -> Response:
    if blocked:
        raise BlockedError()  # Don't do this
```

## Code Style

### Type Hints Required

```python
# All functions must have complete type hints
async def process_url(
    self,
    url: str,
    options: ProcessOptions | None = None
) -> ProcessResult:
    ...
```

### Dataclasses for Models

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class PageStructure:
    domain: str
    page_type: str
    captured_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
```

### Async by Default

```python
# Prefer async for I/O operations
async def fetch(self, url: str) -> Response:
    async with self.client.get(url) as response:
        return await response.read()
```

### Error Handling

```python
# Use specific exceptions from crawler.exceptions
from crawler.exceptions import (
    RobotsBlockedError,
    RateLimitExceededError,
    GDPRViolationError,
)

# Catch specific, re-raise or handle
try:
    result = await self.fetch(url)
except RobotsBlockedError:
    self.metrics.robots_blocked.inc()
    return FetchResult.blocked(url, "robots.txt")
except RateLimitExceededError:
    await asyncio.sleep(delay)
    return await self.fetch(url)  # Retry
```

## Testing Requirements

### Test File Naming

```
tests/
├── unit/
│   ├── test_robots_parser.py      # Unit tests for robots_parser.py
│   ├── test_rate_limiter.py
│   └── adaptive/
│       └── test_change_detector.py
├── integration/
│   └── test_crawl_pipeline.py     # End-to-end tests
└── compliance/
    └── test_gdpr_compliance.py    # Legal compliance tests
```

### Test Pattern

```python
import pytest
from crawler.compliance import RobotsChecker

class TestRobotsChecker:
    @pytest.fixture
    def checker(self):
        return RobotsChecker(cache_ttl=3600)
    
    async def test_allows_permitted_path(self, checker):
        # Arrange
        robots_txt = "User-agent: *\nAllow: /public/"
        
        # Act
        allowed = await checker.is_allowed("/public/page", robots_txt)
        
        # Assert
        assert allowed is True
    
    async def test_blocks_disallowed_path(self, checker):
        robots_txt = "User-agent: *\nDisallow: /private/"
        allowed = await checker.is_allowed("/private/secret", robots_txt)
        assert allowed is False
```

### Coverage Requirements

| Module | Minimum Coverage |
|--------|------------------|
| `compliance/*` | 100% |
| `legal/*` | 100% |
| `adaptive/*` | 95% |
| `core/*` | 90% |
| Overall | 90% |

## Common Tasks

### Adding a New Compliance Check

1. Create checker in `crawler/compliance/` or `crawler/legal/`
2. Add to compliance pipeline in `crawler/core/fetcher.py`
3. Add configuration to `crawler/config.py`
4. Add metrics to `crawler/utils/metrics.py`
5. Add tests with 100% coverage
6. Update AGENTS.md documentation

### Adding a New Change Type

1. Add to `ChangeType` enum in `crawler/adaptive/models.py`
2. Implement detection in `crawler/adaptive/change_detector.py`
3. Add reason template in `crawler/adaptive/change_logger.py`
4. Add test fixtures in `tests/fixtures/structures/`
5. Update `crawler/adaptive/AGENTS.md`

### Modifying Rate Limiting

1. Config: `crawler/config.py` → `RateLimitConfig`
2. Logic: `crawler/compliance/rate_limiter.py`
3. Key methods: `acquire()`, `adapt()`, `set_domain_delay()`
4. Tests: `tests/unit/test_rate_limiter.py`

## Environment Variables

Required:
- `REDIS_URL` - Redis connection string

Optional (see `crawler/config.py` for full list):
- `CRAWLER_USER_AGENT` - Bot identification
- `CRAWLER_DEFAULT_DELAY` - Rate limit base delay
- `GDPR_ENABLED` - Enable GDPR compliance
- `PII_HANDLING` - PII handling mode (redact/pseudonymize)

## Dependencies

When adding dependencies:
1. Add to `pyproject.toml` under appropriate section
2. Pin major version: `httpx>=0.27.0,<0.28`
3. Dev dependencies go in `[project.optional-dependencies]` dev section

## Git Workflow

```bash
# Branch naming
feature/add-ccpa-compliance
fix/rate-limiter-backoff
refactor/extraction-pipeline

# Commit messages
feat(compliance): add CCPA opt-out support
fix(rate-limiter): correct backoff calculation
test(gdpr): add data subject request tests
docs(agents): update ML model documentation
```

## Documentation

- `AGENTS.md` - Root project documentation for AI assistants
- `crawler/adaptive/AGENTS.md` - Adaptive extraction subsystem docs
- `docs/` - Additional documentation
- `legal/` - Legal compliance templates

When modifying behavior, update relevant AGENTS.md files.
