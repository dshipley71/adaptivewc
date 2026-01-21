# Adaptive Web Crawler

An intelligent, ethical web crawler with legal compliance (CFAA/GDPR/CCPA), robots.txt respect, rate limiting, and ML-based structure learning that adapts to website changes.

## Features

- **Legal Compliance**: Built-in CFAA, GDPR, and CCPA compliance with PII detection
- **Robots.txt Respect**: Full RFC 9309 compliance with caching and Crawl-delay support
- **Adaptive Rate Limiting**: Per-domain limits with automatic backoff on 429/503 responses
- **Structure Learning**: ML-based DOM analysis that adapts when websites change
- **Change Detection**: Automatically detects and logs iframe relocations, class renames, and redesigns
- **Anti-Bot Respect**: Treats bot detection as "no bots allowed" (never attempts to evade)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/adaptive-crawler.git
cd adaptive-crawler
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Start Redis (required for adaptive features)
docker run -d -p 6379:6379 redis:7-alpine

# Run a crawl
python -m crawler --seed-url https://example.com --output ./data
```

## Configuration

Configuration is via environment variables (prefix: `CRAWLER_`) or a `.env` file:

```bash
# .env
CRAWLER_USER_AGENT=MyCrawler/1.0 (+https://mysite.com/bot; bot@mysite.com)
CRAWLER_DEFAULT_DELAY=1.0
CRAWLER_MAX_CONCURRENT=10
REDIS_URL=redis://localhost:6379/0
GDPR_ENABLED=true
GDPR_RETENTION_DAYS=365
```

See [Configuration Reference](docs/configuration.md) for all options.

## Architecture

```
crawler/
├── core/           # Orchestrator, scheduler, fetcher
├── compliance/     # robots.txt, rate limiting, bot detection
├── legal/          # CFAA, GDPR, CCPA, PII handling
├── extraction/     # Content parsing and validation
├── adaptive/       # Structure learning, change detection, ML models
├── deduplication/  # Content hashing, URL canonicalization
├── alerting/       # Notifications for changes/failures
├── storage/        # Redis, SQLite, content persistence
└── utils/          # URL utils, metrics, logging
```

## Documentation

- [AGENTS.md](AGENTS.md) - Comprehensive project documentation for developers and AI assistants
- [crawler/adaptive/AGENTS.md](crawler/adaptive/AGENTS.md) - Adaptive extraction subsystem details
- [CLAUDE.md](CLAUDE.md) - Claude Code specific guidance

## Development

```bash
# Run tests
pytest tests/ -v --cov=crawler

# Type checking
mypy crawler/ --strict

# Lint and format
ruff check crawler/ --fix
ruff format crawler/
```

## Legal Notice

This crawler is designed for ethical, legal web data collection. Users are responsible for:

1. Complying with applicable laws (CFAA, GDPR, CCPA, local laws)
2. Respecting website terms of service
3. Obtaining necessary legal advice for their jurisdiction

**This documentation is technical guidance, not legal advice.**

## License

MIT License - See [LICENSE](LICENSE) for details.
