# Adaptive Structure Fingerprinting System

An intelligent web structure fingerprinting system with adaptive learning, Ollama Cloud LLM integration, comprehensive verbose logging, and **ethical compliance** (CFAA, robots.txt RFC 9309, GDPR/CCPA, adaptive rate limiting).

## Features

### Fingerprinting Modes

| Mode | Latency | Use Case |
|------|---------|----------|
| **Rules-based** | ~15ms | Fast, deterministic DOM analysis for stable sites |
| **ML-based** | ~200ms | Semantic similarity using sentence transformer embeddings |
| **Adaptive** | ~15-200ms | Smart mode selection - starts with rules, escalates to ML when needed |

### Ethical Compliance

- **CFAA Compliance** - Only access publicly authorized content
- **robots.txt (RFC 9309)** - Full compliance with Crawl-delay support
- **GDPR** - PII detection with redact/pseudonymize options
- **CCPA** - Respects Global Privacy Control (GPC) signals
- **Adaptive Rate Limiting** - Per-domain delays with automatic backoff
- **Anti-Bot Respect** - Stops on CAPTCHA or block detection

### Change Detection

Automatically classifies structure changes:

| Classification | Similarity | Description |
|----------------|------------|-------------|
| Cosmetic | > 0.95 | Minor styling changes |
| Minor | 0.85 - 0.95 | Small structural updates |
| Moderate | 0.70 - 0.85 | Significant changes |
| Breaking | < 0.70 | Major redesign, selectors likely broken |

### Ollama Cloud Integration

Generate rich, human-readable descriptions of page structures using LLM.

## Installation

### Requirements

- Python 3.11+
- Redis 7+

### Install

```bash
# Clone repository
git clone https://github.com/your-org/adaptive-fingerprint.git
cd adaptive-fingerprint

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install package
pip install -e .
```

### Start Redis

```bash
docker run -d -p 6379:6379 redis:7-alpine
```

## Quick Start

### Set Environment Variables

```bash
export OLLAMA_CLOUD_API_KEY="your-api-key"
export REDIS_URL="redis://localhost:6379/0"
```

### CLI Usage

```bash
# Analyze a URL
fingerprint analyze --url https://example.com

# Compare with stored version (adaptive mode)
fingerprint compare --url https://example.com --mode adaptive

# Generate LLM description
fingerprint describe --url https://example.com

# Verbose output
fingerprint -vvv analyze --url https://example.com
```

### Python API

```python
import asyncio
from fingerprint.config import load_config
from fingerprint.core.analyzer import StructureAnalyzer

async def main():
    config = load_config()
    analyzer = StructureAnalyzer(config)

    # Analyze a URL
    structure = await analyzer.analyze_url("https://example.com/article")
    print(f"Page type: {structure.page_type}")
    print(f"CSS classes: {len(structure.css_class_map)}")
    print(f"Landmarks: {len(structure.semantic_landmarks)}")

    # Compare with stored version
    changes = await analyzer.compare_with_stored("https://example.com/article")
    print(f"Mode used: {changes.mode_used.value}")
    print(f"Similarity: {changes.similarity:.3f}")
    print(f"Classification: {changes.classification.value}")
    print(f"Breaking: {changes.breaking}")

asyncio.run(main())
```

## Configuration

Copy `config.example.yaml` to `config.yaml`:

```yaml
# Fingerprinting mode: "rules", "ml", or "adaptive"
fingerprinting:
  mode: adaptive
  adaptive:
    class_change_threshold: 0.15      # Escalate to ML if >15% classes changed
    rules_uncertainty_threshold: 0.80  # Escalate if rules similarity < 0.80

# Ollama Cloud LLM
ollama_cloud:
  enabled: true
  model: "gemma3:12b"
  timeout: 30

# Redis storage
redis:
  url: "redis://localhost:6379/0"
  key_prefix: "fingerprint"
  ttl_seconds: 604800  # 7 days

# Compliance settings
compliance:
  robots_txt:
    enabled: true
    respect_crawl_delay: true
  rate_limiting:
    enabled: true
    default_delay: 1.0
    backoff_multiplier: 2.0

# Legal compliance
legal:
  cfaa:
    enabled: true
    block_authenticated_areas: true
  gdpr:
    enabled: true
    pii_handling: "redact"

# Verbose logging
verbose:
  enabled: true
  level: 2  # 0=errors, 1=warnings, 2=info, 3=debug
```

## Architecture

```
fingerprint/
├── core/               # Orchestration and fetching
│   ├── analyzer.py     # Main analyzer with mode selection
│   ├── fetcher.py      # Compliance-aware HTTP fetcher
│   └── verbose.py      # Structured logging
│
├── adaptive/           # Rules-based fingerprinting
│   ├── structure_analyzer.py   # DOM structure analysis
│   ├── change_detector.py      # Change detection/classification
│   └── strategy_learner.py     # CSS selector inference
│
├── ml/                 # ML and LLM integration
│   ├── embeddings.py   # Sentence transformer embeddings
│   ├── ollama_client.py # Ollama Cloud API client
│   └── classifier.py   # Page type classification
│
├── storage/            # Redis persistence
│   ├── structure_store.py  # Structure versioning
│   └── embedding_store.py  # Embedding cache
│
├── compliance/         # Ethical compliance
│   ├── robots_parser.py    # RFC 9309 robots.txt
│   ├── rate_limiter.py     # Adaptive rate limiting
│   └── bot_detector.py     # Anti-bot detection
│
└── legal/              # Legal compliance
    ├── cfaa_checker.py     # CFAA authorization
    ├── tos_checker.py      # Terms of Service
    ├── gdpr_handler.py     # GDPR PII handling
    └── ccpa_handler.py     # CCPA opt-out respect
```

## Compliance Pipeline

All URL fetching passes through the compliance pipeline:

```
1. CFAA Check ─────────► Is access authorized?
         │
         ▼
2. robots.txt ─────────► Is path allowed? (RFC 9309)
         │
         ▼
3. Rate Limiter ───────► Acquire slot, respect Crawl-delay
         │
         ▼
4. HTTP Fetch ─────────► Make request with proper headers
         │
         ▼
5. Anti-Bot Check ─────► Detect captcha/block pages
         │
         ▼
6. GDPR/CCPA Check ───► Scan for PII, apply handling
         │
         ▼
7. Return Content ────► Clean content ready for analysis
```

## Adaptive Mode

Adaptive mode intelligently selects the best fingerprinting approach:

1. **Start with Rules** (~15ms) - Fast, deterministic comparison
2. **Check for Escalation Triggers**:
   - `CLASS_VOLATILITY` - >15% of CSS classes changed
   - `RULES_UNCERTAINTY` - Rules similarity < 0.80
   - `KNOWN_VOLATILE` - Domain flagged as frequently changing
   - `RENAME_PATTERN` - Detected class rename patterns
3. **Escalate to ML if triggered** (~200ms) - Semantic similarity

```python
result = await analyzer.compare_with_stored(url)

print(f"Mode used: {result.mode_used.value}")  # "rules" or "ml"
print(f"Escalated: {result.escalated}")

if result.escalated:
    for trigger in result.escalation_triggers:
        print(f"  - {trigger.name}: {trigger.reason}")
```

## Verbose Logging

Enable verbose logging to see detailed operation traces:

```bash
fingerprint -vvv analyze --url https://example.com
```

Output format:
```
[2024-01-15T10:30:00Z] [CFAA:CHECK] Checking authorization for https://example.com
[2024-01-15T10:30:00Z] [CFAA:AUTHORIZED] Access authorized (public page)
[2024-01-15T10:30:00Z] [ROBOTS:CHECK] Checking robots.txt for /
[2024-01-15T10:30:00Z] [ROBOTS:ALLOWED] Path allowed
[2024-01-15T10:30:00Z] [RATELIMIT:ACQUIRE] Acquiring slot for example.com
[2024-01-15T10:30:01Z] [FETCH:REQUEST] GET https://example.com
[2024-01-15T10:30:02Z] [FETCH:RESPONSE] 200 OK (1.2s, 45KB)
[2024-01-15T10:30:02Z] [STRUCTURE:ANALYZE] Analyzing DOM structure
  - tags: 156
  - classes: 89
  - landmarks: 5
[2024-01-15T10:30:02Z] [ADAPTIVE:RESULT] Analysis complete
  - mode: rules
  - similarity: 0.94
  - classification: cosmetic
```

## API Reference

### StructureAnalyzer

```python
class StructureAnalyzer:
    async def analyze_url(self, url: str) -> PageStructure
    async def analyze_html(self, html: str, url: str) -> PageStructure
    async def compare_with_stored(self, url: str) -> ChangeAnalysis
    async def compare_structures(self, old: PageStructure, new: PageStructure) -> ChangeAnalysis
    async def generate_description(self, structure: PageStructure) -> str
```

### PageStructure

```python
@dataclass
class PageStructure:
    domain: str
    page_type: str
    tag_hierarchy: TagHierarchy
    css_class_map: dict[str, int]
    semantic_landmarks: dict[str, str]
    content_regions: list[ContentRegion]
    captured_at: datetime
    version: int
```

### ChangeAnalysis

```python
@dataclass
class ChangeAnalysis:
    similarity: float
    mode_used: FingerprintMode
    classification: ChangeClassification
    breaking: bool
    changes: list[StructureChange]
    escalated: bool
    escalation_triggers: list[EscalationTrigger]
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_CLOUD_API_KEY` | Ollama Cloud API key | (required for ML mode) |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `FINGERPRINT_MODE` | Default mode | `adaptive` |
| `FINGERPRINT_VERBOSE` | Verbosity level (0-3) | `2` |

## Development

See `AGENTS.md` for complete implementation specifications.

### Module Specifications

| Module | Documentation |
|--------|---------------|
| Core | `fingerprint/core/AGENTS.md` |
| Adaptive | `fingerprint/adaptive/AGENTS.md` |
| ML | `fingerprint/ml/AGENTS.md` |
| Storage | `fingerprint/storage/AGENTS.md` |
| Compliance | `fingerprint/compliance/AGENTS.md` |
| Legal | `fingerprint/legal/AGENTS.md` |

## License

MIT
