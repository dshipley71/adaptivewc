# AGENTS.md - Adaptive Structure Fingerprinting System

A complete specification for building an intelligent web structure fingerprinting system with adaptive learning, Ollama Cloud LLM integration, and comprehensive verbose logging.

## Purpose

This document serves as a **complete blueprint** for generating all source code, configuration, and documentation for the Adaptive Structure Fingerprinting System. An AI assistant or developer can use this specification to create the entire application from scratch.

---

## Project Structure

Generate the following directory structure:

```
adaptive-fingerprint/
├── AGENTS.md                       # This file (project specification)
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project metadata and build config
├── config.example.yaml             # Example configuration file
├── .env.example                    # Example environment variables
│
├── fingerprint/                    # Main package
│   ├── __init__.py                 # Package initialization
│   ├── __main__.py                 # CLI entry point
│   ├── config.py                   # Configuration dataclasses
│   ├── models.py                   # Core data models
│   ├── exceptions.py               # Custom exceptions
│   │
│   ├── core/                       # Core orchestration
│   │   ├── __init__.py
│   │   ├── analyzer.py             # Main analyzer orchestrator
│   │   ├── fetcher.py              # HTTP fetching with compliance
│   │   └── verbose.py              # Verbose logging utilities
│   │
│   ├── adaptive/                   # Adaptive fingerprinting
│   │   ├── __init__.py
│   │   ├── structure_analyzer.py   # Rules-based DOM analysis
│   │   ├── change_detector.py      # Change detection and classification
│   │   └── strategy_learner.py     # CSS selector inference
│   │
│   ├── ml/                         # ML and Ollama Cloud integration
│   │   ├── __init__.py
│   │   ├── embeddings.py           # Embedding generation
│   │   ├── ollama_client.py        # Ollama Cloud API client
│   │   └── classifier.py           # Page type classification
│   │
│   ├── storage/                    # Redis persistence
│   │   ├── __init__.py
│   │   ├── structure_store.py      # Structure storage
│   │   ├── embedding_store.py      # Embedding storage
│   │   └── cache.py                # Caching utilities
│   │
│   └── utils/                      # Utilities
│       ├── __init__.py
│       ├── url_utils.py            # URL normalization
│       └── html_utils.py           # HTML parsing utilities
│
└── examples/                       # Example scripts
    ├── basic_fingerprint.py        # Basic usage example
    ├── adaptive_mode.py            # Adaptive mode example
    └── ml_fingerprint.py           # ML mode with Ollama Cloud
```

---

## Dependencies

### requirements.txt

Generate with these exact contents:

```
# Core dependencies
httpx>=0.27.0
beautifulsoup4>=4.12.0
lxml>=5.0.0
redis>=5.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
pyyaml>=6.0.0

# ML dependencies
sentence-transformers>=2.3.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Optional ML backends (install as needed)
# xgboost>=2.0.0
# lightgbm>=4.0.0

# Async support
anyio>=4.0.0

# CLI
click>=8.1.0
rich>=13.0.0
```

### pyproject.toml

```toml
[project]
name = "adaptive-fingerprint"
version = "1.0.0"
description = "Adaptive web structure fingerprinting with ML and Ollama Cloud integration"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}

dependencies = [
    "httpx>=0.27.0",
    "beautifulsoup4>=4.12.0",
    "lxml>=5.0.0",
    "redis>=5.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "pyyaml>=6.0.0",
    "sentence-transformers>=2.3.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "anyio>=4.0.0",
    "click>=8.1.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
ml-backends = [
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
]

[project.scripts]
fingerprint = "fingerprint.__main__:main"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

---

## Configuration

### config.example.yaml

```yaml
# Adaptive Fingerprint Configuration

# Fingerprinting mode: "rules", "ml", or "adaptive"
fingerprinting:
  mode: adaptive

  # Adaptive mode settings
  adaptive:
    class_change_threshold: 0.15      # Trigger ML if >15% classes changed
    rules_uncertainty_threshold: 0.80  # Trigger ML if rules < 0.80
    cache_ml_results: true

  # Change classification thresholds
  thresholds:
    cosmetic: 0.95    # > 0.95 = cosmetic change
    minor: 0.85       # 0.85-0.95 = minor change
    moderate: 0.70    # 0.70-0.85 = moderate change
    breaking: 0.70    # < 0.70 = breaking change

# Ollama Cloud LLM settings
ollama_cloud:
  enabled: true
  model: "gemma3:12b"
  timeout: 30
  max_retries: 3
  temperature: 0.3
  max_tokens: 500

# Embedding model settings
embeddings:
  model: "all-MiniLM-L6-v2"
  cache_embeddings: true

# Redis storage
redis:
  url: "redis://localhost:6379/0"
  key_prefix: "fingerprint"
  ttl_seconds: 604800  # 7 days
  max_versions: 10

# HTTP fetching
http:
  user_agent: "AdaptiveFingerprint/1.0"
  timeout: 30
  max_retries: 3
  respect_robots_txt: true

# Verbose logging
verbose:
  enabled: true
  level: 2  # 0=errors, 1=warnings, 2=info, 3=debug
  format: "structured"  # "structured" or "plain"
  include_timestamp: true
```

### .env.example

```bash
# Required for Ollama Cloud
OLLAMA_CLOUD_API_KEY=your-api-key-here

# Optional overrides
FINGERPRINT_MODE=adaptive
FINGERPRINT_VERBOSE=2
REDIS_URL=redis://localhost:6379/0
```

---

## Core Data Models

### fingerprint/models.py

```python
"""
Core data models for the fingerprinting system.

All models use dataclasses for clean, type-safe data structures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class FingerprintMode(Enum):
    """Fingerprinting mode selection."""
    RULES = "rules"
    ML = "ml"
    ADAPTIVE = "adaptive"


class ChangeClassification(Enum):
    """Change severity classification."""
    COSMETIC = "cosmetic"    # > 0.95 similarity
    MINOR = "minor"          # 0.85 - 0.95
    MODERATE = "moderate"    # 0.70 - 0.85
    BREAKING = "breaking"    # < 0.70


class ChangeType(Enum):
    """Specific types of detected changes."""
    # Tag/Class changes
    TAG_ADDED = "tag_added"
    TAG_REMOVED = "tag_removed"
    CLASS_RENAMED = "class_renamed"
    CLASS_ADDED = "class_added"
    CLASS_REMOVED = "class_removed"
    ID_CHANGED = "id_changed"

    # Structural changes
    STRUCTURE_REORGANIZED = "structure_reorganized"
    CONTENT_RELOCATED = "content_relocated"
    LANDMARK_CHANGED = "landmark_changed"

    # Script/Framework changes
    SCRIPT_ADDED = "script_added"
    SCRIPT_REMOVED = "script_removed"
    FRAMEWORK_CHANGED = "framework_changed"

    # Navigation changes
    NAVIGATION_CHANGED = "navigation_changed"
    PAGINATION_CHANGED = "pagination_changed"

    # Overall
    MINOR_LAYOUT_SHIFT = "minor_layout_shift"
    MAJOR_REDESIGN = "major_redesign"


@dataclass
class TagHierarchy:
    """Tag structure analysis."""
    tag_counts: dict[str, int]
    depth_distribution: dict[int, int]
    parent_child_pairs: dict[str, int]
    max_depth: int = 0


@dataclass
class ContentRegion:
    """Identified content extraction zone."""
    name: str
    primary_selector: str
    fallback_selectors: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class PageStructure:
    """
    Complete fingerprint of a page's DOM structure.

    This is the primary data model for storing and comparing page structures.
    """
    # Identification
    domain: str
    page_type: str
    url_pattern: str = ""
    variant_id: str = "default"

    # Tag structure
    tag_hierarchy: TagHierarchy | None = None

    # CSS analysis
    css_class_map: dict[str, int] = field(default_factory=dict)
    id_attributes: set[str] = field(default_factory=set)

    # Semantic structure
    semantic_landmarks: dict[str, str] = field(default_factory=dict)

    # Content regions
    content_regions: list[ContentRegion] = field(default_factory=list)
    navigation_selectors: list[str] = field(default_factory=list)

    # Script analysis
    script_signatures: list[str] = field(default_factory=list)
    detected_framework: str | None = None

    # Metadata
    captured_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    content_hash: str = ""

    # Description (for ML mode)
    description: str = ""


@dataclass
class StructureEmbedding:
    """Embedding representation of a page structure."""
    domain: str
    page_type: str
    variant_id: str = "default"

    # Embedding vector (numpy array stored as list for serialization)
    vector: list[float] = field(default_factory=list)
    dimensions: int = 384

    # Metadata
    model_name: str = "all-MiniLM-L6-v2"
    description: str = ""
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StructureChange:
    """Single detected change between structures."""
    change_type: ChangeType
    affected_components: list[str] = field(default_factory=list)

    # Details
    old_value: str | None = None
    new_value: str | None = None
    location: str = ""

    # Impact
    breaking: bool = False
    fields_affected: list[str] = field(default_factory=list)
    confidence: float = 0.0

    # Documentation
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class EscalationTrigger:
    """Trigger that caused adaptive mode to escalate to ML."""
    name: str
    reason: str
    threshold: float | None = None
    actual_value: float | None = None


@dataclass
class ChangeAnalysis:
    """
    Complete analysis of changes between structures.

    Contains the result from ONE fingerprinting mode (not combined).
    """
    # Similarity (from one mode)
    similarity: float
    mode_used: FingerprintMode

    # Classification
    classification: ChangeClassification
    breaking: bool

    # Detailed changes
    changes: list[StructureChange] = field(default_factory=list)

    # Impact assessment
    fields_affected: dict[str, str] = field(default_factory=dict)
    can_auto_adapt: bool = False
    adaptation_confidence: float = 0.0

    # Documentation
    reason: str = ""

    # Adaptive mode details
    escalated: bool = False
    escalation_triggers: list[EscalationTrigger] = field(default_factory=list)

    # Timing
    duration_ms: float = 0.0


@dataclass
class SelectorRule:
    """CSS selector extraction rule."""
    primary: str
    fallbacks: list[str] = field(default_factory=list)
    extraction_method: str = "text"  # "text", "html", "attribute"
    attribute_name: str | None = None
    post_processors: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ExtractionStrategy:
    """Rules for extracting content from a page type."""
    domain: str
    page_type: str
    version: int = 1

    # Extraction rules
    title: SelectorRule | None = None
    content: SelectorRule | None = None
    metadata: dict[str, SelectorRule] = field(default_factory=dict)

    # Learning metadata
    learned_at: datetime = field(default_factory=datetime.utcnow)
    learning_source: str = "initial"  # "initial", "adaptation", "manual"
    confidence_scores: dict[str, float] = field(default_factory=dict)
```

---

## Configuration Module

### fingerprint/config.py

```python
"""
Configuration management using Pydantic settings.

Loads configuration from:
1. Environment variables (FINGERPRINT_* prefix)
2. YAML configuration file
3. Default values
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FingerprintSettings(BaseSettings):
    """Environment-based settings with FINGERPRINT_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="FINGERPRINT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core settings
    mode: str = "adaptive"
    verbose: int = 2

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Ollama Cloud
    ollama_cloud_api_key: str = ""
    ollama_cloud_model: str = "gemma3:12b"
    ollama_cloud_timeout: int = 30

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class AdaptiveConfig:
    """Adaptive mode configuration."""
    class_change_threshold: float = 0.15
    rules_uncertainty_threshold: float = 0.80
    cache_ml_results: bool = True


@dataclass
class ThresholdsConfig:
    """Change classification thresholds."""
    cosmetic: float = 0.95
    minor: float = 0.85
    moderate: float = 0.70
    breaking: float = 0.70


@dataclass
class OllamaCloudConfig:
    """Ollama Cloud API configuration."""
    enabled: bool = True
    api_key: str = ""
    model: str = "gemma3:12b"
    timeout: int = 30
    max_retries: int = 3
    temperature: float = 0.3
    max_tokens: int = 500


@dataclass
class EmbeddingsConfig:
    """Embedding model configuration."""
    model: str = "all-MiniLM-L6-v2"
    cache_embeddings: bool = True


@dataclass
class RedisConfig:
    """Redis storage configuration."""
    url: str = "redis://localhost:6379/0"
    key_prefix: str = "fingerprint"
    ttl_seconds: int = 604800  # 7 days
    max_versions: int = 10


@dataclass
class HttpConfig:
    """HTTP client configuration."""
    user_agent: str = "AdaptiveFingerprint/1.0"
    timeout: int = 30
    max_retries: int = 3
    respect_robots_txt: bool = True


@dataclass
class VerboseConfig:
    """Verbose logging configuration."""
    enabled: bool = True
    level: int = 2  # 0=errors, 1=warnings, 2=info, 3=debug
    format: str = "structured"
    include_timestamp: bool = True


@dataclass
class Config:
    """Complete application configuration."""
    mode: str = "adaptive"
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    ollama_cloud: OllamaCloudConfig = field(default_factory=OllamaCloudConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    http: HttpConfig = field(default_factory=HttpConfig)
    verbose: VerboseConfig = field(default_factory=VerboseConfig)


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file and environment.

    Priority: Environment > YAML file > Defaults
    """
    config = Config()

    # Load from YAML if provided
    if config_path and config_path.exists():
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                config = _merge_yaml_config(config, yaml_config)

    # Override with environment variables
    env_settings = FingerprintSettings()
    config.mode = env_settings.mode
    config.verbose.level = env_settings.verbose
    config.redis.url = env_settings.redis_url
    config.ollama_cloud.api_key = env_settings.ollama_cloud_api_key
    config.ollama_cloud.model = env_settings.ollama_cloud_model
    config.embeddings.model = env_settings.embedding_model

    return config


def _merge_yaml_config(config: Config, yaml_data: dict[str, Any]) -> Config:
    """Merge YAML configuration into Config object."""
    if "fingerprinting" in yaml_data:
        fp = yaml_data["fingerprinting"]
        config.mode = fp.get("mode", config.mode)

        if "adaptive" in fp:
            config.adaptive.class_change_threshold = fp["adaptive"].get(
                "class_change_threshold", config.adaptive.class_change_threshold
            )
            config.adaptive.rules_uncertainty_threshold = fp["adaptive"].get(
                "rules_uncertainty_threshold", config.adaptive.rules_uncertainty_threshold
            )

        if "thresholds" in fp:
            config.thresholds.cosmetic = fp["thresholds"].get("cosmetic", config.thresholds.cosmetic)
            config.thresholds.minor = fp["thresholds"].get("minor", config.thresholds.minor)
            config.thresholds.moderate = fp["thresholds"].get("moderate", config.thresholds.moderate)

    if "ollama_cloud" in yaml_data:
        oc = yaml_data["ollama_cloud"]
        config.ollama_cloud.enabled = oc.get("enabled", config.ollama_cloud.enabled)
        config.ollama_cloud.model = oc.get("model", config.ollama_cloud.model)
        config.ollama_cloud.timeout = oc.get("timeout", config.ollama_cloud.timeout)

    if "redis" in yaml_data:
        r = yaml_data["redis"]
        config.redis.url = r.get("url", config.redis.url)
        config.redis.key_prefix = r.get("key_prefix", config.redis.key_prefix)
        config.redis.ttl_seconds = r.get("ttl_seconds", config.redis.ttl_seconds)

    if "verbose" in yaml_data:
        v = yaml_data["verbose"]
        config.verbose.enabled = v.get("enabled", config.verbose.enabled)
        config.verbose.level = v.get("level", config.verbose.level)

    return config
```

---

## Exceptions

### fingerprint/exceptions.py

```python
"""
Custom exception hierarchy for the fingerprinting system.

All exceptions inherit from FingerprintError for easy catching.
"""


class FingerprintError(Exception):
    """Base exception for all fingerprinting errors."""
    pass


# Analysis errors
class AnalysisError(FingerprintError):
    """Error during structure analysis."""
    pass


class InvalidHTMLError(AnalysisError):
    """HTML content is invalid or unparseable."""
    pass


class EmptyContentError(AnalysisError):
    """Content is empty or contains no meaningful structure."""
    pass


# Change detection errors
class ChangeDetectionError(FingerprintError):
    """Error during change detection."""
    pass


class IncompatibleStructuresError(ChangeDetectionError):
    """Structures cannot be compared (different domains/types)."""
    pass


# ML errors
class MLError(FingerprintError):
    """Error in ML operations."""
    pass


class EmbeddingError(MLError):
    """Error generating embeddings."""
    pass


class ModelLoadError(MLError):
    """Error loading ML model."""
    pass


# Ollama Cloud errors
class OllamaCloudError(FingerprintError):
    """Error with Ollama Cloud API."""
    pass


class OllamaAuthError(OllamaCloudError):
    """Authentication failed with Ollama Cloud."""
    pass


class OllamaTimeoutError(OllamaCloudError):
    """Request to Ollama Cloud timed out."""
    pass


class OllamaRateLimitError(OllamaCloudError):
    """Rate limited by Ollama Cloud."""
    pass


# Storage errors
class StorageError(FingerprintError):
    """Error in storage operations."""
    pass


class RedisConnectionError(StorageError):
    """Cannot connect to Redis."""
    pass


class SerializationError(StorageError):
    """Error serializing/deserializing data."""
    pass


# HTTP errors
class FetchError(FingerprintError):
    """Error fetching URL."""
    pass


class HTTPTimeoutError(FetchError):
    """HTTP request timed out."""
    pass


class HTTPStatusError(FetchError):
    """HTTP request returned error status."""
    def __init__(self, status_code: int, message: str = ""):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")
```

---

## Verbose Logging System

### fingerprint/core/verbose.py

```python
"""
Verbose logging system with structured output.

All modules use this for consistent logging format:
[TIMESTAMP] [MODULE:OPERATION] Message
  - detail_1
  - detail_2
"""

from datetime import datetime
from typing import Any

from rich.console import Console

from fingerprint.config import VerboseConfig


class VerboseLogger:
    """
    Structured verbose logger.

    Usage:
        logger = VerboseLogger(config.verbose)
        logger.log("MODULE", "OPERATION", "Message", details={"key": "value"})
    """

    def __init__(self, config: VerboseConfig):
        self.config = config
        self.console = Console()
        self._enabled = config.enabled
        self._level = config.level

    def log(
        self,
        module: str,
        operation: str,
        message: str,
        level: int = 2,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a verbose message.

        Args:
            module: Module name (e.g., "STRUCTURE", "ML", "ADAPTIVE")
            operation: Operation name (e.g., "ANALYZE", "COMPARE", "ESCALATE")
            message: Main message
            level: Log level (0=error, 1=warn, 2=info, 3=debug)
            details: Optional dict of details to display
        """
        if not self._enabled or level > self._level:
            return

        # Build log line
        prefix = f"[{module}:{operation}]"

        if self.config.include_timestamp:
            timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            line = f"[{timestamp}] {prefix} {message}"
        else:
            line = f"{prefix} {message}"

        # Output
        if self.config.format == "structured":
            self._output_structured(line, details, level)
        else:
            self._output_plain(line, details)

    def _output_structured(
        self,
        line: str,
        details: dict[str, Any] | None,
        level: int,
    ) -> None:
        """Output with rich formatting."""
        # Color based on level
        colors = {0: "red", 1: "yellow", 2: "cyan", 3: "dim"}
        color = colors.get(level, "white")

        self.console.print(f"[{color}]{line}[/{color}]")

        if details:
            for key, value in details.items():
                self.console.print(f"  [dim]- {key}: {value}[/dim]")

    def _output_plain(
        self,
        line: str,
        details: dict[str, Any] | None,
    ) -> None:
        """Output plain text."""
        print(line)
        if details:
            for key, value in details.items():
                print(f"  - {key}: {value}")

    # Convenience methods
    def error(self, module: str, operation: str, message: str, **details: Any) -> None:
        self.log(module, operation, message, level=0, details=details or None)

    def warn(self, module: str, operation: str, message: str, **details: Any) -> None:
        self.log(module, operation, message, level=1, details=details or None)

    def info(self, module: str, operation: str, message: str, **details: Any) -> None:
        self.log(module, operation, message, level=2, details=details or None)

    def debug(self, module: str, operation: str, message: str, **details: Any) -> None:
        self.log(module, operation, message, level=3, details=details or None)


# Global logger instance (set by main application)
_logger: VerboseLogger | None = None


def get_logger() -> VerboseLogger:
    """Get the global verbose logger."""
    global _logger
    if _logger is None:
        # Create default logger
        _logger = VerboseLogger(VerboseConfig())
    return _logger


def set_logger(logger: VerboseLogger) -> None:
    """Set the global verbose logger."""
    global _logger
    _logger = logger
```

---

## CLI Entry Point

### fingerprint/__main__.py

```python
"""
CLI entry point for the fingerprinting system.

Usage:
    fingerprint analyze --url https://example.com
    fingerprint compare --url https://example.com --stored-version 1
    fingerprint describe --url https://example.com
"""

import asyncio
from pathlib import Path

import click

from fingerprint.config import load_config
from fingerprint.core.analyzer import StructureAnalyzer
from fingerprint.core.verbose import VerboseLogger, set_logger


@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.option("--verbose", "-v", count=True, help="Increase verbosity")
@click.pass_context
def main(ctx: click.Context, config: str | None, verbose: int) -> None:
    """Adaptive Structure Fingerprinting System."""
    ctx.ensure_object(dict)

    # Load configuration
    config_path = Path(config) if config else None
    ctx.obj["config"] = load_config(config_path)

    # Override verbose level
    if verbose:
        ctx.obj["config"].verbose.level = min(verbose + 1, 3)

    # Set up logger
    logger = VerboseLogger(ctx.obj["config"].verbose)
    set_logger(logger)


@main.command()
@click.option("--url", "-u", required=True, help="URL to analyze")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.pass_context
def analyze(ctx: click.Context, url: str, output: str | None) -> None:
    """Analyze a URL and generate structure fingerprint."""
    config = ctx.obj["config"]

    async def run():
        analyzer = StructureAnalyzer(config)
        result = await analyzer.analyze_url(url)

        if output:
            # Save to file
            import json
            with open(output, "w") as f:
                json.dump(result.__dict__, f, indent=2, default=str)
            click.echo(f"Saved to {output}")
        else:
            # Print summary
            click.echo(f"Domain: {result.domain}")
            click.echo(f"Page type: {result.page_type}")
            click.echo(f"Tags: {len(result.tag_hierarchy.tag_counts) if result.tag_hierarchy else 0}")
            click.echo(f"Classes: {len(result.css_class_map)}")
            click.echo(f"Landmarks: {len(result.semantic_landmarks)}")

    asyncio.run(run())


@main.command()
@click.option("--url", "-u", required=True, help="URL to compare")
@click.option("--mode", "-m", type=click.Choice(["rules", "ml", "adaptive"]), default="adaptive")
@click.pass_context
def compare(ctx: click.Context, url: str, mode: str) -> None:
    """Compare current structure with stored version."""
    config = ctx.obj["config"]
    config.mode = mode

    async def run():
        analyzer = StructureAnalyzer(config)
        result = await analyzer.compare_with_stored(url)

        click.echo(f"Mode: {result.mode_used.value}")
        click.echo(f"Similarity: {result.similarity:.3f}")
        click.echo(f"Classification: {result.classification.value}")
        click.echo(f"Breaking: {result.breaking}")

        if result.escalated:
            click.echo(f"Escalated: Yes")
            for trigger in result.escalation_triggers:
                click.echo(f"  - {trigger.name}: {trigger.reason}")

    asyncio.run(run())


@main.command()
@click.option("--url", "-u", required=True, help="URL to describe")
@click.pass_context
def describe(ctx: click.Context, url: str) -> None:
    """Generate LLM description of page structure."""
    config = ctx.obj["config"]

    async def run():
        analyzer = StructureAnalyzer(config)
        structure = await analyzer.analyze_url(url)
        description = await analyzer.generate_description(structure)

        click.echo("Description:")
        click.echo(description)

    asyncio.run(run())


if __name__ == "__main__":
    main()
```

---

## Fingerprinting Modes

The system provides three independent fingerprinting modes. Each mode operates independently - they are **not combined**.

### Mode 1: Rules-Based

Fast, deterministic fingerprinting using DOM structure analysis.

**When to Use:**
- High-throughput analysis where speed matters
- Sites with stable CSS class names
- When you need deterministic, reproducible results
- Offline environments without embedding model access

**Latency:** ~15ms per comparison

### Mode 2: ML-Based

Semantic fingerprinting using sentence transformer embeddings.

**When to Use:**
- Sites known to frequently rename CSS classes
- When semantic similarity matters more than exact structure
- Sites that undergo regular CSS refactoring
- When you need rich, human-readable change descriptions

**Latency:** ~200ms per comparison

### Mode 3: Adaptive (Recommended)

Intelligent mode selection that starts with fast rules-based comparison and escalates to ML only when needed.

**When to Use:**
- Production environments with mixed site types
- When you don't know site characteristics in advance
- First-time visits to new domains
- When you want optimal speed without sacrificing accuracy

**Latency:** ~15-200ms depending on escalation

**Escalation Triggers:**
| Trigger | Condition | Rationale |
|---------|-----------|-----------|
| CLASS_VOLATILITY | >15% of classes changed | Likely CSS refactor |
| RULES_UNCERTAINTY | Rules similarity < 0.80 | Result is ambiguous |
| KNOWN_VOLATILE | Domain flagged as volatile | Historical data |
| RENAME_PATTERN | Detected rename patterns | e.g., prefix removal |

### Important: Modes are Independent

- **No weighted combination**: We do not combine rules and ML scores
- **No parallel execution**: Adaptive runs rules first, then ML only if triggered
- **Single result**: Returns one similarity score from one mode
- **Clear provenance**: Result indicates which mode produced it

---

## Ollama Cloud Integration

### Endpoint

```
POST https://ollama.com/api/chat
```

### Authentication

```
Authorization: Bearer {OLLAMA_CLOUD_API_KEY}
```

### Request Format

```json
{
    "model": "gemma3:12b",
    "messages": [{"role": "user", "content": "..."}],
    "stream": false,
    "options": {
        "num_predict": 500,
        "temperature": 0.3
    }
}
```

### Response Format

```json
{
    "message": {
        "role": "assistant",
        "content": "Generated description..."
    }
}
```

See `fingerprint/ml/AGENTS.md` for complete implementation details.

---

## Redis Storage Schema

### Key Patterns

```
# Structure storage
{prefix}:structure:{domain}:{page_type}:{variant_id}
{prefix}:structure:{domain}:{page_type}:{variant_id}:v{n}

# Embeddings
{prefix}:embedding:{domain}:{page_type}:{variant_id}

# Extraction strategies
{prefix}:strategy:{domain}:{page_type}:{variant_id}

# Change history
{prefix}:changes:{domain}:{page_type}

# Volatile sites tracking
{prefix}:volatile:{domain}
```

See `fingerprint/storage/AGENTS.md` for complete implementation details.

---

## Verbose Logging Format

All modules use consistent verbose logging:

```
[TIMESTAMP] [MODULE:OPERATION] Message
  - key: value
  - key: value
```

### Module Prefixes

| Module | Prefix | Operations |
|--------|--------|------------|
| Analyzer | `ANALYZER` | INIT, FETCH, ANALYZE, COMPARE, RESULT |
| Structure | `STRUCTURE` | PARSE, TAGS, CLASSES, LANDMARKS, REGIONS, HASH |
| Change | `CHANGE` | COMPARE, SIMILARITY, CLASSIFY, DETAILS |
| Adaptive | `ADAPTIVE` | START, RULES, ANALYZE, TRIGGER, ESCALATE, RESULT |
| ML | `ML` | DESCRIBE, EMBED, SIMILARITY |
| Ollama | `OLLAMA` | INIT, REQUEST, RESPONSE, ERROR |
| Store | `STORE` | GET, SAVE, UPDATE, DELETE |

---

## Module Specifications

The following AGENTS.md files contain detailed implementation specifications for each module:

- `fingerprint/core/AGENTS.md` - Core analyzer and orchestration
- `fingerprint/adaptive/AGENTS.md` - Rules-based fingerprinting and change detection
- `fingerprint/ml/AGENTS.md` - ML embeddings and Ollama Cloud integration
- `fingerprint/storage/AGENTS.md` - Redis storage layer

---

## Example Usage

### Basic Fingerprinting

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
    print(f"Classes: {len(structure.css_class_map)}")

    # Compare with stored version
    changes = await analyzer.compare_with_stored("https://example.com/article")
    print(f"Similarity: {changes.similarity:.3f}")
    print(f"Breaking: {changes.breaking}")

asyncio.run(main())
```

### Using Adaptive Mode

```python
import asyncio
from fingerprint.config import load_config
from fingerprint.core.analyzer import StructureAnalyzer

async def main():
    config = load_config()
    config.mode = "adaptive"

    analyzer = StructureAnalyzer(config)

    # Adaptive mode automatically selects best approach
    result = await analyzer.compare_with_stored("https://example.com")

    print(f"Mode used: {result.mode_used.value}")
    if result.escalated:
        print("Escalated to ML because:")
        for trigger in result.escalation_triggers:
            print(f"  - {trigger.name}: {trigger.reason}")

asyncio.run(main())
```

### Generating Descriptions with Ollama Cloud

```python
import asyncio
from fingerprint.config import load_config
from fingerprint.core.analyzer import StructureAnalyzer

async def main():
    config = load_config()
    config.ollama_cloud.enabled = True

    analyzer = StructureAnalyzer(config)

    structure = await analyzer.analyze_url("https://example.com")
    description = await analyzer.generate_description(structure)

    print("LLM Description:")
    print(description)

asyncio.run(main())
```

---

## README.md

Generate with these contents:

```markdown
# Adaptive Structure Fingerprinting System

An intelligent web structure fingerprinting system with adaptive learning, Ollama Cloud LLM integration, and comprehensive verbose logging.

## Features

- **Three Fingerprinting Modes**: Rules-based (fast), ML-based (semantic), Adaptive (smart selection)
- **Ollama Cloud Integration**: Rich structure descriptions via LLM
- **Change Detection**: Classify changes as cosmetic, minor, moderate, or breaking
- **Redis Storage**: Persistent structure storage with versioning
- **Verbose Logging**: Comprehensive logging for debugging and monitoring

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Set Ollama Cloud API key
export OLLAMA_CLOUD_API_KEY="your-api-key"

# Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# Analyze a URL
fingerprint analyze --url https://example.com

# Compare with stored version
fingerprint compare --url https://example.com --mode adaptive

# Generate LLM description
fingerprint describe --url https://example.com
```

## Configuration

Copy `config.example.yaml` to `config.yaml` and customize as needed.

## Documentation

See `AGENTS.md` for complete specification and implementation details.

## License

MIT
```
