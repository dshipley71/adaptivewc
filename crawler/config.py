"""
Configuration dataclasses for the adaptive web crawler.

All configuration can be set via environment variables with the CRAWLER_ prefix.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PIIHandling(str, Enum):
    """How to handle detected PII."""

    REDACT = "redact"
    PSEUDONYMIZE = "pseudonymize"
    EXCLUDE_PAGE = "exclude_page"
    FLAG_FOR_REVIEW = "flag_for_review"


class LawfulBasis(str, Enum):
    """GDPR lawful basis for processing."""

    LEGITIMATE_INTEREST = "legitimate_interest"
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"


class StructureStoreType(str, Enum):
    """Type of structure store to use."""

    BASIC = "basic"          # Rule-based SemanticDescriptionGenerator
    LLM = "llm"              # LLM-powered descriptions


class LLMProviderType(str, Enum):
    """LLM provider for structure descriptions."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class CrawlerSettings(BaseSettings):
    """Main settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="CRAWLER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Identity
    user_agent: str = "AdaptiveCrawler/1.0 (+https://example.com/bot; bot@example.com)"

    # Rate limiting
    default_delay: float = 1.0
    min_delay: float = 0.5
    max_delay: float = 60.0
    max_concurrent_per_domain: int = 1
    max_concurrent_global: int = 10
    respect_crawl_delay: bool = True
    backoff_multiplier: float = 2.0

    # Robots.txt
    robots_cache_ttl: int = 86400  # 24 hours
    respect_tos: bool = True

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Adaptive extraction
    enable_adaptive: bool = True
    structure_ttl: int = 604800  # 7 days
    change_threshold: float = 0.3

    # Structure store configuration
    structure_store_type: StructureStoreType = StructureStoreType.BASIC
    structure_store_embeddings: bool = False
    structure_store_embedding_model: str = "all-MiniLM-L6-v2"
    
    # LLM provider configuration (only used when structure_store_type = "llm")
    llm_provider: LLMProviderType = LLMProviderType.ANTHROPIC
    llm_model: str = ""  # Empty = use provider default
    llm_api_key: str = ""  # Empty = use environment variable
    ollama_base_url: str = "http://localhost:11434"

    # Safety limits
    max_page_size_mb: float = 10.0
    request_timeout_seconds: float = 30.0
    max_retries: int = 3
    max_redirects: int = 10
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 300.0

    # Security
    verify_ssl: bool = True
    block_private_ips: bool = True

    # GDPR
    gdpr_enabled: bool = True
    gdpr_retention_days: int = 365
    gdpr_lawful_basis: LawfulBasis = LawfulBasis.LEGITIMATE_INTEREST

    # PII
    pii_handling: PIIHandling = PIIHandling.REDACT

    # CCPA
    ccpa_enabled: bool = True
    ccpa_honor_gpc: bool = True

    # Legal
    dpo_email: str = ""
    legal_blocklist_path: str = "/etc/crawler/legal_blocklist.txt"

    # Alerting
    alert_slack_webhook: str = ""
    alert_email_to: str = ""
    alert_throttle_minutes: int = 60

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    default_delay: float = 1.0
    min_delay: float = 0.5
    max_delay: float = 60.0
    respect_crawl_delay: bool = True
    adaptive: bool = True
    backoff_multiplier: float = 2.0
    max_concurrent_per_domain: int = 1
    max_concurrent_global: int = 10


@dataclass
class SafetyLimits:
    """Safety limits to prevent abuse and resource exhaustion."""

    max_page_size_mb: float = 10.0
    request_timeout_seconds: float = 30.0
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 300.0


@dataclass
class SecurityConfig:
    """Security settings."""

    verify_ssl: bool = True
    block_private_ips: bool = True
    max_redirects: int = 10
    allowed_schemes: list[str] = field(default_factory=lambda: ["http", "https"])


@dataclass
class GDPRConfig:
    """GDPR compliance configuration."""

    enabled: bool = True
    lawful_basis: LawfulBasis = LawfulBasis.LEGITIMATE_INTEREST
    legitimate_interest_assessment: str = ""
    collect_only: list[str] = field(
        default_factory=lambda: ["url", "title", "content", "metadata", "published_date"]
    )
    exclude_pii_patterns: bool = True
    retention_days: int = 365
    retention_policy: str = "delete"
    eu_domains_only: bool = False
    process_in_eu: bool = True


@dataclass
class CCPAConfig:
    """CCPA compliance configuration."""

    enabled: bool = True
    disclosure_categories: list[str] = field(
        default_factory=lambda: ["identifiers", "internet_activity", "geolocation"]
    )
    deletion_verification: bool = True
    honor_gpc_header: bool = True
    do_not_sell: bool = True


@dataclass
class PIIHandlingConfig:
    """PII detection and handling configuration."""

    action: PIIHandling = PIIHandling.REDACT
    log_detections: bool = True
    alert_on_sensitive: bool = True
    sensitive_categories: list[str] = field(
        default_factory=lambda: ["health", "financial", "biometric", "racial_ethnic"]
    )


@dataclass
class ProxyConfig:
    """Proxy configuration with ethical constraints."""

    enabled: bool = False
    proxy_urls: list[str] = field(default_factory=list)
    rotation_strategy: str = "round_robin"
    respect_rate_limits_per_proxy: bool = True
    aggregate_rate_limit: bool = True
    max_aggregate_rps: float = 10.0
    health_check_interval: int = 300
    failure_threshold: int = 3


@dataclass
class PolitenessConfig:
    """Advanced politeness settings."""

    prefer_off_peak: bool = True
    off_peak_hours: tuple[int, int] = (1, 6)
    off_peak_rate_multiplier: float = 1.5
    respect_server_timing: bool = True
    max_response_time_ms: int = 5000
    retry_respectful: bool = True
    max_retry_delay: float = 3600.0


@dataclass
class AlertConfig:
    """Alerting configuration."""

    slack_webhook: str = ""
    email_to: str = ""
    pagerduty_key: str = ""
    webhook_url: str = ""
    throttle_minutes: int = 60
    aggregate: bool = True
    min_severity: str = "WARNING"


@dataclass
class StructureStoreConfig:
    """Configuration for structure storage and change detection."""

    store_type: StructureStoreType = StructureStoreType.BASIC
    enable_embeddings: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"
    ttl_seconds: int = 604800  # 7 days
    max_versions: int = 10
    
    # LLM settings (only used when store_type = LLM)
    llm_provider: LLMProviderType = LLMProviderType.ANTHROPIC
    llm_model: str = ""  # Empty = use provider default
    llm_api_key: str = ""  # Empty = use environment variable
    ollama_base_url: str = "http://localhost:11434"


@dataclass
class CrawlConfig:
    """Complete crawl job configuration."""

    seed_urls: list[str]
    output_dir: str
    max_depth: int = 10
    max_pages: int | None = None
    allowed_domains: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)

    # Sub-configurations
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    safety: SafetyLimits = field(default_factory=SafetyLimits)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    gdpr: GDPRConfig = field(default_factory=GDPRConfig)
    ccpa: CCPAConfig = field(default_factory=CCPAConfig)
    pii: PIIHandlingConfig = field(default_factory=PIIHandlingConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    politeness: PolitenessConfig = field(default_factory=PolitenessConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    structure_store: StructureStoreConfig = field(default_factory=StructureStoreConfig)


def load_config() -> CrawlerSettings:
    """Load configuration from environment variables."""
    return CrawlerSettings()
