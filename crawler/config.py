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
    OLLAMA_CLOUD = "ollama-cloud"


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


def load_yaml_config(config_path: str) -> CrawlConfig:
    """
    Load crawl configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        CrawlConfig instance populated from the YAML file.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If required fields are missing.
        
    Example:
        ```python
        from crawler.config import load_yaml_config
        
        config = load_yaml_config("config.yaml")
        crawler = Crawler(config)
        ```
    """
    import yaml
    from pathlib import Path
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, "r") as f:
        data = yaml.safe_load(f)
    
    return _parse_yaml_config(data)


def _parse_yaml_config(data: dict) -> CrawlConfig:
    """Parse YAML data into CrawlConfig."""
    
    # Extract crawl settings (required)
    crawl = data.get("crawl", {})
    if not crawl.get("seed_urls"):
        raise ValueError("crawl.seed_urls is required")
    if not crawl.get("output_dir"):
        raise ValueError("crawl.output_dir is required")
    
    # Build sub-configurations
    rate_limit_data = data.get("rate_limit", {})
    rate_limit = RateLimitConfig(
        default_delay=rate_limit_data.get("default_delay", 1.0),
        min_delay=rate_limit_data.get("min_delay", 0.5),
        max_delay=rate_limit_data.get("max_delay", 60.0),
        respect_crawl_delay=rate_limit_data.get("respect_crawl_delay", True),
        adaptive=rate_limit_data.get("adaptive", True),
        backoff_multiplier=rate_limit_data.get("backoff_multiplier", 2.0),
        max_concurrent_per_domain=rate_limit_data.get("max_concurrent_per_domain", 1),
        max_concurrent_global=rate_limit_data.get("max_concurrent_global", 10),
    )
    
    safety_data = data.get("safety", {})
    safety = SafetyLimits(
        max_page_size_mb=safety_data.get("max_page_size_mb", 10.0),
        request_timeout_seconds=safety_data.get("request_timeout_seconds", 30.0),
        max_retries=safety_data.get("max_retries", 3),
        circuit_breaker_threshold=safety_data.get("circuit_breaker_threshold", 5),
        circuit_breaker_timeout=safety_data.get("circuit_breaker_timeout", 300.0),
    )
    
    security_data = data.get("security", {})
    security = SecurityConfig(
        verify_ssl=security_data.get("verify_ssl", True),
        block_private_ips=security_data.get("block_private_ips", True),
        max_redirects=security_data.get("max_redirects", 10),
        allowed_schemes=security_data.get("allowed_schemes", ["http", "https"]),
    )
    
    gdpr_data = data.get("gdpr", {})
    gdpr = GDPRConfig(
        enabled=gdpr_data.get("enabled", True),
        lawful_basis=LawfulBasis(gdpr_data.get("lawful_basis", "legitimate_interest")),
        legitimate_interest_assessment=gdpr_data.get("legitimate_interest_assessment", ""),
        collect_only=gdpr_data.get("collect_only", ["url", "title", "content", "metadata", "published_date"]),
        exclude_pii_patterns=gdpr_data.get("exclude_pii_patterns", True),
        retention_days=gdpr_data.get("retention_days", 365),
        retention_policy=gdpr_data.get("retention_policy", "delete"),
        eu_domains_only=gdpr_data.get("eu_domains_only", False),
        process_in_eu=gdpr_data.get("process_in_eu", True),
    )
    
    ccpa_data = data.get("ccpa", {})
    ccpa = CCPAConfig(
        enabled=ccpa_data.get("enabled", True),
        disclosure_categories=ccpa_data.get("disclosure_categories", ["identifiers", "internet_activity", "geolocation"]),
        deletion_verification=ccpa_data.get("deletion_verification", True),
        honor_gpc_header=ccpa_data.get("honor_gpc_header", True),
        do_not_sell=ccpa_data.get("do_not_sell", True),
    )
    
    pii_data = data.get("pii", {})
    pii = PIIHandlingConfig(
        action=PIIHandling(pii_data.get("action", "redact")),
        log_detections=pii_data.get("log_detections", True),
        alert_on_sensitive=pii_data.get("alert_on_sensitive", True),
        sensitive_categories=pii_data.get("sensitive_categories", ["health", "financial", "biometric", "racial_ethnic"]),
    )
    
    proxy_data = data.get("proxy", {})
    proxy = ProxyConfig(
        enabled=proxy_data.get("enabled", False),
        proxy_urls=proxy_data.get("proxy_urls", []),
        rotation_strategy=proxy_data.get("rotation_strategy", "round_robin"),
        respect_rate_limits_per_proxy=proxy_data.get("respect_rate_limits_per_proxy", True),
        aggregate_rate_limit=proxy_data.get("aggregate_rate_limit", True),
        max_aggregate_rps=proxy_data.get("max_aggregate_rps", 10.0),
        health_check_interval=proxy_data.get("health_check_interval", 300),
        failure_threshold=proxy_data.get("failure_threshold", 3),
    )
    
    politeness_data = data.get("politeness", {})
    off_peak = politeness_data.get("off_peak_hours", [1, 6])
    politeness = PolitenessConfig(
        prefer_off_peak=politeness_data.get("prefer_off_peak", True),
        off_peak_hours=tuple(off_peak) if isinstance(off_peak, list) else off_peak,
        off_peak_rate_multiplier=politeness_data.get("off_peak_rate_multiplier", 1.5),
        respect_server_timing=politeness_data.get("respect_server_timing", True),
        max_response_time_ms=politeness_data.get("max_response_time_ms", 5000),
        retry_respectful=politeness_data.get("retry_respectful", True),
        max_retry_delay=politeness_data.get("max_retry_delay", 3600.0),
    )
    
    alerts_data = data.get("alerts", {})
    alerts = AlertConfig(
        slack_webhook=alerts_data.get("slack_webhook", ""),
        email_to=alerts_data.get("email_to", ""),
        pagerduty_key=alerts_data.get("pagerduty_key", ""),
        webhook_url=alerts_data.get("webhook_url", ""),
        throttle_minutes=alerts_data.get("throttle_minutes", 60),
        aggregate=alerts_data.get("aggregate", True),
        min_severity=alerts_data.get("min_severity", "WARNING"),
    )
    
    # Structure store configuration
    structure_data = data.get("structure_store", {})
    llm_data = structure_data.get("llm", {})
    
    structure_store = StructureStoreConfig(
        store_type=StructureStoreType(structure_data.get("store_type", "basic")),
        enable_embeddings=structure_data.get("enable_embeddings", False),
        embedding_model=structure_data.get("embedding_model", "all-MiniLM-L6-v2"),
        ttl_seconds=structure_data.get("ttl_seconds", 604800),
        max_versions=structure_data.get("max_versions", 10),
        llm_provider=LLMProviderType(llm_data.get("provider", "anthropic")),
        llm_model=llm_data.get("model", ""),
        llm_api_key=llm_data.get("api_key", ""),
        ollama_base_url=llm_data.get("ollama_base_url", "http://localhost:11434"),
    )
    
    # Build and return CrawlConfig
    return CrawlConfig(
        seed_urls=crawl["seed_urls"],
        output_dir=crawl["output_dir"],
        max_depth=crawl.get("max_depth", 10),
        max_pages=crawl.get("max_pages"),
        allowed_domains=crawl.get("allowed_domains", []),
        exclude_patterns=crawl.get("exclude_patterns", []),
        rate_limit=rate_limit,
        safety=safety,
        security=security,
        gdpr=gdpr,
        ccpa=ccpa,
        pii=pii,
        proxy=proxy,
        politeness=politeness,
        alerts=alerts,
        structure_store=structure_store,
    )


def save_yaml_config(config: CrawlConfig, config_path: str) -> None:
    """
    Save a CrawlConfig to a YAML file.
    
    Args:
        config: CrawlConfig instance to save.
        config_path: Path to write the YAML file.
    """
    import yaml
    from pathlib import Path
    
    data = {
        "crawl": {
            "seed_urls": config.seed_urls,
            "output_dir": config.output_dir,
            "max_depth": config.max_depth,
            "max_pages": config.max_pages,
            "allowed_domains": config.allowed_domains,
            "exclude_patterns": config.exclude_patterns,
        },
        "rate_limit": {
            "default_delay": config.rate_limit.default_delay,
            "min_delay": config.rate_limit.min_delay,
            "max_delay": config.rate_limit.max_delay,
            "respect_crawl_delay": config.rate_limit.respect_crawl_delay,
            "adaptive": config.rate_limit.adaptive,
            "backoff_multiplier": config.rate_limit.backoff_multiplier,
            "max_concurrent_per_domain": config.rate_limit.max_concurrent_per_domain,
            "max_concurrent_global": config.rate_limit.max_concurrent_global,
        },
        "safety": {
            "max_page_size_mb": config.safety.max_page_size_mb,
            "request_timeout_seconds": config.safety.request_timeout_seconds,
            "max_retries": config.safety.max_retries,
            "circuit_breaker_threshold": config.safety.circuit_breaker_threshold,
            "circuit_breaker_timeout": config.safety.circuit_breaker_timeout,
        },
        "security": {
            "verify_ssl": config.security.verify_ssl,
            "block_private_ips": config.security.block_private_ips,
            "max_redirects": config.security.max_redirects,
            "allowed_schemes": config.security.allowed_schemes,
        },
        "structure_store": {
            "store_type": config.structure_store.store_type.value,
            "enable_embeddings": config.structure_store.enable_embeddings,
            "embedding_model": config.structure_store.embedding_model,
            "ttl_seconds": config.structure_store.ttl_seconds,
            "max_versions": config.structure_store.max_versions,
            "llm": {
                "provider": config.structure_store.llm_provider.value,
                "model": config.structure_store.llm_model,
                "api_key": "",  # Don't save API keys
                "ollama_base_url": config.structure_store.ollama_base_url,
            },
        },
        "gdpr": {
            "enabled": config.gdpr.enabled,
            "lawful_basis": config.gdpr.lawful_basis.value,
            "retention_days": config.gdpr.retention_days,
        },
        "ccpa": {
            "enabled": config.ccpa.enabled,
            "honor_gpc_header": config.ccpa.honor_gpc_header,
        },
        "pii": {
            "action": config.pii.action.value,
            "log_detections": config.pii.log_detections,
        },
    }
    
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
