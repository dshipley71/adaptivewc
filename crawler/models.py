"""
Core data models for the adaptive web crawler.

These models are used throughout the codebase for type safety and serialization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# =============================================================================
# Enums
# =============================================================================


class ChangeType(str, Enum):
    """Types of structural changes that can be detected."""

    # Iframe changes
    IFRAME_ADDED = "iframe_added"
    IFRAME_REMOVED = "iframe_removed"
    IFRAME_RELOCATED = "iframe_relocated"
    IFRAME_RESIZED = "iframe_resized"

    # Tag/class changes
    TAG_RENAMED = "tag_renamed"
    CLASS_RENAMED = "class_renamed"
    ID_CHANGED = "id_changed"

    # Structure changes
    STRUCTURE_REORGANIZED = "structure_reorganized"
    CONTENT_RELOCATED = "content_relocated"

    # Script changes
    SCRIPT_ADDED = "script_added"
    SCRIPT_REMOVED = "script_removed"
    SCRIPT_MODIFIED = "script_modified"

    # URL/navigation changes
    URL_PATTERN_CHANGED = "url_pattern_changed"
    PAGINATION_CHANGED = "pagination_changed"

    # Overall changes
    MINOR_LAYOUT_SHIFT = "minor_layout_shift"
    MAJOR_REDESIGN = "major_redesign"


class Severity(str, Enum):
    """Alert severity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class FetchStatus(str, Enum):
    """Status of a fetch operation."""

    SUCCESS = "success"
    BLOCKED_ROBOTS = "blocked_robots"
    BLOCKED_RATE_LIMIT = "blocked_rate_limit"
    BLOCKED_BOT_DETECTION = "blocked_bot_detection"
    BLOCKED_LEGAL = "blocked_legal"
    ERROR_TIMEOUT = "error_timeout"
    ERROR_CONNECTION = "error_connection"
    ERROR_HTTP = "error_http"
    ERROR_CONTENT_TOO_LARGE = "error_content_too_large"


# =============================================================================
# Fetch Models
# =============================================================================


@dataclass
class FetchResult:
    """Result of a URL fetch operation."""

    url: str
    status: FetchStatus
    status_code: int | None = None
    content: bytes | None = None
    html: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    duration_ms: float = 0.0
    redirect_chain: list[str] = field(default_factory=list)
    error_message: str | None = None
    fetched_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def success(
        cls,
        url: str,
        content: bytes,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        duration_ms: float = 0.0,
    ) -> "FetchResult":
        """Create a successful fetch result."""
        html = content.decode("utf-8", errors="replace")
        return cls(
            url=url,
            status=FetchStatus.SUCCESS,
            status_code=status_code,
            content=content,
            html=html,
            headers=headers or {},
            duration_ms=duration_ms,
        )

    @classmethod
    def blocked(cls, url: str, reason: str, status: FetchStatus | None = None) -> "FetchResult":
        """Create a blocked fetch result."""
        if status is None:
            status = FetchStatus.BLOCKED_ROBOTS
        return cls(
            url=url,
            status=status,
            error_message=reason,
        )

    @classmethod
    def error(cls, url: str, message: str, status: FetchStatus) -> "FetchResult":
        """Create an error fetch result."""
        return cls(
            url=url,
            status=status,
            error_message=message,
        )

    def is_success(self) -> bool:
        """Check if fetch was successful."""
        return self.status == FetchStatus.SUCCESS


# =============================================================================
# Structure Models
# =============================================================================


@dataclass
class IframeInfo:
    """Information about an iframe in a page."""

    selector: str
    src_pattern: str
    position: str  # "header", "content", "sidebar", "footer"
    dimensions: tuple[int, int] | None = None
    is_dynamic: bool = False


@dataclass
class ContentRegion:
    """Identified content extraction zone."""

    name: str
    primary_selector: str
    fallback_selectors: list[str] = field(default_factory=list)
    content_type: str = "text"  # "text", "html", "structured"
    confidence: float = 0.0


@dataclass
class PaginationInfo:
    """Information about pagination on a page."""

    next_selector: str | None = None
    prev_selector: str | None = None
    page_number_selector: str | None = None
    pattern: str | None = None  # URL pattern like "?page={n}"


@dataclass
class PageStructure:
    """
    Fingerprint of a page's DOM structure.

    Supports variant tracking: multiple structural variants can exist
    for the same domain/page_type combination (e.g., video articles
    vs text articles on a news site).
    """

    domain: str
    page_type: str
    url_pattern: str

    tag_hierarchy: dict[str, Any] = field(default_factory=dict)
    iframe_locations: list[IframeInfo] = field(default_factory=list)
    script_signatures: list[str] = field(default_factory=list)
    css_class_map: dict[str, int] = field(default_factory=dict)
    id_attributes: set[str] = field(default_factory=set)
    semantic_landmarks: dict[str, str] = field(default_factory=dict)

    content_regions: list[ContentRegion] = field(default_factory=list)
    navigation_selectors: list[str] = field(default_factory=list)
    pagination_pattern: PaginationInfo | None = None

    captured_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    content_hash: str = ""

    # Variant tracking: identifies structural variants within the same page_type
    variant_id: str = "default"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "domain": self.domain,
            "page_type": self.page_type,
            "url_pattern": self.url_pattern,
            "tag_hierarchy": self.tag_hierarchy,
            "iframe_locations": [
                {
                    "selector": i.selector,
                    "src_pattern": i.src_pattern,
                    "position": i.position,
                    "dimensions": i.dimensions,
                    "is_dynamic": i.is_dynamic,
                }
                for i in self.iframe_locations
            ],
            "script_signatures": self.script_signatures,
            "css_class_map": self.css_class_map,
            "id_attributes": list(self.id_attributes),
            "semantic_landmarks": self.semantic_landmarks,
            "content_regions": [
                {
                    "name": r.name,
                    "primary_selector": r.primary_selector,
                    "fallback_selectors": r.fallback_selectors,
                    "content_type": r.content_type,
                    "confidence": r.confidence,
                }
                for r in self.content_regions
            ],
            "navigation_selectors": self.navigation_selectors,
            "captured_at": self.captured_at.isoformat(),
            "version": self.version,
            "content_hash": self.content_hash,
            "variant_id": self.variant_id,
        }


# =============================================================================
# Extraction Strategy Models
# =============================================================================


@dataclass
class SelectorRule:
    """Single extraction rule with fallbacks."""

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

    title: SelectorRule | None = None
    content: SelectorRule | None = None
    metadata: dict[str, SelectorRule] = field(default_factory=dict)
    images: SelectorRule | None = None
    links: SelectorRule | None = None

    wait_for_selectors: list[str] = field(default_factory=list)
    iframe_extraction: dict[str, SelectorRule] = field(default_factory=dict)

    required_fields: list[str] = field(default_factory=lambda: ["title", "content"])
    min_content_length: int = 100

    learned_at: datetime = field(default_factory=datetime.utcnow)
    learning_source: str = "initial"  # "initial", "adaptation", "manual"
    confidence_scores: dict[str, float] = field(default_factory=dict)

    # Variant tracking: identifies structural variants within the same page_type
    variant_id: str = "default"


# =============================================================================
# Change Detection Models
# =============================================================================


@dataclass
class StructureChange:
    """Documented change between structure versions."""

    domain: str
    page_type: str
    detected_at: datetime

    previous_version: int
    new_version: int

    change_type: ChangeType
    affected_components: list[str] = field(default_factory=list)

    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    breaking: bool = False
    fields_affected: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class StructureChanges:
    """Collection of changes between two structures."""

    changes: list[StructureChange] = field(default_factory=list)
    similarity_score: float = 1.0
    has_breaking_changes: bool = False

    def add(self, change: StructureChange) -> None:
        """Add a change to the collection."""
        self.changes.append(change)
        if change.breaking:
            self.has_breaking_changes = True


# =============================================================================
# Extraction Result Models
# =============================================================================


@dataclass
class ExtractedContent:
    """Content extracted from a page."""

    url: str
    title: str | None = None
    content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    images: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)

    extracted_at: datetime = field(default_factory=datetime.utcnow)
    strategy_version: int = 1
    confidence: float = 0.0


@dataclass
class ExtractionResult:
    """Result of content extraction."""

    url: str
    success: bool
    content: ExtractedContent | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    strategy_used: str | None = None
    duration_ms: float = 0.0

    def is_valid(self) -> bool:
        """Check if extraction was successful and valid."""
        if not self.success or self.content is None:
            return False
        return bool(self.content.title and self.content.content)


# =============================================================================
# Authorization Models
# =============================================================================


@dataclass
class AuthorizationResult:
    """Result of CFAA authorization check."""

    authorized: bool
    basis: str  # "robots_txt", "tos_permitted", "public_access"
    restrictions: list[str] = field(default_factory=list)
    documentation: str = ""
    logged_at: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# URL Pattern Models
# =============================================================================


@dataclass
class URLPattern:
    """Tracked URL pattern for a domain."""

    pattern: str
    example_urls: list[str] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    match_count: int = 0
    page_type: str | None = None


@dataclass
class PatternChange:
    """Detected URL pattern change."""

    domain: str
    change_type: str  # "new_pattern", "deprecated_pattern", "redirect_detected"
    old_pattern: str | None = None
    new_pattern: str | None = None
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


# =============================================================================
# JS Detection Models
# =============================================================================


@dataclass
class JSDetectionResult:
    """Result of JavaScript rendering detection."""

    needs_js: bool
    confidence: float
    framework: str | None = None
    signals: list[str] = field(default_factory=list)
    recommendation: str = "static"  # "static", "js_required", "try_static_first"
