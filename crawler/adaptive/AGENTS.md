# AGENTS.md - Adaptive Extraction Subsystem

This module handles dynamic website structure learning, change detection, and extraction strategy adaptation.

## Overview

Websites evolve constantly: iframes move, CSS classes change, JavaScript rewrites DOM structures, and URL patterns shift. This subsystem enables the crawler to detect these changes, adapt extraction strategies automatically, and maintain a persistent memory of site structures in Redis.

## Core Principles

1. **Learn Once, Extract Many**: Capture site structure on first visit, reuse until changes detected
2. **Graceful Degradation**: When structure changes, attempt adaptation before failing
3. **Explainable Changes**: Every adaptation includes a documented reason
4. **Compliance Preserved**: Adaptive behavior never bypasses rate limits or robots.txt

## Architecture

```
adaptive/
├── structure_analyzer.py    # DOM fingerprinting and structure capture
├── change_detector.py       # Diff engine for structure comparison
├── extraction_strategy.py   # Rule-based extraction engine
├── strategy_learner.py      # Infers selectors from page structure
├── change_logger.py         # Documents why changes occurred
├── js_detector.py           # Detects when JS rendering is needed
├── url_pattern_tracker.py   # Tracks URL scheme changes
└── models.py                # Data models for structures and strategies
```

## JavaScript Rendering Detection

Not every page needs Playwright. The crawler intelligently detects when JS rendering is required:

```python
class JSRenderingDetector:
    """
    Determines if a page requires JavaScript execution for content extraction.
    
    Detection signals (weighted scoring):
    - Empty content containers: <div id="app"></div>, <div id="root"></div>
    - Framework signatures: __NEXT_DATA__, window.__INITIAL_STATE__
    - Script-heavy pages: >50% of page is <script> tags
    - Hydration markers: data-reactroot, ng-app, v-cloak
    - Historical: Previous visits needed JS
    
    Decision thresholds:
    - Score > 0.8: Always render with JS
    - Score 0.4-0.8: Try static first, fall back to JS
    - Score < 0.4: Static extraction only
    """
    
    FRAMEWORK_SIGNATURES = {
        "react": ["data-reactroot", "__REACT", "react-root"],
        "nextjs": ["__NEXT_DATA__", "_next"],
        "vue": ["v-cloak", "__VUE__", "data-v-"],
        "angular": ["ng-app", "ng-version", "_nghost"],
        "svelte": ["svelte-", "__svelte"],
    }
    
    def detect(self, html: str, url: str) -> JSDetectionResult:
        """Analyze page to determine JS rendering need."""
    
    def score_js_likelihood(self, html: str) -> float:
        """Return 0-1 score for JS rendering necessity."""
    
    def detect_framework(self, html: str) -> str | None:
        """Identify frontend framework if present."""

@dataclass
class JSDetectionResult:
    needs_js: bool
    confidence: float
    framework: str | None
    signals: list[str]           # What triggered detection
    recommendation: str          # "static", "js_required", "try_static_first"
```

**Integration with fetch pipeline:**

```python
async def smart_fetch(self, url: str) -> FetchResult:
    # 1. Static fetch first (always, for JS detection)
    static_response = await self.fetcher.get(url)
    
    # 2. Check if JS rendering needed
    js_result = self.js_detector.detect(static_response.html, url)
    
    if js_result.recommendation == "static":
        return static_response
    
    if js_result.recommendation == "try_static_first":
        # Attempt static extraction
        extraction = self.extract(static_response.html)
        if extraction.is_valid():
            return static_response
        # Fall through to JS rendering
    
    # 3. JS rendering required - rate limit applies
    await self.rate_limiter.acquire(get_domain(url))
    return await self.fetcher.get_with_js(url)
```

## URL Pattern Tracking

Sites change URL schemes. The crawler detects and adapts:

```python
class URLPatternTracker:
    """
    Tracks URL patterns per domain and detects scheme changes.
    
    Patterns tracked:
    - Path structure: /blog/{slug} vs /articles/{year}/{slug}
    - Query parameters: ?id=123 vs ?article=123
    - Fragment usage: #section vs clean URLs
    
    Change detection:
    - New pattern emerges (>10 URLs match)
    - Old pattern stops appearing
    - Redirect patterns (old → new)
    """
    
    @dataclass
    class URLPattern:
        pattern: str                    # Regex pattern
        example_urls: list[str]         # Sample matches
        first_seen: datetime
        last_seen: datetime
        match_count: int
        page_type: str | None           # Associated page type
    
    def learn_pattern(self, url: str, page_type: str) -> None:
        """Update pattern knowledge from observed URL."""
    
    def detect_pattern_change(self, domain: str) -> list[PatternChange]:
        """Identify URL scheme migrations."""
    
    def match_to_page_type(self, url: str) -> str | None:
        """Predict page type from URL pattern."""

@dataclass
class PatternChange:
    domain: str
    change_type: str              # "new_pattern", "deprecated_pattern", "redirect_detected"
    old_pattern: str | None
    new_pattern: str | None
    confidence: float
    evidence: dict                # Sample URLs, redirect chains
    reason: str                   # Human-readable explanation
```

**Example pattern change:**

```json
{
    "domain": "example.com",
    "change_type": "redirect_detected",
    "old_pattern": "^/blog/([\\w-]+)/?$",
    "new_pattern": "^/articles/(\\d{4})/(\\d{2})/([\\w-]+)/?$",
    "confidence": 0.92,
    "evidence": {
        "redirects_observed": [
            {"/blog/hello-world": "/articles/2025/01/hello-world"},
            {"/blog/another-post": "/articles/2024/12/another-post"}
        ],
        "old_pattern_404_rate": 0.85,
        "new_pattern_success_rate": 0.98
    },
    "reason": "Blog migrated to date-prefixed URL scheme. 301 redirects in place from old URLs."
}
```

## Data Models

### PageStructure (Stored in Redis)

```python
@dataclass
class PageStructure:
    """Fingerprint of a page's DOM structure."""
    
    # Identification
    domain: str
    page_type: str                    # e.g., "article", "listing", "product"
    url_pattern: str                  # Regex pattern for matching URLs
    
    # Structure fingerprint
    tag_hierarchy: dict[str, list]    # Nested tag structure
    iframe_locations: list[IframeInfo]
    script_signatures: list[str]      # Hashes of inline/external scripts
    css_class_map: dict[str, int]     # Class name → occurrence count
    id_attributes: set[str]           # All id attributes found
    semantic_landmarks: dict[str, str] # ARIA landmarks and roles
    
    # Content regions
    content_regions: list[ContentRegion]
    navigation_selectors: list[str]
    pagination_pattern: PaginationInfo | None
    
    # Metadata
    captured_at: datetime
    version: int
    content_hash: str                 # Hash of structure for quick comparison

@dataclass
class IframeInfo:
    """Tracks iframe characteristics for change detection."""
    selector: str                     # CSS selector to locate
    src_pattern: str                  # URL pattern of iframe src
    position: str                     # "header", "content", "sidebar", "footer"
    dimensions: tuple[int, int] | None
    is_dynamic: bool                  # Loaded via JavaScript?

@dataclass
class ContentRegion:
    """Identified content extraction zone."""
    name: str                         # e.g., "main_content", "article_body"
    primary_selector: str             # Best selector
    fallback_selectors: list[str]     # Alternatives if primary fails
    content_type: str                 # "text", "html", "structured"
    confidence: float                 # 0-1, how reliable is this selector
```

### ExtractionStrategy (Stored in Redis)

```python
@dataclass
class ExtractionStrategy:
    """Rules for extracting content from a page type."""
    
    domain: str
    page_type: str
    version: int
    
    # Extraction rules
    title: SelectorRule
    content: SelectorRule
    metadata: dict[str, SelectorRule]  # author, date, tags, etc.
    images: SelectorRule | None
    links: SelectorRule
    
    # Dynamic content handling
    wait_for_selectors: list[str]     # Elements to wait for (JS rendering)
    iframe_extraction: dict[str, SelectorRule]  # iframe name → rules
    
    # Validation
    required_fields: list[str]
    min_content_length: int
    
    # Learning metadata
    learned_at: datetime
    learning_source: str              # "initial", "adaptation", "manual"
    confidence_scores: dict[str, float]

@dataclass
class SelectorRule:
    """Single extraction rule with fallbacks."""
    primary: str                      # CSS selector
    fallbacks: list[str]              # Ordered fallback selectors
    extraction_method: str            # "text", "html", "attribute"
    attribute_name: str | None        # If extraction_method == "attribute"
    post_processors: list[str]        # e.g., ["strip", "normalize_whitespace"]
```

### StructureChange (Stored in Redis)

```python
@dataclass
class StructureChange:
    """Documented change between structure versions."""
    
    domain: str
    page_type: str
    detected_at: datetime
    
    # Version tracking
    previous_version: int
    new_version: int
    
    # Change details
    change_type: ChangeType
    affected_components: list[str]
    
    # Reason documentation
    reason: str                       # Human-readable explanation
    evidence: dict                    # Supporting data for the change
    
    # Impact assessment
    breaking: bool                    # Did extraction strategy need update?
    fields_affected: list[str]
    confidence: float

class ChangeType(Enum):
    IFRAME_ADDED = "iframe_added"
    IFRAME_REMOVED = "iframe_removed"
    IFRAME_RELOCATED = "iframe_relocated"
    IFRAME_RESIZED = "iframe_resized"
    
    TAG_RENAMED = "tag_renamed"           # e.g., div.article → article
    CLASS_RENAMED = "class_renamed"       # e.g., .post-content → .article-body
    ID_CHANGED = "id_changed"
    
    STRUCTURE_REORGANIZED = "structure_reorganized"
    CONTENT_RELOCATED = "content_relocated"
    
    SCRIPT_ADDED = "script_added"
    SCRIPT_REMOVED = "script_removed"
    SCRIPT_MODIFIED = "script_modified"
    
    URL_PATTERN_CHANGED = "url_pattern_changed"
    PAGINATION_CHANGED = "pagination_changed"
    
    MINOR_LAYOUT_SHIFT = "minor_layout_shift"
    MAJOR_REDESIGN = "major_redesign"
```

## Redis Storage Schema

### Key Patterns

```
# Page structure
structure:{domain}:{page_type}              → JSON(PageStructure)
structure:{domain}:{page_type}:history:{v}  → JSON(PageStructure)  # Version history

# Extraction strategy
strategy:{domain}:{page_type}               → JSON(ExtractionStrategy)
strategy:{domain}:{page_type}:history:{v}   → JSON(ExtractionStrategy)

# Change log
changes:{domain}:{page_type}                → LIST[JSON(StructureChange)]
changes:{domain}:{page_type}:{timestamp}    → JSON(StructureChange)

# Quick lookups
domains:active                              → SET[domain]
page_types:{domain}                         → SET[page_type]
```

### Example Redis Data

```json
// Key: structure:example.com:article
{
    "domain": "example.com",
    "page_type": "article",
    "url_pattern": "^/blog/[\\w-]+/?$",
    "tag_hierarchy": {
        "html": {
            "body": {
                "div.container": {
                    "article.post": {
                        "h1.title": {},
                        "div.content": {},
                        "aside.sidebar": {}
                    }
                }
            }
        }
    },
    "iframe_locations": [
        {
            "selector": "div.content iframe.video-embed",
            "src_pattern": "https://youtube.com/embed/*",
            "position": "content",
            "dimensions": [560, 315],
            "is_dynamic": false
        }
    ],
    "css_class_map": {
        "post": 1,
        "title": 1,
        "content": 1,
        "sidebar": 1,
        "video-embed": 1
    },
    "captured_at": "2025-01-15T10:30:00Z",
    "version": 3,
    "content_hash": "a1b2c3d4e5f6"
}
```

```json
// Key: changes:example.com:article (LIST - most recent first)
[
    {
        "domain": "example.com",
        "page_type": "article",
        "detected_at": "2025-01-20T08:15:00Z",
        "previous_version": 2,
        "new_version": 3,
        "change_type": "iframe_relocated",
        "affected_components": ["video-embed"],
        "reason": "Video iframe moved from sidebar (aside.sidebar iframe) to main content area (div.content iframe). Likely due to redesign prioritizing video content visibility.",
        "evidence": {
            "old_selector": "aside.sidebar iframe.video-embed",
            "new_selector": "div.content iframe.video-embed",
            "old_position": "sidebar",
            "new_position": "content"
        },
        "breaking": false,
        "fields_affected": ["embedded_video"],
        "confidence": 0.95
    },
    {
        "domain": "example.com",
        "page_type": "article",
        "detected_at": "2025-01-10T14:22:00Z",
        "previous_version": 1,
        "new_version": 2,
        "change_type": "class_renamed",
        "affected_components": ["main_content"],
        "reason": "Content container class changed from 'post-body' to 'content'. This appears to be part of a CSS refactoring effort (multiple classes follow new naming convention).",
        "evidence": {
            "old_class": "post-body",
            "new_class": "content",
            "selector_updated": "div.post-body → div.content",
            "other_renames_detected": ["post-title → title", "post-meta → meta"]
        },
        "breaking": true,
        "fields_affected": ["content", "title"],
        "confidence": 0.92
    }
]
```

## Key Components

### StructureAnalyzer

```python
class StructureAnalyzer:
    """
    Analyzes HTML to create a PageStructure fingerprint.
    
    Handles:
    - DOM tree traversal and hierarchy mapping
    - Iframe detection (static and dynamically loaded)
    - Script fingerprinting
    - Semantic landmark identification
    - Content region boundary detection
    """
    
    def analyze(self, html: str, url: str) -> PageStructure:
        """Create structure fingerprint from HTML."""
    
    def analyze_with_js(self, url: str) -> PageStructure:
        """Use Playwright to capture post-JavaScript DOM."""
    
    def detect_iframes(self, soup: BeautifulSoup) -> list[IframeInfo]:
        """Find and characterize all iframes."""
    
    def map_tag_hierarchy(self, soup: BeautifulSoup, max_depth: int = 10) -> dict:
        """Build nested tag structure map."""
    
    def identify_content_regions(self, soup: BeautifulSoup) -> list[ContentRegion]:
        """Heuristically identify main content, sidebar, nav, etc."""
```

### ChangeDetector

```python
class ChangeDetector:
    """
    Compares two PageStructure instances to identify changes.
    
    Change detection strategy:
    1. Compare content hashes (fast path for no change)
    2. Diff tag hierarchies for structural changes
    3. Compare iframe lists for additions/removals/relocations
    4. Diff CSS class maps for renames
    5. Compare script signatures for JS changes
    """
    
    def diff(
        self, 
        previous: PageStructure, 
        current: PageStructure
    ) -> StructureChanges:
        """Compute all changes between two structures."""
    
    def has_breaking_changes(self, changes: StructureChanges) -> bool:
        """Determine if changes require strategy re-learning."""
    
    def compute_similarity(
        self, 
        previous: PageStructure, 
        current: PageStructure
    ) -> float:
        """Return 0-1 similarity score."""
    
    # Specialized detectors
    def detect_iframe_changes(self, prev: list[IframeInfo], curr: list[IframeInfo]) -> list[StructureChange]:
        """Detailed iframe change analysis."""
    
    def detect_class_renames(self, prev: dict, curr: dict) -> list[StructureChange]:
        """Identify renamed CSS classes via similarity matching."""
    
    def detect_relocated_content(self, prev: PageStructure, curr: PageStructure) -> list[StructureChange]:
        """Find content that moved to different DOM location."""
```

### StrategyLearner

```python
class StrategyLearner:
    """
    Infers extraction selectors from page structure.
    
    Learning approaches (in order of preference):
    1. Semantic HTML: <article>, <main>, <h1>, etc.
    2. ARIA landmarks: role="main", aria-label
    3. Schema.org/microdata: itemtype, itemprop
    4. Common class patterns: .content, .article, .post
    5. Structural heuristics: largest text block, heading proximity
    6. ML-based selector inference (if training data available)
    """
    
    async def infer(
        self, 
        html: str, 
        previous_strategy: ExtractionStrategy | None = None
    ) -> ExtractionStrategy:
        """Learn extraction strategy for page."""
    
    def infer_title_selector(self, soup: BeautifulSoup) -> SelectorRule:
        """Find best selector for page title."""
    
    def infer_content_selector(self, soup: BeautifulSoup) -> SelectorRule:
        """Find best selector for main content."""
    
    def adapt_selector(
        self, 
        old_selector: str, 
        old_structure: PageStructure,
        new_structure: PageStructure
    ) -> str | None:
        """Attempt to map old selector to new structure."""
    
    def generate_fallbacks(self, primary: str, soup: BeautifulSoup) -> list[str]:
        """Create fallback selectors for robustness."""
```

### ChangeLogger

```python
class ChangeLogger:
    """
    Documents why structure changes occurred.
    
    Generates human-readable explanations by analyzing:
    - Nature of the change (what moved/renamed/added)
    - Patterns in the changes (bulk rename suggests refactoring)
    - Historical changes (is this site frequently changing?)
    - Common redesign signatures
    """
    
    def document(
        self,
        changes: StructureChanges,
        previous: PageStructure,
        current: PageStructure
    ) -> str:
        """Generate human-readable change explanation."""
    
    def categorize_change(self, change: StructureChange) -> str:
        """Classify change as maintenance, redesign, A/B test, etc."""
    
    def assess_impact(self, changes: StructureChanges) -> ImpactAssessment:
        """Determine extraction impact and recovery options."""
```

## Change Detection Examples

### Example 1: Iframe Relocation

```python
# Previous structure
previous_iframe = IframeInfo(
    selector="aside.sidebar iframe.video-embed",
    src_pattern="https://youtube.com/embed/*",
    position="sidebar",
    dimensions=(300, 169),
    is_dynamic=False
)

# Current structure (after site update)
current_iframe = IframeInfo(
    selector="div.content iframe.video-embed",
    src_pattern="https://youtube.com/embed/*",
    position="content",
    dimensions=(560, 315),
    is_dynamic=False
)

# Change detection
change = StructureChange(
    change_type=ChangeType.IFRAME_RELOCATED,
    affected_components=["video-embed"],
    reason="Video iframe moved from sidebar to main content area. "
           "Dimensions increased from 300x169 to 560x315, suggesting "
           "redesign to feature video content more prominently.",
    evidence={
        "old_selector": "aside.sidebar iframe.video-embed",
        "new_selector": "div.content iframe.video-embed",
        "old_position": "sidebar",
        "new_position": "content",
        "dimension_change": "300x169 → 560x315"
    },
    breaking=False,  # Same class name, strategy auto-adapts
    fields_affected=["embedded_video"],
    confidence=0.95
)
```

### Example 2: Class Rename Detection

```python
# Previous CSS class map
previous_classes = {
    "post-title": 5,
    "post-body": 5,
    "post-meta": 5,
    "post-author": 5
}

# Current CSS class map (after refactoring)
current_classes = {
    "title": 5,
    "content": 5,
    "meta": 5,
    "author": 5
}

# Detector identifies pattern: "post-" prefix removed
changes = [
    StructureChange(
        change_type=ChangeType.CLASS_RENAMED,
        reason="Bulk class rename detected: 'post-' prefix removed from 4 classes. "
               "This appears to be a CSS refactoring effort following BEM or "
               "simplified naming conventions.",
        evidence={
            "renames": {
                "post-title": "title",
                "post-body": "content", 
                "post-meta": "meta",
                "post-author": "author"
            },
            "pattern": "prefix_removal:post-"
        },
        breaking=True,  # Selectors need updating
        fields_affected=["title", "content", "metadata"]
    )
]

# Strategy learner auto-updates selectors
new_strategy.title.primary = "h1.title"  # was "h1.post-title"
new_strategy.content.primary = "div.content"  # was "div.post-body"
```

### Example 3: JavaScript-Loaded Content

```python
# Static HTML analysis shows empty container
static_structure = PageStructure(
    content_regions=[
        ContentRegion(
            name="main_content",
            primary_selector="div#app",
            content_type="empty",  # No content in static HTML
            confidence=0.3
        )
    ]
)

# Playwright analysis after JS execution
js_structure = PageStructure(
    content_regions=[
        ContentRegion(
            name="main_content",
            primary_selector="div#app article.post",
            content_type="structured",
            confidence=0.9
        )
    ],
    script_signatures=["react-bundle-abc123", "hydrate-xyz789"]
)

# Strategy includes JS wait conditions
strategy = ExtractionStrategy(
    wait_for_selectors=["div#app article.post"],
    title=SelectorRule(
        primary="div#app article.post h1",
        extraction_method="text"
    )
)

# Change logged
change = StructureChange(
    change_type=ChangeType.SCRIPT_ADDED,
    reason="Site migrated to client-side rendering (React detected). "
           "Content now requires JavaScript execution. Static HTML contains "
           "only empty container div#app.",
    evidence={
        "framework": "React",
        "hydration_detected": True,
        "static_content_length": 0,
        "js_content_length": 15420
    }
)
```

## Integration with Compliance

The adaptive extraction system respects all compliance constraints:

```python
async def extract_with_compliance(
    self,
    url: str,
    html: str,
    rate_limiter: RateLimiter,
    robots_checker: RobotsChecker
) -> ExtractionResult:
    """
    Extract content while maintaining compliance.
    
    CRITICAL: Adaptive behavior never bypasses rate limits.
    If JS rendering is needed, it still goes through the rate limiter.
    """
    domain = get_domain(url)
    
    # Check if we need JS rendering
    static_structure = self.analyzer.analyze(html, url)
    
    if static_structure.needs_js_rendering():
        # Rate limit the Playwright request
        await rate_limiter.acquire(domain)
        
        # Verify still allowed (robots.txt may have changed)
        if not await robots_checker.is_allowed(url, self.user_agent):
            raise RobotsBlockedError(url)
        
        # Now render with JS
        js_structure = await self.analyzer.analyze_with_js(url)
        return self._extract_from_structure(js_structure)
    
    return self._extract_from_structure(static_structure)
```

## Testing Requirements

### Unit Tests

```bash
# Structure analysis
pytest tests/unit/adaptive/test_structure_analyzer.py -v

# Change detection
pytest tests/unit/adaptive/test_change_detector.py -v

# Strategy learning
pytest tests/unit/adaptive/test_strategy_learner.py -v
```

### Required Test Cases

| Component | Test Case | Coverage |
|-----------|-----------|----------|
| StructureAnalyzer | Parse simple HTML | ✓ |
| StructureAnalyzer | Detect nested iframes | ✓ |
| StructureAnalyzer | Handle malformed HTML | ✓ |
| ChangeDetector | Detect iframe relocation | ✓ |
| ChangeDetector | Detect class rename patterns | ✓ |
| ChangeDetector | Compute similarity scores | ✓ |
| StrategyLearner | Infer from semantic HTML | ✓ |
| StrategyLearner | Adapt renamed selectors | ✓ |
| StrategyLearner | Generate robust fallbacks | ✓ |
| ChangeLogger | Document single change | ✓ |
| ChangeLogger | Document bulk changes | ✓ |
| Integration | Redis round-trip | ✓ |
| Integration | Full extract→change→adapt cycle | ✓ |

### Test Fixtures

```python
# tests/fixtures/structures/
article_v1.html          # Original article structure
article_v2_iframe.html   # Iframe relocated
article_v3_classes.html  # Classes renamed
article_v4_spa.html      # Migrated to SPA

# tests/fixtures/expected/
article_v1_structure.json
article_v1_to_v2_changes.json
article_v2_to_v3_changes.json
```

## Common Tasks

### Adding a New Change Type

1. Add enum value to `ChangeType` in `models.py`
2. Implement detection in `ChangeDetector.detect_*` methods
3. Add reason templates to `ChangeLogger`
4. Add test fixtures and test cases
5. Update metrics labels

### Debugging Extraction Failures

```python
# Enable detailed logging
import logging
logging.getLogger("crawler.adaptive").setLevel(logging.DEBUG)

# Inspect stored structure
redis_client = Redis.from_url(REDIS_URL)
structure = json.loads(redis_client.get("structure:example.com:article"))
print(json.dumps(structure, indent=2))

# View change history
changes = redis_client.lrange("changes:example.com:article", 0, -1)
for change in changes:
    print(json.loads(change))

# Force re-learning
await structure_store.delete("example.com", "article")
# Next crawl will learn fresh
```

### Manual Strategy Override

```python
# When automatic learning fails, provide manual rules
manual_strategy = ExtractionStrategy(
    domain="difficult-site.com",
    page_type="product",
    title=SelectorRule(
        primary="span[data-test='product-name']",
        fallbacks=["h1", ".product-title"],
        extraction_method="text"
    ),
    learning_source="manual"
)

await structure_store.save_strategy(manual_strategy)
```

## ML Models for Strategy Learning

The `StrategyLearner` uses an ensemble approach optimized for low-latency inference without GPU requirements.

### Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     StrategyLearner                              │
├─────────────────────────────────────────────────────────────────┤
│  1. Rule-Based Engine (fastest, highest priority)               │
│     - Semantic HTML: <article>, <main>, <h1>                    │
│     - Schema.org: itemtype, itemprop attributes                 │
│     - ARIA landmarks: role="main", aria-label                   │
│                                                                  │
│  2. Tree-Based Classifier (primary ML model)                    │
│     - Model: LightGBM or XGBoost                                │
│     - Task: Score candidate selectors for each field            │
│     - Latency: <1ms per candidate                               │
│                                                                  │
│  3. Semantic Similarity (for selector adaptation)               │
│     - Model: all-MiniLM-L6-v2 (sentence-transformers)           │
│     - Task: Match old selectors to new DOM when classes rename  │
│     - Latency: ~5ms per comparison batch                        │
└─────────────────────────────────────────────────────────────────┘
```

### 1. Tree-Based Selector Scoring (LightGBM)

**Why LightGBM/XGBoost:**
- Tabular features extracted from DOM are ideal for gradient boosting
- Sub-millisecond inference on CPU
- Interpretable feature importance aids debugging
- Small model size (~1-5MB)
- No GPU required

**Features extracted per candidate selector:**

```python
@dataclass
class SelectorFeatures:
    # Structural features
    tag_name: str                    # Categorical: div, article, section, etc.
    depth: int                       # Distance from root
    child_count: int                 # Direct children
    descendant_count: int            # Total descendants
    sibling_index: int               # Position among siblings
    
    # Text features
    text_length: int                 # Character count
    text_density: float              # text_length / descendant_count
    word_count: int
    sentence_count: int
    has_paragraphs: bool             # Contains <p> tags
    
    # Class/ID features
    class_count: int                 # Number of CSS classes
    has_semantic_class: bool         # Contains "content", "article", "body", etc.
    has_layout_class: bool           # Contains "container", "wrapper", "grid"
    id_present: bool
    id_is_semantic: bool             # ID like "main-content" vs "div-42"
    
    # Context features
    preceding_heading_level: int     # h1=1, h2=2, ..., none=0
    heading_distance: int            # DOM distance to nearest heading
    contains_images: bool
    contains_links: bool
    link_density: float              # links / text_length
    
    # Historical features (if available)
    matched_previous_selector: bool  # Did this selector exist before?
    previous_confidence: float       # Confidence from last extraction
```

**Training data sources:**
- Common Crawl with manual annotations
- Open datasets: SWDE, ClueWeb extractions
- Self-supervised: pages with Schema.org markup provide ground truth

**Model training:**

```python
import lightgbm as lgb

# Binary classifier per field type
title_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    min_child_samples=20,
    objective='binary',
    metric='auc'
)

# Features: matrix of SelectorFeatures for all candidates
# Labels: 1 if selector correctly extracts title, 0 otherwise
title_model.fit(X_train, y_train)

# Inference: score all candidate selectors, pick highest
candidate_scores = title_model.predict_proba(candidates)[:, 1]
best_selector = candidates[candidate_scores.argmax()]
```

### 2. Semantic Similarity for Selector Adaptation (MiniLM)

When CSS classes are renamed, the tree model alone can't identify that `div.post-body` and `div.content` refer to the same element. We use sentence embeddings to find semantic matches.

**Why all-MiniLM-L6-v2:**
- 22M parameters, runs efficiently on CPU
- 384-dimensional embeddings
- Optimized for semantic similarity tasks
- ~5ms per batch of comparisons

**Adaptation workflow:**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SelectorAdapter:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def adapt_selector(
        self,
        old_selector: str,
        old_context: str,          # Surrounding HTML context
        new_candidates: list[str],
        new_contexts: list[str]
    ) -> tuple[str, float]:
        """Find best matching selector in new DOM."""
        
        # Create rich descriptions for embedding
        old_description = self._describe_selector(old_selector, old_context)
        new_descriptions = [
            self._describe_selector(sel, ctx) 
            for sel, ctx in zip(new_candidates, new_contexts)
        ]
        
        # Embed all descriptions
        old_embedding = self.model.encode(old_description)
        new_embeddings = self.model.encode(new_descriptions)
        
        # Cosine similarity
        similarities = np.dot(new_embeddings, old_embedding) / (
            np.linalg.norm(new_embeddings, axis=1) * np.linalg.norm(old_embedding)
        )
        
        best_idx = similarities.argmax()
        return new_candidates[best_idx], similarities[best_idx]
    
    def _describe_selector(self, selector: str, context: str) -> str:
        """Create semantic description of selector's purpose."""
        # Example: "div.post-body" + context → 
        # "main content container with class post-body containing article text paragraphs"
        return f"{selector} {self._extract_semantic_hints(context)}"
```

**Example adaptation:**

```python
# Old site structure
old_selector = "div.post-body"
old_context = "<div class='post-body'><p>Article content here...</p></div>"

# New site structure (after redesign)
new_candidates = ["div.content", "div.article-text", "main", "div.sidebar"]
new_contexts = [
    "<div class='content'><p>Article content here...</p></div>",
    "<div class='article-text'>Short summary</div>",
    "<main><nav>...</nav></main>",
    "<div class='sidebar'>Related posts</div>"
]

best_match, confidence = adapter.adapt_selector(
    old_selector, old_context, 
    new_candidates, new_contexts
)
# Result: ("div.content", 0.89)
```

### 3. Rule-Based Fallbacks

When ML confidence is below threshold or models aren't available:

```python
class RuleBasedInference:
    """Deterministic selector inference using HTML semantics."""
    
    TITLE_PRIORITY = [
        "h1",                           # Standard
        "article h1",                   # Scoped to article
        "[itemprop='headline']",        # Schema.org
        ".title", ".headline",          # Common classes
        "meta[property='og:title']",    # OpenGraph fallback
    ]
    
    CONTENT_PRIORITY = [
        "article",                      # Semantic HTML5
        "[itemprop='articleBody']",     # Schema.org
        "main",                         # Landmark
        "[role='main']",                # ARIA
        ".content", ".post-content",    # Common classes
        "#content", "#main-content",    # Common IDs
    ]
    
    def infer_title(self, soup: BeautifulSoup) -> SelectorRule | None:
        for selector in self.TITLE_PRIORITY:
            elements = soup.select(selector)
            if elements and self._has_reasonable_title(elements[0]):
                return SelectorRule(
                    primary=selector,
                    fallbacks=self.TITLE_PRIORITY[self.TITLE_PRIORITY.index(selector)+1:],
                    extraction_method="text",
                    confidence=0.7  # Rule-based confidence cap
                )
        return None
```

### Model Files and Loading

```
crawler/
└── adaptive/
    └── models/
        ├── selector_scorer_title.lgb      # LightGBM model (~2MB)
        ├── selector_scorer_content.lgb
        ├── selector_scorer_metadata.lgb
        └── minilm/                         # Sentence transformer (~90MB)
            ├── config.json
            ├── pytorch_model.bin
            └── tokenizer/
```

**Lazy loading for memory efficiency:**

```python
class StrategyLearner:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._lgb_models: dict[str, lgb.Booster] = {}
        self._sentence_model: SentenceTransformer | None = None
    
    def _get_lgb_model(self, field: str) -> lgb.Booster:
        if field not in self._lgb_models:
            path = self.models_dir / f"selector_scorer_{field}.lgb"
            self._lgb_models[field] = lgb.Booster(model_file=str(path))
        return self._lgb_models[field]
    
    def _get_sentence_model(self) -> SentenceTransformer:
        if self._sentence_model is None:
            self._sentence_model = SentenceTransformer(
                str(self.models_dir / "minilm")
            )
        return self._sentence_model
```

### Training Pipeline

```bash
# Download and prepare training data
python scripts/prepare_training_data.py \
    --common-crawl-sample 100000 \
    --schema-org-filter \
    --output data/training/

# Train selector scoring models
python scripts/train_selector_models.py \
    --data data/training/ \
    --output crawler/adaptive/models/ \
    --fields title,content,author,date

# Evaluate on held-out test set
python scripts/evaluate_models.py \
    --models crawler/adaptive/models/ \
    --test-data data/test/ \
    --report evaluation_report.json
```

### Why Not LLMs?

| Consideration | LLM (e.g., GPT-4) | Our Approach |
|--------------|-------------------|--------------|
| Latency | 500-2000ms | <10ms |
| Cost per inference | $0.01-0.10 | ~$0 (local) |
| GPU required | Often | No |
| Interpretability | Low | High (feature importance) |
| Offline capability | No | Yes |
| Accuracy on this task | Good | Comparable (specialized) |

LLMs are overkill for selector inference. The problem is well-structured, the feature space is bounded, and we need sub-10ms latency for production crawling.

**When LLMs might help:** Generating human-readable change explanations in `ChangeLogger`. This is optional, non-blocking, and can be done async.

## Performance Considerations

| Operation | Typical Latency | Notes |
|-----------|-----------------|-------|
| Structure analysis | 10-50ms | Depends on DOM size |
| Change detection | 5-20ms | Fast hash comparison first |
| Strategy inference | 50-200ms | ML inference if enabled |
| Redis read | 1-5ms | Local Redis |
| Redis write | 1-5ms | Local Redis |

### Caching Strategy

- Structure TTL: 7 days (configurable)
- Strategy TTL: 7 days (same as structure)
- Change log retention: 90 days
- In-memory LRU cache for hot domains: 1000 entries
