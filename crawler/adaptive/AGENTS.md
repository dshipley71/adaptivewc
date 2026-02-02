# AGENTS.md - Adaptive Extraction Subsystem

This module handles dynamic website structure learning, change detection, and extraction strategy adaptation using dual fingerprinting (rules-based + ML-based).

## Overview

Websites evolve constantly: CSS classes change, JavaScript rewrites DOM structures, iframes relocate, and URL patterns shift. This subsystem enables the crawler to:

1. **Detect changes** using dual fingerprinting (rules-based DOM analysis + ML embeddings)
2. **Classify impact** (cosmetic, minor, moderate, breaking)
3. **Adapt extraction strategies** automatically when possible
4. **Log reasons** for all adaptations with verbose output
5. **Persist to Redis** for future crawls

## Core Principles

1. **Learn Once, Extract Many**: Capture site structure on first visit, reuse until changes detected
2. **Independent Fingerprinting Modes**: Choose from Rules, ML, or Adaptive modes (not combined)
3. **Graceful Degradation**: Attempt adaptation before failing
4. **Explainable Changes**: Every adaptation includes documented reasons
5. **Compliance Preserved**: Adaptive behavior never bypasses rate limits or robots.txt
6. **Verbose Logging**: Every operation logged with detailed context

## Architecture

```
adaptive/
├── structure_analyzer.py    # Rules-based DOM fingerprinting
├── change_detector.py       # Diff engine for structure comparison
├── strategy_learner.py      # CSS selector inference and adaptation
└── models.py                # Data models (shared with crawler/models.py)
```

---

## Module Documentation

### 1. Structure Analyzer (`structure_analyzer.py`)

Rules-based DOM fingerprinting that creates a comprehensive structural signature of a web page.

```python
class StructureAnalyzer:
    """
    Creates DOM fingerprints using rules-based analysis.

    Responsibilities:
    - Parse HTML and extract structural features
    - Identify semantic landmarks (HTML5, ARIA)
    - Detect frameworks and scripts
    - Locate content regions
    - Generate fingerprint hash

    Verbose Logging:
    - [STRUCTURE:ANALYZE] Starting analysis with HTML size
    - [STRUCTURE:PARSE] HTML parsed successfully
    - [STRUCTURE:TAGS] Tag distribution extracted
    - [STRUCTURE:CLASSES] CSS classes catalogued
    - [STRUCTURE:IDS] ID attributes collected
    - [STRUCTURE:LANDMARKS] Semantic landmarks identified
    - [STRUCTURE:IFRAMES] Iframes detected with positions
    - [STRUCTURE:SCRIPTS] Scripts analyzed, frameworks detected
    - [STRUCTURE:REGIONS] Content regions identified
    - [STRUCTURE:NAVIGATION] Navigation elements found
    - [STRUCTURE:PAGINATION] Pagination pattern detected
    - [STRUCTURE:HASH] Fingerprint hash generated
    - [STRUCTURE:COMPLETE] Analysis complete with summary
    """
```

#### Fingerprint Components

```python
@dataclass
class PageStructure:
    """Complete fingerprint of a page's DOM structure."""

    # Identification
    domain: str
    page_type: str                          # "article", "listing", "product", etc.
    url_pattern: str                        # Regex pattern for matching URLs
    variant_id: str = "default"             # For structural variants

    # Tag Hierarchy
    tag_hierarchy: TagHierarchy
    # Contains:
    #   tag_counts: dict[str, int]          # {"div": 450, "p": 120}
    #   depth_distribution: dict[int, int]  # {1: 5, 2: 15, 3: 45}
    #   parent_child_pairs: dict[str, int]  # {"article>p": 8}

    # CSS Analysis
    css_class_map: dict[str, int]           # Class name -> occurrence count
    id_attributes: set[str]                 # All id attributes found

    # Semantic Structure
    semantic_landmarks: dict[str, str]      # Landmark -> CSS selector
    # Example: {"header": "header.site-header", "main": "main#content"}

    # Iframes
    iframe_locations: list[IframeInfo]

    # Scripts
    script_signatures: list[str]            # Framework identifiers
    detected_framework: str | None          # "react", "vue", "angular", etc.

    # Content Regions
    content_regions: list[ContentRegion]
    navigation_selectors: list[str]
    pagination_pattern: PaginationInfo | None

    # Metadata
    captured_at: datetime
    version: int
    content_hash: str                       # Hash for quick comparison
```

#### Analysis Process

```python
def analyze(self, html: str, url: str, verbose: bool = True) -> PageStructure:
    """
    Analyze HTML and create structural fingerprint.

    Verbose Output:
    [STRUCTURE:ANALYZE] Analyzing HTML
      - URL: https://example.com/article/123
      - HTML size: 45,230 bytes

    [STRUCTURE:PARSE] Parsing HTML
      - Parser: lxml
      - Parse time: 12ms

    [STRUCTURE:TAGS] Extracting tag distribution
      - Total tags: 847
      - Unique tags: 32
      - Top 5: div(245), p(120), a(89), span(67), li(45)

    [STRUCTURE:CLASSES] Cataloguing CSS classes
      - Total classes: 156
      - Unique classes: 67
      - Semantic classes found: container, article, content, nav, header, footer

    [STRUCTURE:IDS] Collecting ID attributes
      - IDs found: 23
      - Notable: main-content, sidebar, footer, nav-menu

    [STRUCTURE:LANDMARKS] Identifying semantic landmarks
      - HTML5 landmarks:
        - header: header.site-header
        - nav: nav.main-navigation
        - main: main#content
        - article: article.post
        - aside: aside.sidebar
        - footer: footer.site-footer
      - ARIA roles:
        - banner: header[role="banner"]
        - navigation: nav[role="navigation"]
        - main: main[role="main"]

    [STRUCTURE:IFRAMES] Detecting iframes
      - Found: 2 iframes
      - iframe[0]:
        - Selector: div.content iframe.video
        - Src pattern: https://youtube.com/embed/*
        - Position: content
        - Dimensions: 560x315
        - Dynamic: No
      - iframe[1]:
        - Selector: aside.sidebar iframe.ad
        - Src pattern: https://ads.example.com/*
        - Position: sidebar
        - Dimensions: 300x250
        - Dynamic: Yes

    [STRUCTURE:SCRIPTS] Analyzing scripts
      - Total scripts: 12
      - External: 8, Inline: 4
      - Framework detection:
        - React: DETECTED (data-reactroot, __REACT_DEVTOOLS)
        - Next.js: DETECTED (__NEXT_DATA__, _next)
        - Vue: Not detected
        - Angular: Not detected

    [STRUCTURE:REGIONS] Identifying content regions
      - Regions found: 3
      - main_content:
        - Primary selector: article.post
        - Fallbacks: main#content, div.article-body
        - Confidence: 0.92
      - sidebar:
        - Primary selector: aside.sidebar
        - Fallbacks: div.sidebar, div#sidebar
        - Confidence: 0.85
      - comments:
        - Primary selector: section.comments
        - Fallbacks: div#comments
        - Confidence: 0.78

    [STRUCTURE:NAVIGATION] Finding navigation elements
      - Navigation selectors: 5
        - nav.main-navigation
        - nav.footer-nav
        - ul.breadcrumbs
        - div.pagination
        - aside.sidebar nav

    [STRUCTURE:PAGINATION] Detecting pagination pattern
      - Pattern detected: Yes
      - Next selector: a.next-page
      - Prev selector: a.prev-page
      - Page pattern: ?page={n}
      - Current page: 1

    [STRUCTURE:HASH] Generating fingerprint hash
      - Hash algorithm: SHA-256
      - Hash: a1b2c3d4e5f67890

    [STRUCTURE:COMPLETE] Analysis complete
      - Domain: example.com
      - Page type: article
      - Version: 1
      - Content regions: 3
      - Landmarks: 6
      - Total time: 45ms
    """
```

#### Comparison Method

```python
def compare(
    self,
    structure_a: PageStructure,
    structure_b: PageStructure,
    verbose: bool = True
) -> float:
    """
    Compare two structures and return similarity score (0-1).

    Uses weighted combination of:
    - Tag intersection-over-union (weight: 0.25)
    - CSS class Jaccard similarity (weight: 0.25)
    - ID attribute overlap (weight: 0.15)
    - Semantic landmark match (weight: 0.20)
    - Content region overlap (weight: 0.15)

    Verbose Output:
    [STRUCTURE:COMPARE] Comparing structures
      - Structure A: version 2, captured 2025-01-15
      - Structure B: version 3, captured 2025-01-20

    [STRUCTURE:COMPARE:TAGS] Tag comparison
      - Common tags: 28
      - Only in A: 2 (marquee, blink)
      - Only in B: 1 (dialog)
      - IoU similarity: 0.903

    [STRUCTURE:COMPARE:CLASSES] Class comparison
      - Common classes: 58
      - Only in A: 5 (post-body, post-title, ...)
      - Only in B: 5 (content, title, ...)
      - Jaccard similarity: 0.853

    [STRUCTURE:COMPARE:IDS] ID comparison
      - Common IDs: 20
      - Changed: 2
      - Overlap: 0.909

    [STRUCTURE:COMPARE:LANDMARKS] Landmark comparison
      - Matched: 5/6
      - Changed: footer (footer.site-footer -> footer#new-footer)
      - Match rate: 0.833

    [STRUCTURE:COMPARE:REGIONS] Region comparison
      - Matched regions: 2/3
      - Changed: sidebar moved
      - Overlap: 0.667

    [STRUCTURE:COMPARE:RESULT] Final similarity
      - Tag weight (0.25): 0.903 * 0.25 = 0.226
      - Class weight (0.25): 0.853 * 0.25 = 0.213
      - ID weight (0.15): 0.909 * 0.15 = 0.136
      - Landmark weight (0.20): 0.833 * 0.20 = 0.167
      - Region weight (0.15): 0.667 * 0.15 = 0.100
      - TOTAL SIMILARITY: 0.842
    """
```

---

### 2. Change Detector (`change_detector.py`)

Compares structures and classifies changes.

```python
class ChangeDetector:
    """
    Detects and classifies structure changes.

    Responsibilities:
    - Compare two PageStructure instances
    - Identify specific changes (added/removed/moved elements)
    - Classify change severity
    - Determine if changes are breaking

    Classification Thresholds:
    - COSMETIC: > 0.95 similarity
    - MINOR: 0.85 - 0.95 similarity
    - MODERATE: 0.70 - 0.85 similarity
    - BREAKING: < 0.70 similarity

    Verbose Logging:
    - [CHANGE:DETECT] Starting change detection
    - [CHANGE:HASH] Quick hash comparison
    - [CHANGE:SIMILARITY] Computing similarity scores
    - [CHANGE:IFRAMES] Detecting iframe changes
    - [CHANGE:CLASSES] Detecting class renames
    - [CHANGE:STRUCTURE] Detecting structural changes
    - [CHANGE:SCRIPTS] Detecting script changes
    - [CHANGE:CLASSIFY] Classifying change type
    - [CHANGE:IMPACT] Assessing impact on extraction
    - [CHANGE:RESULT] Final change analysis
    """
```

#### Change Types

```python
class ChangeType(Enum):
    """All detectable change types."""

    # Iframe changes
    IFRAME_ADDED = "iframe_added"
    IFRAME_REMOVED = "iframe_removed"
    IFRAME_RELOCATED = "iframe_relocated"
    IFRAME_RESIZED = "iframe_resized"

    # Tag/Class changes
    TAG_RENAMED = "tag_renamed"
    CLASS_RENAMED = "class_renamed"
    ID_CHANGED = "id_changed"

    # Structural changes
    STRUCTURE_REORGANIZED = "structure_reorganized"
    CONTENT_RELOCATED = "content_relocated"

    # Script changes
    SCRIPT_ADDED = "script_added"
    SCRIPT_REMOVED = "script_removed"
    SCRIPT_MODIFIED = "script_modified"
    FRAMEWORK_CHANGED = "framework_changed"

    # URL/Navigation changes
    URL_PATTERN_CHANGED = "url_pattern_changed"
    PAGINATION_CHANGED = "pagination_changed"

    # Overall changes
    MINOR_LAYOUT_SHIFT = "minor_layout_shift"
    MAJOR_REDESIGN = "major_redesign"
```

#### Detection Process

```python
def detect(
    self,
    previous: PageStructure,
    current: PageStructure,
    verbose: bool = True
) -> ChangeAnalysis:
    """
    Detect all changes between two structures.

    Verbose Output:
    [CHANGE:DETECT] Starting change detection
      - Previous: v2, captured 2025-01-15T10:00:00Z
      - Current: v3, captured 2025-01-20T14:30:00Z
      - Domain: example.com
      - Page type: article

    [CHANGE:HASH] Quick hash comparison
      - Previous hash: a1b2c3d4
      - Current hash: e5f6g7h8
      - Hashes match: NO (detailed comparison needed)

    [CHANGE:SIMILARITY] Computing similarity scores
      - Rules-based similarity: 0.78
      - ML-based similarity: 0.82 (via embeddings)
      - Combined similarity: 0.80 (40% rules, 60% ML)

    [CHANGE:IFRAMES] Detecting iframe changes
      - Previous iframes: 2
      - Current iframes: 2
      - Changes detected:
        - IFRAME_RELOCATED: video-embed
          - Old: aside.sidebar iframe.video
          - New: div.content iframe.video
          - Position: sidebar -> content
          - Breaking: No (same class preserved)

    [CHANGE:CLASSES] Detecting class renames
      - Analyzing class frequency patterns...
      - Potential renames detected:
        - CLASS_RENAMED: post-body -> content
          - Frequency match: 5 occurrences each
          - Context similar: 0.91
          - Confidence: 0.87
        - CLASS_RENAMED: post-title -> title
          - Frequency match: 1 occurrence each
          - Context similar: 0.95
          - Confidence: 0.92
        - CLASS_RENAMED: post-meta -> meta
          - Frequency match: 1 occurrence each
          - Context similar: 0.88
          - Confidence: 0.85

    [CHANGE:STRUCTURE] Detecting structural changes
      - Tag hierarchy similarity: 0.89
      - Depth distribution similarity: 0.92
      - Parent-child similarity: 0.85
      - Changes detected:
        - STRUCTURE_REORGANIZED: sidebar moved
          - Old position: right of content
          - New position: below content
          - Impact: Minor (content order changed)

    [CHANGE:SCRIPTS] Detecting script changes
      - Previous framework: React
      - Current framework: React
      - Scripts added: 1 (analytics.js)
      - Scripts removed: 0
      - Changes detected:
        - SCRIPT_ADDED: analytics.js
          - Impact: None (non-essential)

    [CHANGE:CLASSIFY] Classifying change type
      - Similarity: 0.80
      - Classification: MODERATE
      - Threshold applied: 0.70 < 0.80 < 0.85

    [CHANGE:IMPACT] Assessing impact on extraction
      - Fields affected:
        - title: HIGH impact (selector changed)
        - content: HIGH impact (selector changed)
        - metadata: MEDIUM impact (partial change)
      - Strategy needs update: YES
      - Can auto-adapt: YES (class renames detected)

    [CHANGE:RESULT] Final change analysis
      - Total changes: 6
      - Change types:
        - IFRAME_RELOCATED: 1
        - CLASS_RENAMED: 3
        - STRUCTURE_REORGANIZED: 1
        - SCRIPT_ADDED: 1
      - Classification: MODERATE
      - Breaking: YES (extraction selectors invalid)
      - Auto-adaptable: YES
      - Confidence: 0.85
    """
```

#### Change Analysis Result

```python
@dataclass
class ChangeAnalysis:
    """Complete analysis of changes between structures."""

    # Similarity (from one mode - not combined)
    similarity: float
    mode_used: str  # "rules", "ml", or "adaptive"

    # Classification
    classification: str  # "cosmetic", "minor", "moderate", "breaking"
    breaking: bool

    # Detailed changes
    changes: list[StructureChange]

    # Impact assessment
    fields_affected: dict[str, str]  # field -> impact level
    can_auto_adapt: bool
    adaptation_confidence: float

    # Reason documentation
    reason: str  # Human-readable explanation

    # Adaptive mode details (only populated in adaptive mode)
    escalated: bool = False           # Did adaptive escalate to ML?
    escalation_triggers: list[str] = field(default_factory=list)
```

---

### 3. Strategy Learner (`strategy_learner.py`)

Infers and adapts CSS selectors for content extraction.

```python
class StrategyLearner:
    """
    Learns extraction strategies from page structure.

    Responsibilities:
    - Infer CSS selectors for content fields
    - Generate fallback selectors
    - Adapt selectors when structure changes
    - Validate selector effectiveness

    Learning Priority:
    1. Semantic HTML: <article>, <main>, <h1>
    2. ARIA landmarks: role="main", aria-label
    3. Schema.org: itemtype, itemprop
    4. Common patterns: .content, .article, .post
    5. Structural heuristics: largest text block

    Verbose Logging:
    - [LEARN:START] Starting strategy learning
    - [LEARN:SEMANTIC] Checking semantic HTML
    - [LEARN:ARIA] Checking ARIA landmarks
    - [LEARN:SCHEMA] Checking Schema.org markup
    - [LEARN:PATTERNS] Checking common class patterns
    - [LEARN:HEURISTIC] Applying structural heuristics
    - [LEARN:CANDIDATE] Evaluating selector candidate
    - [LEARN:SELECT] Selecting best selector
    - [LEARN:FALLBACK] Generating fallback selectors
    - [LEARN:VALIDATE] Validating selector
    - [LEARN:CONFIDENCE] Computing confidence score
    - [LEARN:COMPLETE] Strategy learning complete
    """
```

#### Learning Process

```python
async def infer(
    self,
    html: str,
    structure: PageStructure,
    previous_strategy: ExtractionStrategy | None = None,
    verbose: bool = True
) -> ExtractionStrategy:
    """
    Infer extraction strategy for a page.

    Verbose Output:
    [LEARN:START] Starting strategy learning
      - Domain: example.com
      - Page type: article
      - Previous strategy: Yes (v2)
      - Mode: Adaptation (previous exists)

    [LEARN:SEMANTIC] Checking semantic HTML elements
      - <article> found: YES at article.post
      - <main> found: YES at main#content
      - <h1> found: YES at article.post h1
      - <time> found: YES at article.post time[datetime]
      - Semantic coverage: 4/5 common elements

    [LEARN:ARIA] Checking ARIA landmarks
      - role="main": YES at main#content
      - role="article": NO
      - role="banner": YES at header
      - aria-label present: 3 elements

    [LEARN:SCHEMA] Checking Schema.org markup
      - itemtype="Article": YES
      - itemprop="headline": YES at h1
      - itemprop="articleBody": YES at div.content
      - itemprop="author": YES at span.author
      - itemprop="datePublished": YES at time
      - Schema.org coverage: EXCELLENT

    [LEARN:PATTERNS] Checking common class patterns
      - Content patterns found:
        - .content: 1 match (div.content)
        - .article-body: 0 matches
        - .post-content: 0 matches
      - Title patterns found:
        - .title: 1 match (h1.title)
        - .headline: 0 matches

    --- FIELD: title ---

    [LEARN:CANDIDATE] Evaluating selector: article.post h1
      - Elements matched: 1
      - Text content: "Example Article Title"
      - Text length: 21 chars
      - Confidence: 0.95 (single match, semantic element)

    [LEARN:CANDIDATE] Evaluating selector: [itemprop="headline"]
      - Elements matched: 1
      - Text content: "Example Article Title"
      - Text length: 21 chars
      - Confidence: 0.98 (Schema.org, explicit semantic)

    [LEARN:CANDIDATE] Evaluating selector: h1.title
      - Elements matched: 1
      - Text content: "Example Article Title"
      - Text length: 21 chars
      - Confidence: 0.90 (class-based)

    [LEARN:SELECT] Selected best selector for title
      - Winner: [itemprop="headline"]
      - Confidence: 0.98
      - Reason: Schema.org provides explicit semantic meaning

    [LEARN:FALLBACK] Generating fallback selectors for title
      - Fallback 1: article.post h1 (confidence: 0.95)
      - Fallback 2: h1.title (confidence: 0.90)
      - Fallback 3: main h1 (confidence: 0.85)

    --- FIELD: content ---

    [LEARN:CANDIDATE] Evaluating selector: [itemprop="articleBody"]
      - Elements matched: 1
      - Text length: 5,230 chars
      - Paragraph count: 12
      - Confidence: 0.97

    [LEARN:CANDIDATE] Evaluating selector: article.post div.content
      - Elements matched: 1
      - Text length: 5,230 chars
      - Paragraph count: 12
      - Confidence: 0.92

    [LEARN:HEURISTIC] Applying largest text block heuristic
      - Candidate: div.content
      - Text length: 5,230 chars (largest)
      - Text density: 0.78 (high)
      - Confidence: 0.88

    [LEARN:SELECT] Selected best selector for content
      - Winner: [itemprop="articleBody"]
      - Confidence: 0.97
      - Reason: Schema.org with validated content

    [LEARN:FALLBACK] Generating fallback selectors for content
      - Fallback 1: article.post div.content (confidence: 0.92)
      - Fallback 2: main .content (confidence: 0.85)

    --- FIELD: author ---

    [LEARN:CANDIDATE] Evaluating selector: [itemprop="author"]
      - Elements matched: 1
      - Text content: "John Smith"
      - Confidence: 0.95

    [LEARN:SELECT] Selected best selector for author
      - Winner: [itemprop="author"]
      - Confidence: 0.95

    --- FIELD: date ---

    [LEARN:CANDIDATE] Evaluating selector: time[datetime]
      - Elements matched: 1
      - Datetime value: 2025-01-20
      - Confidence: 0.98

    [LEARN:SELECT] Selected best selector for date
      - Winner: time[datetime]
      - Confidence: 0.98
      - Extraction: attribute "datetime"

    [LEARN:VALIDATE] Validating complete strategy
      - All required fields present: YES
      - Minimum confidence threshold (0.8): PASSED
      - Selector uniqueness: PASSED

    [LEARN:COMPLETE] Strategy learning complete
      - Domain: example.com
      - Page type: article
      - Version: 3
      - Fields learned: 4 (title, content, author, date)
      - Average confidence: 0.97
      - Total time: 125ms
    """
```

#### Adaptation Process

```python
async def adapt(
    self,
    html: str,
    structure: PageStructure,
    old_strategy: ExtractionStrategy,
    changes: ChangeAnalysis,
    verbose: bool = True
) -> ExtractionStrategy:
    """
    Adapt existing strategy based on detected changes.

    Verbose Output:
    [LEARN:ADAPT] Adapting strategy based on changes
      - Domain: example.com
      - Page type: article
      - Old version: 2
      - Changes: 3 class renames detected

    [LEARN:ADAPT:RENAMES] Processing class renames
      - Rename 1: post-body -> content
        - Old selector: div.post-body
        - New selector: div.content
        - Verification: PASSED (1 element, same content)
        - Confidence: 0.92

      - Rename 2: post-title -> title
        - Old selector: h1.post-title
        - New selector: h1.title
        - Verification: PASSED (1 element, same content)
        - Confidence: 0.94

    [LEARN:ADAPT:UPDATE] Updating selectors
      - title:
        - Old: h1.post-title
        - New: h1.title
        - Fallbacks updated: 2
      - content:
        - Old: div.post-body
        - New: div.content
        - Fallbacks updated: 2

    [LEARN:ADAPT:VERIFY] Verifying adapted strategy
      - All selectors valid: YES
      - Extraction test: PASSED
      - Confidence maintained: 0.93 (was 0.95)

    [LEARN:ADAPT:COMPLETE] Adaptation complete
      - Selectors updated: 2
      - Fallbacks regenerated: 4
      - New version: 3
      - Adaptation confidence: 0.93
    """
```

#### Confidence Scoring

```python
def _compute_confidence(
    self,
    selector: str,
    elements: list,
    context: dict,
    verbose: bool = True
) -> float:
    """
    Compute confidence score for a selector.

    Scoring factors:
    - Element count: 1 = base, 2-3 = *0.9, 4+ = *0.7
    - Semantic bonus: Schema.org = +0.1, HTML5 = +0.05
    - Specificity: Higher specificity = higher confidence
    - Historical: Previously successful = +0.05

    Verbose Output:
    [LEARN:CONFIDENCE] Computing confidence for selector
      - Selector: [itemprop="headline"]
      - Elements matched: 1

    [LEARN:CONFIDENCE:FACTORS]
      - Base score: 1.0 (single element match)
      - Element count adjustment: *1.0 (1 element)
      - Semantic bonus: +0.10 (Schema.org itemprop)
      - Specificity bonus: +0.02 (attribute selector)
      - Historical bonus: +0.00 (new selector)

    [LEARN:CONFIDENCE:RESULT]
      - Raw score: 1.12
      - Capped score: 0.98 (max 1.0)
      - Final confidence: 0.98
    """
```

---

## Fingerprinting Modes

The adaptive module provides three independent fingerprinting modes. You select one mode - they are **not combined**.

### Mode Overview

| Mode | Speed | Best For | Trade-offs |
|------|-------|----------|------------|
| **Rules** | ~15ms | Stable sites, high throughput | Sensitive to class renames |
| **ML** | ~200ms | Sites with frequent CSS changes | Slower, requires embedding model |
| **Adaptive** | ~15-200ms | Unknown/mixed sites | Starts fast, escalates when needed |

### Mode Selection

```yaml
# config.yaml
fingerprinting:
  mode: adaptive  # "rules", "ml", or "adaptive"
```

### Rules-Based Mode

Fast, deterministic comparison using DOM structure analysis.

```
[FINGERPRINT:MODE] Using RULES mode
[COMPARE:RULES] Computing rules-based similarity
  - Tag similarity: 0.92
  - Class similarity: 0.75 (renames detected)
  - ID similarity: 0.95
  - Landmark similarity: 0.90
  - Region similarity: 0.85
[COMPARE:RULES:RESULT] Similarity: 0.874
[COMPARE:CLASSIFY] Classification: MINOR
[COMPARE:RESULT] Breaking: NO
```

### ML-Based Mode

Semantic comparison using embeddings - robust to class renames.

```
[FINGERPRINT:MODE] Using ML mode
[COMPARE:ML] Computing ML-based similarity
  - Generating description for stored structure...
  - Generating description for current structure...
  - Computing embeddings...
    - Stored embedding: 384 dims, norm=1.0
    - Current embedding: 384 dims, norm=1.0
[COMPARE:ML:SIMILARITY] Cosine similarity: 0.912
[COMPARE:ML:RESULT] Similarity: 0.912
[COMPARE:CLASSIFY] Classification: MINOR
[COMPARE:RESULT] Breaking: NO
```

### Adaptive Mode (Recommended)

Starts with fast rules-based, escalates to ML only when needed.

**Escalation Triggers:**
- Class change ratio > 15%
- Rules similarity < 0.80
- Domain flagged as volatile
- Rename pattern detected

```
[ADAPTIVE:START] Starting adaptive comparison
  - Domain: example.com
  - Page type: article

[ADAPTIVE:RULES] Running rules-based comparison
[COMPARE:RULES] Computing similarity...
  - Tag similarity: 0.91
  - Class similarity: 0.52 (significant changes!)
[ADAPTIVE:RULES:RESULT] Similarity: 0.72

[ADAPTIVE:ANALYZE] Checking escalation triggers
[ADAPTIVE:CHECK:CLASSES] Class change ratio: 38% - TRIGGERED
[ADAPTIVE:CHECK:UNCERTAINTY] Similarity 0.72 < 0.80 - TRIGGERED

[ADAPTIVE:TRIGGER] Escalation triggered
  - CLASS_VOLATILITY: 38% of classes changed
  - RULES_UNCERTAINTY: Similarity below threshold

[ADAPTIVE:ESCALATE] Running ML comparison
[COMPARE:ML:SIMILARITY] Cosine similarity: 0.89
[ADAPTIVE:ML:RESULT] Similarity: 0.89

[ADAPTIVE:RESULT] Using ML result
  - Mode used: ML (escalated)
  - Similarity: 0.89
  - Classification: MINOR
  - Total time: 215ms
```

### Important: Modes are Independent

- **No weighted combination**: We do not combine rules and ML scores
- **No parallel execution**: Adaptive runs rules first, then ML only if triggered
- **Single result**: Returns one similarity score from one mode
- **Clear provenance**: Result indicates which mode produced it

---

## Data Models

### PageStructure

```python
@dataclass
class PageStructure:
    """Fingerprint of a page's DOM structure."""

    # Identification
    domain: str
    page_type: str                    # "article", "listing", "product"
    url_pattern: str                  # Regex pattern for matching URLs
    variant_id: str = "default"

    # Structure fingerprint
    tag_hierarchy: TagHierarchy
    iframe_locations: list[IframeInfo]
    script_signatures: list[str]
    detected_framework: str | None
    css_class_map: dict[str, int]
    id_attributes: set[str]
    semantic_landmarks: dict[str, str]

    # Content regions
    content_regions: list[ContentRegion]
    navigation_selectors: list[str]
    pagination_pattern: PaginationInfo | None

    # Metadata
    captured_at: datetime
    version: int
    content_hash: str
```

### ExtractionStrategy

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
    iframe_extraction: dict[str, SelectorRule]

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
    attribute_name: str | None        # If method == "attribute"
    post_processors: list[str]        # ["strip", "normalize_whitespace"]
    confidence: float
```

### StructureChange

```python
@dataclass
class StructureChange:
    """Single detected change between structures."""

    change_type: ChangeType
    affected_components: list[str]

    # Details
    old_value: str | None
    new_value: str | None
    location: str                     # CSS selector or description

    # Impact
    breaking: bool
    fields_affected: list[str]
    confidence: float

    # Documentation
    reason: str                       # Human-readable explanation
    evidence: dict                    # Supporting data
```

---

## Redis Storage Integration

### Key Patterns

```
# Structures
crawler:structure:{domain}:{page_type}:{variant_id}
crawler:structure:{domain}:{page_type}:{variant_id}:v{n}  # History

# Strategies
crawler:strategy:{domain}:{page_type}:{variant_id}
crawler:strategy:{domain}:{page_type}:{variant_id}:v{n}   # History

# Variants
crawler:variants:{domain}:{page_type}

# Changes
crawler:changes:{domain}:{page_type}

# Embeddings (for ML fingerprinting)
crawler:embedding:{domain}:{page_type}:{variant_id}
```

### Storage Verbose Logging

```
[STORE:SAVE] Saving structure
  - Domain: example.com
  - Page type: article
  - Variant: default
  - Version: 3
  - Key: crawler:structure:example.com:article:default

[STORE:HISTORY] Archiving previous version
  - From: crawler:structure:example.com:article:default
  - To: crawler:structure:example.com:article:default:v2
  - Versions retained: 10

[STORE:EMBEDDING] Saving embedding
  - Key: crawler:embedding:example.com:article:default
  - Dimensions: 384
  - Size: 1,536 bytes

[STORE:CHANGE] Logging change
  - Key: crawler:changes:example.com:article
  - Change type: CLASS_RENAMED
  - Version: 2 -> 3
```

---

## Common Tasks

### Adding a New Change Type

1. Add enum value to `ChangeType` in `crawler/models.py`:
   ```python
   class ChangeType(Enum):
       # ... existing types
       NEW_CHANGE_TYPE = "new_change_type"
   ```

2. Implement detection in `change_detector.py`:
   ```python
   def _detect_new_change_type(
       self, prev: PageStructure, curr: PageStructure, verbose: bool
   ) -> list[StructureChange]:
       if verbose:
           self._log("[CHANGE:NEW_TYPE] Detecting new change type...")
       # ... detection logic
   ```

3. Add to main detection method:
   ```python
   def detect(self, ...):
       # ... existing detections
       changes.extend(self._detect_new_change_type(prev, curr, verbose))
   ```

4. Update verbose logging documentation

### Customizing Learning Priority

Modify `strategy_learner.py`:

```python
LEARNING_PRIORITY = [
    ("schema_org", 0.98),    # Schema.org gets highest confidence
    ("semantic_html", 0.95), # HTML5 semantic elements
    ("aria", 0.90),          # ARIA landmarks
    ("class_patterns", 0.85), # Common class names
    ("heuristics", 0.80),    # Structural heuristics
]
```

### Adjusting Change Thresholds

Modify `change_detector.py`:

```python
class ChangeThresholds:
    COSMETIC = 0.95    # > 95% similar = cosmetic
    MINOR = 0.85       # 85-95% = minor
    MODERATE = 0.70    # 70-85% = moderate
    BREAKING = 0.70    # < 70% = breaking
```

---

## Performance Considerations

| Operation | Typical Latency | Notes |
|-----------|-----------------|-------|
| Structure analysis | 20-50ms | Depends on DOM size |
| Rules-based comparison | 5-15ms | Fast hash first |
| ML embedding generation | 100-200ms | First time only, cached |
| ML similarity comparison | 5-10ms | Vector math |
| Strategy learning | 50-150ms | Depends on page complexity |
| Redis read | 1-5ms | Local Redis |
| Redis write | 1-5ms | Local Redis |

### Caching Strategy

- Structure embeddings: Cached in Redis with structure
- Embedding model: Loaded once, kept in memory
- Comparison results: Not cached (fast to compute)
