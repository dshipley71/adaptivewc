# AGENTS.md - ML and Embeddings Module

This module provides ML-based fingerprinting, page classification, and Ollama Cloud LLM integration for intelligent structure descriptions.

## Overview

The ML module enhances the adaptive crawler with:

1. **Semantic Embeddings**: Convert page structures to vector representations for similarity comparison
2. **Ollama Cloud Integration**: Generate rich descriptions of page structures using LLM
3. **Page Classification**: ML-based page type prediction (article, listing, product, etc.)
4. **ML Change Detection**: Embedding-based change detection that's robust to superficial changes

## Architecture

```
ml/
└── embeddings.py    # All ML functionality in single module
    ├── StructureEmbeddingModel      # Embedding generation
    ├── RulesBasedDescriptionGenerator  # Rules-based descriptions
    ├── LLMDescriptionGenerator      # Ollama Cloud descriptions
    ├── StructureClassifier          # Page type classification
    └── MLChangeDetector             # Embedding-based change detection
```

---

## Module Documentation

### 1. Structure Embedding Model

Generates semantic embeddings for page structures using sentence transformers.

```python
class StructureEmbeddingModel:
    """
    Converts page structures to semantic vector embeddings.

    Model: all-MiniLM-L6-v2 (default)
    - 384-dimensional embeddings
    - Optimized for semantic similarity
    - Runs efficiently on CPU

    Alternative: all-mpnet-base-v2
    - 768-dimensional embeddings
    - Higher accuracy
    - Slower inference

    Verbose Logging:
    - [EMBED:INIT] Model initialization
    - [EMBED:LOAD] Loading model weights
    - [EMBED:DESCRIBE] Generating text description
    - [EMBED:ENCODE] Encoding to embedding
    - [EMBED:DIMS] Embedding dimensions
    - [EMBED:NORM] Embedding L2 normalization
    - [EMBED:BATCH] Batch embedding processing
    - [EMBED:CACHE] Cache hit/miss
    - [EMBED:SIMILARITY] Computing similarity
    """
```

#### Initialization

```python
def __init__(
    self,
    model_name: str = "all-MiniLM-L6-v2",
    use_gpu: bool = False,
    cache_embeddings: bool = True,
    verbose: bool = True
):
    """
    Initialize the embedding model.

    Verbose Output:
    [EMBED:INIT] Initializing StructureEmbeddingModel
      - Model: all-MiniLM-L6-v2
      - Device: CPU (GPU not requested)
      - Cache: Enabled
      - Verbose: Enabled

    [EMBED:LOAD] Loading model weights
      - Source: sentence-transformers hub
      - Parameters: 22.7M
      - Load time: 1.2s
      - Ready: YES
    """
```

#### Embedding Generation

```python
def embed_structure(
    self,
    structure: PageStructure,
    description_generator: DescriptionGenerator | None = None,
    verbose: bool = True
) -> StructureEmbedding:
    """
    Generate embedding for a page structure.

    Process:
    1. Generate text description of structure
    2. Encode description to embedding vector
    3. Normalize embedding (L2 norm = 1)

    Verbose Output:
    [EMBED:DESCRIBE] Generating description for structure
      - Domain: example.com
      - Page type: article
      - Generator: RulesBasedDescriptionGenerator

    [EMBED:DESCRIBE:RESULT] Description generated
      - Length: 245 characters
      - Content: "Article page with semantic HTML5 structure.
        Main content in <article> element. Features h1 title,
        content paragraphs, author byline, and publication date.
        Navigation header and footer present. No iframes detected."

    [EMBED:ENCODE] Encoding description
      - Input length: 245 chars
      - Tokenizing...
      - Token count: 52

    [EMBED:DIMS] Embedding generated
      - Dimensions: 384
      - Shape: (384,)

    [EMBED:NORM] Normalizing embedding
      - Raw L2 norm: 12.45
      - Normalized L2 norm: 1.0

    [EMBED:RESULT] Embedding complete
      - Domain: example.com
      - Page type: article
      - Embedding dims: 384
      - First 5 values: [0.023, -0.145, 0.089, 0.012, -0.067]
    """
```

#### Batch Processing

```python
def embed_structures_batch(
    self,
    structures: list[PageStructure],
    verbose: bool = True
) -> list[StructureEmbedding]:
    """
    Generate embeddings for multiple structures efficiently.

    Verbose Output:
    [EMBED:BATCH] Processing batch of structures
      - Count: 10 structures
      - Domains: example.com, news.site, blog.org

    [EMBED:BATCH:DESCRIBE] Generating descriptions
      - Structure 1/10: example.com/article
      - Structure 2/10: example.com/listing
      - ...
      - Structure 10/10: blog.org/post
      - Total descriptions: 10

    [EMBED:BATCH:ENCODE] Batch encoding
      - Input count: 10
      - Batch size: 10 (single batch)
      - Encoding...

    [EMBED:BATCH:RESULT] Batch complete
      - Embeddings generated: 10
      - Average time per embedding: 15ms
      - Total time: 150ms
    """
```

#### Similarity Computation

```python
def compute_similarity(
    self,
    embedding_a: StructureEmbedding,
    embedding_b: StructureEmbedding,
    verbose: bool = True
) -> float:
    """
    Compute cosine similarity between two embeddings.

    Verbose Output:
    [EMBED:SIMILARITY] Computing cosine similarity
      - Embedding A: example.com/article v2
      - Embedding B: example.com/article v3

    [EMBED:SIMILARITY:COMPUTE]
      - Dot product: 0.912
      - Norm A: 1.0
      - Norm B: 1.0
      - Cosine similarity: 0.912

    [EMBED:SIMILARITY:INTERPRET]
      - Similarity: 0.912
      - Interpretation: HIGH (structures very similar)
      - Threshold for breaking: 0.70
      - Is breaking change: NO
    """
```

---

### 2. Description Generators

#### Rules-Based Description Generator

Generates descriptions using deterministic rules.

```python
class RulesBasedDescriptionGenerator:
    """
    Generates text descriptions of page structures using rules.

    Advantages:
    - Fast (no external calls)
    - Deterministic
    - No API costs
    - Works offline

    Verbose Logging:
    - [DESC:RULES] Starting rules-based description
    - [DESC:PAGE_TYPE] Determining page type descriptor
    - [DESC:STRUCTURE] Describing structural elements
    - [DESC:LANDMARKS] Describing semantic landmarks
    - [DESC:CONTENT] Describing content regions
    - [DESC:IFRAMES] Describing iframes
    - [DESC:SCRIPTS] Describing detected frameworks
    - [DESC:RESULT] Final description
    """
```

```python
def generate(
    self,
    structure: PageStructure,
    verbose: bool = True
) -> str:
    """
    Generate description from page structure.

    Verbose Output:
    [DESC:RULES] Generating rules-based description
      - Domain: example.com
      - Page type: article

    [DESC:PAGE_TYPE] Determining page type descriptor
      - Input: article
      - Descriptor: "Article page"

    [DESC:STRUCTURE] Describing structural elements
      - Total tags: 847
      - Key tags: article(1), h1(1), p(45), a(89)
      - Depth: max 8 levels
      - Complexity: MEDIUM

    [DESC:LANDMARKS] Describing semantic landmarks
      - header: YES (header.site-header)
      - nav: YES (nav.main-navigation)
      - main: YES (main#content)
      - article: YES (article.post)
      - aside: YES (aside.sidebar)
      - footer: YES (footer.site-footer)
      - Coverage: EXCELLENT (6/6)

    [DESC:CONTENT] Describing content regions
      - Regions: 3
      - main_content: article.post (confidence 0.92)
      - sidebar: aside.sidebar (confidence 0.85)
      - comments: section.comments (confidence 0.78)

    [DESC:IFRAMES] Describing iframes
      - Count: 1
      - video: YouTube embed in content area

    [DESC:SCRIPTS] Describing detected frameworks
      - Framework: React + Next.js
      - Client-side rendering: YES

    [DESC:RESULT] Description generated
      - Length: 289 characters
      - Content: "Article page with excellent semantic HTML5 structure.
        Main content in <article> element with clear heading hierarchy.
        Features header navigation, sidebar, and footer. Single YouTube
        video embed. Built with React and Next.js framework."
    """
```

#### LLM Description Generator (Ollama Cloud)

Generates rich descriptions using Ollama Cloud LLM.

```python
class LLMDescriptionGenerator:
    """
    Generates descriptions using Ollama Cloud LLM.

    Provider: Ollama Cloud (https://ollama.com)
    Endpoint: https://ollama.com/api/chat
    Authentication: Bearer token (OLLAMA_CLOUD_API_KEY)

    Models Supported:
    - gemma3:12b (default, good balance)
    - llama3.2 (fast, efficient)
    - mistral (good for technical content)

    Verbose Logging:
    - [LLM:INIT] Client initialization
    - [LLM:CONFIG] Configuration details
    - [LLM:PROMPT] Prompt generation
    - [LLM:REQUEST] API request details
    - [LLM:RESPONSE] API response
    - [LLM:PARSE] Response parsing
    - [LLM:DESCRIPTION] Final description
    - [LLM:ERROR] Error handling
    - [LLM:RETRY] Retry attempts
    """
```

##### Initialization

```python
def __init__(
    self,
    model: str = "gemma3:12b",
    api_key: str | None = None,
    timeout: int = 30,
    max_retries: int = 3,
    temperature: float = 0.3,
    max_tokens: int = 500,
    verbose: bool = True
):
    """
    Initialize Ollama Cloud client.

    Verbose Output:
    [LLM:INIT] Initializing LLMDescriptionGenerator
      - Provider: ollama-cloud
      - Model: gemma3:12b
      - Endpoint: https://ollama.com/api/chat

    [LLM:CONFIG] Configuration
      - API key: ****...***** (set via OLLAMA_CLOUD_API_KEY)
      - Timeout: 30s
      - Max retries: 3
      - Temperature: 0.3
      - Max tokens: 500

    [LLM:INIT:READY] Client ready
      - Status: READY
      - Health check: PASSED
    """
```

##### Description Generation

```python
async def generate(
    self,
    structure: PageStructure,
    verbose: bool = True
) -> str:
    """
    Generate description using Ollama Cloud.

    Verbose Output:
    [LLM:PROMPT] Creating prompt for structure
      - Domain: example.com
      - Page type: article
      - Structure summary:
        - Tags: 847 total, 32 unique
        - Classes: 67 unique
        - Landmarks: 6/6 present
        - Regions: 3 content areas
        - Iframes: 1 (video)
        - Framework: React

    [LLM:PROMPT:CONTENT]
      "Analyze this web page structure and provide a concise 2-3 sentence
      description focusing on: page type, layout organization, semantic
      structure quality, and notable features.

      Structure data:
      - Domain: example.com
      - Page type: article
      - Tag counts: div(245), p(45), a(89), article(1), h1(1)
      - Semantic landmarks: header, nav, main, article, aside, footer
      - Content regions: main_content (0.92), sidebar (0.85)
      - Iframes: 1 video embed
      - Framework: React + Next.js

      Provide only the description, no explanations."

    [LLM:REQUEST] Sending request to Ollama Cloud
      - Endpoint: https://ollama.com/api/chat
      - Model: gemma3:12b
      - Prompt length: 650 characters
      - Headers: Authorization: Bearer ****

    [LLM:REQUEST:PAYLOAD]
      {
        "model": "gemma3:12b",
        "messages": [{"role": "user", "content": "..."}],
        "stream": false,
        "options": {
          "num_predict": 500,
          "temperature": 0.3
        }
      }

    [LLM:RESPONSE] Response received
      - Status: 200 OK
      - Response time: 1.8s
      - Content length: 312 characters

    [LLM:RESPONSE:CONTENT]
      {
        "message": {
          "role": "assistant",
          "content": "Well-structured article page using modern React and
            Next.js framework with excellent semantic HTML5 markup.
            Features clear content hierarchy with article element containing
            title, body paragraphs, and author metadata. Includes
            responsive sidebar navigation and embedded video player."
        }
      }

    [LLM:PARSE] Parsing response
      - Raw content: 312 chars
      - Cleaned content: 312 chars

    [LLM:DESCRIPTION] Description generated
      - Final length: 312 characters
      - Quality: HIGH (specific, technical, informative)
    """
```

##### Error Handling

```python
async def _handle_error(
    self,
    error: Exception,
    attempt: int,
    verbose: bool = True
) -> None:
    """
    Handle API errors with retry logic.

    Verbose Output (on error):
    [LLM:ERROR] API request failed
      - Attempt: 1/3
      - Error type: HTTPStatusError
      - Status code: 429 (Too Many Requests)
      - Message: Rate limit exceeded

    [LLM:RETRY] Retrying request
      - Wait time: 2s (exponential backoff)
      - Next attempt: 2/3

    [LLM:REQUEST] Sending request to Ollama Cloud (retry)
      - Attempt: 2/3
      ...

    (On persistent failure):
    [LLM:ERROR:FINAL] All retries exhausted
      - Total attempts: 3
      - Final error: 429 Rate limit exceeded
      - Fallback: Using rules-based description

    [LLM:FALLBACK] Falling back to rules-based generator
      - Reason: API unavailable
      - Generator: RulesBasedDescriptionGenerator
    """
```

##### Change Detection Descriptions

```python
async def generate_for_change_detection(
    self,
    old_structure: PageStructure,
    new_structure: PageStructure,
    verbose: bool = True
) -> str:
    """
    Generate description of changes between structures.

    Verbose Output:
    [LLM:CHANGE] Generating change description
      - Old version: v2 (2025-01-15)
      - New version: v3 (2025-01-20)
      - Domain: example.com

    [LLM:CHANGE:DIFF] Computing differences
      - Tag changes: 5 added, 2 removed
      - Class changes: 12 renamed
      - Landmark changes: 1 (footer selector changed)
      - Region changes: 1 (sidebar moved)

    [LLM:PROMPT] Creating change analysis prompt
      - Focus: Structural differences
      - Context: Previous and current structures

    [LLM:REQUEST] Sending request to Ollama Cloud
      ...

    [LLM:DESCRIPTION] Change description generated
      - Content: "Site underwent CSS refactoring with class renames
        following simplified naming convention (post-* prefix removed).
        Sidebar relocated from right column to below content area.
        Footer selector updated but structure preserved. Video embed
        moved to main content area for better visibility."
    """
```

---

### 3. Structure Classifier

ML-based page type classification.

```python
class StructureClassifier:
    """
    Classifies page structures into types using ML.

    Backends:
    - LogisticRegression (default, fast, interpretable)
    - XGBoost (higher accuracy, feature importance)
    - LightGBM (fast training, handles large features)

    Page Types:
    - article: News articles, blog posts
    - listing: Search results, category pages
    - product: E-commerce product pages
    - homepage: Site landing pages
    - profile: User profile pages
    - form: Contact, login, registration pages
    - error: 404, 500 error pages

    Verbose Logging:
    - [CLASSIFY:INIT] Classifier initialization
    - [CLASSIFY:LOAD] Loading trained model
    - [CLASSIFY:FEATURES] Feature extraction
    - [CLASSIFY:PREDICT] Prediction execution
    - [CLASSIFY:RESULT] Classification result
    - [CLASSIFY:CONFIDENCE] Confidence analysis
    - [CLASSIFY:TRAIN] Training operations
    """
```

##### Feature Extraction

```python
def extract_features(
    self,
    structure: PageStructure,
    verbose: bool = True
) -> np.ndarray:
    """
    Extract feature vector from page structure.

    Features (42 total):
    - Tag distribution (15 features): counts of key tags
    - Class patterns (10 features): semantic class indicators
    - Structure metrics (10 features): depth, breadth, ratios
    - Content features (7 features): text density, link ratio

    Verbose Output:
    [CLASSIFY:FEATURES] Extracting features from structure
      - Domain: example.com
      - Page type (if known): article

    [CLASSIFY:FEATURES:TAGS] Tag features (15)
      - article_count: 1
      - h1_count: 1
      - p_count: 45
      - ul_count: 3
      - li_count: 28
      - a_count: 89
      - img_count: 5
      - form_count: 0
      - input_count: 0
      - table_count: 0
      - div_count: 245
      - span_count: 67
      - section_count: 4
      - aside_count: 1
      - nav_count: 2

    [CLASSIFY:FEATURES:CLASSES] Class pattern features (10)
      - has_article_class: YES (article.post)
      - has_content_class: YES (div.content)
      - has_product_class: NO
      - has_listing_class: NO
      - has_search_class: NO
      - has_form_class: NO
      - has_error_class: NO
      - has_profile_class: NO
      - has_nav_class: YES
      - has_sidebar_class: YES

    [CLASSIFY:FEATURES:STRUCTURE] Structure metrics (10)
      - max_depth: 8
      - avg_depth: 4.2
      - tag_diversity: 0.78
      - class_diversity: 0.65
      - landmark_count: 6
      - region_count: 3
      - iframe_count: 1
      - script_count: 12
      - text_node_ratio: 0.35
      - link_density: 0.12

    [CLASSIFY:FEATURES:CONTENT] Content features (7)
      - has_pagination: YES
      - has_comments: YES
      - has_author: YES
      - has_date: YES
      - content_length_bucket: LONG (>5000 chars)
      - image_ratio: 0.06
      - form_field_count: 0

    [CLASSIFY:FEATURES:RESULT] Feature vector
      - Dimensions: 42
      - Non-zero: 35
      - Vector: [1, 1, 45, 3, 28, 89, 5, 0, ...]
    """
```

##### Prediction

```python
def predict(
    self,
    structure: PageStructure,
    verbose: bool = True
) -> tuple[str, float]:
    """
    Predict page type with confidence.

    Verbose Output:
    [CLASSIFY:PREDICT] Predicting page type
      - Domain: example.com
      - Features extracted: 42

    [CLASSIFY:PREDICT:PROBA] Class probabilities
      - article: 0.89
      - listing: 0.05
      - product: 0.02
      - homepage: 0.02
      - profile: 0.01
      - form: 0.00
      - error: 0.01

    [CLASSIFY:RESULT] Prediction result
      - Predicted type: article
      - Confidence: 0.89
      - Second choice: listing (0.05)

    [CLASSIFY:CONFIDENCE] Confidence analysis
      - Confidence level: HIGH (>0.80)
      - Decision: ACCEPT prediction
      - Margin: 0.84 (89% - 5%)
    """
```

---

### 4. ML Change Detector

Embedding-based change detection.

```python
class MLChangeDetector:
    """
    Detects changes using embedding similarity.

    Advantages over rules-based:
    - Robust to class renames
    - Semantic understanding
    - Handles superficial changes

    Thresholds:
    - COSMETIC: > 0.95 similarity
    - MINOR: 0.85 - 0.95
    - MODERATE: 0.70 - 0.85
    - BREAKING: < 0.70

    Verbose Logging:
    - [ML:CHANGE] Starting ML change detection
    - [ML:EMBED] Generating embeddings
    - [ML:SIMILARITY] Computing similarity
    - [ML:THRESHOLD] Applying thresholds
    - [ML:BREAKING] Breaking change determination
    - [ML:IMPACT] Impact prediction
    - [ML:BASELINE] Baseline operations
    - [ML:DRIFT] Drift detection
    """
```

##### Change Detection

```python
def detect_change(
    self,
    old_structure: PageStructure,
    new_structure: PageStructure,
    old_embedding: StructureEmbedding | None = None,
    new_embedding: StructureEmbedding | None = None,
    verbose: bool = True
) -> MLChangeResult:
    """
    Detect changes using embeddings.

    Verbose Output:
    [ML:CHANGE] Starting ML-based change detection
      - Old: example.com/article v2
      - New: example.com/article v3

    [ML:EMBED] Generating embeddings
      - Old embedding: Computing...
        [EMBED:DESCRIBE] Generating description...
        [EMBED:ENCODE] Encoding...
        - Dimensions: 384
        - Norm: 1.0
      - New embedding: Computing...
        [EMBED:DESCRIBE] Generating description...
        [EMBED:ENCODE] Encoding...
        - Dimensions: 384
        - Norm: 1.0

    [ML:SIMILARITY] Computing cosine similarity
      - Dot product: 0.823
      - Similarity: 0.823

    [ML:THRESHOLD] Applying thresholds
      - Similarity: 0.823
      - Thresholds:
        - COSMETIC: > 0.95 (NO)
        - MINOR: 0.85-0.95 (NO)
        - MODERATE: 0.70-0.85 (YES)
        - BREAKING: < 0.70 (NO)
      - Classification: MODERATE

    [ML:BREAKING] Determining if breaking
      - Threshold: 0.70
      - Similarity: 0.823
      - Is breaking: NO (above threshold)

    [ML:IMPACT] Predicting impact
      - Semantic shift: MODERATE
      - Likely causes:
        - Class renames (detected similarity pattern)
        - Layout reorganization
      - Recommended action: ADAPT selectors

    [ML:RESULT] ML change detection complete
      - Similarity: 0.823
      - Classification: MODERATE
      - Breaking: NO
      - Confidence: 0.87
    """
```

##### Baseline and Drift Detection

```python
def set_site_baseline(
    self,
    domain: str,
    structure: PageStructure,
    verbose: bool = True
) -> None:
    """
    Set baseline structure for drift detection.

    Verbose Output:
    [ML:BASELINE] Setting site baseline
      - Domain: example.com
      - Page type: article
      - Version: 1 (initial)

    [ML:BASELINE:EMBED] Generating baseline embedding
      - Embedding: 384 dimensions
      - Stored in memory cache

    [ML:BASELINE:SAVE] Baseline saved
      - Key: baseline:example.com:article
      - Timestamp: 2025-01-15T10:00:00Z
    """

def detect_drift_from_baseline(
    self,
    domain: str,
    structure: PageStructure,
    verbose: bool = True
) -> DriftAnalysis:
    """
    Detect drift from baseline structure.

    Verbose Output:
    [ML:DRIFT] Detecting drift from baseline
      - Domain: example.com
      - Baseline: v1 (2025-01-15)
      - Current: v5 (2025-02-01)

    [ML:DRIFT:COMPARE] Comparing to baseline
      - Baseline embedding: loaded from cache
      - Current embedding: generating...

    [ML:DRIFT:SIMILARITY] Computing drift
      - Similarity to baseline: 0.72
      - Cumulative drift: 0.28 (1 - 0.72)

    [ML:DRIFT:ANALYSIS] Drift analysis
      - Drift level: HIGH (>0.20)
      - Trend: INCREASING (was 0.15 last week)
      - Recommendation: Consider re-baselining

    [ML:DRIFT:RESULT] Drift detection complete
      - Current similarity: 0.72
      - Drift amount: 0.28
      - Level: HIGH
      - Action needed: REVIEW
    """
```

---

## Ollama Cloud API Reference

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
    "messages": [
        {
            "role": "user",
            "content": "Your prompt here"
        }
    ],
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
    "model": "gemma3:12b",
    "created_at": "2025-01-20T10:30:00Z",
    "message": {
        "role": "assistant",
        "content": "Generated response..."
    },
    "done": true
}
```

### Error Responses

| Status | Meaning | Action |
|--------|---------|--------|
| 401 | Invalid API key | Check OLLAMA_CLOUD_API_KEY |
| 429 | Rate limited | Wait and retry with backoff |
| 500 | Server error | Retry with backoff |
| 503 | Service unavailable | Retry or use fallback |

### Verbose API Logging

```
[OLLAMA:REQUEST] Sending API request
  - Endpoint: https://ollama.com/api/chat
  - Model: gemma3:12b
  - Prompt length: 650 chars

[OLLAMA:HEADERS]
  - Authorization: Bearer ****
  - Content-Type: application/json
  - User-Agent: AdaptiveCrawler/1.0

[OLLAMA:RESPONSE] Response received
  - Status: 200 OK
  - Latency: 1.8s
  - Content-Length: 412

[OLLAMA:METRICS] Request metrics
  - Model: gemma3:12b
  - Input tokens: ~150
  - Output tokens: ~80
  - Total time: 1.8s
```

---

## Configuration

### Environment Variables

```bash
# Required
export OLLAMA_CLOUD_API_KEY="your-api-key"

# Optional
export OLLAMA_CLOUD_MODEL="gemma3:12b"      # Default model
export OLLAMA_CLOUD_TIMEOUT="30"            # Request timeout (seconds)
export OLLAMA_CLOUD_MAX_RETRIES="3"         # Max retry attempts
export OLLAMA_CLOUD_TEMPERATURE="0.3"       # Generation temperature
export OLLAMA_CLOUD_MAX_TOKENS="500"        # Max output tokens

# Embedding model
export CRAWLER_EMBEDDING_MODEL="all-MiniLM-L6-v2"
export CRAWLER_ENABLE_ML="true"
export CRAWLER_BREAKING_THRESHOLD="0.70"
```

### YAML Configuration

```yaml
ml:
  # Embeddings
  embedding_model: all-MiniLM-L6-v2
  enable_embeddings: true
  cache_embeddings: true

  # Ollama Cloud LLM
  llm:
    provider: ollama-cloud
    model: gemma3:12b
    api_key: ${OLLAMA_CLOUD_API_KEY}
    timeout: 30
    max_retries: 3
    temperature: 0.3
    max_tokens: 500

  # Classification
  classifier:
    backend: lightgbm  # or xgboost, logistic_regression
    model_path: ./models/classifier.pkl

  # Change detection
  change_detection:
    breaking_threshold: 0.70
    cosmetic_threshold: 0.95
    enable_drift_detection: true
```

---

## Data Models

### StructureEmbedding

```python
@dataclass
class StructureEmbedding:
    """Embedding representation of a page structure."""

    domain: str
    page_type: str
    variant_id: str

    # Embedding vector
    vector: np.ndarray          # Shape: (384,) or (768,)
    dimensions: int             # 384 or 768

    # Metadata
    model_name: str             # "all-MiniLM-L6-v2"
    description: str            # Text used to generate embedding
    generated_at: datetime

    # Normalization
    normalized: bool = True     # L2 normalized
```

### MLChangeResult

```python
@dataclass
class MLChangeResult:
    """Result of ML-based change detection."""

    # Similarity
    similarity: float           # 0-1, cosine similarity
    classification: str         # cosmetic, minor, moderate, breaking

    # Impact
    breaking: bool
    confidence: float

    # Analysis
    semantic_shift: str         # description of change
    likely_causes: list[str]
    recommended_action: str

    # Embeddings used
    old_embedding: StructureEmbedding
    new_embedding: StructureEmbedding
```

### DriftAnalysis

```python
@dataclass
class DriftAnalysis:
    """Analysis of structure drift from baseline."""

    domain: str
    page_type: str

    # Drift metrics
    current_similarity: float   # Similarity to baseline
    drift_amount: float         # 1 - similarity
    drift_level: str            # low, medium, high

    # Trend
    trend: str                  # stable, increasing, decreasing
    previous_drift: float | None

    # Baseline info
    baseline_date: datetime
    current_date: datetime
    days_since_baseline: int

    # Recommendations
    action_needed: bool
    recommendation: str
```

---

## Usage Examples

### Basic Embedding Generation

```python
from crawler.ml.embeddings import StructureEmbeddingModel

# Initialize
model = StructureEmbeddingModel(verbose=True)

# Generate embedding
embedding = model.embed_structure(page_structure)
print(f"Embedding shape: {embedding.vector.shape}")
```

### Using Ollama Cloud for Descriptions

```python
from crawler.ml.embeddings import LLMDescriptionGenerator

# Initialize
generator = LLMDescriptionGenerator(
    model="gemma3:12b",
    api_key=os.environ["OLLAMA_CLOUD_API_KEY"],
    verbose=True
)

# Generate description
description = await generator.generate(page_structure)
print(description)
```

### ML Change Detection

```python
from crawler.ml.embeddings import MLChangeDetector, StructureEmbeddingModel

# Initialize
embedding_model = StructureEmbeddingModel(verbose=True)
detector = MLChangeDetector(embedding_model=embedding_model, verbose=True)

# Detect changes
result = detector.detect_change(old_structure, new_structure)

if result.breaking:
    print(f"Breaking change detected! Similarity: {result.similarity}")
    print(f"Causes: {result.likely_causes}")
else:
    print(f"Non-breaking change. Classification: {result.classification}")
```

### Page Classification

```python
from crawler.ml.embeddings import StructureClassifier

# Initialize (loads pre-trained model)
classifier = StructureClassifier(backend="lightgbm", verbose=True)

# Predict page type
page_type, confidence = classifier.predict(page_structure)
print(f"Page type: {page_type} (confidence: {confidence:.2f})")
```

---

## Performance

| Operation | Typical Latency | Notes |
|-----------|-----------------|-------|
| Embedding model load | 1-2s | Once at startup |
| Embedding generation | 50-100ms | Per structure |
| Batch embedding (10) | 150-200ms | More efficient |
| Similarity computation | <1ms | Vector math |
| Ollama Cloud request | 1-3s | Network dependent |
| Classification | 5-10ms | After feature extraction |
| Feature extraction | 10-20ms | Per structure |

### Optimization Tips

1. **Batch embeddings**: Use `embed_structures_batch()` for multiple structures
2. **Cache embeddings**: Enable `cache_embeddings=True` to avoid recomputation
3. **Preload model**: Initialize model at startup, not per-request
4. **Use rules fallback**: Fall back to rules-based descriptions if Ollama Cloud is slow

---

## Troubleshooting

### Common Issues

**Ollama Cloud Authentication Failed**
```
[LLM:ERROR] API request failed
  - Status: 401 Unauthorized
  - Message: Invalid API key

Solution:
1. Verify OLLAMA_CLOUD_API_KEY is set
2. Check key is valid at https://ollama.com/settings
3. Ensure no extra whitespace in key
```

**Embedding Model Load Failed**
```
[EMBED:ERROR] Failed to load model
  - Model: all-MiniLM-L6-v2
  - Error: Connection timeout

Solution:
1. Check internet connection
2. Model will be cached after first download
3. Set HF_HUB_OFFLINE=1 to use cached model
```

**Low Classification Confidence**
```
[CLASSIFY:RESULT] Low confidence prediction
  - Confidence: 0.45
  - Recommendation: Manual review

Solution:
1. Page may have unusual structure
2. Retrain classifier with more examples
3. Add page to training data
```
