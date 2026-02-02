# AGENTS.md - Core Module

Complete specification for the core orchestration module of the Adaptive Structure Fingerprinting System.

---

## Module Purpose

The core module provides:
- Main analyzer orchestrator
- HTTP fetching with compliance
- Verbose logging utilities

---

## Files to Generate

```
fingerprint/core/
├── __init__.py
├── analyzer.py         # Main orchestrator
├── fetcher.py          # HTTP fetching
└── verbose.py          # Logging utilities (defined in main AGENTS.md)
```

---

## fingerprint/core/__init__.py

```python
"""
Core module - Main orchestration for fingerprinting system.
"""

from fingerprint.core.analyzer import StructureAnalyzer
from fingerprint.core.fetcher import HTTPFetcher
from fingerprint.core.verbose import VerboseLogger, get_logger, set_logger

__all__ = [
    "StructureAnalyzer",
    "HTTPFetcher",
    "VerboseLogger",
    "get_logger",
    "set_logger",
]
```

---

## fingerprint/core/analyzer.py

```python
"""
Main analyzer orchestrator.

Coordinates fingerprinting operations across all modes:
- Rules-based: Uses StructureAnalyzer from adaptive module
- ML-based: Uses embeddings and Ollama Cloud from ml module
- Adaptive: Intelligently selects mode based on triggers

Verbose logging pattern:
[ANALYZER:OPERATION] Message
"""

import time
from urllib.parse import urlparse

from fingerprint.config import Config
from fingerprint.core.fetcher import HTTPFetcher
from fingerprint.core.verbose import get_logger
from fingerprint.models import (
    ChangeAnalysis,
    ChangeClassification,
    EscalationTrigger,
    FingerprintMode,
    PageStructure,
)
from fingerprint.adaptive.structure_analyzer import DOMStructureAnalyzer
from fingerprint.adaptive.change_detector import ChangeDetector
from fingerprint.ml.embeddings import EmbeddingGenerator
from fingerprint.ml.ollama_client import OllamaCloudClient
from fingerprint.storage.structure_store import StructureStore


class StructureAnalyzer:
    """
    Main orchestrator for all fingerprinting operations.

    Usage:
        config = load_config()
        analyzer = StructureAnalyzer(config)

        # Analyze a URL
        structure = await analyzer.analyze_url("https://example.com")

        # Compare with stored version
        changes = await analyzer.compare_with_stored("https://example.com")
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger()

        # Initialize components
        self.fetcher = HTTPFetcher(config.http)
        self.dom_analyzer = DOMStructureAnalyzer()
        self.change_detector = ChangeDetector(config.thresholds)
        self.embedding_generator = EmbeddingGenerator(config.embeddings)
        self.ollama_client = OllamaCloudClient(config.ollama_cloud)
        self.structure_store = StructureStore(config.redis)

        self.logger.info(
            "ANALYZER", "INIT",
            f"Initialized with mode={config.mode}",
            redis_url=config.redis.url,
            ollama_enabled=config.ollama_cloud.enabled,
        )

    async def analyze_url(self, url: str) -> PageStructure:
        """
        Fetch and analyze URL, returning structure fingerprint.

        Args:
            url: URL to analyze

        Returns:
            PageStructure with complete fingerprint

        Verbose output:
            [ANALYZER:FETCH] Fetching https://example.com
              - status: 200
              - content_length: 45230
            [ANALYZER:ANALYZE] Analyzing structure
              - tags: 156
              - classes: 89
              - landmarks: 5
        """
        start_time = time.perf_counter()

        # Parse domain
        parsed = urlparse(url)
        domain = parsed.netloc

        self.logger.info("ANALYZER", "FETCH", f"Fetching {url}")

        # Fetch HTML
        html = await self.fetcher.fetch(url)

        self.logger.debug(
            "ANALYZER", "FETCH",
            "Fetch complete",
            content_length=len(html),
        )

        # Analyze structure
        self.logger.info("ANALYZER", "ANALYZE", "Analyzing structure")

        structure = self.dom_analyzer.analyze(html, domain)

        duration_ms = (time.perf_counter() - start_time) * 1000

        self.logger.info(
            "ANALYZER", "RESULT",
            f"Analysis complete in {duration_ms:.1f}ms",
            domain=structure.domain,
            page_type=structure.page_type,
            tags=len(structure.tag_hierarchy.tag_counts) if structure.tag_hierarchy else 0,
            classes=len(structure.css_class_map),
            landmarks=len(structure.semantic_landmarks),
        )

        return structure

    async def compare_with_stored(self, url: str) -> ChangeAnalysis:
        """
        Compare current URL structure with stored version.

        Uses configured mode (rules, ml, adaptive) to perform comparison.

        Args:
            url: URL to compare

        Returns:
            ChangeAnalysis with similarity and change details

        Verbose output (adaptive mode):
            [ANALYZER:COMPARE] Comparing https://example.com (mode=adaptive)
            [ADAPTIVE:START] Running rules-based comparison
            [ADAPTIVE:ANALYZE] Checking escalation triggers
              - class_volatility: 0.12 (threshold: 0.15)
              - rules_similarity: 0.92 (threshold: 0.80)
            [ADAPTIVE:RESULT] No escalation needed
            [ANALYZER:RESULT] Comparison complete
              - mode_used: rules
              - similarity: 0.92
              - classification: cosmetic
        """
        start_time = time.perf_counter()

        self.logger.info(
            "ANALYZER", "COMPARE",
            f"Comparing {url} (mode={self.config.mode})",
        )

        # Get current structure
        current = await self.analyze_url(url)

        # Get stored structure
        stored = await self.structure_store.get(
            current.domain, current.page_type
        )

        if stored is None:
            # No stored version - save current and return
            await self.structure_store.save(current)
            self.logger.info(
                "ANALYZER", "RESULT",
                "No stored version found, saved current",
            )
            return ChangeAnalysis(
                similarity=1.0,
                mode_used=FingerprintMode(self.config.mode),
                classification=ChangeClassification.COSMETIC,
                breaking=False,
                reason="No previous version to compare",
            )

        # Compare based on mode
        mode = FingerprintMode(self.config.mode)

        if mode == FingerprintMode.RULES:
            result = await self._compare_rules(stored, current)
        elif mode == FingerprintMode.ML:
            result = await self._compare_ml(stored, current)
        else:  # adaptive
            result = await self._compare_adaptive(stored, current)

        duration_ms = (time.perf_counter() - start_time) * 1000
        result.duration_ms = duration_ms

        self.logger.info(
            "ANALYZER", "RESULT",
            f"Comparison complete in {duration_ms:.1f}ms",
            mode_used=result.mode_used.value,
            similarity=f"{result.similarity:.3f}",
            classification=result.classification.value,
            breaking=result.breaking,
            escalated=result.escalated,
        )

        # Update stored structure if breaking change
        if result.breaking:
            await self.structure_store.save(current)
            self.logger.info(
                "ANALYZER", "UPDATE",
                "Breaking change detected, updated stored structure",
            )

        return result

    async def _compare_rules(
        self,
        stored: PageStructure,
        current: PageStructure,
    ) -> ChangeAnalysis:
        """Compare using rules-based fingerprinting."""
        self.logger.debug("ANALYZER", "RULES", "Using rules-based comparison")

        analysis = self.change_detector.detect_changes(stored, current)
        analysis.mode_used = FingerprintMode.RULES

        return analysis

    async def _compare_ml(
        self,
        stored: PageStructure,
        current: PageStructure,
    ) -> ChangeAnalysis:
        """Compare using ML embeddings."""
        self.logger.debug("ANALYZER", "ML", "Using ML-based comparison")

        # Generate embeddings
        stored_embedding = await self.embedding_generator.generate(stored)
        current_embedding = await self.embedding_generator.generate(current)

        # Calculate similarity
        similarity = self.embedding_generator.cosine_similarity(
            stored_embedding.vector,
            current_embedding.vector,
        )

        # Classify change
        classification = self.change_detector.classify_similarity(similarity)
        breaking = classification == ChangeClassification.BREAKING

        return ChangeAnalysis(
            similarity=similarity,
            mode_used=FingerprintMode.ML,
            classification=classification,
            breaking=breaking,
        )

    async def _compare_adaptive(
        self,
        stored: PageStructure,
        current: PageStructure,
    ) -> ChangeAnalysis:
        """
        Adaptive comparison - starts with rules, escalates to ML if needed.

        Escalation triggers:
        - CLASS_VOLATILITY: >15% of classes changed
        - RULES_UNCERTAINTY: Rules similarity < 0.80
        - KNOWN_VOLATILE: Domain flagged as volatile
        - RENAME_PATTERN: Detected rename patterns
        """
        self.logger.info("ADAPTIVE", "START", "Running rules-based comparison")

        # Step 1: Run rules-based comparison
        rules_result = await self._compare_rules(stored, current)

        # Step 2: Check escalation triggers
        self.logger.debug("ADAPTIVE", "ANALYZE", "Checking escalation triggers")

        triggers: list[EscalationTrigger] = []

        # Check class volatility
        class_volatility = self._calculate_class_volatility(stored, current)
        class_threshold = self.config.adaptive.class_change_threshold

        self.logger.debug(
            "ADAPTIVE", "TRIGGER",
            f"CLASS_VOLATILITY: {class_volatility:.2f} (threshold: {class_threshold})",
        )

        if class_volatility > class_threshold:
            triggers.append(EscalationTrigger(
                name="CLASS_VOLATILITY",
                reason=f"{class_volatility:.0%} of CSS classes changed",
                threshold=class_threshold,
                actual_value=class_volatility,
            ))

        # Check rules uncertainty
        rules_threshold = self.config.adaptive.rules_uncertainty_threshold

        self.logger.debug(
            "ADAPTIVE", "TRIGGER",
            f"RULES_UNCERTAINTY: {rules_result.similarity:.2f} (threshold: {rules_threshold})",
        )

        if rules_result.similarity < rules_threshold:
            triggers.append(EscalationTrigger(
                name="RULES_UNCERTAINTY",
                reason=f"Rules similarity {rules_result.similarity:.2f} below threshold",
                threshold=rules_threshold,
                actual_value=rules_result.similarity,
            ))

        # Check for known volatile domain
        is_volatile = await self.structure_store.is_volatile(current.domain)
        if is_volatile:
            triggers.append(EscalationTrigger(
                name="KNOWN_VOLATILE",
                reason="Domain flagged as volatile from history",
            ))

        # Check for rename patterns
        rename_detected = self._detect_rename_pattern(stored, current)
        if rename_detected:
            triggers.append(EscalationTrigger(
                name="RENAME_PATTERN",
                reason="CSS class rename pattern detected",
            ))

        # Step 3: Escalate if triggers fired
        if triggers:
            self.logger.info(
                "ADAPTIVE", "ESCALATE",
                f"Escalating to ML ({len(triggers)} triggers)",
            )
            for t in triggers:
                self.logger.debug("ADAPTIVE", "TRIGGER", f"  - {t.name}: {t.reason}")

            ml_result = await self._compare_ml(stored, current)
            ml_result.escalated = True
            ml_result.escalation_triggers = triggers

            return ml_result

        self.logger.info("ADAPTIVE", "RESULT", "No escalation needed, using rules result")
        rules_result.escalated = False
        return rules_result

    def _calculate_class_volatility(
        self,
        stored: PageStructure,
        current: PageStructure,
    ) -> float:
        """Calculate percentage of CSS classes that changed."""
        stored_classes = set(stored.css_class_map.keys())
        current_classes = set(current.css_class_map.keys())

        if not stored_classes:
            return 0.0

        added = current_classes - stored_classes
        removed = stored_classes - current_classes
        changed = len(added) + len(removed)

        return changed / len(stored_classes)

    def _detect_rename_pattern(
        self,
        stored: PageStructure,
        current: PageStructure,
    ) -> bool:
        """
        Detect if classes were renamed vs removed.

        Patterns checked:
        - Prefix removal: 'legacy-header' -> 'header'
        - Prefix addition: 'header' -> 'v2-header'
        - Hash suffix changes: 'btn_abc123' -> 'btn_def456'
        """
        stored_classes = set(stored.css_class_map.keys())
        current_classes = set(current.css_class_map.keys())

        removed = stored_classes - current_classes
        added = current_classes - stored_classes

        if not removed or not added:
            return False

        # Check for hash suffix pattern (CSS modules)
        import re
        hash_pattern = re.compile(r'^(.+?)_[a-f0-9]{6,}$')

        removed_bases = set()
        for cls in removed:
            match = hash_pattern.match(cls)
            if match:
                removed_bases.add(match.group(1))

        added_bases = set()
        for cls in added:
            match = hash_pattern.match(cls)
            if match:
                added_bases.add(match.group(1))

        # If same base names with different hashes, it's a rename
        common_bases = removed_bases & added_bases
        if len(common_bases) > len(removed) * 0.5:
            return True

        return False

    async def generate_description(self, structure: PageStructure) -> str:
        """
        Generate LLM description of page structure using Ollama Cloud.

        Args:
            structure: PageStructure to describe

        Returns:
            Human-readable description from LLM

        Verbose output:
            [OLLAMA:REQUEST] Generating description
              - model: gemma3:12b
              - domain: example.com
            [OLLAMA:RESPONSE] Description generated
              - tokens: 156
              - duration_ms: 1234
        """
        if not self.config.ollama_cloud.enabled:
            return "Ollama Cloud not enabled"

        description = await self.ollama_client.describe_structure(structure)
        structure.description = description

        return description
```

---

## fingerprint/core/fetcher.py

```python
"""
HTTP fetcher with full compliance pipeline.

All fetches pass through the ethical compliance pipeline:
1. CFAA authorization check
2. ToS check
3. robots.txt check (RFC 9309)
4. Rate limiting with Crawl-delay
5. HTTP fetch
6. Anti-bot detection
7. GDPR/CCPA compliance

Verbose logging pattern:
[FETCH:OPERATION] Message
"""

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from fingerprint.config import Config
from fingerprint.core.verbose import get_logger
from fingerprint.compliance.robots_parser import RobotsChecker
from fingerprint.compliance.rate_limiter import RateLimiter
from fingerprint.compliance.bot_detector import BotDetector
from fingerprint.legal.cfaa_checker import CFAAChecker
from fingerprint.legal.tos_checker import ToSChecker
from fingerprint.legal.gdpr_handler import GDPRHandler
from fingerprint.legal.ccpa_handler import CCPAHandler
from fingerprint.exceptions import (
    FetchError,
    HTTPStatusError,
    HTTPTimeoutError,
    RobotsBlockedError,
    CFAAViolationError,
    ToSViolationError,
    BotDetectedError,
)


@dataclass
class FetchResult:
    """Result of a fetch operation."""
    url: str
    success: bool
    content: str = ""
    status_code: int = 0
    blocked: bool = False
    block_reason: str = ""
    response_time: float = 0.0

    @classmethod
    def success_result(cls, url: str, content: str, status_code: int, response_time: float) -> "FetchResult":
        return cls(url=url, success=True, content=content, status_code=status_code, response_time=response_time)

    @classmethod
    def blocked_result(cls, url: str, reason: str) -> "FetchResult":
        return cls(url=url, success=False, blocked=True, block_reason=reason)

    @classmethod
    def error_result(cls, url: str, error: str) -> "FetchResult":
        return cls(url=url, success=False, block_reason=error)


class ComplianceFetcher:
    """
    HTTP fetcher with full ethical compliance pipeline.

    IMPORTANT: All fetches MUST go through this fetcher.
    Direct HTTP requests bypass compliance and are forbidden.

    Usage:
        fetcher = ComplianceFetcher(config)
        result = await fetcher.fetch("https://example.com")
        if result.success:
            html = result.content
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger()

        # HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.http.timeout),
            headers={"User-Agent": config.http.user_agent},
            follow_redirects=True,
        )

        # Compliance components
        self.cfaa_checker = CFAAChecker(config)
        self.tos_checker = ToSChecker(config)
        self.robots_checker = RobotsChecker(config)
        self.rate_limiter = RateLimiter(config)
        self.bot_detector = BotDetector(config)
        self.gdpr_handler = GDPRHandler(config)
        self.ccpa_handler = CCPAHandler(config)

        self.logger.info(
            "FETCH", "INIT",
            "Compliance fetcher initialized",
            user_agent=config.http.user_agent,
        )

    async def fetch(self, url: str) -> FetchResult:
        """
        Fetch URL through full compliance pipeline.

        Pipeline order:
        1. CFAA check -> 2. ToS check -> 3. robots.txt -> 4. Rate limit
        5. HTTP fetch -> 6. Anti-bot -> 7. GDPR/CCPA

        Args:
            url: URL to fetch

        Returns:
            FetchResult with content or block reason

        Verbose output:
            [CFAA:CHECK] Checking authorization
            [CFAA:AUTHORIZED] Access authorized
            [TOS:CHECK] Checking Terms of Service
            [TOS:ALLOWED] No restrictive directives
            [ROBOTS:CHECK] Checking robots.txt
            [ROBOTS:ALLOWED] Path allowed
            [RATELIMIT:ACQUIRE] Acquiring slot
            [RATELIMIT:WAIT] Waiting 1.5s
            [FETCH:REQUEST] GET https://example.com
            [FETCH:RESPONSE] 200 OK (1.2s, 45KB)
            [ANTIBOT:CHECK] Scanning for bot detection
            [ANTIBOT:CLEAR] No detection found
            [GDPR:SCAN] Scanning for PII
            [GDPR:CLEAR] No PII detected
        """
        domain = self._get_domain(url)

        # ===== 1. CFAA AUTHORIZATION CHECK =====
        cfaa_result = await self.cfaa_checker.is_authorized(url)
        if not cfaa_result.authorized:
            self.logger.warn(
                "FETCH", "CFAA_BLOCKED",
                f"CFAA blocked: {cfaa_result.reason}",
            )
            return FetchResult.blocked_result(url, f"CFAA: {cfaa_result.reason}")

        # ===== 2. robots.txt CHECK (RFC 9309) =====
        if not await self.robots_checker.is_allowed(url):
            self.logger.info("FETCH", "ROBOTS_BLOCKED", "Blocked by robots.txt")
            return FetchResult.blocked_result(url, "robots.txt")

        # Get Crawl-delay from robots.txt
        crawl_delay = await self.robots_checker.get_crawl_delay(domain)
        if crawl_delay:
            self.rate_limiter.set_crawl_delay(domain, crawl_delay)

        # ===== 3. RATE LIMITING =====
        await self.rate_limiter.acquire(domain)

        # ===== 4. HTTP FETCH =====
        try:
            import time
            start_time = time.time()

            self.logger.info("FETCH", "REQUEST", f"GET {url}")
            response = await self.client.get(url)
            response_time = time.time() - start_time

            self.logger.info(
                "FETCH", "RESPONSE",
                f"Status {response.status_code} ({response_time:.2f}s, {len(response.content)} bytes)",
            )

            # Check for HTTP errors
            if response.status_code >= 400:
                self.rate_limiter.report_error(domain, response.status_code)

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        self.rate_limiter.backoff(domain, float(retry_after))
                    else:
                        self.rate_limiter.backoff(domain)
                    return FetchResult.blocked_result(url, f"Rate limited (429)")

                return FetchResult.error_result(url, f"HTTP {response.status_code}")

        except httpx.TimeoutException as e:
            self.rate_limiter.report_error(domain)
            self.logger.error("FETCH", "TIMEOUT", str(e))
            raise HTTPTimeoutError(f"Timeout fetching {url}")

        except httpx.HTTPError as e:
            self.rate_limiter.report_error(domain)
            self.logger.error("FETCH", "ERROR", str(e))
            raise FetchError(f"Error fetching {url}: {e}")

        # ===== 5. ANTI-BOT DETECTION =====
        bot_check = await self.bot_detector.check(response)
        if bot_check.detected:
            self.rate_limiter.backoff(domain)
            self.logger.warn(
                "FETCH", "BOT_DETECTED",
                f"Bot detection: {bot_check.detection_type}",
            )

            if self.bot_detector.should_stop(bot_check):
                return FetchResult.blocked_result(
                    url,
                    f"Bot detected: {bot_check.detection_type}",
                )

        # ===== 6. ToS CHECK (on response) =====
        tos_result = await self.tos_checker.check(url, response)
        if not tos_result.allowed:
            self.logger.info(
                "FETCH", "TOS_BLOCKED",
                f"ToS directive: {tos_result.directive}",
            )
            return FetchResult.blocked_result(url, f"ToS: {tos_result.directive}")

        # ===== 7. GDPR/CCPA COMPLIANCE =====
        if self.config.legal.gdpr.enabled:
            response = await self.gdpr_handler.process(response)

        if self.config.legal.ccpa.enabled:
            response = await self.ccpa_handler.process(response)

        # ===== SUCCESS =====
        self.rate_limiter.report_success(domain, response_time)

        return FetchResult.success_result(
            url=url,
            content=response.text,
            status_code=response.status_code,
            response_time=response_time,
        )

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc

    async def close(self) -> None:
        """Close the HTTP client and cleanup."""
        await self.client.aclose()
        self.logger.debug("FETCH", "CLOSE", "Compliance fetcher closed")


# IMPORTANT: Legacy HTTPFetcher for internal use only
# All external fetches MUST use ComplianceFetcher

class HTTPFetcher:
    """
    Low-level HTTP client for internal use ONLY.

    WARNING: This bypasses the compliance pipeline.
    Only use for:
    - Fetching robots.txt
    - Internal health checks

    For all other fetches, use ComplianceFetcher.
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger()

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.http.timeout),
            headers={"User-Agent": config.http.user_agent},
            follow_redirects=True,
        )

    async def fetch_raw(self, url: str) -> httpx.Response:
        """
        Fetch URL without compliance checks.

        WARNING: Only for internal use (robots.txt, etc.)
        """
        self.logger.debug("FETCH", "RAW_REQUEST", f"GET {url} (bypassing compliance)")
        return await self.client.get(url)

    async def close(self) -> None:
        await self.client.aclose()
```

---

## Verbose Logging

The verbose logging system is defined in the main AGENTS.md under `fingerprint/core/verbose.py`. All core module operations use the following pattern:

```python
self.logger.info("MODULE", "OPERATION", "Message", key=value)
```

### Core Module Operations

| Module | Operations |
|--------|------------|
| ANALYZER | INIT, FETCH, ANALYZE, COMPARE, RESULT, UPDATE |
| FETCH | INIT, REQUEST, RESPONSE, RETRY, CLOSE |
| ADAPTIVE | START, ANALYZE, TRIGGER, ESCALATE, RESULT |

### Example Output

```
[2024-01-15T10:30:00Z] [ANALYZER:INIT] Initialized with mode=adaptive
  - redis_url: redis://localhost:6379/0
  - ollama_enabled: true

[2024-01-15T10:30:01Z] [ANALYZER:FETCH] Fetching https://example.com

[2024-01-15T10:30:02Z] [FETCH:REQUEST] GET https://example.com

[2024-01-15T10:30:03Z] [FETCH:RESPONSE] Status 200, 45230 bytes
  - status: 200
  - content_type: text/html

[2024-01-15T10:30:03Z] [ANALYZER:ANALYZE] Analyzing structure

[2024-01-15T10:30:03Z] [ANALYZER:RESULT] Analysis complete in 156.3ms
  - domain: example.com
  - page_type: article
  - tags: 156
  - classes: 89
  - landmarks: 5
```
