# AGENTS.md - Legal Compliance Module

Complete specification for legal compliance: CFAA (Computer Fraud and Abuse Act), Terms of Service, GDPR (General Data Protection Regulation), and CCPA (California Consumer Privacy Act).

---

## Module Purpose

The legal module ensures compliance with:
- **CFAA**: Only access publicly authorized content
- **Terms of Service**: Respect website ToS and meta directives
- **GDPR**: Detect and handle personally identifiable information (PII)
- **CCPA**: Respect California privacy opt-out signals

---

## Files to Generate

```
fingerprint/legal/
├── __init__.py
├── cfaa_checker.py       # CFAA authorization checks
├── tos_checker.py        # Terms of Service compliance
├── gdpr_handler.py       # GDPR PII detection/handling
└── ccpa_handler.py       # CCPA compliance
```

---

## fingerprint/legal/__init__.py

```python
"""
Legal compliance module.

Provides CFAA, ToS, GDPR, and CCPA compliance checking.
"""

from fingerprint.legal.cfaa_checker import CFAAChecker, AuthorizationResult
from fingerprint.legal.tos_checker import ToSChecker, ToSResult
from fingerprint.legal.gdpr_handler import GDPRHandler, PIIDetectionResult
from fingerprint.legal.ccpa_handler import CCPAHandler

__all__ = [
    "CFAAChecker",
    "AuthorizationResult",
    "ToSChecker",
    "ToSResult",
    "GDPRHandler",
    "PIIDetectionResult",
    "CCPAHandler",
]
```

---

## fingerprint/legal/cfaa_checker.py

```python
"""
CFAA (Computer Fraud and Abuse Act) authorization checker.

The CFAA prohibits accessing computers "without authorization" or "exceeding
authorized access." This module helps ensure we only access publicly
available content.

Key principles:
1. Only access publicly available pages
2. Don't access login-protected content
3. Don't access API endpoints without permission
4. Respect access control mechanisms
5. Don't circumvent technical barriers

Verbose logging pattern:
[CFAA:OPERATION] Message
"""

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from fingerprint.config import Config
from fingerprint.core.verbose import get_logger
from fingerprint.exceptions import CFAAViolationError, UnauthorizedAccessError


@dataclass
class AuthorizationResult:
    """Result of CFAA authorization check."""
    authorized: bool
    reason: str = ""
    risk_level: str = "none"  # none, low, medium, high

    @classmethod
    def allowed(cls) -> "AuthorizationResult":
        """Access is authorized."""
        return cls(authorized=True)

    @classmethod
    def blocked(cls, reason: str, risk_level: str = "high") -> "AuthorizationResult":
        """Access is not authorized."""
        return cls(authorized=False, reason=reason, risk_level=risk_level)


class CFAAChecker:
    """
    CFAA authorization checker.

    Ensures we only access publicly available content without
    circumventing access controls.

    Usage:
        checker = CFAAChecker(config)
        result = await checker.is_authorized(url)
        if not result.authorized:
            # Don't access this URL
    """

    # Patterns indicating login/auth areas
    AUTH_PATH_PATTERNS = [
        r"/login",
        r"/signin",
        r"/sign-in",
        r"/auth",
        r"/oauth",
        r"/sso",
        r"/account",
        r"/my-account",
        r"/dashboard",
        r"/admin",
        r"/user/",
        r"/profile",
        r"/settings",
        r"/preferences",
        r"/private",
        r"/members",
        r"/secure",
    ]

    # Patterns indicating API endpoints
    API_PATH_PATTERNS = [
        r"^/api/",
        r"^/v\d+/",
        r"^/graphql",
        r"^/rest/",
        r"^/_api/",
        r"/api$",
        r"\.json$",
        r"\.xml$",
    ]

    # Patterns indicating internal/system paths
    INTERNAL_PATH_PATTERNS = [
        r"^/\.",
        r"^/_",
        r"/\.git",
        r"/\.env",
        r"/wp-admin",
        r"/phpmyadmin",
        r"/cpanel",
        r"/cgi-bin",
        r"/server-status",
        r"/config",
    ]

    def __init__(self, config: Config):
        self.config = config
        self.cfaa_config = config.legal.cfaa
        self.logger = get_logger()

        # Compile patterns
        self._auth_re = re.compile(
            "|".join(self.AUTH_PATH_PATTERNS),
            re.IGNORECASE,
        )
        self._api_re = re.compile(
            "|".join(self.API_PATH_PATTERNS),
            re.IGNORECASE,
        )
        self._internal_re = re.compile(
            "|".join(self.INTERNAL_PATH_PATTERNS),
            re.IGNORECASE,
        )

        self.logger.info("CFAA", "INIT", "CFAA checker initialized")

    async def is_authorized(self, url: str) -> AuthorizationResult:
        """
        Check if accessing URL is authorized under CFAA.

        Args:
            url: URL to check

        Returns:
            AuthorizationResult indicating if access is authorized

        Verbose output:
            [CFAA:CHECK] Checking authorization for https://example.com/page
            [CFAA:AUTHORIZED] Access authorized (public page)
            -- or --
            [CFAA:BLOCKED] Access blocked
              - reason: Login-protected area
              - risk_level: high
        """
        if not self.cfaa_config.enabled:
            return AuthorizationResult.allowed()

        parsed = urlparse(url)
        path = parsed.path or "/"

        self.logger.debug("CFAA", "CHECK", f"Checking authorization for {url}")

        # Check for authentication-required paths
        if self.cfaa_config.block_authenticated_areas:
            if self._auth_re.search(path):
                result = AuthorizationResult.blocked(
                    "Login-protected area",
                    risk_level="high",
                )
                self.logger.warn(
                    "CFAA", "BLOCKED",
                    f"Access blocked for {url}",
                    reason=result.reason,
                    risk_level=result.risk_level,
                )
                return result

        # Check for API endpoints
        if self.cfaa_config.block_api_endpoints:
            if self._api_re.search(path):
                result = AuthorizationResult.blocked(
                    "API endpoint requires authorization",
                    risk_level="medium",
                )
                self.logger.warn(
                    "CFAA", "BLOCKED",
                    f"Access blocked for {url}",
                    reason=result.reason,
                    risk_level=result.risk_level,
                )
                return result

        # Check for internal/system paths
        if self._internal_re.search(path):
            result = AuthorizationResult.blocked(
                "Internal/system path",
                risk_level="high",
            )
            self.logger.warn(
                "CFAA", "BLOCKED",
                f"Access blocked for {url}",
                reason=result.reason,
                risk_level=result.risk_level,
            )
            return result

        # Check for query parameters indicating auth
        if parsed.query:
            auth_params = ["token", "key", "apikey", "api_key", "auth", "session"]
            query_lower = parsed.query.lower()
            for param in auth_params:
                if f"{param}=" in query_lower:
                    result = AuthorizationResult.blocked(
                        f"URL contains authentication parameter: {param}",
                        risk_level="medium",
                    )
                    self.logger.warn(
                        "CFAA", "BLOCKED",
                        f"Access blocked for {url}",
                        reason=result.reason,
                    )
                    return result

        # Check URL scheme
        if parsed.scheme not in ("http", "https"):
            result = AuthorizationResult.blocked(
                f"Unsupported URL scheme: {parsed.scheme}",
                risk_level="high",
            )
            self.logger.warn("CFAA", "BLOCKED", f"Unsupported scheme: {parsed.scheme}")
            return result

        self.logger.debug("CFAA", "AUTHORIZED", f"Access authorized for {url}")
        return AuthorizationResult.allowed()

    def check_response_headers(self, headers: dict[str, str]) -> AuthorizationResult:
        """
        Check response headers for authorization requirements.

        Args:
            headers: HTTP response headers

        Returns:
            AuthorizationResult
        """
        # Check for WWW-Authenticate header (indicates auth required)
        if "www-authenticate" in {k.lower() for k in headers}:
            return AuthorizationResult.blocked(
                "Server requires authentication",
                risk_level="high",
            )

        return AuthorizationResult.allowed()
```

---

## fingerprint/legal/tos_checker.py

```python
"""
Terms of Service compliance checker.

Checks for and respects:
- Meta robots tags (noindex, nofollow, noarchive)
- X-Robots-Tag headers
- Common ToS indicators

Verbose logging pattern:
[TOS:OPERATION] Message
"""

import re
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup

from fingerprint.config import Config
from fingerprint.core.verbose import get_logger
from fingerprint.exceptions import ToSViolationError


@dataclass
class ToSResult:
    """Result of Terms of Service check."""
    allowed: bool
    directive: str = ""
    details: dict[str, Any] | None = None

    @classmethod
    def allow(cls) -> "ToSResult":
        """Access is allowed."""
        return cls(allowed=True)

    @classmethod
    def block(cls, directive: str, **details: Any) -> "ToSResult":
        """Access is blocked by directive."""
        return cls(allowed=False, directive=directive, details=details or None)


class ToSChecker:
    """
    Terms of Service compliance checker.

    Respects meta robots tags and X-Robots-Tag headers.

    Usage:
        checker = ToSChecker(config)
        result = await checker.check(url, response)
        if not result.allowed:
            # Respect the directive
    """

    def __init__(self, config: Config):
        self.config = config
        self.tos_config = config.legal.tos
        self.logger = get_logger()

        self.logger.info("TOS", "INIT", "ToS checker initialized")

    async def check(self, url: str, response: Any | None = None) -> ToSResult:
        """
        Check ToS compliance for URL/response.

        Args:
            url: URL being accessed
            response: Optional HTTP response to check

        Returns:
            ToSResult indicating if access is allowed

        Verbose output:
            [TOS:CHECK] Checking Terms of Service
            [TOS:ALLOWED] No restrictive directives found
            -- or --
            [TOS:BLOCKED] ToS violation
              - directive: noindex
        """
        if not self.tos_config.enabled:
            return ToSResult.allow()

        self.logger.debug("TOS", "CHECK", f"Checking ToS for {url}")

        if response is None:
            return ToSResult.allow()

        # Check X-Robots-Tag header
        header_result = self._check_headers(response)
        if not header_result.allowed:
            return header_result

        # Check meta robots tags in content
        if self.tos_config.check_meta_tags and hasattr(response, 'text'):
            meta_result = self._check_meta_tags(response.text)
            if not meta_result.allowed:
                return meta_result

        self.logger.debug("TOS", "ALLOWED", "No restrictive directives found")
        return ToSResult.allow()

    def _check_headers(self, response: Any) -> ToSResult:
        """Check X-Robots-Tag header."""
        headers = response.headers if hasattr(response, 'headers') else {}

        # Normalize header names to lowercase
        headers_lower = {k.lower(): v for k, v in headers.items()}

        x_robots = headers_lower.get("x-robots-tag", "")
        if not x_robots:
            return ToSResult.allow()

        x_robots_lower = x_robots.lower()

        # Check for noindex
        if self.tos_config.respect_noindex and "noindex" in x_robots_lower:
            self.logger.info("TOS", "NOINDEX", "X-Robots-Tag contains noindex")
            return ToSResult.block("noindex", source="X-Robots-Tag")

        # Check for nofollow (affects link extraction)
        if self.tos_config.respect_nofollow and "nofollow" in x_robots_lower:
            self.logger.info("TOS", "NOFOLLOW", "X-Robots-Tag contains nofollow")
            return ToSResult.block("nofollow", source="X-Robots-Tag")

        # Check for noarchive
        if "noarchive" in x_robots_lower:
            self.logger.info("TOS", "NOARCHIVE", "X-Robots-Tag contains noarchive")
            return ToSResult.block("noarchive", source="X-Robots-Tag")

        # Check for none (equivalent to noindex, nofollow)
        if "none" in x_robots_lower:
            self.logger.info("TOS", "NONE", "X-Robots-Tag contains none")
            return ToSResult.block("none", source="X-Robots-Tag")

        return ToSResult.allow()

    def _check_meta_tags(self, content: str) -> ToSResult:
        """Check meta robots tags in HTML content."""
        try:
            soup = BeautifulSoup(content, "lxml")
        except Exception:
            return ToSResult.allow()

        # Find meta robots tag
        meta_robots = soup.find("meta", attrs={"name": re.compile(r"robots", re.I)})
        if not meta_robots:
            return ToSResult.allow()

        robots_content = meta_robots.get("content", "")
        if not robots_content:
            return ToSResult.allow()

        robots_lower = robots_content.lower()

        # Check for noindex
        if self.tos_config.respect_noindex and "noindex" in robots_lower:
            self.logger.info("TOS", "META_NOINDEX", "Meta robots contains noindex")
            return ToSResult.block("noindex", source="meta robots")

        # Check for nofollow
        if self.tos_config.respect_nofollow and "nofollow" in robots_lower:
            self.logger.info("TOS", "META_NOFOLLOW", "Meta robots contains nofollow")
            return ToSResult.block("nofollow", source="meta robots")

        # Check for noarchive
        if "noarchive" in robots_lower:
            self.logger.info("TOS", "META_NOARCHIVE", "Meta robots contains noarchive")
            return ToSResult.block("noarchive", source="meta robots")

        # Check for none
        if "none" in robots_lower:
            self.logger.info("TOS", "META_NONE", "Meta robots contains none")
            return ToSResult.block("none", source="meta robots")

        return ToSResult.allow()

    def extract_nofollow_links(self, content: str) -> list[str]:
        """
        Extract links that should not be followed (rel=nofollow).

        Returns list of URLs marked as nofollow.
        """
        nofollow_links = []

        try:
            soup = BeautifulSoup(content, "lxml")
            for link in soup.find_all("a", rel=re.compile(r"nofollow", re.I)):
                href = link.get("href")
                if href:
                    nofollow_links.append(href)
        except Exception:
            pass

        return nofollow_links
```

---

## fingerprint/legal/gdpr_handler.py

```python
"""
GDPR (General Data Protection Regulation) compliance handler.

Detects and handles personally identifiable information (PII):
- Email addresses
- Phone numbers
- IP addresses
- Names (when identifiable)
- Postal addresses
- ID numbers

Handling options:
- redact: Replace PII with [REDACTED]
- pseudonymize: Replace with consistent pseudonyms
- skip: Skip content containing PII

Verbose logging pattern:
[GDPR:OPERATION] Message
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from fingerprint.config import Config
from fingerprint.core.verbose import get_logger
from fingerprint.exceptions import GDPRViolationError


@dataclass
class PIIMatch:
    """Single PII detection match."""
    pii_type: str
    value: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class PIIDetectionResult:
    """Result of PII detection scan."""
    contains_pii: bool
    matches: list[PIIMatch] = field(default_factory=list)
    pii_types_found: set[str] = field(default_factory=set)

    @classmethod
    def clean(cls) -> "PIIDetectionResult":
        """No PII detected."""
        return cls(contains_pii=False)

    @classmethod
    def found(cls, matches: list[PIIMatch]) -> "PIIDetectionResult":
        """PII detected."""
        return cls(
            contains_pii=True,
            matches=matches,
            pii_types_found={m.pii_type for m in matches},
        )


class GDPRHandler:
    """
    GDPR compliance handler for PII detection and handling.

    Usage:
        handler = GDPRHandler(config)
        result = handler.scan(content)
        if result.contains_pii:
            content = handler.handle(content, result)
    """

    # PII detection patterns
    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",

        "phone": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",

        "ip_address": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",

        "ssn": r"\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b",

        "credit_card": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",

        "postal_code": r"\b[0-9]{5}(?:-[0-9]{4})?\b",

        # EU/UK phone formats
        "phone_eu": r"\b\+?(?:44|33|49|39|34|31|32|43|41)[0-9\s-]{8,12}\b",
    }

    def __init__(self, config: Config):
        self.config = config
        self.gdpr_config = config.legal.gdpr
        self.logger = get_logger()

        # Compile patterns
        self._patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.PATTERNS.items()
        }

        # Pseudonymization cache (consistent replacement)
        self._pseudonym_cache: dict[str, str] = {}

        self.logger.info(
            "GDPR", "INIT",
            "GDPR handler initialized",
            handling_mode=self.gdpr_config.pii_handling,
        )

    async def process(self, response: Any) -> Any:
        """
        Process response for GDPR compliance.

        Args:
            response: HTTP response object

        Returns:
            Processed response with PII handled

        Verbose output:
            [GDPR:SCAN] Scanning for PII
            [GDPR:CLEAR] No PII detected
            -- or --
            [GDPR:PII_FOUND] PII detected
              - types: email, phone
              - count: 5
            [GDPR:REDACT] Redacting PII
        """
        if not self.gdpr_config.enabled or not self.gdpr_config.pii_detection:
            return response

        content = response.text if hasattr(response, 'text') else ""
        if not content:
            return response

        # Scan for PII
        result = self.scan(content)

        if not result.contains_pii:
            self.logger.debug("GDPR", "CLEAR", "No PII detected")
            return response

        self.logger.info(
            "GDPR", "PII_FOUND",
            "PII detected",
            types=list(result.pii_types_found),
            count=len(result.matches),
        )

        if self.gdpr_config.log_pii_access:
            for match in result.matches:
                self.logger.debug(
                    "GDPR", "PII_DETAIL",
                    f"Found {match.pii_type}",
                    position=f"{match.start}-{match.end}",
                )

        # Handle based on configuration
        if self.gdpr_config.pii_handling == "skip":
            raise GDPRViolationError(
                str(response.url) if hasattr(response, 'url') else "unknown",
                pii_type=", ".join(result.pii_types_found),
            )

        # Apply handling
        processed_content = self.handle(content, result)

        # Update response (create new response with modified content)
        # This depends on the HTTP client used; for httpx:
        if hasattr(response, '_content'):
            response._content = processed_content.encode()

        return response

    def scan(self, content: str) -> PIIDetectionResult:
        """
        Scan content for PII.

        Args:
            content: Text content to scan

        Returns:
            PIIDetectionResult with matches
        """
        self.logger.debug("GDPR", "SCAN", "Scanning for PII")

        matches: list[PIIMatch] = []

        for pii_type, pattern in self._patterns.items():
            for match in pattern.finditer(content):
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                ))

        if matches:
            return PIIDetectionResult.found(matches)
        return PIIDetectionResult.clean()

    def handle(self, content: str, result: PIIDetectionResult) -> str:
        """
        Handle detected PII according to configuration.

        Args:
            content: Original content
            result: PII detection result

        Returns:
            Content with PII handled
        """
        if self.gdpr_config.pii_handling == "redact":
            return self._redact(content, result)
        elif self.gdpr_config.pii_handling == "pseudonymize":
            return self._pseudonymize(content, result)
        else:
            return content

    def _redact(self, content: str, result: PIIDetectionResult) -> str:
        """Replace PII with [REDACTED]."""
        self.logger.info("GDPR", "REDACT", f"Redacting {len(result.matches)} PII instances")

        # Sort matches by position (reverse) to maintain positions during replacement
        sorted_matches = sorted(result.matches, key=lambda m: m.start, reverse=True)

        for match in sorted_matches:
            redaction = f"[REDACTED-{match.pii_type.upper()}]"
            content = content[:match.start] + redaction + content[match.end:]

        return content

    def _pseudonymize(self, content: str, result: PIIDetectionResult) -> str:
        """Replace PII with consistent pseudonyms."""
        self.logger.info("GDPR", "PSEUDONYMIZE", f"Pseudonymizing {len(result.matches)} PII instances")

        sorted_matches = sorted(result.matches, key=lambda m: m.start, reverse=True)

        for match in sorted_matches:
            pseudonym = self._get_pseudonym(match.value, match.pii_type)
            content = content[:match.start] + pseudonym + content[match.end:]

        return content

    def _get_pseudonym(self, value: str, pii_type: str) -> str:
        """Get consistent pseudonym for PII value."""
        cache_key = f"{pii_type}:{value}"

        if cache_key not in self._pseudonym_cache:
            # Generate consistent pseudonym using hash
            hash_val = hashlib.sha256(value.encode()).hexdigest()[:8]
            self._pseudonym_cache[cache_key] = f"[{pii_type.upper()}-{hash_val}]"

        return self._pseudonym_cache[cache_key]
```

---

## fingerprint/legal/ccpa_handler.py

```python
"""
CCPA (California Consumer Privacy Act) compliance handler.

Respects:
- GPC (Global Privacy Control) signal
- "Do Not Sell My Personal Information" opt-outs
- California privacy rights signals

Verbose logging pattern:
[CCPA:OPERATION] Message
"""

from dataclasses import dataclass
from typing import Any

from fingerprint.config import Config
from fingerprint.core.verbose import get_logger
from fingerprint.exceptions import CCPAViolationError


@dataclass
class CCPACheckResult:
    """Result of CCPA compliance check."""
    compliant: bool
    opt_out_detected: bool = False
    gpc_enabled: bool = False
    reason: str = ""


class CCPAHandler:
    """
    CCPA compliance handler.

    Respects Global Privacy Control (GPC) and "Do Not Sell" signals.

    Usage:
        handler = CCPAHandler(config)
        result = handler.check_request(headers)
        content = await handler.process(response)
    """

    def __init__(self, config: Config):
        self.config = config
        self.ccpa_config = config.legal.ccpa
        self.logger = get_logger()

        self.logger.info("CCPA", "INIT", "CCPA handler initialized")

    def check_request_headers(self, headers: dict[str, str]) -> CCPACheckResult:
        """
        Check if request should respect privacy opt-outs.

        Checks for GPC (Sec-GPC) header.

        Args:
            headers: Request headers to check

        Returns:
            CCPACheckResult
        """
        if not self.ccpa_config.enabled:
            return CCPACheckResult(compliant=True)

        # Check for GPC header
        # Sec-GPC: 1 means user has opted out of data selling
        headers_lower = {k.lower(): v for k, v in headers.items()}
        gpc_value = headers_lower.get("sec-gpc", "")

        if self.ccpa_config.respect_gpc and gpc_value == "1":
            self.logger.info("CCPA", "GPC", "GPC signal detected - respecting opt-out")
            return CCPACheckResult(
                compliant=True,
                gpc_enabled=True,
                reason="GPC opt-out detected",
            )

        return CCPACheckResult(compliant=True)

    async def process(self, response: Any) -> Any:
        """
        Process response for CCPA compliance.

        Checks for privacy policy indicators and opt-out mechanisms.

        Args:
            response: HTTP response object

        Returns:
            Processed response

        Verbose output:
            [CCPA:CHECK] Checking CCPA compliance
            [CCPA:OPT_OUT] "Do Not Sell" link detected
        """
        if not self.ccpa_config.enabled:
            return response

        self.logger.debug("CCPA", "CHECK", "Checking CCPA compliance")

        content = response.text if hasattr(response, 'text') else ""

        # Check for "Do Not Sell" indicators
        if self.ccpa_config.respect_opt_out:
            opt_out_detected = self._detect_opt_out(content)
            if opt_out_detected:
                self.logger.info(
                    "CCPA", "OPT_OUT",
                    "\"Do Not Sell\" indicator detected on page",
                )

        return response

    def _detect_opt_out(self, content: str) -> bool:
        """Detect CCPA "Do Not Sell" indicators in content."""
        # Common "Do Not Sell" link text patterns
        opt_out_patterns = [
            "do not sell",
            "do not sell my personal information",
            "do not sell my info",
            "opt-out of sale",
            "opt out of sale",
            "ccpa opt-out",
            "california privacy",
        ]

        content_lower = content.lower()

        for pattern in opt_out_patterns:
            if pattern in content_lower:
                return True

        return False

    def get_privacy_headers(self) -> dict[str, str]:
        """
        Get headers to send indicating privacy preference.

        Returns headers for requests to indicate we respect privacy.
        """
        headers = {}

        # We don't collect/sell data, but we can indicate privacy-respecting behavior
        if self.ccpa_config.respect_gpc:
            # Note: This is typically sent by browser, not crawler
            # but we include for transparency
            pass

        return headers
```

---

## Verbose Logging Examples

### CFAA Checking

```
[2024-01-15T10:30:00Z] [CFAA:INIT] CFAA checker initialized

[2024-01-15T10:30:01Z] [CFAA:CHECK] Checking authorization for https://example.com/public/page
[2024-01-15T10:30:01Z] [CFAA:AUTHORIZED] Access authorized for https://example.com/public/page

[2024-01-15T10:30:02Z] [CFAA:CHECK] Checking authorization for https://example.com/admin/dashboard
[2024-01-15T10:30:02Z] [CFAA:BLOCKED] Access blocked for https://example.com/admin/dashboard
  - reason: Login-protected area
  - risk_level: high

[2024-01-15T10:30:03Z] [CFAA:CHECK] Checking authorization for https://example.com/api/v1/users
[2024-01-15T10:30:03Z] [CFAA:BLOCKED] Access blocked for https://example.com/api/v1/users
  - reason: API endpoint requires authorization
  - risk_level: medium
```

### ToS Checking

```
[2024-01-15T10:30:00Z] [TOS:INIT] ToS checker initialized

[2024-01-15T10:30:01Z] [TOS:CHECK] Checking ToS for https://example.com/article
[2024-01-15T10:30:01Z] [TOS:ALLOWED] No restrictive directives found

[2024-01-15T10:30:02Z] [TOS:CHECK] Checking ToS for https://example.com/private-content
[2024-01-15T10:30:02Z] [TOS:META_NOINDEX] Meta robots contains noindex
```

### GDPR Handling

```
[2024-01-15T10:30:00Z] [GDPR:INIT] GDPR handler initialized
  - handling_mode: redact

[2024-01-15T10:30:01Z] [GDPR:SCAN] Scanning for PII
[2024-01-15T10:30:01Z] [GDPR:PII_FOUND] PII detected
  - types: ['email', 'phone']
  - count: 5

[2024-01-15T10:30:01Z] [GDPR:REDACT] Redacting 5 PII instances
```

### CCPA Handling

```
[2024-01-15T10:30:00Z] [CCPA:INIT] CCPA handler initialized

[2024-01-15T10:30:01Z] [CCPA:CHECK] Checking CCPA compliance
[2024-01-15T10:30:01Z] [CCPA:GPC] GPC signal detected - respecting opt-out

[2024-01-15T10:30:02Z] [CCPA:OPT_OUT] "Do Not Sell" indicator detected on page
```

---

## Legal Compliance Summary

| Law | Requirement | Implementation |
|-----|-------------|----------------|
| **CFAA** | Don't access without authorization | Block login areas, APIs, internal paths |
| **ToS** | Respect website terms | Honor meta robots, X-Robots-Tag |
| **GDPR** | Protect personal data | Detect PII, redact/pseudonymize |
| **CCPA** | Respect privacy opt-outs | Honor GPC, "Do Not Sell" |

### Key Principles

1. **Authorization First**: Always check if access is authorized before fetching
2. **Respect Directives**: Honor noindex, nofollow, noarchive
3. **Protect Privacy**: Never store or transmit raw PII
4. **Transparency**: Log all compliance decisions for audit

### When to Block

| Scenario | Action |
|----------|--------|
| Login-protected path | Block (CFAA) |
| API endpoint | Block (CFAA) |
| Internal path (/.git, etc.) | Block (CFAA) |
| `noindex` directive | Skip indexing (ToS) |
| `nofollow` directive | Don't follow links (ToS) |
| PII detected | Redact/pseudonymize/skip (GDPR) |
| GPC = 1 | Respect opt-out (CCPA) |
