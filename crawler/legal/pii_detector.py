"""
PII (Personally Identifiable Information) detector for GDPR/CCPA compliance.

Detects and handles PII in crawled content.
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from crawler.config import PIIHandling, PIIHandlingConfig
from crawler.utils.logging import CrawlerLogger
from crawler.utils import metrics


class PIIType(str, Enum):
    """Types of PII that can be detected."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    NATIONAL_ID = "national_id"
    HEALTH_INFO = "health_info"
    FINANCIAL = "financial"
    BIOMETRIC = "biometric"


@dataclass
class PIIMatch:
    """A detected PII match."""

    pii_type: PIIType
    value: str
    start: int
    end: int
    confidence: float
    context: str = ""


@dataclass
class PIIDetectionResult:
    """Result of PII detection."""

    url: str
    has_pii: bool
    matches: list[PIIMatch] = field(default_factory=list)
    pii_types_found: list[PIIType] = field(default_factory=list)
    sensitive_categories: list[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "has_pii": self.has_pii,
            "pii_types_found": [t.value for t in self.pii_types_found],
            "sensitive_categories": self.sensitive_categories,
            "match_count": len(self.matches),
            "detected_at": self.detected_at.isoformat(),
        }


class PIIDetector:
    """
    Detector for personally identifiable information.

    Supports multiple PII types and provides options for
    redaction, pseudonymization, or flagging.
    """

    # Regex patterns for PII detection
    PATTERNS = {
        PIIType.EMAIL: re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        ),
        PIIType.PHONE: re.compile(
            r"\b(?:\+?1[-.\s]?)?"
            r"(?:\(?\d{3}\)?[-.\s]?)?"
            r"\d{3}[-.\s]?\d{4}\b"
        ),
        PIIType.SSN: re.compile(
            r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
        ),
        PIIType.CREDIT_CARD: re.compile(
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
        ),
        PIIType.IP_ADDRESS: re.compile(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        ),
        PIIType.DATE_OF_BIRTH: re.compile(
            r"\b(?:born|dob|birth\s*date)[:\s]*"
            r"(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})\b",
            re.IGNORECASE,
        ),
    }

    # Sensitive categories that require special handling
    SENSITIVE_CATEGORIES = {
        PIIType.SSN: "identity",
        PIIType.CREDIT_CARD: "financial",
        PIIType.HEALTH_INFO: "health",
        PIIType.BIOMETRIC: "biometric",
        PIIType.NATIONAL_ID: "identity",
    }

    def __init__(
        self,
        config: PIIHandlingConfig | None = None,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the PII detector.

        Args:
            config: PII handling configuration.
            logger: Logger instance.
        """
        self.config = config or PIIHandlingConfig()
        self.logger = logger or CrawlerLogger("pii_detector")

        # Pseudonymization salt (should be securely stored in production)
        self._salt = "adaptive_crawler_pii_salt"

    def detect(self, text: str, url: str = "") -> PIIDetectionResult:
        """
        Detect PII in text.

        Args:
            text: Text to scan for PII.
            url: URL the text came from (for logging).

        Returns:
            PIIDetectionResult with all matches.
        """
        matches: list[PIIMatch] = []
        pii_types: set[PIIType] = set()
        sensitive: set[str] = set()

        for pii_type, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                value = match.group()

                # Skip false positives
                if self._is_false_positive(pii_type, value):
                    continue

                # Get surrounding context
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end]

                pii_match = PIIMatch(
                    pii_type=pii_type,
                    value=value,
                    start=match.start(),
                    end=match.end(),
                    confidence=self._calculate_confidence(pii_type, value, context),
                    context=context,
                )
                matches.append(pii_match)
                pii_types.add(pii_type)

                # Check for sensitive categories
                if pii_type in self.SENSITIVE_CATEGORIES:
                    category = self.SENSITIVE_CATEGORIES[pii_type]
                    if category in self.config.sensitive_categories:
                        sensitive.add(category)

        result = PIIDetectionResult(
            url=url,
            has_pii=len(matches) > 0,
            matches=matches,
            pii_types_found=list(pii_types),
            sensitive_categories=list(sensitive),
        )

        # Log if PII found
        if result.has_pii and self.config.log_detections:
            self.logger.pii_detected(
                url=url,
                pii_types=[t.value for t in pii_types],
                action=self.config.action.value,
            )
            # Record metrics
            for pii_type in pii_types:
                metrics.record_pii_detection(pii_type.value, self.config.action.value)

        return result

    def _is_false_positive(self, pii_type: PIIType, value: str) -> bool:
        """Check if a match is a likely false positive."""
        if pii_type == PIIType.PHONE:
            # Filter out dates that look like phone numbers
            if re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}$", value):
                return True
            # Filter out version numbers
            if re.match(r"^\d+\.\d+\.\d+$", value):
                return True

        if pii_type == PIIType.IP_ADDRESS:
            # Check for valid IP range
            parts = value.split(".")
            if any(int(p) > 255 for p in parts):
                return True
            # Filter out version numbers
            if value.count(".") > 3:
                return True

        if pii_type == PIIType.SSN:
            # SSN should not start with 000, 666, or 900-999
            first_three = value[:3].replace("-", "").replace(" ", "")
            if first_three in ("000", "666") or first_three >= "900":
                return True

        if pii_type == PIIType.CREDIT_CARD:
            # Validate with Luhn algorithm
            if not self._validate_luhn(value):
                return True

        return False

    def _validate_luhn(self, number: str) -> bool:
        """Validate a number using the Luhn algorithm."""
        digits = [int(d) for d in number if d.isdigit()]
        if len(digits) < 13:
            return False

        checksum = 0
        for i, digit in enumerate(reversed(digits)):
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit
        return checksum % 10 == 0

    def _calculate_confidence(
        self,
        pii_type: PIIType,
        value: str,
        context: str,
    ) -> float:
        """Calculate confidence score for a PII match."""
        base_confidence = 0.7

        # Adjust based on context keywords
        context_lower = context.lower()

        positive_keywords = {
            PIIType.EMAIL: ["email", "contact", "mail"],
            PIIType.PHONE: ["phone", "tel", "call", "mobile"],
            PIIType.SSN: ["ssn", "social security", "social-security"],
            PIIType.CREDIT_CARD: ["card", "credit", "payment", "cc"],
        }

        if pii_type in positive_keywords:
            for keyword in positive_keywords[pii_type]:
                if keyword in context_lower:
                    base_confidence += 0.15
                    break

        return min(1.0, base_confidence)

    def process(
        self,
        text: str,
        url: str = "",
    ) -> tuple[str, PIIDetectionResult]:
        """
        Detect and handle PII in text.

        Args:
            text: Text to process.
            url: URL the text came from.

        Returns:
            Tuple of (processed_text, detection_result).
        """
        result = self.detect(text, url)

        if not result.has_pii:
            return text, result

        # Handle based on configuration
        action = self.config.action

        if action == PIIHandling.REDACT:
            processed = self._redact(text, result.matches)
        elif action == PIIHandling.PSEUDONYMIZE:
            processed = self._pseudonymize(text, result.matches)
        elif action == PIIHandling.EXCLUDE_PAGE:
            processed = ""  # Return empty to exclude
        else:  # FLAG_FOR_REVIEW
            processed = text

        return processed, result

    def _redact(self, text: str, matches: list[PIIMatch]) -> str:
        """Redact PII from text."""
        # Sort matches by position in reverse order
        sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)

        result = text
        for match in sorted_matches:
            redacted = f"[REDACTED:{match.pii_type.value}]"
            result = result[:match.start] + redacted + result[match.end:]

        return result

    def _pseudonymize(self, text: str, matches: list[PIIMatch]) -> str:
        """Pseudonymize PII in text."""
        sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)

        result = text
        for match in sorted_matches:
            # Create consistent pseudonym from hash
            hash_input = f"{self._salt}:{match.value}"
            hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
            pseudonym = f"[{match.pii_type.value}:{hash_value}]"
            result = result[:match.start] + pseudonym + result[match.end:]

        return result

    def should_exclude_page(self, result: PIIDetectionResult) -> bool:
        """
        Check if a page should be excluded due to sensitive PII.

        Args:
            result: PII detection result.

        Returns:
            True if page should be excluded.
        """
        if self.config.action == PIIHandling.EXCLUDE_PAGE and result.has_pii:
            return True

        # Check for sensitive categories
        if self.config.alert_on_sensitive and result.sensitive_categories:
            return True

        return False
