"""
PII (Personally Identifiable Information) detector for GDPR/CCPA compliance.

Detects and handles PII in crawled content.
"""

import hashlib
import os
import re
import secrets
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

    # Regex patterns for PII detection - all 14 types
    PATTERNS = {
        # Email addresses
        PIIType.EMAIL: re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        ),
        # Phone numbers (US/International)
        PIIType.PHONE: re.compile(
            r"\b(?:\+?1[-.\s]?)?"
            r"(?:\(?\d{3}\)?[-.\s]?)?"
            r"\d{3}[-.\s]?\d{4}\b"
        ),
        # US Social Security Numbers
        PIIType.SSN: re.compile(
            r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
        ),
        # Credit card numbers
        PIIType.CREDIT_CARD: re.compile(
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
        ),
        # IP addresses
        PIIType.IP_ADDRESS: re.compile(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        ),
        # Date of birth with context
        PIIType.DATE_OF_BIRTH: re.compile(
            r"\b(?:born|dob|birth\s*date|date\s*of\s*birth|birthday)[:\s]*"
            r"(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})\b",
            re.IGNORECASE,
        ),
        # Names with common prefixes/patterns
        PIIType.NAME: re.compile(
            r"\b(?:(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+)?"
            r"(?:[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)\b"
        ),
        # US Street addresses
        PIIType.ADDRESS: re.compile(
            r"\b\d+\s+(?:[A-Za-z]+\s+){1,4}"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Circle|Cir)"
            r"\.?(?:\s*,?\s*(?:Apt|Suite|Unit|#)\s*\d+)?\b",
            re.IGNORECASE,
        ),
        # Passport numbers (various formats)
        PIIType.PASSPORT: re.compile(
            r"\b(?:passport\s*(?:no|number|#)?[:\s]*)?[A-Z]{1,2}\d{6,9}\b",
            re.IGNORECASE,
        ),
        # US Driver's license (various state formats)
        PIIType.DRIVER_LICENSE: re.compile(
            r"\b(?:driver'?s?\s*(?:license|lic)\s*(?:no|number|#)?[:\s]*)?"
            r"[A-Z]{0,2}\d{5,12}\b",
            re.IGNORECASE,
        ),
        # National ID numbers (various formats)
        PIIType.NATIONAL_ID: re.compile(
            r"\b(?:national\s*id|id\s*(?:no|number|#))[:\s]*[A-Z0-9]{6,15}\b",
            re.IGNORECASE,
        ),
        # Health-related information keywords
        PIIType.HEALTH_INFO: re.compile(
            r"\b(?:diagnosis|diagnosed|patient|medical\s*record|"
            r"prescription|medication|treatment|condition|"
            r"blood\s*type|allergy|allergies|disease|disorder|"
            r"symptoms?|chronic|health\s*(?:condition|issue|problem))[:\s]+"
            r"[A-Za-z0-9\s,.-]{3,50}\b",
            re.IGNORECASE,
        ),
        # Financial information (account numbers, etc.)
        PIIType.FINANCIAL: re.compile(
            r"\b(?:account\s*(?:no|number|#)|routing\s*(?:no|number)|"
            r"bank\s*account|iban|swift|bic)[:\s]*[A-Z0-9]{8,34}\b",
            re.IGNORECASE,
        ),
        # Biometric identifiers
        PIIType.BIOMETRIC: re.compile(
            r"\b(?:fingerprint|retina|iris|facial\s*recognition|"
            r"voice\s*print|dna|biometric)\s*(?:id|data|scan|sample)[:\s]+"
            r"[A-Za-z0-9-]{5,}\b",
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
        PIIType.PASSPORT: "identity",
        PIIType.DRIVER_LICENSE: "identity",
        PIIType.FINANCIAL: "financial",
    }

    # Context keywords for confidence boosting
    POSITIVE_KEYWORDS = {
        PIIType.EMAIL: ["email", "contact", "mail", "e-mail"],
        PIIType.PHONE: ["phone", "tel", "call", "mobile", "cell", "fax"],
        PIIType.SSN: ["ssn", "social security", "social-security", "ss#"],
        PIIType.CREDIT_CARD: ["card", "credit", "payment", "cc", "visa", "mastercard"],
        PIIType.NAME: ["name", "customer", "user", "member", "subscriber"],
        PIIType.ADDRESS: ["address", "location", "residence", "home", "mailing"],
        PIIType.DATE_OF_BIRTH: ["born", "birthday", "dob", "age"],
        PIIType.PASSPORT: ["passport", "travel document"],
        PIIType.DRIVER_LICENSE: ["driver", "license", "dmv", "driving"],
        PIIType.HEALTH_INFO: ["medical", "health", "patient", "doctor", "hospital"],
        PIIType.FINANCIAL: ["bank", "account", "routing", "wire", "transfer"],
    }

    def __init__(
        self,
        config: PIIHandlingConfig | None = None,
        logger: CrawlerLogger | None = None,
        salt: str | None = None,
    ):
        """
        Initialize the PII detector.

        Args:
            config: PII handling configuration.
            logger: Logger instance.
            salt: Pseudonymization salt. If not provided, uses environment
                  variable PII_PSEUDONYMIZATION_SALT or generates a random one.
        """
        self.config = config or PIIHandlingConfig()
        self.logger = logger or CrawlerLogger("pii_detector")

        # Secure salt management
        if salt:
            self._salt = salt
        else:
            # Try environment variable first
            env_salt = os.environ.get("PII_PSEUDONYMIZATION_SALT")
            if env_salt:
                self._salt = env_salt
            else:
                # Generate random salt for this session
                # In production, this should be persisted securely
                self._salt = secrets.token_hex(32)
                self.logger.warning(
                    "Using random pseudonymization salt - pseudonyms will not be consistent across sessions. "
                    "Set PII_PSEUDONYMIZATION_SALT environment variable for consistent pseudonymization."
                )

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
            try:
                for match in pattern.finditer(text):
                    value = match.group()

                    # Skip false positives
                    if self._is_false_positive(pii_type, value, text, match.start(), match.end()):
                        continue

                    # Get surrounding context
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end]

                    confidence = self._calculate_confidence(pii_type, value, context)

                    # Skip low confidence matches
                    if confidence < 0.5:
                        continue

                    pii_match = PIIMatch(
                        pii_type=pii_type,
                        value=value,
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                        context=context,
                    )
                    matches.append(pii_match)
                    pii_types.add(pii_type)

                    # Check for sensitive categories
                    if pii_type in self.SENSITIVE_CATEGORIES:
                        category = self.SENSITIVE_CATEGORIES[pii_type]
                        if category in self.config.sensitive_categories:
                            sensitive.add(category)
            except re.error:
                self.logger.debug(f"Regex error for PII type {pii_type.value}")
                continue

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

    def _is_false_positive(
        self,
        pii_type: PIIType,
        value: str,
        text: str,
        start: int,
        end: int,
    ) -> bool:
        """Check if a match is a likely false positive."""
        if pii_type == PIIType.PHONE:
            # Filter out dates that look like phone numbers
            if re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}$", value):
                return True
            # Filter out version numbers
            if re.match(r"^\d+\.\d+\.\d+$", value):
                return True
            # Filter out very short numbers
            digits = re.sub(r"\D", "", value)
            if len(digits) < 7:
                return True

        if pii_type == PIIType.IP_ADDRESS:
            # Check for valid IP range
            parts = value.split(".")
            try:
                if any(int(p) > 255 for p in parts):
                    return True
            except ValueError:
                return True
            # Filter out version numbers (more than 3 dots)
            if value.count(".") > 3:
                return True
            # Check if it's in a code context
            context_before = text[max(0, start - 20):start]
            if re.search(r"version|v\d|\.jar|\.dll", context_before, re.IGNORECASE):
                return True

        if pii_type == PIIType.SSN:
            # SSN should not start with 000, 666, or 900-999
            first_three = re.sub(r"\D", "", value)[:3]
            if first_three in ("000", "666") or first_three >= "900":
                return True
            # Should have exactly 9 digits
            digits = re.sub(r"\D", "", value)
            if len(digits) != 9:
                return True

        if pii_type == PIIType.CREDIT_CARD:
            # Validate with Luhn algorithm
            if not self._validate_luhn(value):
                return True

        if pii_type == PIIType.NAME:
            # Filter out common non-name patterns
            common_phrases = [
                "New York", "Los Angeles", "San Francisco", "San Diego",
                "United States", "North America", "South America",
            ]
            if value in common_phrases:
                return True

        if pii_type == PIIType.ADDRESS:
            # Filter out very short matches
            if len(value) < 15:
                return True

        if pii_type == PIIType.DRIVER_LICENSE:
            # Need context to confirm it's a license number
            context = text[max(0, start - 50):end + 50].lower()
            if "license" not in context and "driver" not in context:
                return True

        if pii_type == PIIType.PASSPORT:
            # Need context to confirm it's a passport number
            context = text[max(0, start - 50):end + 50].lower()
            if "passport" not in context:
                return True

        return False

    def _validate_luhn(self, number: str) -> bool:
        """Validate a number using the Luhn algorithm."""
        digits = [int(d) for d in number if d.isdigit()]
        if len(digits) < 13 or len(digits) > 19:
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
        base_confidence = 0.6

        # Adjust based on PII type (some patterns are more reliable)
        type_confidence = {
            PIIType.EMAIL: 0.85,
            PIIType.CREDIT_CARD: 0.9,
            PIIType.SSN: 0.85,
            PIIType.IP_ADDRESS: 0.7,
            PIIType.PHONE: 0.7,
            PIIType.NAME: 0.5,
            PIIType.ADDRESS: 0.6,
            PIIType.DATE_OF_BIRTH: 0.8,
            PIIType.PASSPORT: 0.75,
            PIIType.DRIVER_LICENSE: 0.7,
            PIIType.NATIONAL_ID: 0.7,
            PIIType.HEALTH_INFO: 0.75,
            PIIType.FINANCIAL: 0.8,
            PIIType.BIOMETRIC: 0.8,
        }
        base_confidence = type_confidence.get(pii_type, 0.6)

        # Adjust based on context keywords
        context_lower = context.lower()

        if pii_type in self.POSITIVE_KEYWORDS:
            for keyword in self.POSITIVE_KEYWORDS[pii_type]:
                if keyword in context_lower:
                    base_confidence = min(1.0, base_confidence + 0.15)
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
        """Pseudonymize PII in text using secure hashing."""
        sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)

        result = text
        for match in sorted_matches:
            # Create consistent pseudonym from HMAC for security
            hash_input = f"{self._salt}:{match.pii_type.value}:{match.value}"
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

    def get_salt_hash(self) -> str:
        """
        Get a hash of the salt for verification purposes.

        Returns:
            SHA256 hash of the salt (first 16 chars).
        """
        return hashlib.sha256(self._salt.encode()).hexdigest()[:16]
