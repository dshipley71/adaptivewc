"""
Exception hierarchy for the adaptive web crawler.

All exceptions inherit from CrawlerError to allow catching all crawler-related errors.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


class CrawlerError(Exception):
    """Base exception for all crawler errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.utcnow()


# =============================================================================
# Compliance Errors
# =============================================================================


class ComplianceError(CrawlerError):
    """Raised when a compliance check fails."""

    pass


class RobotsBlockedError(ComplianceError):
    """URL is blocked by robots.txt."""

    def __init__(self, url: str, user_agent: str, directive: str | None = None):
        super().__init__(
            f"URL blocked by robots.txt: {url}",
            {"url": url, "user_agent": user_agent, "directive": directive},
        )
        self.url = url
        self.user_agent = user_agent
        self.directive = directive


class RateLimitExceededError(ComplianceError):
    """Rate limit would be exceeded."""

    def __init__(self, domain: str, current_delay: float, required_delay: float):
        super().__init__(
            f"Rate limit exceeded for {domain}",
            {
                "domain": domain,
                "current_delay": current_delay,
                "required_delay": required_delay,
            },
        )
        self.domain = domain
        self.current_delay = current_delay
        self.required_delay = required_delay


class MetaRobotsBlockedError(ComplianceError):
    """URL blocked by meta robots tag or X-Robots-Tag header."""

    def __init__(self, url: str, directive: str, source: str):
        super().__init__(
            f"URL blocked by {source}: {url} ({directive})",
            {"url": url, "directive": directive, "source": source},
        )
        self.url = url
        self.directive = directive
        self.source = source


class BotDetectionBlockedError(ComplianceError):
    """Request blocked by anti-bot detection."""

    def __init__(self, url: str, detection_type: str):
        super().__init__(
            f"Blocked by bot detection ({detection_type}): {url}",
            {"url": url, "detection_type": detection_type},
        )
        self.url = url
        self.detection_type = detection_type


# =============================================================================
# Fetch Errors
# =============================================================================


class FetchError(CrawlerError):
    """HTTP fetch failed."""

    def __init__(self, url: str, message: str, status_code: int | None = None):
        super().__init__(
            f"Fetch failed for {url}: {message}",
            {"url": url, "status_code": status_code},
        )
        self.url = url
        self.status_code = status_code


class TimeoutError(FetchError):
    """Request timed out."""

    def __init__(self, url: str, timeout_seconds: float):
        super().__init__(url, f"Timeout after {timeout_seconds}s")
        self.timeout_seconds = timeout_seconds


class CircuitOpenError(FetchError):
    """Circuit breaker is open for domain."""

    def __init__(self, domain: str, opens_at: datetime, failure_count: int):
        super().__init__(
            f"https://{domain}/",
            f"Circuit breaker open until {opens_at}",
        )
        self.domain = domain
        self.opens_at = opens_at
        self.failure_count = failure_count


class TooManyRedirectsError(FetchError):
    """Too many redirects encountered."""

    def __init__(self, url: str, redirect_count: int, max_redirects: int):
        super().__init__(
            url,
            f"Too many redirects: {redirect_count} > {max_redirects}",
        )
        self.redirect_count = redirect_count
        self.max_redirects = max_redirects


class ContentTooLargeError(FetchError):
    """Content exceeds size limit."""

    def __init__(self, url: str, content_length: int, max_size: int):
        super().__init__(
            url,
            f"Content too large: {content_length} > {max_size}",
        )
        self.content_length = content_length
        self.max_size = max_size


# =============================================================================
# Extraction Errors
# =============================================================================


class ExtractionError(CrawlerError):
    """Content extraction failed."""

    def __init__(self, url: str, message: str, selector: str | None = None):
        super().__init__(
            f"Extraction failed for {url}: {message}",
            {"url": url, "selector": selector},
        )
        self.url = url
        self.selector = selector


class StructureChangeError(ExtractionError):
    """Page structure changed beyond adaptation capability."""

    def __init__(
        self,
        url: str,
        domain: str,
        page_type: str,
        similarity_score: float,
        threshold: float,
    ):
        super().__init__(
            url,
            f"Structure changed too much: similarity {similarity_score:.2f} < {threshold}",
        )
        self.domain = domain
        self.page_type = page_type
        self.similarity_score = similarity_score
        self.threshold = threshold


class StrategyInferenceError(ExtractionError):
    """Failed to infer extraction strategy for new structure."""

    def __init__(self, url: str, reason: str):
        super().__init__(url, f"Strategy inference failed: {reason}")
        self.reason = reason


class ValidationError(ExtractionError):
    """Extracted content failed validation."""

    def __init__(self, url: str, issues: list[str]):
        super().__init__(url, f"Validation failed: {', '.join(issues)}")
        self.issues = issues


# =============================================================================
# Storage Errors
# =============================================================================


class StorageError(CrawlerError):
    """Storage operation failed."""

    pass


class StructureStoreError(StorageError):
    """Redis structure storage operation failed."""

    def __init__(self, operation: str, key: str, message: str):
        super().__init__(
            f"Structure store {operation} failed for {key}: {message}",
            {"operation": operation, "key": key},
        )
        self.operation = operation
        self.key = key


class ContentStoreError(StorageError):
    """Content storage operation failed."""

    def __init__(self, operation: str, url: str, message: str):
        super().__init__(
            f"Content store {operation} failed for {url}: {message}",
            {"operation": operation, "url": url},
        )
        self.operation = operation
        self.url = url


# =============================================================================
# Legal Compliance Errors
# =============================================================================


class LegalComplianceError(CrawlerError):
    """Legal compliance requirement not met."""

    pass


class UnauthorizedAccessError(LegalComplianceError):
    """Access not authorized under CFAA analysis."""

    def __init__(self, url: str, reason: str, authorization_sources: list[str]):
        super().__init__(
            f"Unauthorized access to {url}: {reason}",
            {"url": url, "authorization_sources": authorization_sources},
        )
        self.url = url
        self.reason = reason
        self.authorization_sources = authorization_sources


class GDPRViolationError(LegalComplianceError):
    """Operation would violate GDPR requirements."""

    def __init__(self, operation: str, article: str, reason: str):
        super().__init__(
            f"GDPR violation (Article {article}): {reason}",
            {"operation": operation, "article": article},
        )
        self.operation = operation
        self.article = article


class DataSubjectRequestError(LegalComplianceError):
    """Failed to process data subject request."""

    def __init__(self, request_type: str, subject_id: str, reason: str):
        super().__init__(
            f"Failed to process {request_type} request for {subject_id}: {reason}",
            {"request_type": request_type, "subject_id": subject_id},
        )
        self.request_type = request_type
        self.subject_id = subject_id


class CeaseAndDesistError(LegalComplianceError):
    """Domain blocked due to legal request."""

    def __init__(self, domain: str, received_at: datetime, reference: str):
        super().__init__(
            f"Domain {domain} blocked due to cease and desist",
            {"domain": domain, "received_at": received_at.isoformat(), "reference": reference},
        )
        self.domain = domain
        self.received_at = received_at
        self.reference = reference


class PIIExposureError(LegalComplianceError):
    """Sensitive PII detected, handling failed."""

    def __init__(self, url: str, pii_types: list[str], action_failed: str):
        super().__init__(
            f"PII exposure risk at {url}: {', '.join(pii_types)}",
            {"url": url, "pii_types": pii_types, "action_failed": action_failed},
        )
        self.url = url
        self.pii_types = pii_types
        self.action_failed = action_failed


class RetentionViolationError(LegalComplianceError):
    """Data retention policy would be violated."""

    def __init__(self, data_type: str, retention_days: int, actual_days: int):
        super().__init__(
            f"Retention violation: {data_type} kept {actual_days} days, limit {retention_days}",
            {"data_type": data_type, "retention_days": retention_days, "actual_days": actual_days},
        )
        self.data_type = data_type
        self.retention_days = retention_days
        self.actual_days = actual_days
