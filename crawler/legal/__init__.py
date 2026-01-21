"""Legal compliance modules for CFAA, GDPR, and CCPA."""

from crawler.legal.cfaa_checker import CFAAChecker
from crawler.legal.pii_detector import PIIDetector, PIIDetectionResult, PIIType

__all__ = [
    "CFAAChecker",
    "PIIDetectionResult",
    "PIIDetector",
    "PIIType",
]
