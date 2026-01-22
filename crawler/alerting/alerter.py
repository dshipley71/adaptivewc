"""
Alerting system for the adaptive web crawler.

Sends notifications for important events like structure changes,
failures, and compliance issues.
"""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import httpx

from crawler.utils.logging import CrawlerLogger


class AlertSeverity(str, Enum):
    """Severity levels for alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of alerts."""

    STRUCTURE_CHANGE = "structure_change"
    EXTRACTION_FAILURE = "extraction_failure"
    COMPLIANCE_BLOCK = "compliance_block"
    RATE_LIMIT = "rate_limit"
    CONNECTION_ERROR = "connection_error"
    PII_DETECTED = "pii_detected"
    CRAWL_COMPLETE = "crawl_complete"
    CRAWL_ERROR = "crawl_error"


@dataclass
class Alert:
    """An alert to be sent."""

    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    domain: str | None = None
    url: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "domain": self.domain,
            "url": self.url,
            "details": self.details,
            "created_at": self.created_at.isoformat(),
        }


class AlertChannel(ABC):
    """Abstract base class for alert channels."""

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Send an alert through this channel."""
        pass


class SlackChannel(AlertChannel):
    """Slack webhook alert channel."""

    SEVERITY_COLORS = {
        AlertSeverity.INFO: "#36a64f",
        AlertSeverity.WARNING: "#ffcc00",
        AlertSeverity.ERROR: "#ff6600",
        AlertSeverity.CRITICAL: "#ff0000",
    }

    def __init__(self, webhook_url: str):
        """Initialize Slack channel with webhook URL."""
        self.webhook_url = webhook_url

    async def send(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        if not self.webhook_url:
            return False

        payload = {
            "attachments": [
                {
                    "color": self.SEVERITY_COLORS.get(alert.severity, "#808080"),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "Type", "value": alert.alert_type.value, "short": True},
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                    ],
                    "footer": "Adaptive Web Crawler",
                    "ts": int(alert.created_at.timestamp()),
                }
            ]
        }

        if alert.domain:
            payload["attachments"][0]["fields"].append(
                {"title": "Domain", "value": alert.domain, "short": True}
            )

        if alert.url:
            payload["attachments"][0]["fields"].append(
                {"title": "URL", "value": alert.url, "short": False}
            )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    timeout=10.0,
                )
                return response.status_code == 200
        except Exception:
            return False


class WebhookChannel(AlertChannel):
    """Generic webhook alert channel."""

    def __init__(self, webhook_url: str, headers: dict[str, str] | None = None):
        """Initialize webhook channel."""
        self.webhook_url = webhook_url
        self.headers = headers or {}

    async def send(self, alert: Alert) -> bool:
        """Send alert to webhook."""
        if not self.webhook_url:
            return False

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=alert.to_dict(),
                    headers=self.headers,
                    timeout=10.0,
                )
                return response.status_code in (200, 201, 202, 204)
        except Exception:
            return False


class LogChannel(AlertChannel):
    """Log-based alert channel (for testing/development)."""

    def __init__(self, logger: CrawlerLogger | None = None):
        """Initialize log channel."""
        self.logger = logger or CrawlerLogger("alerter")

    async def send(self, alert: Alert) -> bool:
        """Log the alert."""
        log_func = {
            AlertSeverity.INFO: self.logger.info,
            AlertSeverity.WARNING: self.logger.warning,
            AlertSeverity.ERROR: self.logger.error,
            AlertSeverity.CRITICAL: self.logger.error,
        }.get(alert.severity, self.logger.info)

        log_func(
            f"ALERT: {alert.title}",
            alert_type=alert.alert_type.value,
            message=alert.message,
            domain=alert.domain,
        )
        return True


class Alerter:
    """
    Main alerter that manages channels and throttling.

    Aggregates alerts and sends through configured channels
    with rate limiting to prevent alert fatigue.
    """

    def __init__(
        self,
        channels: list[AlertChannel] | None = None,
        throttle_minutes: int = 60,
        min_severity: AlertSeverity = AlertSeverity.WARNING,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the alerter.

        Args:
            channels: List of alert channels.
            throttle_minutes: Minimum time between similar alerts.
            min_severity: Minimum severity to send.
            logger: Logger instance.
        """
        self.channels = channels or []
        self.throttle_minutes = throttle_minutes
        self.min_severity = min_severity
        self.logger = logger or CrawlerLogger("alerter")

        # Track sent alerts for throttling
        self._sent_alerts: dict[str, datetime] = {}
        self._pending_alerts: list[Alert] = []
        self._lock = asyncio.Lock()

    def add_channel(self, channel: AlertChannel) -> None:
        """Add an alert channel."""
        self.channels.append(channel)

    async def alert(self, alert: Alert) -> bool:
        """
        Send an alert.

        Args:
            alert: Alert to send.

        Returns:
            True if alert was sent (not throttled).
        """
        # Check severity threshold
        severity_order = [
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.ERROR,
            AlertSeverity.CRITICAL,
        ]
        if severity_order.index(alert.severity) < severity_order.index(self.min_severity):
            return False

        # Check throttling
        throttle_key = f"{alert.alert_type.value}:{alert.domain or 'global'}"
        async with self._lock:
            if throttle_key in self._sent_alerts:
                last_sent = self._sent_alerts[throttle_key]
                if datetime.utcnow() - last_sent < timedelta(minutes=self.throttle_minutes):
                    # Aggregate instead of sending
                    self._pending_alerts.append(alert)
                    return False

            self._sent_alerts[throttle_key] = datetime.utcnow()

        # Send through all channels
        results = await asyncio.gather(
            *[channel.send(alert) for channel in self.channels],
            return_exceptions=True,
        )

        success = any(r is True for r in results)
        if not success:
            self.logger.warning("Failed to send alert through any channel", alert=alert.title)

        return success

    async def alert_structure_change(
        self,
        domain: str,
        page_type: str,
        similarity: float,
        breaking: bool,
    ) -> bool:
        """Send alert for structure change."""
        severity = AlertSeverity.ERROR if breaking else AlertSeverity.WARNING

        alert = Alert(
            alert_type=AlertType.STRUCTURE_CHANGE,
            severity=severity,
            title=f"Structure change detected: {domain}",
            message=f"Page type '{page_type}' structure changed. Similarity: {similarity:.1%}. "
            f"{'Requires re-learning extraction strategy.' if breaking else 'Minor change detected.'}",
            domain=domain,
            details={
                "page_type": page_type,
                "similarity": similarity,
                "breaking": breaking,
            },
        )
        return await self.alert(alert)

    async def alert_compliance_block(
        self,
        url: str,
        reason: str,
        block_type: str,
    ) -> bool:
        """Send alert for compliance block."""
        alert = Alert(
            alert_type=AlertType.COMPLIANCE_BLOCK,
            severity=AlertSeverity.INFO,
            title=f"URL blocked: {block_type}",
            message=f"URL was blocked due to compliance check: {reason}",
            url=url,
            details={
                "block_type": block_type,
                "reason": reason,
            },
        )
        return await self.alert(alert)

    async def alert_pii_detected(
        self,
        url: str,
        pii_types: list[str],
        action: str,
    ) -> bool:
        """Send alert for PII detection."""
        alert = Alert(
            alert_type=AlertType.PII_DETECTED,
            severity=AlertSeverity.WARNING,
            title="PII detected in crawled content",
            message=f"Found PII types: {', '.join(pii_types)}. Action taken: {action}",
            url=url,
            details={
                "pii_types": pii_types,
                "action": action,
            },
        )
        return await self.alert(alert)

    async def alert_crawl_complete(
        self,
        stats: dict[str, Any],
    ) -> bool:
        """Send alert when crawl completes."""
        alert = Alert(
            alert_type=AlertType.CRAWL_COMPLETE,
            severity=AlertSeverity.INFO,
            title="Crawl completed",
            message=f"Crawled {stats.get('pages_crawled', 0)} pages, "
            f"{stats.get('pages_failed', 0)} failed, "
            f"{stats.get('pages_blocked', 0)} blocked.",
            details=stats,
        )
        return await self.alert(alert)

    async def flush_pending(self) -> int:
        """
        Send summary of pending throttled alerts.

        Returns:
            Number of pending alerts flushed.
        """
        async with self._lock:
            if not self._pending_alerts:
                return 0

            # Group by type and domain
            groups: dict[str, list[Alert]] = {}
            for alert in self._pending_alerts:
                key = f"{alert.alert_type.value}:{alert.domain or 'global'}"
                if key not in groups:
                    groups[key] = []
                groups[key].append(alert)

            count = len(self._pending_alerts)
            self._pending_alerts.clear()

        # Send summary for each group
        for key, alerts in groups.items():
            if len(alerts) > 1:
                first = alerts[0]
                summary = Alert(
                    alert_type=first.alert_type,
                    severity=first.severity,
                    title=f"{first.title} (x{len(alerts)})",
                    message=f"{len(alerts)} similar alerts were throttled. Last: {first.message}",
                    domain=first.domain,
                    details={"count": len(alerts)},
                )
                await self._send_direct(summary)

        return count

    async def _send_direct(self, alert: Alert) -> bool:
        """Send alert directly without throttling."""
        results = await asyncio.gather(
            *[channel.send(alert) for channel in self.channels],
            return_exceptions=True,
        )
        return any(r is True for r in results)


def create_alerter_from_config(
    slack_webhook: str | None = None,
    webhook_url: str | None = None,
    throttle_minutes: int = 60,
    min_severity: str = "warning",
    logger: CrawlerLogger | None = None,
) -> Alerter:
    """
    Create an alerter from configuration.

    Args:
        slack_webhook: Slack webhook URL.
        webhook_url: Generic webhook URL.
        throttle_minutes: Throttle interval.
        min_severity: Minimum severity level.
        logger: Logger instance.

    Returns:
        Configured Alerter instance.
    """
    channels: list[AlertChannel] = []

    # Add configured channels
    if slack_webhook:
        channels.append(SlackChannel(slack_webhook))

    if webhook_url:
        channels.append(WebhookChannel(webhook_url))

    # Always add log channel as fallback
    channels.append(LogChannel(logger))

    severity_map = {
        "info": AlertSeverity.INFO,
        "warning": AlertSeverity.WARNING,
        "error": AlertSeverity.ERROR,
        "critical": AlertSeverity.CRITICAL,
    }

    return Alerter(
        channels=channels,
        throttle_minutes=throttle_minutes,
        min_severity=severity_map.get(min_severity.lower(), AlertSeverity.WARNING),
        logger=logger,
    )
