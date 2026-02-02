# AGENTS.md - Alerting Module

Complete specification for change alerts, manual review queue, and notifications.

---

## Module Purpose

The alerting module provides:
- Change detection monitoring
- Alert generation for breaking/significant changes
- Manual review queue for human approval
- Multiple notification channels (log, webhook, email)
- Review workflow management

---

## Files to Generate

```
fingerprint/alerting/
├── __init__.py
├── change_monitor.py    # Monitor for breaking changes
├── review_queue.py      # Manual review queue
└── notifiers.py         # Alert notifications
```

---

## fingerprint/alerting/__init__.py

```python
"""
Alerting module - Change alerts and manual review.
"""

from fingerprint.alerting.change_monitor import ChangeMonitor
from fingerprint.alerting.review_queue import ReviewQueue, ReviewItem
from fingerprint.alerting.notifiers import (
    Notifier,
    LogNotifier,
    WebhookNotifier,
    EmailNotifier,
    NotificationManager,
)

__all__ = [
    "ChangeMonitor",
    "ReviewQueue",
    "ReviewItem",
    "Notifier",
    "LogNotifier",
    "WebhookNotifier",
    "EmailNotifier",
    "NotificationManager",
]
```

---

## fingerprint/alerting/change_monitor.py

```python
"""
Change monitor for detecting and alerting on structure changes.

Monitors comparison results and triggers alerts when thresholds are exceeded.

Verbose logging pattern:
[ALERT:OPERATION] Message
"""

import uuid
from datetime import datetime
from typing import Any

from fingerprint.config import Config
from fingerprint.core.verbose import get_logger
from fingerprint.models import (
    ChangeAnalysis,
    ChangeAlert,
    AlertSeverity,
    ReviewItem,
    ReviewStatus,
    ChangeClassification,
)
from fingerprint.alerting.review_queue import ReviewQueue
from fingerprint.alerting.notifiers import NotificationManager


class ChangeMonitor:
    """
    Monitor structure changes and generate alerts.

    Usage:
        monitor = ChangeMonitor(config, review_queue, notifier)
        alert = await monitor.process_change(url, change_analysis)
    """

    def __init__(
        self,
        config: Config,
        review_queue: "ReviewQueue",
        notifier: "NotificationManager",
    ):
        self.config = config
        self.alerting_config = config.alerting
        self.review_queue = review_queue
        self.notifier = notifier
        self.logger = get_logger()

        self.logger.info(
            "ALERT", "INIT",
            "Change monitor initialized",
            alert_threshold=self.alerting_config.alert_threshold,
        )

    async def process_change(
        self,
        url: str,
        domain: str,
        page_type: str,
        change_analysis: ChangeAnalysis,
        old_version: int,
        new_version: int,
    ) -> ChangeAlert | None:
        """
        Process a detected change and determine if alert is needed.

        Args:
            url: URL that was analyzed
            domain: Domain name
            page_type: Page type
            change_analysis: Result of structure comparison
            old_version: Previous structure version
            new_version: New structure version

        Returns:
            ChangeAlert if alert was triggered, None otherwise

        Verbose output:
            [ALERT:DETECT] Change detected for example.com/article
              - similarity: 0.65
              - classification: breaking
            [ALERT:CREATE] Creating alert
              - severity: critical
            [ALERT:SEND] Sending notifications
              - channels: ['log', 'webhook']
        """
        if not self.alerting_config.enabled:
            return None

        self.logger.debug(
            "ALERT", "DETECT",
            f"Change detected for {domain}/{page_type}",
            similarity=f"{change_analysis.similarity:.2f}",
            classification=change_analysis.classification.value,
        )

        # Determine if we should alert
        should_alert = self._should_alert(change_analysis)

        if not should_alert:
            self.logger.debug(
                "ALERT", "SKIP",
                f"Change does not require alert",
            )

            # Auto-approve if configured
            if self._should_auto_approve(change_analysis):
                await self._auto_approve(
                    domain, page_type, change_analysis, old_version, new_version
                )

            return None

        # Create alert
        alert = self._create_alert(url, domain, page_type, change_analysis)

        self.logger.info(
            "ALERT", "CREATE",
            f"Creating alert for {domain}/{page_type}",
            severity=alert.severity.value,
            alert_id=alert.id,
        )

        # Add to review queue if breaking
        if change_analysis.breaking and self.alerting_config.review_queue.require_review_breaking:
            review_item = await self._add_to_review_queue(
                domain, page_type, change_analysis, old_version, new_version
            )
            self.logger.info(
                "ALERT", "QUEUE",
                f"Added to review queue",
                review_id=review_item.id,
            )

        # Send notifications
        await self._send_notifications(alert)

        return alert

    def _should_alert(self, change_analysis: ChangeAnalysis) -> bool:
        """Determine if change should trigger alert."""
        # Always alert on breaking changes if configured
        if change_analysis.breaking and self.alerting_config.alert_on_breaking:
            return True

        # Alert on moderate changes if configured
        if (change_analysis.classification == ChangeClassification.MODERATE
            and self.alerting_config.alert_on_moderate):
            return True

        # Alert if below threshold
        if change_analysis.similarity < self.alerting_config.alert_threshold:
            return True

        return False

    def _should_auto_approve(self, change_analysis: ChangeAnalysis) -> bool:
        """Determine if change should be auto-approved."""
        review_config = self.alerting_config.review_queue

        if not review_config.enabled:
            return True

        if (change_analysis.classification == ChangeClassification.COSMETIC
            and review_config.auto_approve_cosmetic):
            return True

        if (change_analysis.classification == ChangeClassification.MINOR
            and review_config.auto_approve_minor):
            return True

        return False

    def _create_alert(
        self,
        url: str,
        domain: str,
        page_type: str,
        change_analysis: ChangeAnalysis,
    ) -> ChangeAlert:
        """Create alert from change analysis."""
        # Determine severity
        if change_analysis.breaking:
            severity = AlertSeverity.CRITICAL
        elif change_analysis.classification == ChangeClassification.MODERATE:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO

        return ChangeAlert(
            id=str(uuid.uuid4()),
            domain=domain,
            page_type=page_type,
            url=url,
            change_analysis=change_analysis,
            severity=severity,
        )

    async def _add_to_review_queue(
        self,
        domain: str,
        page_type: str,
        change_analysis: ChangeAnalysis,
        old_version: int,
        new_version: int,
    ) -> ReviewItem:
        """Add change to manual review queue."""
        review_item = ReviewItem(
            id=str(uuid.uuid4()),
            domain=domain,
            page_type=page_type,
            old_structure_version=old_version,
            new_structure_version=new_version,
            change_analysis=change_analysis,
        )

        await self.review_queue.add(review_item)
        return review_item

    async def _auto_approve(
        self,
        domain: str,
        page_type: str,
        change_analysis: ChangeAnalysis,
        old_version: int,
        new_version: int,
    ) -> None:
        """Auto-approve a minor/cosmetic change."""
        self.logger.info(
            "REVIEW", "AUTO_APPROVE",
            f"Auto-approving {change_analysis.classification.value} change",
            domain=domain,
            page_type=page_type,
        )

        # Create and immediately approve review item
        review_item = ReviewItem(
            id=str(uuid.uuid4()),
            domain=domain,
            page_type=page_type,
            old_structure_version=old_version,
            new_structure_version=new_version,
            change_analysis=change_analysis,
            status=ReviewStatus.AUTO_APPROVED,
            reviewer="system",
            reviewed_at=datetime.utcnow(),
        )

        await self.review_queue.add(review_item)

    async def _send_notifications(self, alert: ChangeAlert) -> None:
        """Send alert notifications to configured channels."""
        self.logger.debug(
            "ALERT", "SEND",
            f"Sending notifications for alert {alert.id}",
        )

        await self.notifier.notify(alert)
```

---

## fingerprint/alerting/review_queue.py

```python
"""
Manual review queue for structure changes.

Stores pending reviews in Redis for manual approval/rejection.

Verbose logging pattern:
[REVIEW:OPERATION] Message
"""

import json
from datetime import datetime
from typing import Any

import redis.asyncio as redis

from fingerprint.config import Config
from fingerprint.core.verbose import get_logger
from fingerprint.models import ReviewItem, ReviewStatus, ChangeAnalysis, ChangeClassification
from fingerprint.exceptions import StorageError


class ReviewQueue:
    """
    Manual review queue backed by Redis.

    Usage:
        queue = ReviewQueue(config)
        await queue.add(review_item)
        pending = await queue.get_pending()
        await queue.approve(item_id, reviewer="admin")
    """

    def __init__(self, config: Config):
        self.config = config
        self.redis_config = config.redis
        self.queue_config = config.alerting.review_queue
        self.logger = get_logger()

        self._client: redis.Redis | None = None

        self.logger.info(
            "REVIEW", "INIT",
            "Review queue initialized",
            max_size=self.queue_config.max_queue_size,
        )

    async def _get_client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._client = redis.from_url(self.redis_config.url)
        return self._client

    def _key(self, suffix: str = "") -> str:
        """Generate Redis key."""
        base = f"{self.redis_config.key_prefix}:review"
        return f"{base}:{suffix}" if suffix else base

    async def add(self, item: ReviewItem) -> None:
        """
        Add item to review queue.

        Verbose output:
            [REVIEW:ADD] Adding to review queue
              - domain: example.com
              - page_type: article
              - status: pending
        """
        if not self.queue_config.enabled:
            return

        client = await self._get_client()

        self.logger.info(
            "REVIEW", "ADD",
            f"Adding to review queue: {item.domain}/{item.page_type}",
            status=item.status.value,
        )

        # Check queue size
        current_size = await client.llen(self._key("pending"))
        if current_size >= self.queue_config.max_queue_size:
            self.logger.warn(
                "REVIEW", "QUEUE_FULL",
                f"Review queue full ({current_size}/{self.queue_config.max_queue_size})",
            )
            # Remove oldest item
            await client.lpop(self._key("pending"))

        # Serialize and add
        data = self._serialize(item)

        # Add to appropriate list based on status
        if item.status == ReviewStatus.PENDING:
            await client.rpush(self._key("pending"), data)
        else:
            await client.rpush(self._key("completed"), data)

        # Also store by ID for quick lookup
        await client.setex(
            self._key(f"item:{item.id}"),
            self.redis_config.ttl_seconds,
            data,
        )

    async def get(self, item_id: str) -> ReviewItem | None:
        """Get review item by ID."""
        client = await self._get_client()
        data = await client.get(self._key(f"item:{item_id}"))

        if data is None:
            return None

        return self._deserialize(data)

    async def get_pending(self, limit: int = 50) -> list[ReviewItem]:
        """
        Get pending review items.

        Verbose output:
            [REVIEW:QUEUE] Retrieved pending items
              - count: 15
        """
        client = await self._get_client()
        items_data = await client.lrange(self._key("pending"), 0, limit - 1)

        items = [self._deserialize(data) for data in items_data]

        self.logger.debug(
            "REVIEW", "QUEUE",
            f"Retrieved {len(items)} pending items",
        )

        return items

    async def get_by_domain(self, domain: str) -> list[ReviewItem]:
        """Get all pending reviews for a domain."""
        pending = await self.get_pending(limit=1000)
        return [item for item in pending if item.domain == domain]

    async def approve(
        self,
        item_id: str,
        reviewer: str,
        notes: str = "",
        auto_adapt: bool = True,
    ) -> ReviewItem | None:
        """
        Approve a review item.

        Verbose output:
            [REVIEW:APPROVE] Approving review
              - id: abc-123
              - reviewer: admin
              - auto_adapt: true
        """
        item = await self.get(item_id)
        if item is None:
            self.logger.warn("REVIEW", "NOT_FOUND", f"Review item not found: {item_id}")
            return None

        self.logger.info(
            "REVIEW", "APPROVE",
            f"Approving review {item_id}",
            reviewer=reviewer,
            auto_adapt=auto_adapt,
        )

        # Update item
        item.status = ReviewStatus.APPROVED
        item.reviewer = reviewer
        item.review_notes = notes
        item.reviewed_at = datetime.utcnow()
        item.auto_adapt = auto_adapt

        # Move from pending to completed
        await self._move_to_completed(item)

        return item

    async def reject(
        self,
        item_id: str,
        reviewer: str,
        notes: str = "",
    ) -> ReviewItem | None:
        """
        Reject a review item.

        Verbose output:
            [REVIEW:REJECT] Rejecting review
              - id: abc-123
              - reviewer: admin
        """
        item = await self.get(item_id)
        if item is None:
            self.logger.warn("REVIEW", "NOT_FOUND", f"Review item not found: {item_id}")
            return None

        self.logger.info(
            "REVIEW", "REJECT",
            f"Rejecting review {item_id}",
            reviewer=reviewer,
        )

        # Update item
        item.status = ReviewStatus.REJECTED
        item.reviewer = reviewer
        item.review_notes = notes
        item.reviewed_at = datetime.utcnow()

        # Move from pending to completed
        await self._move_to_completed(item)

        return item

    async def get_stats(self) -> dict[str, Any]:
        """Get review queue statistics."""
        client = await self._get_client()

        pending_count = await client.llen(self._key("pending"))
        completed_count = await client.llen(self._key("completed"))

        return {
            "pending": pending_count,
            "completed": completed_count,
            "max_size": self.queue_config.max_queue_size,
        }

    async def _move_to_completed(self, item: ReviewItem) -> None:
        """Move item from pending to completed."""
        client = await self._get_client()

        # Remove from pending (by value)
        old_data = await client.get(self._key(f"item:{item.id}"))
        if old_data:
            await client.lrem(self._key("pending"), 1, old_data)

        # Add to completed
        new_data = self._serialize(item)
        await client.rpush(self._key("completed"), new_data)

        # Update item store
        await client.setex(
            self._key(f"item:{item.id}"),
            self.redis_config.ttl_seconds,
            new_data,
        )

    def _serialize(self, item: ReviewItem) -> str:
        """Serialize review item to JSON."""
        data = {
            "id": item.id,
            "domain": item.domain,
            "page_type": item.page_type,
            "old_structure_version": item.old_structure_version,
            "new_structure_version": item.new_structure_version,
            "change_analysis": {
                "similarity": item.change_analysis.similarity,
                "classification": item.change_analysis.classification.value,
                "breaking": item.change_analysis.breaking,
                "mode_used": item.change_analysis.mode_used.value,
            },
            "status": item.status.value,
            "reviewer": item.reviewer,
            "review_notes": item.review_notes,
            "created_at": item.created_at.isoformat(),
            "reviewed_at": item.reviewed_at.isoformat() if item.reviewed_at else None,
            "auto_adapt": item.auto_adapt,
        }
        return json.dumps(data)

    def _deserialize(self, data: bytes | str) -> ReviewItem:
        """Deserialize JSON to review item."""
        if isinstance(data, bytes):
            data = data.decode("utf-8")

        obj = json.loads(data)

        # Reconstruct ChangeAnalysis (simplified)
        from fingerprint.models import FingerprintMode
        change_analysis = ChangeAnalysis(
            similarity=obj["change_analysis"]["similarity"],
            classification=ChangeClassification(obj["change_analysis"]["classification"]),
            breaking=obj["change_analysis"]["breaking"],
            mode_used=FingerprintMode(obj["change_analysis"]["mode_used"]),
        )

        return ReviewItem(
            id=obj["id"],
            domain=obj["domain"],
            page_type=obj["page_type"],
            old_structure_version=obj["old_structure_version"],
            new_structure_version=obj["new_structure_version"],
            change_analysis=change_analysis,
            status=ReviewStatus(obj["status"]),
            reviewer=obj["reviewer"],
            review_notes=obj["review_notes"],
            created_at=datetime.fromisoformat(obj["created_at"]),
            reviewed_at=datetime.fromisoformat(obj["reviewed_at"]) if obj["reviewed_at"] else None,
            auto_adapt=obj["auto_adapt"],
        )

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
```

---

## fingerprint/alerting/notifiers.py

```python
"""
Alert notification handlers.

Sends alerts to various channels: log, webhook, email.

Verbose logging pattern:
[NOTIFY:OPERATION] Message
"""

import json
from abc import ABC, abstractmethod
from typing import Any

import httpx

from fingerprint.config import Config
from fingerprint.core.verbose import get_logger
from fingerprint.models import ChangeAlert


class Notifier(ABC):
    """Base class for notification handlers."""

    @abstractmethod
    async def send(self, alert: ChangeAlert) -> bool:
        """Send alert notification. Returns True on success."""
        pass


class LogNotifier(Notifier):
    """Log-based notifier (always enabled)."""

    def __init__(self):
        self.logger = get_logger()

    async def send(self, alert: ChangeAlert) -> bool:
        """Log alert to verbose output."""
        self.logger.info(
            "NOTIFY", "LOG",
            f"ALERT [{alert.severity.value.upper()}]: {alert.domain}/{alert.page_type}",
            alert_id=alert.id,
            similarity=f"{alert.change_analysis.similarity:.2f}",
            classification=alert.change_analysis.classification.value,
            breaking=alert.change_analysis.breaking,
        )
        return True


class WebhookNotifier(Notifier):
    """Webhook-based notifier."""

    def __init__(self, url: str, timeout: int = 30):
        self.url = url
        self.timeout = timeout
        self.logger = get_logger()

    async def send(self, alert: ChangeAlert) -> bool:
        """Send alert to webhook URL."""
        if not self.url:
            return False

        payload = self._build_payload(alert)

        self.logger.debug(
            "NOTIFY", "WEBHOOK",
            f"Sending webhook to {self.url}",
            alert_id=alert.id,
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

            if response.status_code < 300:
                self.logger.info(
                    "NOTIFY", "WEBHOOK_SUCCESS",
                    f"Webhook sent successfully",
                    status=response.status_code,
                )
                return True
            else:
                self.logger.warn(
                    "NOTIFY", "WEBHOOK_FAILED",
                    f"Webhook failed",
                    status=response.status_code,
                )
                return False

        except Exception as e:
            self.logger.error("NOTIFY", "WEBHOOK_ERROR", str(e))
            return False

    def _build_payload(self, alert: ChangeAlert) -> dict[str, Any]:
        """Build webhook payload."""
        return {
            "alert_id": alert.id,
            "severity": alert.severity.value,
            "domain": alert.domain,
            "page_type": alert.page_type,
            "url": alert.url,
            "similarity": alert.change_analysis.similarity,
            "classification": alert.change_analysis.classification.value,
            "breaking": alert.change_analysis.breaking,
            "detected_at": alert.detected_at.isoformat(),
            "changes_count": len(alert.change_analysis.changes),
        }


class EmailNotifier(Notifier):
    """Email-based notifier."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        recipients: list[str],
        sender: str = "fingerprint@localhost",
        username: str | None = None,
        password: str | None = None,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.recipients = recipients
        self.sender = sender
        self.username = username
        self.password = password
        self.logger = get_logger()

    async def send(self, alert: ChangeAlert) -> bool:
        """Send alert via email."""
        if not self.recipients or not self.smtp_host:
            return False

        self.logger.debug(
            "NOTIFY", "EMAIL",
            f"Sending email to {len(self.recipients)} recipients",
            alert_id=alert.id,
        )

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # Build email
            msg = MIMEMultipart()
            msg["From"] = self.sender
            msg["To"] = ", ".join(self.recipients)
            msg["Subject"] = self._build_subject(alert)

            body = self._build_body(alert)
            msg.attach(MIMEText(body, "plain"))

            # Send
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.username and self.password:
                    server.starttls()
                    server.login(self.username, self.password)
                server.send_message(msg)

            self.logger.info(
                "NOTIFY", "EMAIL_SUCCESS",
                f"Email sent to {len(self.recipients)} recipients",
            )
            return True

        except Exception as e:
            self.logger.error("NOTIFY", "EMAIL_ERROR", str(e))
            return False

    def _build_subject(self, alert: ChangeAlert) -> str:
        """Build email subject."""
        severity = alert.severity.value.upper()
        return f"[{severity}] Structure change detected: {alert.domain}/{alert.page_type}"

    def _build_body(self, alert: ChangeAlert) -> str:
        """Build email body."""
        return f"""
Structure Change Alert
======================

Domain: {alert.domain}
Page Type: {alert.page_type}
URL: {alert.url}

Change Details:
- Severity: {alert.severity.value.upper()}
- Similarity: {alert.change_analysis.similarity:.2f}
- Classification: {alert.change_analysis.classification.value}
- Breaking: {alert.change_analysis.breaking}
- Changes Detected: {len(alert.change_analysis.changes)}

Alert ID: {alert.id}
Detected At: {alert.detected_at.isoformat()}

---
This is an automated alert from Adaptive Structure Fingerprinting System.
"""


class NotificationManager:
    """
    Manages multiple notification channels.

    Usage:
        manager = NotificationManager(config)
        await manager.notify(alert)
    """

    def __init__(self, config: Config):
        self.config = config
        self.notif_config = config.alerting.notifications
        self.logger = get_logger()

        # Initialize notifiers
        self._notifiers: list[Notifier] = []

        # Always add log notifier
        if self.notif_config.log:
            self._notifiers.append(LogNotifier())

        # Add webhook if configured
        if self.notif_config.webhook.enabled and self.notif_config.webhook.url:
            self._notifiers.append(WebhookNotifier(self.notif_config.webhook.url))

        # Add email if configured
        if self.notif_config.email.enabled and self.notif_config.email.smtp_host:
            self._notifiers.append(EmailNotifier(
                smtp_host=self.notif_config.email.smtp_host,
                smtp_port=self.notif_config.email.smtp_port,
                recipients=self.notif_config.email.recipients,
            ))

        self.logger.info(
            "NOTIFY", "INIT",
            f"Notification manager initialized with {len(self._notifiers)} channels",
        )

    async def notify(self, alert: ChangeAlert) -> dict[str, bool]:
        """
        Send alert to all configured channels.

        Returns dict of channel -> success status.
        """
        results = {}

        for notifier in self._notifiers:
            channel = notifier.__class__.__name__
            try:
                success = await notifier.send(alert)
                results[channel] = success
            except Exception as e:
                self.logger.error(
                    "NOTIFY", "ERROR",
                    f"Failed to send via {channel}: {e}",
                )
                results[channel] = False

        return results

    def add_notifier(self, notifier: Notifier) -> None:
        """Add custom notifier."""
        self._notifiers.append(notifier)
```

---

## Verbose Logging Examples

### Change Detection and Alerting

```
[2024-01-15T10:30:00Z] [ALERT:INIT] Change monitor initialized
  - alert_threshold: 0.70

[2024-01-15T10:30:01Z] [ALERT:DETECT] Change detected for example.com/article
  - similarity: 0.65
  - classification: breaking

[2024-01-15T10:30:01Z] [ALERT:CREATE] Creating alert for example.com/article
  - severity: critical
  - alert_id: abc-123-def

[2024-01-15T10:30:01Z] [ALERT:QUEUE] Added to review queue
  - review_id: xyz-456-uvw

[2024-01-15T10:30:01Z] [NOTIFY:LOG] ALERT [CRITICAL]: example.com/article
  - alert_id: abc-123-def
  - similarity: 0.65
  - classification: breaking
  - breaking: true

[2024-01-15T10:30:02Z] [NOTIFY:WEBHOOK] Sending webhook to https://hooks.example.com/alerts
  - alert_id: abc-123-def

[2024-01-15T10:30:02Z] [NOTIFY:WEBHOOK_SUCCESS] Webhook sent successfully
  - status: 200
```

### Review Queue

```
[2024-01-15T10:30:00Z] [REVIEW:INIT] Review queue initialized
  - max_size: 1000

[2024-01-15T10:30:01Z] [REVIEW:ADD] Adding to review queue: example.com/article
  - status: pending

[2024-01-15T11:00:00Z] [REVIEW:APPROVE] Approving review xyz-456-uvw
  - reviewer: admin
  - auto_adapt: true

[2024-01-15T11:00:01Z] [REVIEW:AUTO_APPROVE] Auto-approving cosmetic change
  - domain: another-site.com
  - page_type: blog
```

---

## Redis Key Schema

```
# Pending reviews (list)
{prefix}:review:pending

# Completed reviews (list)
{prefix}:review:completed

# Individual review items (hash)
{prefix}:review:item:{item_id}
```

---

## Review Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                      REVIEW WORKFLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Change Detected                                                │
│       │                                                         │
│       ▼                                                         │
│  Classification ──────┬─────────────────────────────────────┐  │
│       │               │                                     │  │
│       │          COSMETIC/MINOR              MODERATE/BREAKING │
│       │               │                          │          │  │
│       │               ▼                          ▼          │  │
│       │      Auto-Approve?              Add to Review Queue │  │
│       │          │    │                          │          │  │
│       │        YES   NO                          │          │  │
│       │          │    │                          │          │  │
│       │          ▼    └──────────────────────────┤          │  │
│       │   AUTO_APPROVED                          │          │  │
│       │          │                               │          │  │
│       │          │                               ▼          │  │
│       │          │                     Manual Review        │  │
│       │          │                        │      │          │  │
│       │          │                   APPROVE   REJECT       │  │
│       │          │                        │      │          │  │
│       │          │                        ▼      ▼          │  │
│       │          │                   Adapt    Revert        │  │
│       │          │                   Strategy               │  │
│       │          │                        │                 │  │
│       └──────────┴────────────────────────┴─────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Usage Example

```python
from fingerprint.alerting import ChangeMonitor, ReviewQueue, NotificationManager
from fingerprint.config import load_config

async def setup_alerting():
    config = load_config()

    # Initialize components
    review_queue = ReviewQueue(config)
    notifier = NotificationManager(config)
    monitor = ChangeMonitor(config, review_queue, notifier)

    return monitor, review_queue

async def process_comparison(url: str, change_analysis: ChangeAnalysis):
    monitor, review_queue = await setup_alerting()

    # Process change and generate alert if needed
    alert = await monitor.process_change(
        url=url,
        domain="example.com",
        page_type="article",
        change_analysis=change_analysis,
        old_version=3,
        new_version=4,
    )

    if alert:
        print(f"Alert generated: {alert.id}")

    # Check review queue
    pending = await review_queue.get_pending()
    print(f"Pending reviews: {len(pending)}")

    # Approve a review
    if pending:
        item = await review_queue.approve(
            pending[0].id,
            reviewer="admin",
            auto_adapt=True,
        )
        print(f"Approved: {item.id}")
```
