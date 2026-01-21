"""
Tests for the rate limiter compliance module.

Demonstrates testing patterns for the adaptive web crawler.
"""

import asyncio
from datetime import datetime, timedelta

import pytest

from crawler.config import RateLimitConfig
from crawler.models import FetchResult, FetchStatus


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that default configuration values are sensible."""
        config = RateLimitConfig()

        assert config.default_delay == 1.0
        assert config.min_delay == 0.5
        assert config.max_delay == 60.0
        assert config.respect_crawl_delay is True
        assert config.adaptive is True
        assert config.max_concurrent_per_domain == 1
        assert config.max_concurrent_global == 10

    def test_custom_values(self) -> None:
        """Test that custom values are accepted."""
        config = RateLimitConfig(
            default_delay=2.0,
            min_delay=1.0,
            max_delay=120.0,
            respect_crawl_delay=False,
        )

        assert config.default_delay == 2.0
        assert config.min_delay == 1.0
        assert config.max_delay == 120.0
        assert config.respect_crawl_delay is False


class TestFetchResult:
    """Tests for FetchResult model."""

    def test_success_creation(self) -> None:
        """Test creating a successful fetch result."""
        result = FetchResult.success(
            url="https://example.com/page",
            content=b"<html><body>Hello</body></html>",
            status_code=200,
            headers={"content-type": "text/html"},
            duration_ms=150.0,
        )

        assert result.is_success()
        assert result.status == FetchStatus.SUCCESS
        assert result.status_code == 200
        assert result.url == "https://example.com/page"
        assert result.html is not None
        assert "Hello" in result.html
        assert result.duration_ms == 150.0

    def test_blocked_creation(self) -> None:
        """Test creating a blocked fetch result."""
        result = FetchResult.blocked(
            url="https://example.com/private",
            reason="robots.txt",
            status=FetchStatus.BLOCKED_ROBOTS,
        )

        assert not result.is_success()
        assert result.status == FetchStatus.BLOCKED_ROBOTS
        assert result.error_message == "robots.txt"

    def test_error_creation(self) -> None:
        """Test creating an error fetch result."""
        result = FetchResult.error(
            url="https://example.com/slow",
            message="Timeout after 30s",
            status=FetchStatus.ERROR_TIMEOUT,
        )

        assert not result.is_success()
        assert result.status == FetchStatus.ERROR_TIMEOUT
        assert "Timeout" in result.error_message


# =============================================================================
# Example async test (for when rate limiter is implemented)
# =============================================================================


class TestRateLimiterPlaceholder:
    """Placeholder tests for rate limiter implementation."""

    @pytest.mark.asyncio
    async def test_acquire_respects_delay(self) -> None:
        """Test that acquiring a rate limit slot respects configured delay."""
        # TODO: Implement when RateLimiter class is created
        #
        # config = RateLimitConfig(default_delay=0.1)
        # limiter = RateLimiter(config)
        #
        # start = datetime.utcnow()
        # await limiter.acquire("example.com")
        # await limiter.acquire("example.com")
        # elapsed = (datetime.utcnow() - start).total_seconds()
        #
        # assert elapsed >= 0.1
        pass

    @pytest.mark.asyncio
    async def test_different_domains_independent(self) -> None:
        """Test that rate limits are independent per domain."""
        # TODO: Implement when RateLimiter class is created
        #
        # config = RateLimitConfig(default_delay=1.0)
        # limiter = RateLimiter(config)
        #
        # # These should not wait for each other
        # start = datetime.utcnow()
        # await asyncio.gather(
        #     limiter.acquire("example.com"),
        #     limiter.acquire("other.com"),
        # )
        # elapsed = (datetime.utcnow() - start).total_seconds()
        #
        # # Should be nearly instant since different domains
        # assert elapsed < 0.1
        pass


# =============================================================================
# Fixtures for future tests
# =============================================================================


@pytest.fixture
def sample_robots_txt() -> str:
    """Sample robots.txt for testing."""
    return """
User-agent: *
Disallow: /private/
Disallow: /api/
Crawl-delay: 2

User-agent: AdaptiveCrawler
Allow: /api/public/
Disallow: /api/internal/
    """.strip()


@pytest.fixture
def sample_html() -> str:
    """Sample HTML page for extraction testing."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Test Article</title>
    <meta name="author" content="Test Author">
</head>
<body>
    <article class="post">
        <h1 class="title">Test Article Title</h1>
        <div class="content">
            <p>This is the main content of the article.</p>
            <p>It has multiple paragraphs.</p>
        </div>
        <aside class="sidebar">
            <iframe src="https://youtube.com/embed/123" class="video-embed"></iframe>
        </aside>
    </article>
</body>
</html>
    """.strip()


@pytest.fixture
def sample_html_redesigned() -> str:
    """Same content with different structure (for change detection tests)."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Test Article</title>
    <meta name="author" content="Test Author">
</head>
<body>
    <main>
        <article>
            <header>
                <h1>Test Article Title</h1>
            </header>
            <section class="article-body">
                <p>This is the main content of the article.</p>
                <p>It has multiple paragraphs.</p>
                <div class="video-container">
                    <iframe src="https://youtube.com/embed/123"></iframe>
                </div>
            </section>
        </article>
    </main>
</body>
</html>
    """.strip()
