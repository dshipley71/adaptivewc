"""
Pytest configuration and shared fixtures.
"""

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio as the async backend."""
    return "asyncio"


# =============================================================================
# Mock Redis (using fakeredis when available)
# =============================================================================


@pytest_asyncio.fixture
async def mock_redis() -> AsyncGenerator[Any, None]:
    """
    Provide a mock Redis client for testing.

    Uses fakeredis if available, otherwise skips tests requiring Redis.
    """
    try:
        import fakeredis.aioredis

        redis = fakeredis.aioredis.FakeRedis()
        yield redis
        await redis.close()
    except ImportError:
        pytest.skip("fakeredis not installed")


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_robots_txt_permissive() -> str:
    """Robots.txt that allows most crawling."""
    return """
User-agent: *
Allow: /
Crawl-delay: 1
Sitemap: https://example.com/sitemap.xml
    """.strip()


@pytest.fixture
def sample_robots_txt_restrictive() -> str:
    """Robots.txt that blocks most crawling."""
    return """
User-agent: *
Disallow: /

User-agent: Googlebot
Allow: /
    """.strip()


@pytest.fixture
def sample_html_article() -> str:
    """Sample article HTML for extraction testing."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Sample Article Title</title>
    <meta name="author" content="John Doe">
    <meta name="description" content="This is a sample article for testing.">
    <meta property="og:title" content="Sample Article Title">
</head>
<body>
    <header>
        <nav>
            <a href="/">Home</a>
            <a href="/about">About</a>
        </nav>
    </header>
    <main>
        <article itemscope itemtype="https://schema.org/Article">
            <h1 itemprop="headline">Sample Article Title</h1>
            <div class="meta">
                <span itemprop="author">John Doe</span>
                <time itemprop="datePublished" datetime="2025-01-15">January 15, 2025</time>
            </div>
            <div class="content" itemprop="articleBody">
                <p>This is the first paragraph of the article content.</p>
                <p>This is the second paragraph with more details.</p>
                <p>And a third paragraph to make it substantial.</p>
            </div>
        </article>
    </main>
    <footer>
        <p>&copy; 2025 Example Site</p>
    </footer>
</body>
</html>
    """.strip()


@pytest.fixture
def sample_html_spa() -> str:
    """Sample SPA HTML that needs JavaScript rendering."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Loading...</title>
    <script src="/static/react.js"></script>
</head>
<body>
    <div id="root"></div>
    <script>
        window.__INITIAL_STATE__ = {};
        ReactDOM.render(App, document.getElementById('root'));
    </script>
</body>
</html>
    """.strip()


@pytest.fixture
def sample_page_structure() -> dict[str, Any]:
    """Sample PageStructure as a dictionary."""
    return {
        "domain": "example.com",
        "page_type": "article",
        "url_pattern": r"^/blog/[\w-]+/?$",
        "tag_hierarchy": {
            "html": {
                "body": {
                    "main": {
                        "article": {
                            "h1": {},
                            "div.content": {},
                        }
                    }
                }
            }
        },
        "iframe_locations": [],
        "css_class_map": {"content": 1, "meta": 1},
        "captured_at": "2025-01-15T10:00:00Z",
        "version": 1,
        "content_hash": "abc123",
    }


@pytest.fixture
def sample_extraction_strategy() -> dict[str, Any]:
    """Sample ExtractionStrategy as a dictionary."""
    return {
        "domain": "example.com",
        "page_type": "article",
        "version": 1,
        "title": {
            "primary": "article h1",
            "fallbacks": ["h1", "title"],
            "extraction_method": "text",
        },
        "content": {
            "primary": "article .content",
            "fallbacks": ["article", "main"],
            "extraction_method": "text",
        },
        "learned_at": "2025-01-15T10:00:00Z",
        "learning_source": "initial",
    }
