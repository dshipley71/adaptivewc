"""
JavaScript rendering for the adaptive web crawler.

Uses Playwright to render JavaScript-heavy pages (SPAs, dynamic content).

Features:
- Automatic detection of JS-required pages
- Configurable wait strategies (networkidle, domcontentloaded, load)
- Screenshot capture for debugging
- Resource blocking (images, fonts) for performance
- Headless browser management
- Concurrent rendering with browser pool
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal
from urllib.parse import urlparse

from crawler.utils.logging import CrawlerLogger


class WaitStrategy(str, Enum):
    """Page load wait strategies."""

    LOAD = "load"  # Wait for 'load' event
    DOMCONTENTLOADED = "domcontentloaded"  # Wait for DOMContentLoaded
    NETWORKIDLE = "networkidle"  # Wait until no network activity for 500ms
    COMMIT = "commit"  # Wait for initial response


class ResourceType(str, Enum):
    """Browser resource types that can be blocked."""

    DOCUMENT = "document"
    STYLESHEET = "stylesheet"
    IMAGE = "image"
    MEDIA = "media"
    FONT = "font"
    SCRIPT = "script"
    TEXTTRACK = "texttrack"
    XHR = "xhr"
    FETCH = "fetch"
    EVENTSOURCE = "eventsource"
    WEBSOCKET = "websocket"
    MANIFEST = "manifest"
    OTHER = "other"


@dataclass
class RenderConfig:
    """Configuration for JavaScript rendering."""

    # Wait strategy
    wait_until: WaitStrategy = WaitStrategy.NETWORKIDLE
    timeout: float = 30.0  # seconds

    # Browser settings
    headless: bool = True
    viewport_width: int = 1920
    viewport_height: int = 1080

    # Resource blocking for performance
    block_resources: list[ResourceType] = field(
        default_factory=lambda: [
            ResourceType.IMAGE,
            ResourceType.MEDIA,
            ResourceType.FONT,
        ]
    )

    # JavaScript execution
    wait_for_selector: str | None = None  # CSS selector to wait for
    wait_for_timeout: float | None = None  # Additional wait time (seconds)
    evaluate_script: str | None = None  # Script to run after page load

    # Screenshots for debugging
    capture_screenshot: bool = False
    screenshot_path: str | None = None

    # User agent override
    user_agent: str | None = None

    # Extra HTTP headers
    extra_headers: dict[str, str] = field(default_factory=dict)

    # Cookies to set before navigation
    cookies: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RenderResult:
    """Result of rendering a page."""

    url: str
    html: str
    status_code: int
    success: bool
    rendered_at: datetime = field(default_factory=datetime.utcnow)
    error: str | None = None
    screenshot: bytes | None = None
    title: str | None = None
    final_url: str | None = None  # After redirects
    render_time_ms: float = 0.0
    js_errors: list[str] = field(default_factory=list)
    console_messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "success": self.success,
            "status_code": self.status_code,
            "rendered_at": self.rendered_at.isoformat(),
            "error": self.error,
            "title": self.title,
            "final_url": self.final_url,
            "render_time_ms": self.render_time_ms,
            "html_length": len(self.html) if self.html else 0,
            "js_errors": self.js_errors,
        }


class JSRequirementDetector:
    """
    Detects whether a page likely requires JavaScript rendering.

    Uses heuristics based on HTML content to determine if JS is needed.
    """

    # Indicators that suggest JS rendering is needed
    JS_FRAMEWORK_PATTERNS = [
        # React
        'id="root"',
        'id="app"',
        'data-reactroot',
        "__NEXT_DATA__",
        # Vue
        'id="__nuxt"',
        "data-v-",
        # Angular
        "ng-app",
        "ng-view",
        "_nghost",
        # Svelte
        "svelte-",
        # Generic SPA patterns
        "window.__INITIAL_STATE__",
        "window.__PRELOADED_STATE__",
        "application/json",
    ]

    # Minimal content indicators
    MINIMAL_CONTENT_PATTERNS = [
        "<body></body>",
        "<body>\n</body>",
        '<div id="root"></div>',
        '<div id="app"></div>',
    ]

    # Script-heavy indicators
    SCRIPT_HEAVY_THRESHOLD = 5  # Number of script tags

    def __init__(self, logger: CrawlerLogger | None = None):
        """Initialize the detector."""
        self.logger = logger or CrawlerLogger("js_detector")

    def requires_js(self, html: str, url: str | None = None) -> tuple[bool, str]:
        """
        Determine if a page requires JavaScript rendering.

        Args:
            html: Raw HTML content from initial fetch.
            url: Optional URL for context.

        Returns:
            Tuple of (requires_js, reason).
        """
        html_lower = html.lower()

        # Check for framework patterns
        for pattern in self.JS_FRAMEWORK_PATTERNS:
            if pattern.lower() in html_lower:
                return True, f"Framework pattern detected: {pattern}"

        # Check for minimal content (empty body with JS)
        for pattern in self.MINIMAL_CONTENT_PATTERNS:
            if pattern.lower() in html_lower:
                return True, f"Minimal content pattern: {pattern}"

        # Count script tags
        script_count = html_lower.count("<script")
        if script_count >= self.SCRIPT_HEAVY_THRESHOLD:
            # Also check if there's minimal visible content
            text_content = self._extract_visible_text(html)
            if len(text_content) < 500:  # Less than 500 chars of visible text
                return True, f"Script-heavy ({script_count} scripts) with minimal content"

        # Check for noscript warnings
        if "<noscript" in html_lower:
            noscript_content = self._extract_noscript(html)
            if any(
                warn in noscript_content.lower()
                for warn in ["enable javascript", "javascript required", "javascript is required"]
            ):
                return True, "Noscript warning detected"

        return False, "No JS indicators found"

    def _extract_visible_text(self, html: str) -> str:
        """Extract approximate visible text from HTML."""
        # Simple extraction - remove tags and scripts
        import re

        # Remove script and style content
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Remove tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_noscript(self, html: str) -> str:
        """Extract noscript content."""
        import re

        matches = re.findall(
            r"<noscript[^>]*>(.*?)</noscript>", html, flags=re.DOTALL | re.IGNORECASE
        )
        return " ".join(matches)


class BrowserPool:
    """
    Manages a pool of browser contexts for concurrent rendering.

    Provides efficient reuse of browser instances while allowing
    concurrent page rendering.
    """

    def __init__(
        self,
        max_contexts: int = 5,
        headless: bool = True,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the browser pool.

        Args:
            max_contexts: Maximum number of concurrent browser contexts.
            headless: Whether to run browsers in headless mode.
            logger: Logger instance.
        """
        self.max_contexts = max_contexts
        self.headless = headless
        self.logger = logger or CrawlerLogger("browser_pool")

        self._browser = None
        self._playwright = None
        self._semaphore = asyncio.Semaphore(max_contexts)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the browser pool."""
        if self._initialized:
            return

        try:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
            )
            self._initialized = True
            self.logger.info(
                "Browser pool initialized",
                headless=self.headless,
                max_contexts=self.max_contexts,
            )

        except ImportError:
            raise ImportError(
                "Playwright is required for JS rendering. "
                "Install with: pip install playwright && playwright install chromium"
            )

    async def close(self) -> None:
        """Close the browser pool."""
        if self._browser:
            await self._browser.close()
            self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        self._initialized = False
        self.logger.info("Browser pool closed")

    async def acquire_context(self, config: RenderConfig):
        """
        Acquire a browser context from the pool.

        Args:
            config: Render configuration.

        Returns:
            Browser context (must be closed when done).
        """
        await self._semaphore.acquire()

        if not self._initialized:
            await self.initialize()

        context = await self._browser.new_context(
            viewport={
                "width": config.viewport_width,
                "height": config.viewport_height,
            },
            user_agent=config.user_agent,
            extra_http_headers=config.extra_headers or None,
        )

        # Set cookies if provided
        if config.cookies:
            await context.add_cookies(config.cookies)

        return context

    def release_context(self) -> None:
        """Release a context back to the pool."""
        self._semaphore.release()

    async def __aenter__(self) -> "BrowserPool":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


class JSRenderer:
    """
    JavaScript renderer using Playwright.

    Provides async rendering of JavaScript-heavy pages with
    configurable wait strategies and resource management.
    """

    def __init__(
        self,
        config: RenderConfig | None = None,
        browser_pool: BrowserPool | None = None,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the renderer.

        Args:
            config: Default render configuration.
            browser_pool: Shared browser pool (creates own if None).
            logger: Logger instance.
        """
        self.config = config or RenderConfig()
        self._external_pool = browser_pool is not None
        self.browser_pool = browser_pool or BrowserPool(
            headless=self.config.headless,
        )
        self.logger = logger or CrawlerLogger("js_renderer")
        self.detector = JSRequirementDetector(logger=self.logger)

    async def __aenter__(self) -> "JSRenderer":
        """Async context manager entry."""
        if not self._external_pool:
            await self.browser_pool.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if not self._external_pool:
            await self.browser_pool.close()

    async def render(
        self,
        url: str,
        config: RenderConfig | None = None,
    ) -> RenderResult:
        """
        Render a page with JavaScript.

        Args:
            url: URL to render.
            config: Optional config override for this render.

        Returns:
            RenderResult with rendered HTML.
        """
        cfg = config or self.config
        start_time = asyncio.get_event_loop().time()

        result = RenderResult(
            url=url,
            html="",
            status_code=0,
            success=False,
        )

        context = None
        page = None

        try:
            context = await self.browser_pool.acquire_context(cfg)
            page = await context.new_page()

            # Set up resource blocking
            if cfg.block_resources:
                blocked_types = [r.value for r in cfg.block_resources]
                await page.route(
                    "**/*",
                    lambda route: (
                        route.abort()
                        if route.request.resource_type in blocked_types
                        else route.continue_()
                    ),
                )

            # Capture console messages and errors
            page.on("console", lambda msg: result.console_messages.append(msg.text))
            page.on(
                "pageerror",
                lambda err: result.js_errors.append(str(err)),
            )

            # Navigate to the page
            response = await page.goto(
                url,
                wait_until=cfg.wait_until.value,
                timeout=cfg.timeout * 1000,  # Convert to ms
            )

            if response:
                result.status_code = response.status
                result.final_url = response.url

            # Wait for specific selector if configured
            if cfg.wait_for_selector:
                await page.wait_for_selector(
                    cfg.wait_for_selector,
                    timeout=cfg.timeout * 1000,
                )

            # Additional wait time if configured
            if cfg.wait_for_timeout:
                await asyncio.sleep(cfg.wait_for_timeout)

            # Execute custom script if configured
            if cfg.evaluate_script:
                await page.evaluate(cfg.evaluate_script)

            # Get the rendered HTML
            result.html = await page.content()
            result.title = await page.title()

            # Capture screenshot if configured
            if cfg.capture_screenshot:
                result.screenshot = await page.screenshot(
                    path=cfg.screenshot_path,
                    full_page=True,
                )

            result.success = True

        except Exception as e:
            error_msg = str(e)
            result.error = error_msg
            self.logger.warning("Render failed", url=url, error=error_msg)

        finally:
            if page:
                await page.close()
            if context:
                await context.close()
                self.browser_pool.release_context()

        result.render_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

        self.logger.debug(
            "Render complete",
            url=url,
            success=result.success,
            render_time_ms=result.render_time_ms,
        )

        return result

    async def render_if_needed(
        self,
        url: str,
        initial_html: str,
        config: RenderConfig | None = None,
    ) -> RenderResult:
        """
        Render a page only if JavaScript is detected as required.

        Args:
            url: URL of the page.
            initial_html: HTML from initial HTTP fetch.
            config: Optional render configuration.

        Returns:
            RenderResult (either from rendering or wrapped initial HTML).
        """
        requires_js, reason = self.detector.requires_js(initial_html, url)

        if requires_js:
            self.logger.info(
                "JS rendering required",
                url=url,
                reason=reason,
            )
            return await self.render(url, config)

        # Return initial HTML as render result
        return RenderResult(
            url=url,
            html=initial_html,
            status_code=200,
            success=True,
            final_url=url,
        )

    def check_js_required(self, html: str, url: str | None = None) -> tuple[bool, str]:
        """
        Check if a page requires JavaScript rendering.

        Args:
            html: HTML content to check.
            url: Optional URL for context.

        Returns:
            Tuple of (requires_js, reason).
        """
        return self.detector.requires_js(html, url)


class HybridFetcher:
    """
    Hybrid fetcher that automatically switches between HTTP and JS rendering.

    Attempts regular HTTP fetch first, then falls back to JS rendering
    if the page is detected as requiring JavaScript.
    """

    def __init__(
        self,
        http_client,
        renderer: JSRenderer | None = None,
        render_config: RenderConfig | None = None,
        always_render_patterns: list[str] | None = None,
        never_render_patterns: list[str] | None = None,
        logger: CrawlerLogger | None = None,
    ):
        """
        Initialize the hybrid fetcher.

        Args:
            http_client: HTTP client for regular fetches.
            renderer: JS renderer (creates if None).
            render_config: Default render configuration.
            always_render_patterns: URL patterns to always render.
            never_render_patterns: URL patterns to never render.
            logger: Logger instance.
        """
        self.http_client = http_client
        self._external_renderer = renderer is not None
        self.renderer = renderer
        self.render_config = render_config or RenderConfig()
        self.always_render = always_render_patterns or []
        self.never_render = never_render_patterns or []
        self.logger = logger or CrawlerLogger("hybrid_fetcher")
        self._detector = JSRequirementDetector(logger=self.logger)

    async def __aenter__(self) -> "HybridFetcher":
        """Async context manager entry."""
        if self.renderer is None:
            self.renderer = JSRenderer(config=self.render_config, logger=self.logger)
        if not self._external_renderer:
            await self.renderer.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if not self._external_renderer and self.renderer:
            await self.renderer.__aexit__(exc_type, exc_val, exc_tb)

    def _matches_patterns(self, url: str, patterns: list[str]) -> bool:
        """Check if URL matches any pattern."""
        import re

        for pattern in patterns:
            if re.search(pattern, url):
                return True
        return False

    async def fetch(
        self,
        url: str,
        render_config: RenderConfig | None = None,
    ) -> tuple[str, int, bool]:
        """
        Fetch a URL, using JS rendering if needed.

        Args:
            url: URL to fetch.
            render_config: Optional render configuration.

        Returns:
            Tuple of (html, status_code, was_rendered).
        """
        # Check never-render patterns first
        if self._matches_patterns(url, self.never_render):
            return await self._http_fetch(url)

        # Check always-render patterns
        if self._matches_patterns(url, self.always_render):
            result = await self.renderer.render(url, render_config)
            return result.html, result.status_code, True

        # Try HTTP fetch first
        html, status_code = await self._http_fetch(url)

        if status_code != 200:
            return html, status_code, False

        # Check if JS rendering is needed
        requires_js, reason = self._detector.requires_js(html, url)

        if requires_js:
            self.logger.info("Falling back to JS rendering", url=url, reason=reason)
            result = await self.renderer.render(url, render_config)
            return result.html, result.status_code, True

        return html, status_code, False

    async def _http_fetch(self, url: str) -> tuple[str, int]:
        """Perform regular HTTP fetch."""
        try:
            response = await self.http_client.get(url)
            return response.text, response.status_code
        except Exception as e:
            self.logger.warning("HTTP fetch failed", url=url, error=str(e))
            return "", 0
