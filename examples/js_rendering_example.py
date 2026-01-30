#!/usr/bin/env python3
"""
JavaScript Rendering Example

Demonstrates how to use Playwright integration to render JavaScript-heavy
pages, including SPAs (Single Page Applications) built with React, Vue,
Angular, and similar frameworks.

Features demonstrated:
- Rendering pages with JavaScript execution
- Automatic JS requirement detection
- Wait strategies (networkidle, selector, function)
- Browser pool management for high-volume rendering
- Hybrid fetching (HTTP + JS when needed)
- Screenshot capture

Usage:
    # First, install Playwright
    pip install playwright
    playwright install chromium

    # Basic usage - render a JavaScript-heavy page
    python examples/js_rendering_example.py --url https://example-spa.com

    # With screenshot
    python examples/js_rendering_example.py --url https://example.com --screenshot

    # Compare HTTP vs JS rendering
    python examples/js_rendering_example.py --url https://example.com --compare

Requirements:
    - pip install -e ".[js-rendering]"
    - playwright install chromium
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from crawler.core.renderer import (
    JSRenderer,
    BrowserPool,
    HybridFetcher,
    JSRequirementDetector,
    RenderConfig,
    RenderResult,
    WaitStrategy,
)
from crawler.utils.logging import CrawlerLogger, setup_logging


class JSRenderingDemo:
    """
    Demonstrates JavaScript rendering capabilities.

    Key concepts:
    - Modern SPAs render content via JavaScript
    - Simple HTTP fetch returns only the initial HTML (often empty)
    - JS rendering executes JavaScript to get the full content
    - Auto-detection identifies when JS rendering is needed
    """

    def __init__(self, verbose: bool = False):
        """Initialize the demo."""
        self.logger = CrawlerLogger("js_rendering_demo")
        self.verbose = verbose

    async def render_page(
        self,
        url: str,
        wait_strategy: WaitStrategy = WaitStrategy.NETWORKIDLE,
        capture_screenshot: bool = False,
        timeout_ms: int = 30000,
    ) -> RenderResult:
        """
        Render a page using Playwright.

        This demonstrates basic JS rendering with configurable options.

        Args:
            url: URL to render
            wait_strategy: When to consider page "loaded"
            capture_screenshot: Whether to capture a screenshot
            timeout_ms: Timeout for page load

        Returns:
            RenderResult with HTML, status, timing, etc.
        """
        config = RenderConfig(
            wait_strategy=wait_strategy,
            timeout_ms=timeout_ms,
            capture_screenshot=capture_screenshot,
            capture_console=True,
            viewport_width=1280,
            viewport_height=720,
        )

        async with JSRenderer() as renderer:
            result = await renderer.render(url, config)

            self.logger.info(
                "Page rendered",
                url=url,
                status=result.status_code,
                html_length=len(result.html),
                render_time_ms=result.render_time_ms,
                console_logs=len(result.console_logs),
            )

            if result.console_logs and self.verbose:
                print("\nConsole logs from page:")
                for log in result.console_logs[:5]:
                    print(f"  [{log['type']}] {log['text'][:100]}")

            return result

    async def compare_http_vs_js(self, url: str) -> dict:
        """
        Compare HTTP fetch vs JS rendering for the same URL.

        This demonstrates why JS rendering is sometimes necessary.

        Args:
            url: URL to compare

        Returns:
            Comparison results
        """
        import httpx

        print(f"\n{'='*60}")
        print("COMPARING HTTP FETCH vs JAVASCRIPT RENDERING")
        print(f"{'='*60}")
        print(f"URL: {url}\n")

        # HTTP fetch
        print("1. Fetching with HTTP (no JavaScript)...")
        async with httpx.AsyncClient(follow_redirects=True) as client:
            http_start = datetime.now()
            response = await client.get(url)
            http_time = (datetime.now() - http_start).total_seconds() * 1000
            http_html = response.text

        print(f"   Status: {response.status_code}")
        print(f"   HTML length: {len(http_html)} characters")
        print(f"   Time: {http_time:.0f}ms")

        # JS rendering
        print("\n2. Rendering with JavaScript (Playwright)...")
        async with JSRenderer() as renderer:
            js_start = datetime.now()
            js_result = await renderer.render(url)
            js_time = (datetime.now() - js_start).total_seconds() * 1000

        print(f"   Status: {js_result.status_code}")
        print(f"   HTML length: {len(js_result.html)} characters")
        print(f"   Time: {js_time:.0f}ms")

        # Analysis
        http_length = len(http_html)
        js_length = len(js_result.html)
        length_ratio = js_length / http_length if http_length > 0 else 0

        print(f"\n{'='*60}")
        print("ANALYSIS")
        print(f"{'='*60}")

        if length_ratio > 1.5:
            print(f"JS rendering produced {length_ratio:.1f}x more content!")
            print("This page likely requires JavaScript rendering.")
        elif length_ratio > 1.1:
            print(f"JS rendering added {(length_ratio-1)*100:.0f}% more content.")
            print("JavaScript may be used for some dynamic content.")
        else:
            print("HTTP and JS rendering produced similar content.")
            print("This page may not require JavaScript rendering.")

        # Check for SPA indicators
        detector = JSRequirementDetector()
        needs_js, indicators = detector.analyze(http_html)

        if needs_js:
            print(f"\nSPA Framework Indicators Found:")
            for indicator in indicators:
                print(f"  - {indicator}")

        return {
            "url": url,
            "http_length": http_length,
            "js_length": js_length,
            "length_ratio": length_ratio,
            "http_time_ms": http_time,
            "js_time_ms": js_time,
            "needs_js": needs_js,
            "indicators": indicators,
        }

    async def demo_smart_rendering(self, url: str) -> RenderResult:
        """
        Demonstrate smart rendering that only uses JS when needed.

        The HybridFetcher first tries HTTP, then checks if JS is needed.

        Args:
            url: URL to fetch

        Returns:
            Render result (from HTTP or JS rendering)
        """
        print(f"\n{'='*60}")
        print("SMART RENDERING DEMO")
        print(f"{'='*60}")
        print("Using HybridFetcher to automatically detect JS requirements\n")

        async with JSRenderer() as renderer:
            async with HybridFetcher(js_renderer=renderer) as fetcher:
                result = await fetcher.fetch(url)

                if result.used_js_rendering:
                    print(f"Result: JavaScript rendering was REQUIRED")
                    print(f"  Reason: {result.js_reason}")
                else:
                    print(f"Result: HTTP fetch was SUFFICIENT")
                    print(f"  JavaScript rendering was not needed")

                print(f"\nFinal HTML length: {len(result.html)} characters")

                return result

    async def demo_wait_strategies(self, url: str) -> dict:
        """
        Demonstrate different wait strategies.

        Wait strategies determine when a page is considered "loaded":
        - load: Window onload event
        - domcontentloaded: DOM ready
        - networkidle: No network activity for 500ms
        - selector: Specific element appears

        Args:
            url: URL to test

        Returns:
            Results for each strategy
        """
        print(f"\n{'='*60}")
        print("WAIT STRATEGY COMPARISON")
        print(f"{'='*60}")
        print(f"URL: {url}\n")

        strategies = [
            (WaitStrategy.DOMCONTENTLOADED, "DOM Content Loaded"),
            (WaitStrategy.LOAD, "Window Load"),
            (WaitStrategy.NETWORKIDLE, "Network Idle"),
        ]

        results = {}

        async with JSRenderer() as renderer:
            for strategy, name in strategies:
                config = RenderConfig(
                    wait_strategy=strategy,
                    timeout_ms=30000,
                )

                try:
                    start = datetime.now()
                    result = await renderer.render(url, config)
                    elapsed = (datetime.now() - start).total_seconds() * 1000

                    results[name] = {
                        "html_length": len(result.html),
                        "time_ms": elapsed,
                        "status": result.status_code,
                    }

                    print(f"{name}:")
                    print(f"  Time: {elapsed:.0f}ms")
                    print(f"  HTML: {len(result.html)} chars")
                    print()

                except Exception as e:
                    results[name] = {"error": str(e)}
                    print(f"{name}: Error - {e}\n")

        return results

    async def demo_browser_pool(self, urls: list[str]) -> list[RenderResult]:
        """
        Demonstrate browser pool for concurrent rendering.

        A browser pool manages multiple browser instances for
        high-throughput rendering.

        Args:
            urls: List of URLs to render concurrently

        Returns:
            List of render results
        """
        print(f"\n{'='*60}")
        print("BROWSER POOL DEMO")
        print(f"{'='*60}")
        print(f"Rendering {len(urls)} pages concurrently\n")

        results = []

        async with BrowserPool(
            max_browsers=2,
            max_contexts_per_browser=3,
            browser_type="chromium",
        ) as pool:
            async def render_one(url: str) -> RenderResult:
                async with pool.acquire() as context:
                    page = await context.new_page()
                    try:
                        await page.goto(url, wait_until="networkidle")
                        html = await page.content()
                        return RenderResult(
                            url=url,
                            html=html,
                            status_code=200,
                            render_time_ms=0,
                            console_logs=[],
                        )
                    finally:
                        await page.close()

            # Render all URLs concurrently
            start = datetime.now()
            results = await asyncio.gather(
                *[render_one(url) for url in urls],
                return_exceptions=True,
            )
            elapsed = (datetime.now() - start).total_seconds()

            print(f"Rendered {len(urls)} pages in {elapsed:.1f}s")
            print(f"Average: {elapsed/len(urls):.2f}s per page")

            for url, result in zip(urls, results):
                if isinstance(result, Exception):
                    print(f"  {url}: Error - {result}")
                else:
                    print(f"  {url}: {len(result.html)} chars")

        return results


def print_render_result(result: RenderResult) -> None:
    """Pretty print a render result."""
    print(f"\n{'='*60}")
    print("RENDER RESULT")
    print(f"{'='*60}")
    print(f"URL: {result.url}")
    print(f"Status: {result.status_code}")
    print(f"HTML Length: {len(result.html)} characters")
    print(f"Render Time: {result.render_time_ms}ms")

    if result.console_logs:
        print(f"Console Logs: {len(result.console_logs)}")

    if result.screenshot:
        print(f"Screenshot: {len(result.screenshot)} bytes")

    # Preview of content
    print(f"\nHTML Preview (first 500 chars):")
    print("-" * 40)
    preview = result.html[:500].replace('\n', ' ')
    print(preview)
    print("-" * 40)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="JavaScript rendering example with Playwright"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://example.com",
        help="URL to render",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare HTTP fetch vs JS rendering",
    )
    parser.add_argument(
        "--smart",
        action="store_true",
        help="Use smart rendering (auto-detect JS need)",
    )
    parser.add_argument(
        "--strategies",
        action="store_true",
        help="Compare different wait strategies",
    )
    parser.add_argument(
        "--pool",
        action="store_true",
        help="Demo browser pool with multiple URLs",
    )
    parser.add_argument(
        "--screenshot",
        action="store_true",
        help="Capture screenshot",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./js_render_output",
        help="Output directory for screenshots",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(
        level="DEBUG" if args.verbose else "INFO",
        format_type="console",
    )

    print("""
    ===============================================
       JAVASCRIPT RENDERING EXAMPLE
       Using Playwright for SPA crawling
    ===============================================

    This example shows how to:
    1. Render JavaScript-heavy pages (SPAs)
    2. Detect when JS rendering is needed
    3. Use different wait strategies
    4. Manage browser pools for concurrency
    5. Capture screenshots

    Prerequisites:
      pip install playwright
      playwright install chromium

    """)

    # Check Playwright is installed
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("Error: Playwright is not installed.")
        print("\nInstall it with:")
        print("  pip install playwright")
        print("  playwright install chromium")
        sys.exit(1)

    demo = JSRenderingDemo(verbose=args.verbose)

    try:
        if args.compare:
            await demo.compare_http_vs_js(args.url)

        elif args.smart:
            result = await demo.demo_smart_rendering(args.url)
            print_render_result(result)

        elif args.strategies:
            await demo.demo_wait_strategies(args.url)

        elif args.pool:
            urls = [
                "https://example.com",
                "https://httpbin.org/html",
                "https://httpbin.org/robots.txt",
            ]
            await demo.demo_browser_pool(urls)

        else:
            # Basic rendering
            result = await demo.render_page(
                args.url,
                capture_screenshot=args.screenshot,
            )
            print_render_result(result)

            if args.screenshot and result.screenshot:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                screenshot_path = output_dir / "screenshot.png"
                with open(screenshot_path, "wb") as f:
                    f.write(result.screenshot)
                print(f"\nScreenshot saved to: {screenshot_path}")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            raise


if __name__ == "__main__":
    asyncio.run(main())
