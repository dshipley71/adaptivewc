#!/usr/bin/env python3
"""
Demonstration of Adaptive Web Crawler: Learning and Adaptation Phases

This script demonstrates:
1. Initial crawl (Learning Phase) - System learns extraction strategies
2. Website redesign simulation
3. Subsequent crawl (Adaptation Phase) - System detects changes and adapts

Usage:
    python examples/demo_adaptive_extraction.py

Requirements:
    - Redis running on localhost:6379
    - pip install -e .
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from crawler.adaptive.change_detector import ChangeDetector
from crawler.adaptive.strategy_learner import StrategyLearner, LearnedStrategy
from crawler.adaptive.structure_analyzer import StructureAnalyzer
from crawler.extraction.content_extractor import ContentExtractor
from crawler.models import ExtractionStrategy, PageStructure
from crawler.storage.structure_store import StructureStore
from crawler.utils.logging import CrawlerLogger, setup_logging

import redis.asyncio as redis


# =============================================================================
# Mock Website HTML - Before and After Redesign
# =============================================================================

INITIAL_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Tech News - Breaking: AI Breakthrough</title>
</head>
<body>
    <header>
        <nav>
            <a href="/">Home</a>
            <a href="/tech">Tech</a>
            <a href="/science">Science</a>
        </nav>
    </header>

    <main>
        <article>
            <h1 class="headline">Breaking: Major AI Breakthrough Announced</h1>
            <div class="meta">
                <span class="author">By Sarah Johnson</span>
                <time datetime="2024-01-28">January 28, 2024</time>
            </div>
            <div class="article-body">
                <p>In a groundbreaking development, researchers at TechLab have
                announced a major breakthrough in artificial intelligence that could
                revolutionize the industry.</p>
                <p>The new algorithm demonstrates unprecedented capabilities in
                natural language understanding, achieving human-level performance
                on several benchmark tests.</p>
                <p>Dr. Emily Chen, lead researcher, stated: "This represents a
                significant leap forward in AI capabilities. We're excited about
                the potential applications."</p>
                <p>The research team plans to publish their findings in the
                upcoming issue of Nature AI. Industry experts are calling this
                one of the most important developments in recent years.</p>
            </div>
            <div class="tags">
                <a href="/tag/ai">AI</a>
                <a href="/tag/research">Research</a>
                <a href="/tag/technology">Technology</a>
            </div>
            <img src="/images/ai-lab.jpg" alt="AI Research Lab">
            <img src="/images/team.jpg" alt="Research Team">
        </article>
    </main>

    <footer>
        <p>&copy; 2024 Tech News</p>
    </footer>
</body>
</html>
"""

REDESIGNED_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Tech News - Breaking: AI Breakthrough</title>
</head>
<body>
    <header class="site-header">
        <nav class="main-navigation">
            <a href="/">Home</a>
            <a href="/tech">Tech</a>
            <a href="/science">Science</a>
        </nav>
    </header>

    <main class="content-wrapper">
        <article class="news-article">
            <!-- REDESIGN: Changed from "headline" to "article-title" -->
            <h1 class="article-title">Breaking: Major AI Breakthrough Announced</h1>

            <!-- REDESIGN: Changed structure of metadata -->
            <div class="article-metadata">
                <div class="byline">
                    <span class="author-name">By Sarah Johnson</span>
                </div>
                <div class="publish-date">
                    <time datetime="2024-01-28">January 28, 2024</time>
                </div>
            </div>

            <!-- REDESIGN: Changed from "article-body" to "article-content" -->
            <div class="article-content">
                <p>In a groundbreaking development, researchers at TechLab have
                announced a major breakthrough in artificial intelligence that could
                revolutionize the industry.</p>
                <p>The new algorithm demonstrates unprecedented capabilities in
                natural language understanding, achieving human-level performance
                on several benchmark tests.</p>
                <p>Dr. Emily Chen, lead researcher, stated: "This represents a
                significant leap forward in AI capabilities. We're excited about
                the potential applications."</p>
                <p>The research team plans to publish their findings in the
                upcoming issue of Nature AI. Industry experts are calling this
                one of the most important developments in recent years.</p>
            </div>

            <!-- REDESIGN: Changed tag structure -->
            <div class="article-tags">
                <span class="tag">AI</span>
                <span class="tag">Research</span>
                <span class="tag">Technology</span>
            </div>

            <!-- REDESIGN: Images moved to gallery -->
            <div class="image-gallery">
                <img src="/images/ai-lab.jpg" alt="AI Research Lab">
                <img src="/images/team.jpg" alt="Research Team">
            </div>
        </article>
    </main>

    <footer class="site-footer">
        <p>&copy; 2024 Tech News</p>
    </footer>
</body>
</html>
"""


# =============================================================================
# Demo Functions
# =============================================================================

class AdaptiveExtractionDemo:
    """Demonstrates the adaptive extraction workflow."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize demo components."""
        self.redis_url = redis_url
        self.logger = CrawlerLogger("demo")

        # Initialize components
        self.structure_analyzer = StructureAnalyzer(logger=self.logger)
        self.change_detector = ChangeDetector(logger=self.logger)
        self.strategy_learner = StrategyLearner(logger=self.logger)
        self.content_extractor = ContentExtractor(logger=self.logger)

        self.redis_client: redis.Redis | None = None
        self.structure_store: StructureStore | None = None

        # Demo data
        self.domain = "technews.example.com"
        self.page_type = "article"
        self.url = "https://technews.example.com/ai-breakthrough"

    async def setup(self) -> None:
        """Setup Redis connection and storage."""
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            await self.redis_client.ping()
            self.structure_store = StructureStore(
                redis_client=self.redis_client,
                logger=self.logger
            )
            self.logger.info("✓ Connected to Redis")
        except Exception as e:
            self.logger.error("Failed to connect to Redis", error=str(e))
            raise

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.redis_client:
            await self.redis_client.aclose()

    def print_section(self, title: str) -> None:
        """Print a section header."""
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}\n")

    def print_structure_summary(self, structure: PageStructure) -> None:
        """Print summary of page structure."""
        tag_counts = structure.tag_hierarchy.get("tag_counts", {})
        print(f"Domain: {structure.domain}")
        print(f"Page Type: {structure.page_type}")
        print(f"Version: {structure.version}")
        print(f"Captured: {structure.captured_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nTag Hierarchy:")
        for tag, count in sorted(tag_counts.items())[:10]:
            print(f"  {tag}: {count}")
        print(f"\nContent Regions: {len(structure.content_regions)}")
        for region in structure.content_regions[:3]:
            print(f"  - {region.name}: {region.primary_selector} (confidence: {region.confidence:.2%})")
        print(f"\nSemantic Landmarks: {', '.join(structure.semantic_landmarks.keys())}")

    def print_strategy_summary(self, strategy: ExtractionStrategy) -> None:
        """Print summary of extraction strategy."""
        print(f"Domain: {strategy.domain}")
        print(f"Page Type: {strategy.page_type}")
        print(f"Version: {strategy.version}")
        print(f"Learning Source: {strategy.learning_source}")
        print(f"Learned At: {strategy.learned_at.strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\nExtraction Rules:")
        if strategy.title:
            print(f"  Title: {strategy.title.primary} (confidence: {strategy.title.confidence:.2%})")
            if strategy.title.fallbacks:
                print(f"    Fallbacks: {', '.join(strategy.title.fallbacks)}")

        if strategy.content:
            print(f"  Content: {strategy.content.primary} (confidence: {strategy.content.confidence:.2%})")
            if strategy.content.fallbacks:
                print(f"    Fallbacks: {', '.join(strategy.content.fallbacks)}")

        if strategy.metadata:
            print(f"  Metadata Fields:")
            for key, rule in strategy.metadata.items():
                print(f"    {key}: {rule.primary} (confidence: {rule.confidence:.2%})")

        if strategy.images:
            print(f"  Images: {strategy.images.primary}")

    def print_extraction_result(self, result: Any) -> None:
        """Print extraction result."""
        print(f"Success: {'✓' if result.success else '✗'}")
        print(f"Confidence: {result.content.confidence:.2%}" if result.content else "N/A")
        print(f"Duration: {result.duration_ms:.2f}ms")

        if result.content:
            print(f"\nExtracted Content:")
            print(f"  Title: {result.content.title}")
            print(f"  Content Length: {len(result.content.content)} characters")
            print(f"  Content Preview: {result.content.content[:150]}...")

            if result.content.metadata:
                print(f"\n  Metadata:")
                for key, value in result.content.metadata.items():
                    print(f"    {key}: {value}")

            if result.content.images:
                print(f"\n  Images: {len(result.content.images)}")
                for img in result.content.images[:3]:
                    print(f"    - {img}")

        if result.errors:
            print(f"\n  Errors: {', '.join(result.errors)}")

        if result.warnings:
            print(f"  Warnings: {', '.join(result.warnings)}")

    def print_changes(self, analysis: Any) -> None:
        """Print detected changes."""
        print(f"Similarity Score: {analysis.similarity_score:.2%}")
        print(f"Requires Re-learning: {'Yes' if analysis.requires_relearning else 'No'}")
        print(f"Number of Changes: {len(analysis.changes)}")

        if analysis.changes:
            print(f"\nDetected Changes:")
            for change in analysis.changes:
                breaking_marker = "🔴" if change.breaking else "🟡"
                print(f"  {breaking_marker} {change.change_type}: {change.reason}")
                print(f"     Confidence: {change.confidence:.2%}")
                if change.affected_components:
                    print(f"     Affected: {', '.join(change.affected_components)}")

    async def phase1_initial_crawl(self) -> tuple[PageStructure, ExtractionStrategy]:
        """Phase 1: Initial crawl - Learning phase."""
        self.print_section("PHASE 1: INITIAL CRAWL (Learning Phase)")

        print("📥 Fetching page (simulated)...")
        html = INITIAL_HTML

        print("🔍 Analyzing page structure...")
        structure = self.structure_analyzer.analyze(html, self.url, self.page_type)
        print("\n📊 Structure Analysis:")
        self.print_structure_summary(structure)

        print("\n🧠 Learning extraction strategy...")
        learning_result: LearnedStrategy = self.strategy_learner.infer(html, structure)
        strategy = learning_result.strategy

        print("\n📋 Learned Strategy:")
        self.print_strategy_summary(strategy)

        print("\n💾 Saving structure and strategy to Redis...")
        assert self.structure_store is not None
        success = await self.structure_store.save_structure(structure, strategy, "default")

        if success:
            print("✓ Saved successfully")
        else:
            print("✗ Failed to save")

        print("\n🎯 Extracting content using learned strategy...")
        extraction_result = self.content_extractor.extract(self.url, html, strategy)

        print("\n📄 Extraction Result:")
        self.print_extraction_result(extraction_result)

        return structure, strategy

    async def phase2_redesigned_crawl(
        self,
        old_structure: PageStructure,
        old_strategy: ExtractionStrategy
    ) -> tuple[PageStructure, ExtractionStrategy]:
        """Phase 2: Crawl after redesign - Adaptation phase."""
        self.print_section("PHASE 2: WEBSITE REDESIGN")

        print("🎨 Website has been redesigned!")
        print("\nChanges made:")
        print("  • .headline → .article-title")
        print("  • .article-body → .article-content")
        print("  • .author → .author-name (inside .byline)")
        print("  • .tags → .article-tags with .tag spans")
        print("  • Images moved to .image-gallery")

        self.print_section("PHASE 3: SUBSEQUENT CRAWL (Adaptation Phase)")

        print("📥 Fetching redesigned page...")
        html = REDESIGNED_HTML

        print("🔍 Analyzing new page structure...")
        new_structure = self.structure_analyzer.analyze(html, self.url, self.page_type)

        print("\n🔬 Detecting changes...")
        analysis = self.change_detector.detect_changes(old_structure, new_structure)

        print("\n📊 Change Analysis:")
        self.print_changes(analysis)

        if analysis.requires_relearning:
            print("\n🔄 Adapting extraction strategy...")
            adapted = self.strategy_learner.adapt(old_strategy, new_structure, html)
            new_strategy = adapted.strategy

            print("\n📋 Adapted Strategy:")
            self.print_strategy_summary(new_strategy)

            print("\n📊 Strategy Changes:")
            print(f"  Title selector: {old_strategy.title.primary} → {new_strategy.title.primary}")
            print(f"  Content selector: {old_strategy.content.primary} → {new_strategy.content.primary}")

            # Compare metadata
            old_author = old_strategy.metadata.get("author")
            new_author = new_strategy.metadata.get("author")
            if old_author and new_author:
                print(f"  Author selector: {old_author.primary} → {new_author.primary}")

            print("\n💾 Saving updated structure and strategy...")
            assert self.structure_store is not None
            new_structure.version = old_structure.version + 1
            new_strategy.version = old_strategy.version + 1
            success = await self.structure_store.save_structure(
                new_structure, new_strategy, "default"
            )

            if success:
                print("✓ Saved successfully")
            else:
                print("✗ Failed to save")
        else:
            print("\n✓ No significant changes detected, using existing strategy")
            new_strategy = old_strategy

        print("\n🎯 Extracting content using adapted strategy...")
        extraction_result = self.content_extractor.extract(self.url, html, new_strategy)

        print("\n📄 Extraction Result:")
        self.print_extraction_result(extraction_result)

        return new_structure, new_strategy

    async def show_redis_state(self) -> None:
        """Show what's stored in Redis."""
        self.print_section("REDIS STATE")

        assert self.structure_store is not None

        # Get structure
        structure = await self.structure_store.get_structure(
            self.domain, self.page_type, "default"
        )

        # Get strategy
        strategy = await self.structure_store.get_strategy(
            self.domain, self.page_type, "default"
        )

        # Get history
        history = await self.structure_store.get_history(
            self.domain, self.page_type, limit=5
        )

        print(f"Stored Structure: {'✓' if structure else '✗'}")
        if structure:
            print(f"  Version: {structure.version}")
            print(f"  Captured: {structure.captured_at.strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\nStored Strategy: {'✓' if strategy else '✗'}")
        if strategy:
            print(f"  Version: {strategy.version}")
            print(f"  Learning Source: {strategy.learning_source}")
            print(f"  Title Selector: {strategy.title.primary if strategy.title else 'N/A'}")
            print(f"  Content Selector: {strategy.content.primary if strategy.content else 'N/A'}")

        print(f"\nHistory: {len(history)} versions")
        for i, hist in enumerate(history, 1):
            print(f"  {i}. Version {hist.version} - {hist.captured_at.strftime('%Y-%m-%d %H:%M:%S')}")

    async def run(self) -> None:
        """Run the complete demo."""
        try:
            await self.setup()

            # Phase 1: Initial crawl
            old_structure, old_strategy = await self.phase1_initial_crawl()

            # Simulate time passing
            await asyncio.sleep(1)

            # Phase 2: Redesigned crawl
            new_structure, new_strategy = await self.phase2_redesigned_crawl(
                old_structure, old_strategy
            )

            # Show Redis state
            await self.show_redis_state()

            self.print_section("DEMO COMPLETE")
            print("✓ Successfully demonstrated:")
            print("  1. Initial structure learning")
            print("  2. Extraction strategy inference")
            print("  3. Content extraction")
            print("  4. Change detection after redesign")
            print("  5. Strategy adaptation")
            print("  6. Continued extraction with adapted strategy")
            print("\nThe system successfully adapted to website changes!")

        except Exception as e:
            self.logger.error("Demo failed", error=str(e))
            raise
        finally:
            await self.cleanup()


# =============================================================================
# Main
# =============================================================================

async def main() -> None:
    """Run the demo."""
    # Setup logging
    setup_logging(level="INFO", format_type="console")

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                   ADAPTIVE WEB CRAWLER DEMONSTRATION                         ║
║                                                                              ║
║  This demo shows how the crawler learns extraction strategies and adapts    ║
║  when website structure changes.                                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Check Redis
    print("🔧 Checking prerequisites...")
    try:
        redis_client = await redis.from_url("redis://localhost:6379/0")
        await redis_client.ping()
        await redis_client.aclose()
        print("✓ Redis is running")
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        print("\nPlease start Redis:")
        print("  docker run -d -p 6379:6379 redis:7-alpine")
        print("  # or")
        print("  redis-server")
        sys.exit(1)

    # Run demo
    demo = AdaptiveExtractionDemo()
    await demo.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
