"""
CLI entry point for the adaptive web crawler.

Usage:
    python -m crawler --seed-url https://example.com --output ./data

    # With ML embeddings and LLM descriptions:
    python -m crawler --seed-url https://example.com --output ./data \
        --enable-embeddings --embedding-model all-MiniLM-L6-v2 \
        --llm-provider ollama-cloud --llm-model gemma2:27b
"""

import asyncio
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from crawler.config import (
    CrawlConfig,
    GDPRConfig,
    LLMProviderType,
    PIIHandlingConfig,
    RateLimitConfig,
    SafetyLimits,
    SecurityConfig,
    StructureStoreConfig,
    StructureStoreType,
    load_config,
)
from crawler.core.crawler import Crawler
from crawler.utils.logging import setup_logging

console = Console()


@click.command()
@click.option(
    "--seed-url",
    "-u",
    multiple=True,
    required=True,
    help="Starting URL(s) to crawl. Can be specified multiple times.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output directory for crawled content.",
)
@click.option(
    "--rate-limit",
    "-r",
    type=float,
    default=1.0,
    help="Requests per second per domain (default: 1.0).",
)
@click.option(
    "--max-depth",
    "-d",
    type=int,
    default=10,
    help="Maximum crawl depth (default: 10).",
)
@click.option(
    "--max-pages",
    "-p",
    type=int,
    default=None,
    help="Maximum total pages to crawl across all domains (default: unlimited).",
)
@click.option(
    "--max-pages-per-domain",
    type=int,
    default=None,
    help="Maximum pages to crawl per domain (default: unlimited).",
)
@click.option(
    "--allowed-domains",
    "-a",
    multiple=True,
    help="Restrict crawling to specific domains.",
)
@click.option(
    "--user-agent",
    type=str,
    default="AdaptiveCrawler/1.0 (+https://github.com/adaptivecrawler)",
    help="Custom user agent string.",
)
@click.option(
    "--respect-robots/--ignore-robots",
    default=True,
    help="Respect robots.txt (default: respect).",
)
@click.option(
    "--redis-url",
    type=str,
    default=None,
    help="Redis connection URL (default: redis://localhost:6379/0).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging.",
)
# ML and LLM options
@click.option(
    "--enable-embeddings",
    is_flag=True,
    default=False,
    help="Enable ML embeddings for structure analysis.",
)
@click.option(
    "--embedding-model",
    type=str,
    default="all-MiniLM-L6-v2",
    help="Embedding model name (default: all-MiniLM-L6-v2).",
)
@click.option(
    "--llm-provider",
    type=click.Choice(["none", "openai", "anthropic", "ollama", "ollama-cloud"]),
    default="none",
    help="LLM provider for structure descriptions (default: none).",
)
@click.option(
    "--llm-model",
    type=str,
    default="",
    help="LLM model name (default: provider-specific, e.g., gpt-4o-mini, llama3.2, gemma2:27b).",
)
@click.option(
    "--ollama-url",
    type=str,
    default="http://localhost:11434",
    help="Ollama base URL (default: http://localhost:11434).",
)
def main(
    seed_url: tuple[str, ...],
    output: str,
    rate_limit: float,
    max_depth: int,
    max_pages: int | None,
    max_pages_per_domain: int | None,
    allowed_domains: tuple[str, ...],
    user_agent: str,
    respect_robots: bool,
    redis_url: str | None,
    verbose: bool,
    enable_embeddings: bool,
    embedding_model: str,
    llm_provider: str,
    llm_model: str,
    ollama_url: str,
) -> None:
    """
    Adaptive Web Crawler - Ethical, compliant web crawling with ML-based adaptation.

    Example:
        python -m crawler -u https://example.com -o ./data
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level, format_type="console")

    console.print("[bold blue]Adaptive Web Crawler[/bold blue]")
    console.print(f"Seed URLs: {', '.join(seed_url)}")
    console.print(f"Output: {output}")
    console.print(f"Rate limit: {rate_limit} req/s")
    console.print(f"Max depth: {max_depth}")
    if max_pages:
        console.print(f"Max pages (total): {max_pages}")
    if max_pages_per_domain:
        console.print(f"Max pages per domain: {max_pages_per_domain}")
    if allowed_domains:
        console.print(f"Allowed domains: {', '.join(allowed_domains)}")

    if not respect_robots:
        console.print("[yellow]Warning: robots.txt will be ignored![/yellow]")

    # Display ML settings
    if enable_embeddings:
        console.print(f"[cyan]ML Embeddings: enabled ({embedding_model})[/cyan]")
    if llm_provider != "none":
        model_display = llm_model or "(provider default)"
        console.print(f"[cyan]LLM Descriptions: {llm_provider} {model_display}[/cyan]")
        if llm_provider in ("ollama", "ollama-cloud"):
            console.print(f"[cyan]Ollama URL: {ollama_url}[/cyan]")

    # Determine Redis URL
    redis_connection = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    # Create structure store configuration
    structure_store_config = StructureStoreConfig(
        enable_embeddings=enable_embeddings,
        embedding_model=embedding_model,
        store_type=StructureStoreType.LLM if llm_provider != "none" else StructureStoreType.BASIC,
        llm_provider=LLMProviderType(llm_provider) if llm_provider != "none" else LLMProviderType.ANTHROPIC,
        llm_model=llm_model,
        ollama_base_url=ollama_url,
    )

    # Create crawl configuration
    config = CrawlConfig(
        seed_urls=list(seed_url),
        output_dir=output,
        max_depth=max_depth,
        max_pages=max_pages,
        max_pages_per_domain=max_pages_per_domain,
        allowed_domains=list(allowed_domains) if allowed_domains else [],
        rate_limit=RateLimitConfig(
            default_delay=1.0 / rate_limit,
            respect_crawl_delay=respect_robots,
        ),
        safety=SafetyLimits(),
        security=SecurityConfig(),
        gdpr=GDPRConfig(enabled=False),  # Disabled by default for CLI
        pii=PIIHandlingConfig(),
        structure_store=structure_store_config,
    )

    # Run the crawler
    console.print("\n[green]Starting crawl...[/green]\n")

    try:
        stats = asyncio.run(_run_crawler(config, redis_connection, user_agent))

        # Print results
        console.print("\n[bold green]Crawl completed![/bold green]")
        console.print(f"Pages crawled: {stats.pages_crawled}")
        console.print(f"Pages failed: {stats.pages_failed}")
        console.print(f"Pages blocked: {stats.pages_blocked}")
        console.print(f"Links discovered: {stats.links_discovered}")
        console.print(f"Domains crawled: {len(stats.domains_crawled)}")
        console.print(f"Bytes downloaded: {stats.bytes_downloaded:,}")
        duration = (stats.finished_at - stats.started_at).total_seconds() if stats.finished_at else 0
        console.print(f"Duration: {duration:.1f}s")

    except ConnectionError as e:
        console.print(f"\n[bold red]Error: Could not connect to Redis[/bold red]")
        console.print(f"Make sure Redis is running at: {redis_connection}")
        console.print("\nTo start Redis with Docker:")
        console.print("  docker run -d -p 6379:6379 redis:7-alpine")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Crawl interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


async def _run_crawler(config: CrawlConfig, redis_url: str, user_agent: str):
    """Run the crawler asynchronously."""
    async with Crawler(config, redis_url=redis_url, user_agent=user_agent) as crawler:
        return await crawler.crawl()


if __name__ == "__main__":
    main()
