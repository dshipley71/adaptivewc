"""
CLI entry point for the adaptive web crawler.

Usage:
    python -m crawler --seed-url https://example.com --output ./data
"""

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from crawler.config import load_config

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
    help="Maximum pages to crawl (default: unlimited).",
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
    default=None,
    help="Custom user agent string.",
)
@click.option(
    "--respect-robots/--ignore-robots",
    default=True,
    help="Respect robots.txt (default: respect).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging.",
)
def main(
    seed_url: tuple[str, ...],
    output: str,
    rate_limit: float,
    max_depth: int,
    max_pages: int | None,
    allowed_domains: tuple[str, ...],
    user_agent: str | None,
    respect_robots: bool,
    verbose: bool,
) -> None:
    """
    Adaptive Web Crawler - Ethical, compliant web crawling with ML-based adaptation.

    Example:
        python -m crawler -u https://example.com -o ./data
    """
    console.print("[bold blue]Adaptive Web Crawler[/bold blue]")
    console.print(f"Seed URLs: {', '.join(seed_url)}")
    console.print(f"Output: {output}")
    console.print(f"Rate limit: {rate_limit} req/s")

    # Load configuration
    config = load_config()

    if user_agent:
        config.user_agent = user_agent

    if not respect_robots:
        console.print("[yellow]Warning: robots.txt will be ignored![/yellow]")

    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    # TODO: Initialize and run crawler
    console.print("\n[yellow]Crawler implementation pending. Project structure ready.[/yellow]")
    console.print("\nNext steps:")
    console.print("1. Implement crawler/core/crawler.py")
    console.print("2. Implement crawler/compliance/robots_parser.py")
    console.print("3. Implement crawler/compliance/rate_limiter.py")
    console.print("4. Implement crawler/adaptive/structure_analyzer.py")
    console.print("\nSee AGENTS.md for detailed implementation guidance.")


if __name__ == "__main__":
    main()
