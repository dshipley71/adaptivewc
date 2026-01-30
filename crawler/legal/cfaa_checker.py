"""
CFAA (Computer Fraud and Abuse Act) authorization checker.

Provides authorization verification to ensure crawling complies with CFAA.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse, urljoin

from crawler.models import AuthorizationResult
from crawler.utils.logging import CrawlerLogger
from crawler.utils.url_utils import get_domain

if TYPE_CHECKING:
    from crawler.config import CFAAConfig


@dataclass
class DomainAuthorization:
    """Authorization status for a domain."""

    domain: str
    authorized: bool
    basis: str
    restrictions: list[str] = field(default_factory=list)
    tos_url: str | None = None
    tos_last_checked: datetime | None = None
    tos_analysis: dict[str, Any] | None = None
    cease_desist_received: bool = False
    cease_desist_date: datetime | None = None
    notes: str = ""


@dataclass
class ToSAnalysisResult:
    """Result of Terms of Service analysis."""

    domain: str
    tos_url: str | None
    is_restrictive: bool
    restrictions: list[str]
    concerns: list[str]
    recommendation: str  # "allow", "block", "proceed_with_caution"
    analyzed_at: datetime
    raw_analysis: dict[str, Any] | None = None


class CFAAChecker:
    """
    CFAA compliance checker for web crawling.

    Evaluates authorization to crawl based on:
    - robots.txt permission (implied authorization)
    - Public access (no authentication required)
    - Terms of Service analysis (enabled by default)
    - Explicit consent/API agreements
    - Cease and desist records

    This is a conservative implementation that errs on the side of caution.
    """

    # Default paths to check for Terms of Service
    DEFAULT_TOS_PATHS = [
        "/terms",
        "/terms-of-service",
        "/tos",
        "/legal/terms",
        "/terms-and-conditions",
        "/terms-of-use",
    ]

    def __init__(
        self,
        user_agent: str = "AdaptiveCrawler",
        blocklist_path: str | None = None,
        logger: CrawlerLogger | None = None,
        config: CFAAConfig | None = None,
    ):
        """
        Initialize the CFAA checker.

        Args:
            user_agent: User agent string for identification.
            blocklist_path: Path to legal blocklist file.
            logger: Logger instance.
            config: CFAA configuration (enables ToS analysis by default).
        """
        self.user_agent = user_agent
        self.logger = logger or CrawlerLogger("cfaa_checker")
        self.config = config

        # ToS analysis is enabled by default
        self.tos_analysis_enabled = True
        self.tos_cache_ttl = 86400  # 24 hours
        self.block_on_restrictive_tos = True
        self.tos_paths = self.DEFAULT_TOS_PATHS.copy()

        # Apply config if provided
        if config:
            self.tos_analysis_enabled = config.tos_analysis_enabled
            self.tos_cache_ttl = config.tos_cache_ttl
            self.block_on_restrictive_tos = config.block_on_restrictive_tos
            if config.common_tos_paths:
                self.tos_paths = config.common_tos_paths
            if config.blocklist_path:
                blocklist_path = config.blocklist_path

        self._domain_cache: dict[str, DomainAuthorization] = {}
        self._tos_cache: dict[str, ToSAnalysisResult] = {}
        self._blocklist: set[str] = set()
        self._allowlist: set[str] = set()

        if blocklist_path:
            self._load_blocklist(blocklist_path)

    def _load_blocklist(self, path: str) -> None:
        """Load domain blocklist from file."""
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        self._blocklist.add(line.lower())
        except FileNotFoundError:
            self.logger.debug("Blocklist file not found", path=path)

    def add_to_blocklist(self, domain: str, reason: str = "") -> None:
        """
        Add a domain to the blocklist.

        Args:
            domain: Domain to block.
            reason: Reason for blocking.
        """
        self._blocklist.add(domain.lower())
        self.logger.warning(
            "Domain added to blocklist",
            domain=domain,
            reason=reason,
        )

    def add_to_allowlist(self, domain: str) -> None:
        """
        Add a domain to the allowlist (explicit permission).

        Args:
            domain: Domain with explicit crawling permission.
        """
        self._allowlist.add(domain.lower())

    async def is_authorized(
        self,
        url: str,
        robots_allows: bool = True,
        requires_auth: bool = False,
        tos_text: str | None = None,
    ) -> AuthorizationResult:
        """
        Check if crawling a URL is authorized under CFAA analysis.

        Args:
            url: The URL to check.
            robots_allows: Whether robots.txt permits crawling.
            requires_auth: Whether the URL requires authentication.
            tos_text: Optional Terms of Service text to analyze.

        Returns:
            AuthorizationResult with authorization decision.
        """
        domain = get_domain(url)

        # Check blocklist first
        if self._is_blocked(domain):
            return AuthorizationResult(
                authorized=False,
                basis="blocklist",
                restrictions=["domain_blocked"],
                documentation=f"Domain {domain} is on legal blocklist",
            )

        # Check allowlist (explicit permission)
        if domain.lower() in self._allowlist:
            return AuthorizationResult(
                authorized=True,
                basis="explicit_permission",
                restrictions=[],
                documentation=f"Domain {domain} has explicit crawling permission",
            )

        # Check if authentication is required
        if requires_auth:
            return AuthorizationResult(
                authorized=False,
                basis="authentication_required",
                restrictions=["login_required"],
                documentation="Page requires authentication - no authorization to access",
            )

        # Check robots.txt permission
        if not robots_allows:
            return AuthorizationResult(
                authorized=False,
                basis="robots_txt",
                restrictions=["robots_disallow"],
                documentation="robots.txt disallows crawling this path",
            )

        # Check for cease and desist
        cached = self._domain_cache.get(domain.lower())
        if cached and cached.cease_desist_received:
            return AuthorizationResult(
                authorized=False,
                basis="cease_desist",
                restrictions=["legal_notice"],
                documentation=f"Cease and desist received on {cached.cease_desist_date}",
            )

        # Check Terms of Service (enabled by default)
        if self.tos_analysis_enabled:
            tos_result = await self._check_tos(domain, url, tos_text)
            if tos_result and tos_result.is_restrictive and self.block_on_restrictive_tos:
                return AuthorizationResult(
                    authorized=False,
                    basis="terms_of_service",
                    restrictions=tos_result.restrictions,
                    documentation=f"Terms of Service prohibits crawling: {', '.join(tos_result.restrictions)}",
                )

        # Public access with robots.txt permission implies authorization
        return AuthorizationResult(
            authorized=True,
            basis="public_access",
            restrictions=self._get_standard_restrictions(),
            documentation="Public page with robots.txt permission",
        )

    async def _check_tos(
        self,
        domain: str,
        url: str,
        tos_text: str | None = None,
    ) -> ToSAnalysisResult | None:
        """
        Check Terms of Service for crawling restrictions.

        Args:
            domain: The domain to check.
            url: The URL being crawled.
            tos_text: Optional ToS text if already fetched.

        Returns:
            ToSAnalysisResult or None if ToS not found/analyzed.
        """
        domain_lower = domain.lower()

        # Check cache first
        cached = self._tos_cache.get(domain_lower)
        if cached:
            cache_age = (datetime.utcnow() - cached.analyzed_at).total_seconds()
            if cache_age < self.tos_cache_ttl:
                return cached

        # Analyze provided ToS text
        if tos_text:
            analysis = self.analyze_tos(tos_text, domain)
            result = ToSAnalysisResult(
                domain=domain,
                tos_url=None,
                is_restrictive=analysis["is_restrictive"],
                restrictions=analysis["restrictions"],
                concerns=analysis["concerns"],
                recommendation=analysis["recommendation"],
                analyzed_at=datetime.utcnow(),
                raw_analysis=analysis,
            )
            self._tos_cache[domain_lower] = result

            if result.is_restrictive:
                self.logger.warning(
                    "Restrictive ToS detected",
                    domain=domain,
                    restrictions=result.restrictions,
                )

            return result

        # If no ToS text provided, return None (ToS will be fetched separately if needed)
        return None

    def get_tos_urls(self, base_url: str) -> list[str]:
        """
        Get potential Terms of Service URLs for a domain.

        Args:
            base_url: Base URL of the domain.

        Returns:
            List of potential ToS URLs to check.
        """
        parsed = urlparse(base_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        return [urljoin(base, path) for path in self.tos_paths]

    def is_tos_analysis_enabled(self) -> bool:
        """Check if ToS analysis is enabled."""
        return self.tos_analysis_enabled

    def get_cached_tos_analysis(self, domain: str) -> ToSAnalysisResult | None:
        """Get cached ToS analysis for a domain."""
        return self._tos_cache.get(domain.lower())

    def cache_tos_analysis(self, result: ToSAnalysisResult) -> None:
        """Cache a ToS analysis result."""
        self._tos_cache[result.domain.lower()] = result

    def _is_blocked(self, domain: str) -> bool:
        """Check if domain is blocked."""
        domain = domain.lower()

        # Direct match
        if domain in self._blocklist:
            return True

        # Check for parent domain match
        parts = domain.split(".")
        for i in range(len(parts) - 1):
            parent = ".".join(parts[i:])
            if parent in self._blocklist:
                return True

        return False

    def _get_standard_restrictions(self) -> list[str]:
        """Get standard restrictions for public crawling."""
        return [
            "respect_rate_limits",
            "respect_robots_txt",
            "no_circumvention",
            "public_pages_only",
        ]

    def record_cease_desist(
        self,
        domain: str,
        received_date: datetime,
        reference: str = "",
    ) -> None:
        """
        Record a cease and desist notice for a domain.

        Args:
            domain: The domain that sent the notice.
            received_date: Date the notice was received.
            reference: Reference number or identifier.
        """
        domain = domain.lower()
        self._blocklist.add(domain)

        auth = DomainAuthorization(
            domain=domain,
            authorized=False,
            basis="cease_desist",
            cease_desist_received=True,
            cease_desist_date=received_date,
            notes=f"Reference: {reference}" if reference else "",
        )
        self._domain_cache[domain] = auth

        self.logger.warning(
            "Cease and desist recorded",
            domain=domain,
            received_date=received_date.isoformat(),
            reference=reference,
        )

    def analyze_tos(self, tos_text: str, domain: str) -> dict[str, Any]:
        """
        Analyze Terms of Service for crawling restrictions.

        This is a basic keyword-based analysis. For production use,
        consider more sophisticated NLP analysis.

        Args:
            tos_text: Terms of Service text.
            domain: Domain the ToS applies to.

        Returns:
            Analysis result with restrictions found.
        """
        tos_lower = tos_text.lower()
        restrictions: list[str] = []
        concerns: list[str] = []

        # Common crawling prohibition patterns
        prohibition_patterns = [
            (r"prohibit.*scraping", "scraping_prohibited"),
            (r"prohibit.*crawling", "crawling_prohibited"),
            (r"prohibit.*automated.*access", "automated_access_prohibited"),
            (r"no.*robots", "robots_prohibited"),
            (r"no.*spiders", "spiders_prohibited"),
            (r"must.*not.*scrape", "scraping_prohibited"),
            (r"shall.*not.*harvest", "harvesting_prohibited"),
        ]

        for pattern, restriction in prohibition_patterns:
            if re.search(pattern, tos_lower):
                restrictions.append(restriction)

        # Check for API requirement
        if re.search(r"api.*only|only.*api|must.*use.*api", tos_lower):
            concerns.append("api_required")

        # Check for rate limit mentions
        if re.search(r"rate.*limit|request.*limit", tos_lower):
            concerns.append("rate_limits_mentioned")

        is_restrictive = len(restrictions) > 0

        return {
            "domain": domain,
            "is_restrictive": is_restrictive,
            "restrictions": restrictions,
            "concerns": concerns,
            "recommendation": "block" if is_restrictive else "proceed_with_caution",
        }

    def get_authorization_report(self, domain: str) -> dict[str, Any]:
        """
        Get authorization status report for a domain.

        Args:
            domain: The domain to report on.

        Returns:
            Authorization report dictionary.
        """
        cached = self._domain_cache.get(domain.lower())
        is_blocked = self._is_blocked(domain)
        is_allowed = domain.lower() in self._allowlist

        return {
            "domain": domain,
            "blocked": is_blocked,
            "explicitly_allowed": is_allowed,
            "cached_authorization": cached.authorized if cached else None,
            "basis": cached.basis if cached else None,
            "cease_desist": cached.cease_desist_received if cached else False,
        }
