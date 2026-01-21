"""
Robots.txt parser and checker for the adaptive web crawler.

Provides parsing and compliance checking for robots.txt files.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from crawler.utils.url_utils import get_domain, get_path


@dataclass
class RobotsRule:
    """A single rule from robots.txt."""

    pattern: str
    allow: bool
    line_number: int = 0

    def matches(self, path: str) -> bool:
        """Check if this rule matches the given path."""
        # Convert robots.txt pattern to regex
        regex = self._pattern_to_regex(self.pattern)
        return bool(re.match(regex, path))

    def _pattern_to_regex(self, pattern: str) -> str:
        """Convert robots.txt pattern to regex."""
        # Escape special regex characters except * and $
        escaped = re.escape(pattern)
        # Convert * to .*
        escaped = escaped.replace(r"\*", ".*")
        # Handle $ at end (robots.txt end-of-path anchor)
        if escaped.endswith(r"\$"):
            escaped = escaped[:-2] + "$"
        else:
            # Pattern matches prefix
            escaped = escaped + ".*"
        return "^" + escaped


@dataclass
class RobotsGroup:
    """A user-agent group from robots.txt."""

    user_agents: list[str] = field(default_factory=list)
    rules: list[RobotsRule] = field(default_factory=list)
    crawl_delay: float | None = None
    request_rate: tuple[int, int] | None = None  # (requests, seconds)
    sitemaps: list[str] = field(default_factory=list)


@dataclass
class RobotsTxt:
    """Parsed robots.txt file."""

    domain: str
    groups: list[RobotsGroup] = field(default_factory=list)
    sitemaps: list[str] = field(default_factory=list)
    raw_content: str = ""
    parsed_at: datetime = field(default_factory=datetime.utcnow)
    fetch_status: int = 200

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "domain": self.domain,
            "groups": [
                {
                    "user_agents": g.user_agents,
                    "rules": [
                        {"pattern": r.pattern, "allow": r.allow}
                        for r in g.rules
                    ],
                    "crawl_delay": g.crawl_delay,
                }
                for g in self.groups
            ],
            "sitemaps": self.sitemaps,
            "parsed_at": self.parsed_at.isoformat(),
            "fetch_status": self.fetch_status,
        }


class RobotsParser:
    """Parser for robots.txt files."""

    def parse(self, content: str, domain: str, status_code: int = 200) -> RobotsTxt:
        """
        Parse robots.txt content.

        Args:
            content: Raw robots.txt content.
            domain: Domain this robots.txt applies to.
            status_code: HTTP status code when fetching.

        Returns:
            Parsed RobotsTxt object.
        """
        robots = RobotsTxt(
            domain=domain,
            raw_content=content,
            fetch_status=status_code,
        )

        # Handle non-200 status codes
        if status_code == 404:
            # No robots.txt means everything is allowed
            return robots
        elif status_code >= 500:
            # Server error - be conservative, treat as disallow all
            group = RobotsGroup(
                user_agents=["*"],
                rules=[RobotsRule(pattern="/", allow=False, line_number=0)],
            )
            robots.groups.append(group)
            return robots
        elif status_code == 401 or status_code == 403:
            # Auth required - disallow all
            group = RobotsGroup(
                user_agents=["*"],
                rules=[RobotsRule(pattern="/", allow=False, line_number=0)],
            )
            robots.groups.append(group)
            return robots

        current_group: RobotsGroup | None = None
        line_number = 0

        for line in content.split("\n"):
            line_number += 1
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Handle inline comments
            if "#" in line:
                line = line.split("#")[0].strip()

            # Parse directive
            if ":" not in line:
                continue

            directive, value = line.split(":", 1)
            directive = directive.strip().lower()
            value = value.strip()

            if directive == "user-agent":
                # Start a new group if we have rules or user-agent changed
                if current_group is None or current_group.rules:
                    current_group = RobotsGroup()
                    robots.groups.append(current_group)
                current_group.user_agents.append(value.lower())

            elif directive == "disallow" and current_group is not None:
                if value:  # Empty Disallow means allow all
                    rule = RobotsRule(
                        pattern=value,
                        allow=False,
                        line_number=line_number,
                    )
                    current_group.rules.append(rule)

            elif directive == "allow" and current_group is not None:
                rule = RobotsRule(
                    pattern=value,
                    allow=True,
                    line_number=line_number,
                )
                current_group.rules.append(rule)

            elif directive == "crawl-delay" and current_group is not None:
                try:
                    current_group.crawl_delay = float(value)
                except ValueError:
                    pass

            elif directive == "sitemap":
                robots.sitemaps.append(value)
                if current_group is not None:
                    current_group.sitemaps.append(value)

            elif directive == "request-rate" and current_group is not None:
                # Format: requests/seconds (e.g., "1/10")
                try:
                    parts = value.split("/")
                    if len(parts) == 2:
                        requests = int(parts[0])
                        seconds = int(parts[1])
                        current_group.request_rate = (requests, seconds)
                except ValueError:
                    pass

        return robots


class RobotsChecker:
    """
    Checker for robots.txt compliance.

    Uses parsed robots.txt to determine if a URL can be crawled.
    """

    def __init__(
        self,
        user_agent: str = "AdaptiveCrawler",
        default_allow: bool = True,
    ):
        """
        Initialize the checker.

        Args:
            user_agent: The user agent string to check rules for.
            default_allow: Whether to allow by default if no rule matches.
        """
        self.user_agent = user_agent.lower()
        self.default_allow = default_allow
        self._parser = RobotsParser()

    def is_allowed(self, url: str, robots_txt: RobotsTxt) -> bool:
        """
        Check if a URL is allowed by robots.txt.

        Args:
            url: The URL to check.
            robots_txt: Parsed robots.txt object.

        Returns:
            True if crawling is allowed.
        """
        path = get_path(url)

        # Find the most specific matching group
        group = self._find_matching_group(robots_txt)
        if group is None:
            return self.default_allow

        # Check rules in order - most specific match wins
        return self._check_rules(path, group.rules)

    def _find_matching_group(self, robots_txt: RobotsTxt) -> RobotsGroup | None:
        """Find the group that matches our user agent."""
        specific_match = None
        wildcard_match = None

        for group in robots_txt.groups:
            for ua in group.user_agents:
                # Check for exact or prefix match
                if ua == self.user_agent or self.user_agent.startswith(ua):
                    specific_match = group
                    break
                # Check for wildcard
                if ua == "*":
                    wildcard_match = group

            if specific_match:
                break

        return specific_match or wildcard_match

    def _check_rules(self, path: str, rules: list[RobotsRule]) -> bool:
        """
        Check rules for a path.

        Rules are evaluated in order of specificity (longer pattern first).
        If multiple rules match, the most specific one wins.
        """
        # Sort rules by pattern length (more specific first)
        sorted_rules = sorted(rules, key=lambda r: len(r.pattern), reverse=True)

        for rule in sorted_rules:
            if rule.matches(path):
                return rule.allow

        return self.default_allow

    def get_crawl_delay(self, robots_txt: RobotsTxt) -> float | None:
        """
        Get the crawl-delay directive for our user agent.

        Args:
            robots_txt: Parsed robots.txt object.

        Returns:
            Crawl delay in seconds, or None if not specified.
        """
        group = self._find_matching_group(robots_txt)
        if group:
            return group.crawl_delay
        return None

    def get_sitemaps(self, robots_txt: RobotsTxt) -> list[str]:
        """
        Get sitemap URLs from robots.txt.

        Args:
            robots_txt: Parsed robots.txt object.

        Returns:
            List of sitemap URLs.
        """
        return robots_txt.sitemaps

    def parse_and_check(
        self,
        url: str,
        robots_content: str,
        status_code: int = 200,
    ) -> tuple[bool, RobotsTxt]:
        """
        Parse robots.txt and check a URL in one operation.

        Args:
            url: The URL to check.
            robots_content: Raw robots.txt content.
            status_code: HTTP status code when fetching robots.txt.

        Returns:
            Tuple of (is_allowed, parsed_robots_txt).
        """
        domain = get_domain(url)
        robots_txt = self._parser.parse(robots_content, domain, status_code)
        allowed = self.is_allowed(url, robots_txt)
        return allowed, robots_txt
