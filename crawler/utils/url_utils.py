"""
URL manipulation utilities for the adaptive web crawler.

Provides URL normalization, domain extraction, and validation.
"""

import ipaddress
import re
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

# Common tracking parameters to strip during normalization
TRACKING_PARAMS = frozenset([
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "gclsrc", "dclid", "msclkid",
    "mc_cid", "mc_eid", "ref", "affiliate", "partner",
    "source", "medium", "campaign", "content",
])

# Session ID parameter patterns
SESSION_PATTERNS = re.compile(
    r"(session|sess|sid|jsessionid|phpsessid|aspsessionid|cfid|cftoken)[\w]*",
    re.IGNORECASE,
)


def get_domain(url: str) -> str:
    """
    Extract the domain (host) from a URL.

    Args:
        url: The URL to extract domain from.

    Returns:
        The domain/host portion of the URL.
    """
    parsed = urlparse(url)
    return parsed.netloc.lower()


def get_scheme(url: str) -> str:
    """
    Extract the scheme (protocol) from a URL.

    Args:
        url: The URL to extract scheme from.

    Returns:
        The scheme (e.g., 'http', 'https').
    """
    parsed = urlparse(url)
    return parsed.scheme.lower()


def get_path(url: str) -> str:
    """
    Extract the path from a URL.

    Args:
        url: The URL to extract path from.

    Returns:
        The path portion of the URL.
    """
    parsed = urlparse(url)
    return parsed.path or "/"


def normalize_url(
    url: str,
    strip_tracking: bool = True,
    strip_session: bool = True,
    sort_params: bool = True,
    lowercase_domain: bool = True,
    remove_trailing_slash: bool = False,
    remove_default_port: bool = True,
    remove_fragment: bool = True,
) -> str:
    """
    Normalize a URL to a canonical form for deduplication.

    Args:
        url: The URL to normalize.
        strip_tracking: Remove common tracking parameters.
        strip_session: Remove session ID parameters.
        sort_params: Sort query parameters alphabetically.
        lowercase_domain: Convert domain to lowercase.
        remove_trailing_slash: Remove trailing slash from path.
        remove_default_port: Remove default ports (80, 443).
        remove_fragment: Remove URL fragment (anchor).

    Returns:
        The normalized URL string.
    """
    parsed = urlparse(url)

    # Normalize scheme
    scheme = parsed.scheme.lower()

    # Normalize netloc (domain)
    netloc = parsed.netloc
    if lowercase_domain:
        netloc = netloc.lower()

    # Remove default ports
    if remove_default_port:
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        elif netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]

    # Normalize path
    path = parsed.path or "/"
    # Collapse multiple slashes
    while "//" in path:
        path = path.replace("//", "/")

    if remove_trailing_slash and path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # Normalize query parameters
    query = parsed.query
    if query:
        params = parse_qs(query, keep_blank_values=True)

        # Strip tracking parameters
        if strip_tracking:
            params = {k: v for k, v in params.items() if k.lower() not in TRACKING_PARAMS}

        # Strip session parameters
        if strip_session:
            params = {k: v for k, v in params.items() if not SESSION_PATTERNS.match(k)}

        # Sort and encode
        if params:
            if sort_params:
                params = dict(sorted(params.items()))
            # Flatten single-value lists
            flat_params = {k: v[0] if len(v) == 1 else v for k, v in params.items()}
            query = urlencode(flat_params, doseq=True)
        else:
            query = ""

    # Handle fragment
    fragment = "" if remove_fragment else parsed.fragment

    return urlunparse((scheme, netloc, path, "", query, fragment))


def resolve_url(base_url: str, relative_url: str) -> str:
    """
    Resolve a relative URL against a base URL.

    Args:
        base_url: The base URL to resolve against.
        relative_url: The relative URL to resolve.

    Returns:
        The resolved absolute URL.
    """
    return urljoin(base_url, relative_url)


def is_same_domain(url1: str, url2: str, include_subdomains: bool = False) -> bool:
    """
    Check if two URLs belong to the same domain.

    Args:
        url1: First URL.
        url2: Second URL.
        include_subdomains: Whether to consider subdomains as same domain.

    Returns:
        True if URLs are from the same domain.
    """
    domain1 = get_domain(url1)
    domain2 = get_domain(url2)

    if not include_subdomains:
        return domain1 == domain2

    # Extract base domain (handle cases like foo.example.com vs bar.example.com)
    base1 = _get_base_domain(domain1)
    base2 = _get_base_domain(domain2)
    return base1 == base2


def _get_base_domain(domain: str) -> str:
    """
    Extract base domain from a full domain.

    Note: This is a simple implementation. For production use,
    consider using the publicsuffix library.
    """
    parts = domain.split(".")
    if len(parts) <= 2:
        return domain
    # Simple heuristic: take last two parts
    # This doesn't handle co.uk, com.au etc properly
    return ".".join(parts[-2:])


def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid and has an HTTP(S) scheme.

    Args:
        url: The URL to validate.

    Returns:
        True if URL is valid for crawling.
    """
    try:
        parsed = urlparse(url)
        return bool(
            parsed.scheme in ("http", "https")
            and parsed.netloc
            and not _is_empty_path_only(parsed)
        )
    except Exception:
        return False


def _is_empty_path_only(parsed) -> bool:
    """Check if parsed URL has only empty components."""
    return not parsed.netloc and not parsed.path


def is_private_ip(url: str) -> bool:
    """
    Check if a URL resolves to a private/internal IP address.

    This is a security check to prevent SSRF attacks.

    Args:
        url: The URL to check.

    Returns:
        True if URL appears to point to a private IP.
    """
    try:
        domain = get_domain(url)

        # Remove port if present
        if ":" in domain:
            domain = domain.split(":")[0]

        # Check if it's an IP address
        try:
            ip = ipaddress.ip_address(domain)
            return ip.is_private or ip.is_loopback or ip.is_reserved
        except ValueError:
            # Not an IP address, check for localhost
            return domain in ("localhost", "127.0.0.1", "::1", "0.0.0.0")

    except Exception:
        return False


def get_url_depth(url: str) -> int:
    """
    Calculate the depth of a URL based on path segments.

    Args:
        url: The URL to analyze.

    Returns:
        Number of path segments (depth).
    """
    path = get_path(url)
    # Filter out empty segments
    segments = [s for s in path.split("/") if s]
    return len(segments)


def extract_url_pattern(url: str) -> str:
    """
    Extract a URL pattern by replacing variable segments with placeholders.

    For example:
        /articles/2024/01/my-post -> /articles/{year}/{month}/{slug}
        /users/12345/profile -> /users/{id}/profile

    Args:
        url: The URL to extract pattern from.

    Returns:
        URL pattern with placeholders.
    """
    path = get_path(url)
    segments = path.split("/")
    pattern_segments = []

    for segment in segments:
        if not segment:
            pattern_segments.append("")
            continue

        # Check for numeric ID
        if segment.isdigit():
            pattern_segments.append("{id}")
        # Check for UUID
        elif _is_uuid(segment):
            pattern_segments.append("{uuid}")
        # Check for date-like patterns
        elif re.match(r"^\d{4}$", segment):
            pattern_segments.append("{year}")
        elif re.match(r"^\d{2}$", segment):
            pattern_segments.append("{num}")
        # Check for slug (alphanumeric with hyphens)
        elif re.match(r"^[\w-]+$", segment) and "-" in segment:
            pattern_segments.append("{slug}")
        else:
            pattern_segments.append(segment)

    return "/".join(pattern_segments)


def _is_uuid(value: str) -> bool:
    """Check if a string looks like a UUID."""
    uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )
    return bool(uuid_pattern.match(value))


def matches_pattern(url: str, patterns: list[str]) -> bool:
    """
    Check if a URL matches any of the given patterns.

    Patterns support:
        - * for single path segment wildcard
        - ** for multi-segment wildcard
        - {var} for variable segments

    Args:
        url: The URL to check.
        patterns: List of patterns to match against.

    Returns:
        True if URL matches any pattern.
    """
    path = get_path(url)

    for pattern in patterns:
        if _match_pattern(path, pattern):
            return True

    return False


def _match_pattern(path: str, pattern: str) -> bool:
    """Match a single path against a pattern."""
    # Convert pattern to regex
    regex_pattern = pattern
    regex_pattern = regex_pattern.replace("**", "__DOUBLE_STAR__")
    regex_pattern = regex_pattern.replace("*", "[^/]+")
    regex_pattern = regex_pattern.replace("__DOUBLE_STAR__", ".*")
    regex_pattern = re.sub(r"\{[^}]+\}", "[^/]+", regex_pattern)
    regex_pattern = f"^{regex_pattern}$"

    try:
        return bool(re.match(regex_pattern, path))
    except re.error:
        return False
