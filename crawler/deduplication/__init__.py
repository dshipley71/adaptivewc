"""Deduplication module for content and URL deduplication."""

from crawler.deduplication.content_hasher import (
    ContentHasher,
    DuplicateResult,
)

__all__ = [
    "ContentHasher",
    "DuplicateResult",
]
