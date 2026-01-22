"""
Content hashing for near-duplicate detection.

Provides SimHash and other fingerprinting algorithms.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ContentFingerprint:
    """Fingerprint of page content."""

    url: str
    simhash: int
    md5: str
    word_count: int
    char_count: int

    def similarity(self, other: "ContentFingerprint") -> float:
        """Calculate similarity with another fingerprint using Hamming distance."""
        if self.simhash == other.simhash:
            return 1.0

        # Calculate Hamming distance
        xor = self.simhash ^ other.simhash
        hamming = bin(xor).count("1")

        # Convert to similarity (64-bit hash)
        return 1.0 - (hamming / 64.0)


class ContentHasher:
    """
    Content hasher for duplicate detection.

    Uses SimHash for near-duplicate detection and MD5 for exact duplicates.
    """

    def __init__(
        self,
        hash_bits: int = 64,
        shingle_size: int = 3,
    ):
        """
        Initialize the content hasher.

        Args:
            hash_bits: Number of bits for SimHash (default 64).
            shingle_size: Size of word shingles for SimHash.
        """
        self.hash_bits = hash_bits
        self.shingle_size = shingle_size

    def fingerprint(self, content: str, url: str = "") -> ContentFingerprint:
        """
        Create a fingerprint of content.

        Args:
            content: Text content to fingerprint.
            url: URL of the content.

        Returns:
            ContentFingerprint with hash values.
        """
        # Normalize content
        normalized = self._normalize(content)

        # Calculate hashes
        md5_hash = hashlib.md5(normalized.encode()).hexdigest()
        simhash = self._simhash(normalized)

        # Count metrics
        words = normalized.split()
        word_count = len(words)
        char_count = len(normalized)

        return ContentFingerprint(
            url=url,
            simhash=simhash,
            md5=md5_hash,
            word_count=word_count,
            char_count=char_count,
        )

    def _normalize(self, content: str) -> str:
        """Normalize content for hashing."""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", content)

        # Lowercase
        text = text.lower()

        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove punctuation (keep alphanumeric and spaces)
        text = re.sub(r"[^\w\s]", "", text)

        return text

    def _simhash(self, text: str) -> int:
        """
        Calculate SimHash of text.

        SimHash creates a fingerprint that allows for similarity comparison
        using Hamming distance.
        """
        words = text.split()
        if not words:
            return 0

        # Create shingles (n-grams of words)
        shingles = []
        for i in range(len(words) - self.shingle_size + 1):
            shingle = " ".join(words[i : i + self.shingle_size])
            shingles.append(shingle)

        if not shingles:
            shingles = words  # Use individual words if too short

        # Initialize vector
        vector = [0] * self.hash_bits

        # Process each shingle
        for shingle in shingles:
            # Get hash of shingle
            h = int(hashlib.md5(shingle.encode()).hexdigest(), 16)

            # Update vector
            for i in range(self.hash_bits):
                if h & (1 << i):
                    vector[i] += 1
                else:
                    vector[i] -= 1

        # Create fingerprint from vector
        fingerprint = 0
        for i in range(self.hash_bits):
            if vector[i] > 0:
                fingerprint |= 1 << i

        return fingerprint

    def is_duplicate(
        self,
        fp1: ContentFingerprint,
        fp2: ContentFingerprint,
        threshold: float = 0.9,
    ) -> bool:
        """
        Check if two fingerprints represent duplicate content.

        Args:
            fp1: First fingerprint.
            fp2: Second fingerprint.
            threshold: Similarity threshold for duplicate detection.

        Returns:
            True if content is likely duplicate.
        """
        # Exact match
        if fp1.md5 == fp2.md5:
            return True

        # Near-duplicate check
        return fp1.similarity(fp2) >= threshold

    def is_near_duplicate(
        self,
        fp1: ContentFingerprint,
        fp2: ContentFingerprint,
        threshold: float = 0.85,
    ) -> bool:
        """
        Check if two fingerprints represent near-duplicate content.

        Args:
            fp1: First fingerprint.
            fp2: Second fingerprint.
            threshold: Similarity threshold.

        Returns:
            True if content is similar.
        """
        return fp1.similarity(fp2) >= threshold
