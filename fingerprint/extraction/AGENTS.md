# AGENTS.md - Extraction Module

Complete specification for content extraction and file saving.

---

## Module Purpose

The extraction module provides:
- Content extraction using learned strategies
- Multiple output formats (JSON, CSV, etc.)
- File persistence with configurable paths
- Extraction metadata and confidence tracking

---

## Files to Generate

```
fingerprint/extraction/
├── __init__.py
├── extractor.py        # Content extraction engine
├── file_writer.py      # Save content to files
└── formats.py          # Output format handlers
```

---

## fingerprint/extraction/__init__.py

```python
"""
Extraction module - Content extraction and file saving.
"""

from fingerprint.extraction.extractor import ContentExtractor, ExtractionResult
from fingerprint.extraction.file_writer import FileWriter
from fingerprint.extraction.formats import JSONFormatter, CSVFormatter, OutputFormatter

__all__ = [
    "ContentExtractor",
    "ExtractionResult",
    "FileWriter",
    "JSONFormatter",
    "CSVFormatter",
    "OutputFormatter",
]
```

---

## fingerprint/extraction/extractor.py

```python
"""
Content extraction engine.

Extracts content from HTML using learned extraction strategies.

Verbose logging pattern:
[EXTRACT:OPERATION] Message
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup

from fingerprint.config import Config
from fingerprint.core.verbose import get_logger
from fingerprint.models import (
    ExtractedContent,
    ExtractionResult,
    ExtractionStrategy,
    SelectorRule,
    PageStructure,
)
from fingerprint.storage.structure_store import StructureStore


class ContentExtractor:
    """
    Extract content from HTML using learned strategies.

    Usage:
        extractor = ContentExtractor(config)
        result = await extractor.extract(url, html, strategy)
        if result.success:
            print(result.content.title)
    """

    def __init__(self, config: Config):
        self.config = config
        self.extraction_config = config.extraction
        self.logger = get_logger()

        self.logger.info(
            "EXTRACT", "INIT",
            "Content extractor initialized",
            output_dir=self.extraction_config.output_dir,
        )

    async def extract(
        self,
        url: str,
        html: str,
        strategy: ExtractionStrategy,
    ) -> ExtractionResult:
        """
        Extract content from HTML using strategy.

        Args:
            url: Source URL
            html: HTML content
            strategy: Extraction strategy with selectors

        Returns:
            ExtractionResult with extracted content

        Verbose output:
            [EXTRACT:START] Extracting content from https://example.com
              - strategy_version: 3
            [EXTRACT:FIELD] Extracted title
              - selector: h1.article-title
              - length: 45
            [EXTRACT:FIELD] Extracted content
              - selector: article.main-content
              - length: 5420
            [EXTRACT:COMPLETE] Extraction complete
              - fields: 5
              - duration_ms: 45.2
        """
        if not self.extraction_config.enabled:
            return ExtractionResult(success=False, error="Extraction disabled")

        start_time = time.time()

        self.logger.info(
            "EXTRACT", "START",
            f"Extracting content from {url}",
            strategy_version=strategy.version,
        )

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception as e:
            self.logger.error("EXTRACT", "PARSE_ERROR", str(e))
            return ExtractionResult(success=False, error=f"HTML parse error: {e}")

        # Initialize extracted content
        from urllib.parse import urlparse
        parsed = urlparse(url)

        content = ExtractedContent(
            url=url,
            domain=parsed.netloc,
            page_type=strategy.page_type,
            strategy_version=strategy.version,
        )

        fields_extracted = 0
        total_confidence = 0.0

        # Extract title
        if strategy.title:
            title_result = self._extract_field(soup, strategy.title, "title")
            if title_result:
                content.title = title_result
                fields_extracted += 1
                total_confidence += strategy.title.confidence

        # Extract main content
        if strategy.content:
            content_result = self._extract_field(soup, strategy.content, "content")
            if content_result:
                content.content = content_result
                fields_extracted += 1
                total_confidence += strategy.content.confidence

        # Extract metadata fields
        for field_name, rule in strategy.metadata.items():
            field_result = self._extract_field(soup, rule, field_name)
            if field_result:
                content.metadata[field_name] = field_result
                fields_extracted += 1
                total_confidence += rule.confidence

        # Include raw HTML if configured
        if self.extraction_config.include_html:
            content.html = html[:self.extraction_config.max_content_length]

        # Calculate confidence
        if fields_extracted > 0:
            content.extraction_confidence = total_confidence / fields_extracted

        duration_ms = (time.time() - start_time) * 1000

        self.logger.info(
            "EXTRACT", "COMPLETE",
            f"Extraction complete for {url}",
            fields=fields_extracted,
            duration_ms=f"{duration_ms:.1f}",
            confidence=f"{content.extraction_confidence:.2f}",
        )

        return ExtractionResult(
            success=True,
            content=content,
            fields_extracted=fields_extracted,
            content_length=len(content.content),
            duration_ms=duration_ms,
        )

    def _extract_field(
        self,
        soup: BeautifulSoup,
        rule: SelectorRule,
        field_name: str,
    ) -> str | None:
        """Extract single field using selector rule."""
        # Try primary selector
        element = soup.select_one(rule.primary)

        # Try fallbacks if primary fails
        if element is None:
            for fallback in rule.fallbacks:
                element = soup.select_one(fallback)
                if element:
                    break

        if element is None:
            self.logger.debug(
                "EXTRACT", "FIELD_MISSING",
                f"Field not found: {field_name}",
                selector=rule.primary,
            )
            return None

        # Extract value based on method
        if rule.extraction_method == "text":
            value = element.get_text(strip=True)
        elif rule.extraction_method == "html":
            value = str(element)
        elif rule.extraction_method == "attribute" and rule.attribute_name:
            value = element.get(rule.attribute_name, "")
        else:
            value = element.get_text(strip=True)

        # Apply post-processors
        value = self._apply_post_processors(value, rule.post_processors)

        self.logger.debug(
            "EXTRACT", "FIELD",
            f"Extracted {field_name}",
            selector=rule.primary,
            length=len(value),
        )

        return value

    def _apply_post_processors(
        self,
        value: str,
        processors: list[str],
    ) -> str:
        """Apply post-processing transformations."""
        for processor in processors:
            if processor == "strip":
                value = value.strip()
            elif processor == "lower":
                value = value.lower()
            elif processor == "upper":
                value = value.upper()
            elif processor == "normalize_whitespace":
                import re
                value = re.sub(r'\s+', ' ', value).strip()
            elif processor == "remove_html":
                from bs4 import BeautifulSoup
                value = BeautifulSoup(value, "lxml").get_text()
            # Add more processors as needed

        return value

    async def extract_with_structure(
        self,
        url: str,
        html: str,
        structure: PageStructure,
    ) -> ExtractionResult:
        """
        Extract content using inferred selectors from structure.

        Used when no explicit strategy exists - infers from content regions.
        """
        self.logger.info(
            "EXTRACT", "INFER",
            f"Inferring extraction from structure for {url}",
        )

        # Build strategy from structure's content regions
        strategy = ExtractionStrategy(
            domain=structure.domain,
            page_type=structure.page_type,
            version=structure.version,
        )

        for region in structure.content_regions:
            rule = SelectorRule(
                primary=region.primary_selector,
                fallbacks=region.fallback_selectors,
                confidence=region.confidence,
            )

            if region.name == "title":
                strategy.title = rule
            elif region.name == "content":
                strategy.content = rule
            else:
                strategy.metadata[region.name] = rule

        return await self.extract(url, html, strategy)
```

---

## fingerprint/extraction/file_writer.py

```python
"""
File writer for extracted content.

Saves extracted content to files in various formats.

Verbose logging pattern:
[FILEWRITER:OPERATION] Message
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fingerprint.config import Config
from fingerprint.core.verbose import get_logger
from fingerprint.models import ExtractedContent
from fingerprint.extraction.formats import OutputFormatter, JSONFormatter, CSVFormatter


class FileWriter:
    """
    Save extracted content to files.

    Usage:
        writer = FileWriter(config)
        path = await writer.save(content, format="json")
    """

    def __init__(self, config: Config):
        self.config = config
        self.extraction_config = config.extraction
        self.logger = get_logger()

        # Initialize formatters
        self._formatters: dict[str, OutputFormatter] = {
            "json": JSONFormatter(),
            "csv": CSVFormatter(),
        }

        # Ensure output directory exists
        self.output_dir = Path(self.extraction_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "FILEWRITER", "INIT",
            "File writer initialized",
            output_dir=str(self.output_dir),
            formats=list(self._formatters.keys()),
        )

    async def save(
        self,
        content: ExtractedContent,
        format: str = "json",
        filename: str | None = None,
    ) -> str:
        """
        Save extracted content to file.

        Args:
            content: ExtractedContent to save
            format: Output format (json, csv)
            filename: Optional custom filename

        Returns:
            Path to saved file

        Verbose output:
            [FILEWRITER:WRITE] Saving content
              - domain: example.com
              - format: json
            [FILEWRITER:PATH] File path generated
              - path: ./extracted/example.com/article/2024-01-15_abc123.json
            [FILEWRITER:COMPLETE] Content saved
              - size: 4523 bytes
        """
        self.logger.info(
            "FILEWRITER", "WRITE",
            f"Saving content from {content.domain}",
            format=format,
        )

        # Get formatter
        formatter = self._formatters.get(format)
        if not formatter:
            raise ValueError(f"Unknown format: {format}")

        # Generate file path
        if filename:
            file_path = self.output_dir / filename
        else:
            file_path = self._generate_path(content, format)

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.debug(
            "FILEWRITER", "PATH",
            f"File path: {file_path}",
        )

        # Format and write content
        formatted_data = formatter.format(content, self.extraction_config.include_metadata)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(formatted_data)

        # Update content with output path
        content.output_file = str(file_path)

        file_size = file_path.stat().st_size

        self.logger.info(
            "FILEWRITER", "COMPLETE",
            f"Content saved to {file_path}",
            size=f"{file_size} bytes",
        )

        return str(file_path)

    async def save_batch(
        self,
        contents: list[ExtractedContent],
        format: str = "json",
    ) -> list[str]:
        """
        Save multiple extracted contents.

        Returns list of file paths.
        """
        paths = []
        for content in contents:
            path = await self.save(content, format)
            paths.append(path)
        return paths

    def _generate_path(self, content: ExtractedContent, format: str) -> Path:
        """Generate file path for content."""
        # Structure: output_dir/domain/page_type/date_hash.format
        date_str = content.extracted_at.strftime("%Y-%m-%d")
        url_hash = self._hash_url(content.url)[:8]

        filename = f"{date_str}_{url_hash}.{format}"

        return self.output_dir / content.domain / content.page_type / filename

    def _hash_url(self, url: str) -> str:
        """Generate short hash for URL."""
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()

    def register_formatter(self, name: str, formatter: OutputFormatter) -> None:
        """Register custom output formatter."""
        self._formatters[name] = formatter
        self.logger.debug(
            "FILEWRITER", "REGISTER",
            f"Registered formatter: {name}",
        )
```

---

## fingerprint/extraction/formats.py

```python
"""
Output format handlers for extracted content.

Verbose logging pattern:
[FORMAT:OPERATION] Message
"""

import csv
import json
import io
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime
from typing import Any

from fingerprint.models import ExtractedContent


class OutputFormatter(ABC):
    """Base class for output formatters."""

    @abstractmethod
    def format(self, content: ExtractedContent, include_metadata: bool = True) -> str:
        """Format content as string."""
        pass

    @abstractmethod
    def format_batch(self, contents: list[ExtractedContent], include_metadata: bool = True) -> str:
        """Format multiple contents as string."""
        pass


class JSONFormatter(OutputFormatter):
    """JSON output formatter."""

    def format(self, content: ExtractedContent, include_metadata: bool = True) -> str:
        """Format content as JSON."""
        data = self._to_dict(content, include_metadata)
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)

    def format_batch(self, contents: list[ExtractedContent], include_metadata: bool = True) -> str:
        """Format multiple contents as JSON array."""
        data = [self._to_dict(c, include_metadata) for c in contents]
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)

    def _to_dict(self, content: ExtractedContent, include_metadata: bool) -> dict[str, Any]:
        """Convert content to dictionary."""
        data = {
            "url": content.url,
            "domain": content.domain,
            "page_type": content.page_type,
            "title": content.title,
            "content": content.content,
            "metadata": content.metadata,
        }

        if include_metadata:
            data["_extraction"] = {
                "extracted_at": content.extracted_at.isoformat(),
                "strategy_version": content.strategy_version,
                "confidence": content.extraction_confidence,
            }

        if content.html:
            data["html"] = content.html

        return data


class CSVFormatter(OutputFormatter):
    """CSV output formatter."""

    def format(self, content: ExtractedContent, include_metadata: bool = True) -> str:
        """Format content as CSV row with header."""
        return self.format_batch([content], include_metadata)

    def format_batch(self, contents: list[ExtractedContent], include_metadata: bool = True) -> str:
        """Format multiple contents as CSV."""
        if not contents:
            return ""

        output = io.StringIO()

        # Determine all fields
        base_fields = ["url", "domain", "page_type", "title", "content"]
        metadata_fields = set()

        for content in contents:
            metadata_fields.update(content.metadata.keys())

        metadata_fields = sorted(metadata_fields)

        # Build header
        fields = base_fields + list(metadata_fields)
        if include_metadata:
            fields += ["extracted_at", "strategy_version", "confidence"]

        writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()

        # Write rows
        for content in contents:
            row = {
                "url": content.url,
                "domain": content.domain,
                "page_type": content.page_type,
                "title": content.title,
                "content": content.content,
            }

            # Add metadata fields
            for field in metadata_fields:
                row[field] = content.metadata.get(field, "")

            if include_metadata:
                row["extracted_at"] = content.extracted_at.isoformat()
                row["strategy_version"] = content.strategy_version
                row["confidence"] = content.extraction_confidence

            writer.writerow(row)

        return output.getvalue()


class MarkdownFormatter(OutputFormatter):
    """Markdown output formatter (for human-readable reports)."""

    def format(self, content: ExtractedContent, include_metadata: bool = True) -> str:
        """Format content as Markdown."""
        lines = [
            f"# {content.title}",
            "",
            f"**URL:** {content.url}",
            f"**Domain:** {content.domain}",
            f"**Page Type:** {content.page_type}",
            "",
        ]

        if content.metadata:
            lines.append("## Metadata")
            lines.append("")
            for key, value in content.metadata.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        lines.append("## Content")
        lines.append("")
        lines.append(content.content)
        lines.append("")

        if include_metadata:
            lines.append("---")
            lines.append("")
            lines.append(f"*Extracted: {content.extracted_at.isoformat()}*")
            lines.append(f"*Strategy Version: {content.strategy_version}*")
            lines.append(f"*Confidence: {content.extraction_confidence:.2f}*")

        return "\n".join(lines)

    def format_batch(self, contents: list[ExtractedContent], include_metadata: bool = True) -> str:
        """Format multiple contents as Markdown document."""
        sections = []
        for i, content in enumerate(contents, 1):
            sections.append(f"# Document {i}")
            sections.append("")
            sections.append(self.format(content, include_metadata))
            sections.append("")
            sections.append("---")
            sections.append("")

        return "\n".join(sections)
```

---

## Verbose Logging Examples

```
[2024-01-15T10:30:00Z] [EXTRACT:INIT] Content extractor initialized
  - output_dir: ./extracted

[2024-01-15T10:30:01Z] [EXTRACT:START] Extracting content from https://example.com/article/123
  - strategy_version: 3

[2024-01-15T10:30:01Z] [EXTRACT:FIELD] Extracted title
  - selector: h1.article-title
  - length: 45

[2024-01-15T10:30:01Z] [EXTRACT:FIELD] Extracted content
  - selector: article.main-content
  - length: 5420

[2024-01-15T10:30:01Z] [EXTRACT:FIELD] Extracted author
  - selector: .author-name
  - length: 18

[2024-01-15T10:30:01Z] [EXTRACT:COMPLETE] Extraction complete for https://example.com/article/123
  - fields: 5
  - duration_ms: 45.2
  - confidence: 0.89

[2024-01-15T10:30:02Z] [FILEWRITER:WRITE] Saving content from example.com
  - format: json

[2024-01-15T10:30:02Z] [FILEWRITER:PATH] File path: ./extracted/example.com/article/2024-01-15_abc12345.json

[2024-01-15T10:30:02Z] [FILEWRITER:COMPLETE] Content saved to ./extracted/example.com/article/2024-01-15_abc12345.json
  - size: 4523 bytes
```

---

## Output File Structure

```
extracted/
├── example.com/
│   ├── article/
│   │   ├── 2024-01-15_abc12345.json
│   │   ├── 2024-01-15_def67890.json
│   │   └── 2024-01-15_ghi11111.csv
│   └── product/
│       └── 2024-01-15_jkl22222.json
└── another-site.com/
    └── blog/
        └── 2024-01-15_mno33333.json
```

---

## JSON Output Format

```json
{
  "url": "https://example.com/article/123",
  "domain": "example.com",
  "page_type": "article",
  "title": "Example Article Title",
  "content": "Full article content here...",
  "metadata": {
    "author": "John Smith",
    "date": "2024-01-15",
    "category": "Technology"
  },
  "_extraction": {
    "extracted_at": "2024-01-15T10:30:01Z",
    "strategy_version": 3,
    "confidence": 0.89
  }
}
```

---

## Usage Example

```python
from fingerprint.extraction import ContentExtractor, FileWriter
from fingerprint.storage import StructureStore

async def extract_and_save(url: str, html: str):
    config = load_config()

    # Get extraction strategy from storage
    store = StructureStore(config.redis)
    strategy = await store.get_strategy(domain, page_type)

    # Extract content
    extractor = ContentExtractor(config)
    result = await extractor.extract(url, html, strategy)

    if result.success:
        # Save to file
        writer = FileWriter(config)
        path = await writer.save(result.content, format="json")
        print(f"Saved to: {path}")

        # Also save embedding for the extracted content
        # (handled by ML module)
```
