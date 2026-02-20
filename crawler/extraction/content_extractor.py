"""
Content extraction engine that applies learned strategies to extract structured data.

Uses BeautifulSoup to apply CSS selector rules from ExtractionStrategy to HTML,
with fallback handling, confidence scoring, and validation.
"""

from datetime import datetime, timezone
from dateutil import parser as dateutil_parser

import dateparser
import json
import os
import re
import time
from typing import Any, Literal, Optional, Dict
import re

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


from bs4 import BeautifulSoup, Tag

from crawler.models import (
    ExtractedContent,
    ExtractionResult,
    ExtractionStrategy,
    SelectorRule,
)
from crawler.utils.logging import CrawlerLogger



class ContentExtractor:
    """
    Applies extraction strategies to HTML to extract structured content.

    Uses learned CSS selectors with fallback chains to robustly extract
    titles, content, metadata, images, and links from web pages.
    """

    def __init__(self, logger: CrawlerLogger | None = None):
        """
        Initialize the content extractor.

        Args:
            logger: Logger instance.
        """
        self.logger = logger or CrawlerLogger("content_extractor")

    def extract(
        self,
        url: str,
        html: str,
        strategy: ExtractionStrategy,
        llm_provider: str | None = None,
        ollama_base_url: str | None = None,
        llm_api_key: str | None = None 
    ) -> ExtractionResult:
        """
        Extract content from HTML using the provided strategy.

        Args:
            url: URL of the page being extracted.
            html: HTML content to extract from.
            strategy: Extraction strategy with selector rules.

        Returns:
            ExtractionResult with extracted content or errors.
        """
        start_time = time.time()
        errors = []
        warnings = []

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception as e:
            self.logger.error("Failed to parse HTML", url=url, error=str(e))
            return ExtractionResult(
                url=url,
                success=False,
                errors=[f"HTML parsing failed: {str(e)}"],
                duration_ms=(time.time() - start_time) * 1000,
            )

        # Extract title
        title = None
        if strategy.title:
            title, title_confidence = self._extract_with_rule(soup, strategy.title)
            if not title:
                warnings.append("Title extraction failed - no matching elements")

        # Extract content
        content = None
        content_confidence = 0.0
        if strategy.content:
            content, content_confidence = self._extract_with_rule(
                soup, strategy.content
            )
            if not content:
                errors.append("Content extraction failed - no matching elements")

        # Extract metadata
        metadata = {}
        metadata_confidences = {}
        if strategy.metadata:
            for key, rule in strategy.metadata.items():
              value, confidence = self._extract_with_rule(soup, rule)
              if value:
                  metadata[key] = value
                  metadata_confidences[key] = confidence
              else:
                  warnings.append(f"Metadata '{key}' extraction failed")

        # Extract date
        if "date" not in metadata:
          detected_date, date_confidence = self._extract_date(soup)
          if detected_date:
            metadata["date"] = detected_date
            metadata_confidences["date"] = date_confidence
          else:
            extractor = get_date_extractor(
              provider=llm_provider, 
              base_url=ollama_base_url,
              api_key=llm_api_key
            )

            result = extractor.extract(html)

            metadata["date"] =  self._parse_date(result.publish_date)
            metadata_confidences["date"] = result.confidence
            warnings.append("Date extraction failed")

        # Extract images
        images = []
        if strategy.images:
            images, _ = self._extract_images(soup, strategy.images)

        # Extract links
        links = []
        if strategy.links:
            links, _ = self._extract_links(soup, strategy.links)

        # Calculate overall confidence
        confidences = [content_confidence]
        if title:
            confidences.append(title_confidence if 'title_confidence' in locals() else 0.0)
        confidences.extend(metadata_confidences.values())

        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Validate required fields
        success = True
        if "title" in strategy.required_fields and not title:
            errors.append("Required field 'title' missing")
            success = False
        if "content" in strategy.required_fields and not content:
            errors.append("Required field 'content' missing")
            success = False

        # Validate content length
        if content and len(content) < strategy.min_content_length:
            warnings.append(
                f"Content length ({len(content)}) below minimum ({strategy.min_content_length})"
            )
            # Don't mark as failure, just warn

        # Build extracted content
        extracted = ExtractedContent(
            url=url,
            title=title,
            content=content,
            metadata=metadata,
            images=images,
            links=links,
            strategy_version=strategy.version,
            confidence=overall_confidence,
        )

        duration_ms = (time.time() - start_time) * 1000

        self.logger.debug(
            "Content extraction completed",
            url=url,
            success=success,
            title_length=len(title) if title else 0,
            content_length=len(content) if content else 0,
            metadata_fields=len(metadata),
            confidence=f"{overall_confidence:.2%}",
            duration_ms=f"{duration_ms:.2f}",
        )

        return ExtractionResult(
            url=url,
            success=success,
            content=extracted,
            errors=errors,
            warnings=warnings,
            strategy_used=f"{strategy.domain}:{strategy.page_type}:v{strategy.version}",
            duration_ms=duration_ms,
        )

    def _extract_with_rule(
        self,
        soup: BeautifulSoup,
        rule: SelectorRule,
    ) -> tuple[str | None, float]:
        """
        Apply a selector rule to extract text with fallback handling.

        Args:
            soup: Parsed HTML.
            rule: Selector rule with primary and fallback selectors.

        Returns:
            Tuple of (extracted_text, confidence).
        """
        # Try primary selector
        try:

          elements = soup.select(rule.primary)
            
          if elements:
              # Extract from all matching elements and join
              texts = []
              for elem in elements:
                text = self._extract_text_from_element(
                    elem, rule.extraction_method, rule.attribute_name
                )
                if "time" in rule.primary or "date" in rule.primary:
                  text = self._parse_date(text)
                if text:
                  texts.append(text)
              if texts:
                if "time" in rule.primary or "date" in rule.primary:
                  # Find the earliest date of the ones that have already happened. Eliminate the future dates.
                  earliest = [min(
                    (x for x in texts),
                    default=None
                    )]
                  texts = earliest
                combined = "\n\n".join(set(texts))
                return combined, rule.confidence
        except Exception as e:
            self.logger.warning(
                "Primary selector failed",
                selector=rule.primary,
                error=str(e),
            )

        # Try fallback selectors
        for i, fallback in enumerate(rule.fallbacks):
            try:
                elements = soup.select(fallback)
                if elements:
                    # Extract from all matching elements and join
                    texts = []
                    for elem in elements:
                        text = self._extract_text_from_element(
                            elem, rule.extraction_method, rule.attribute_name
                        )
                        if text:
                            texts.append(text)
                    if texts:
                        combined = "\n\n".join(texts)
                        # Reduce confidence for using fallback
                        fallback_confidence = rule.confidence * (0.9 ** (i + 1))
                        return combined, fallback_confidence
            except Exception as e:
                self.logger.warning(
                    "Fallback selector failed",
                    selector=fallback,
                    error=str(e),
                )

        return None, 0.0


    def _extract_date(self, soup: BeautifulSoup) -> tuple[str | None, float]:
        """
        Attempt to extract a publication date from the page using
        structured tags and intelligent fallback scanning.
        """

        # 1️⃣ Check common meta tags
        meta_properties = [
            "article:published_time",
            "article:modified_time",
            "og:published_time",
            "pubdate",
            "publish-date",
            "date",
        ]

        for prop in meta_properties:
          tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
          if tag and tag.get("content"):
              parsed = self._parse_date(tag["content"])
              if parsed:
                  return parsed, 0.9


        # 2️⃣ Check <time> elements first (highest confidence)
        for time_tag in soup.find_all("time"):
          # Try datetime attribute first
          datetime_attr = time_tag.get("datetime")
          if datetime_attr:
              parsed = self._parse_date(datetime_attr)
              if parsed:
                return parsed, 0.95

          # Fallback to visible text
          text = time_tag.get_text(strip=True)
          parsed = self._parse_date(text)
          if parsed:
              return parsed, 0.9


        # 3️⃣ Look for elements with date-like class/id names
        date_pattern = re.compile(r"(date|time|publish|posted)", re.IGNORECASE)
        for element in soup.find_all(attrs={"class": date_pattern}) + soup.find_all(attrs={"id": date_pattern}):
            text = element.get_text(strip=True)
            parsed = self._parse_date(text)
            if parsed:
                return parsed, 0.75

        # 4️⃣ Fallback: Scan full page text for date-like patterns
        full_text = soup.get_text(separator=" ", strip=True)

        # Broad regex for date candidates
        date_candidates = re.findall(
            r"\b(?:\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|\d{1,2}\s+\w+\s+\d{2,4}|\w+\s+\d{1,2},?\s+\d{2,4})\b",
            full_text,
        )

        for candidate in date_candidates[:10]:  # limit attempts
            parsed = self._parse_date(candidate)
            if parsed:
                return parsed, 0.6

        return None, 0.0

    def _parse_date(self, value: str) -> str | None:
      if not value or len(value) < 4:
          return None

      now = datetime.now(timezone.utc)

      def valid(dt: datetime) -> str | None:
          if dt.tzinfo is None:
              dt = dt.replace(tzinfo=timezone.utc)
          if dt.astimezone(timezone.utc) <= now:
              return dt.isoformat()
          return None

      # 1️⃣ Try normal parsing
      try:
          dt = dateutil_parser.parse(value, fuzzy=True)
          result = valid(dt)
          if result:
              return result
      except Exception:
          pass

      # 2️⃣ ISO inside string
      iso = re.search(
          r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?',
          value
      )
      if iso:
          try:
              dt = datetime.fromisoformat(iso.group(0).replace("Z", "+00:00"))
              result = valid(dt)
              if result:
                  return result
          except Exception:
              pass

      # 3️⃣ Unix timestamp
      unix = re.search(r'\b\d{10,13}\b', value)
      if unix:
          try:
              ts = int(unix.group(0))
              if len(unix.group(0)) == 13:
                  ts /= 1000
              dt = datetime.fromtimestamp(ts, tz=timezone.utc)
              result = valid(dt)
              if result:
                  return result
          except Exception:
              pass

      return None


    def _extract_text_from_element(
        self,
        element: Tag,
        extraction_method: str = "text",
        attribute_name: str | None = None,
    ) -> str | None:
        """
        Extract text from an element based on the extraction method.

        Args:
            element: BeautifulSoup element.
            extraction_method: How to extract ("text", "html", "attribute").
            attribute_name: Attribute name if method is "attribute".

        Returns:
            Extracted string or None.
        """
        try:
          if extraction_method == "text":
              text = element.get_text(strip=True, separator=" ")
              return text if text else None
          elif extraction_method == "html":
              html = str(element)
              return html if html else None
          elif extraction_method == "attribute" and attribute_name:
              attr = element.get(attribute_name)
              return str(attr) if attr else None
          else:
              # Default to text
              text = element.get_text(strip=True, separator=" ")
              return text if text else None
        except Exception as e:
            self.logger.warning(
                "Text extraction failed",
                method=extraction_method,
                error=str(e),
            )
            return None

    def _extract_images(
        self,
        soup: BeautifulSoup,
        rule: SelectorRule,
    ) -> tuple[list[str], float]:
        """
        Extract image URLs using a selector rule.

        Args:
            soup: Parsed HTML.
            rule: Selector rule for images.

        Returns:
            Tuple of (image_urls, confidence).
        """
        images = []

        # Try primary selector
        try:
            elements = soup.select(rule.primary)
            for elem in elements:
                src = elem.get("src") or elem.get("data-src")
                if src:
                    images.append(str(src))

            if images:
                return images, rule.confidence
        except Exception as e:
            self.logger.warning(
                "Image extraction failed",
                selector=rule.primary,
                error=str(e),
            )

        # Try fallbacks
        for i, fallback in enumerate(rule.fallbacks):
            try:
                elements = soup.select(fallback)
                for elem in elements:
                    src = elem.get("src") or elem.get("data-src")
                    if src:
                        images.append(str(src))

                if images:
                    fallback_confidence = rule.confidence * (0.9 ** (i + 1))
                    return images, fallback_confidence
            except Exception as e:
                self.logger.warning(
                    "Image fallback failed",
                    selector=fallback,
                    error=str(e),
                )

        return images, 0.0

    def _extract_links(
        self,
        soup: BeautifulSoup,
        rule: SelectorRule,
    ) -> tuple[list[str], float]:
        """
        Extract link URLs using a selector rule.

        Args:
            soup: Parsed HTML.
            rule: Selector rule for links.

        Returns:
            Tuple of (link_urls, confidence).
        """
        links = []

        # Try primary selector
        try:
            elements = soup.select(rule.primary)
            for elem in elements:
                href = elem.get("href")
                if href:
                    links.append(str(href))

            if links:
                return links, rule.confidence
        except Exception as e:
            self.logger.warning(
                "Link extraction failed",
                selector=rule.primary,
                error=str(e),
            )

        # Try fallbacks
        for i, fallback in enumerate(rule.fallbacks):
            try:
                elements = soup.select(fallback)
                for elem in elements:
                    href = elem.get("href")
                    if href:
                        links.append(str(href))

                if links:
                    fallback_confidence = rule.confidence * (0.9 ** (i + 1))
                    return links, fallback_confidence
            except Exception as e:
                self.logger.warning(
                    "Link fallback failed",
                    selector=fallback,
                    error=str(e),
                )

        return links, 0.0

    def validate_extraction(
        self,
        result: ExtractionResult,
        min_title_length: int = 5,
        min_content_length: int = 100,
    ) -> bool:
        """
        Validate that extraction meets quality thresholds.

        Args:
            result: Extraction result to validate.
            min_title_length: Minimum acceptable title length.
            min_content_length: Minimum acceptable content length.

        Returns:
            True if extraction is valid.
        """
        if not result.success or not result.content:
            return False

        content = result.content

        # Check title
        if content.title and len(content.title) < min_title_length:
            self.logger.warning(
                "Title too short",
                url=result.url,
                length=len(content.title),
            )
            return False

        # Check content
        if content.content and len(content.content) < min_content_length:
            self.logger.warning(
                "Content too short",
                url=result.url,
                length=len(content.content),
            )
            return False

        # Check confidence
        if content.confidence < 0.5:
            self.logger.warning(
                "Confidence too low",
                url=result.url,
                confidence=content.confidence,
            )
            return False

        return True


@dataclass
class PublishDateResult:
    """Structured publish date extraction result."""

    publish_date: Optional[str]
    source_hint: Optional[str]
    confidence: float
    extraction_method: str  # LLM provider name

    def to_dict(self) -> dict[str, Any]:
        return {
            "publish_date": self.publish_date,
            "source_hint": self.source_hint,
            "confidence": self.confidence,
            "extraction_method": self.extraction_method,
        }


class BaseDateExtractor(ABC):
    @abstractmethod
    def extract(self, html: str) -> PublishDateResult:
        pass


class LLMDateExtractor(BaseDateExtractor):
    """
    Uses an LLM provider to determine:
    - The publish date
    - Where it was found (meta tag, class, property, JSON-LD, etc.)
    """

    DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
        "ollama": "llama3.2",
        "ollama-cloud": "gemma3:12b-cloud",
    }

    # Default Ollama endpoints
    OLLAMA_LOCAL_URL = "http://localhost:11434"
    # Ollama cloud uses native Ollama API format at /api/chat
    OLLAMA_CLOUD_URL = "https://ollama.com"

    def __init__(
        self,
        provider: Literal["openai", "anthropic", "ollama", "ollama-cloud"] = "ollama-cloud",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,  # useful for ollama-cloud
    ):
        self.provider = provider
        self.model = model or self.DEFAULT_MODELS.get(provider)
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client:
            return self._client

        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY")
            )

        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            )

        elif self.provider == "ollama":
              # Local Ollama - no API key needed
              try:
                  import httpx
                  base_url = self.base_url or os.environ.get(
                      "OLLAMA_BASE_URL", self.OLLAMA_LOCAL_URL
                  )
                  self._client = {"base_url": base_url, "httpx": httpx}
              except ImportError:
                  raise ImportError("httpx package required: pip install httpx")

        elif self.provider == "ollama-cloud":
            # Ollama cloud - uses native Ollama API format at /api/chat
            # API key required for authentication
            try:
                import httpx
                # API key from parameter or environment variable
                api_key = self.api_key if self.api_key is not None else os.environ.get("OLLAMA_API_KEY", "")
                base_url = self.base_url or os.environ.get("OLLAMA_BASE_URL", self.OLLAMA_CLOUD_URL)
                self._client = {"base_url": base_url, "api_key": api_key, "httpx": httpx}
            except ImportError:
                raise ImportError("httpx package required: pip install httpx")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        return self._client

    def _call_ollama(self, prompt: str, max_tokens: int = 200) -> str:
        """Make a request to Ollama API (local or cloud)."""
        client = self._get_client()
        httpx = client["httpx"]
        base_url = client["base_url"]

        headers = {"Content-Type": "application/json"}
        # Only add Authorization header if API key is non-empty
        if client.get("api_key"):
            headers["Authorization"] = f"Bearer {client['api_key']}"

        # Both local Ollama and Ollama Cloud use native Ollama API format
        # Ollama Cloud: https://ollama.com/api/chat
        # Local Ollama: http://localhost:11434/api/generate
        if self.provider == "ollama-cloud":
            # Ollama Cloud uses /api/chat with messages format
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.3,
                },
            }
            endpoint = f"{base_url}/api/chat"
            response = httpx.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=60.0,
            )
            response.raise_for_status()
            result = response.json()
            # Ollama chat response format: {"message": {"content": "..."}}
            return result.get("message", {}).get("content", "").strip()
        else:
            # Local Ollama uses /api/generate with prompt format
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.3,
                },
            }
            endpoint = f"{base_url}/api/generate"
            response = httpx.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=60.0,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()

    def extract(self, html: str) -> PublishDateResult:
        client = self._get_client()

        prompt = f"""
          You are analyzing raw HTML from a news article.

          Task:
          1. Identify the article publish date.
          2. Identify the exact HTML element or property where it was found
            (e.g., meta[property="article:published_time"],
              div class="post-date", JSON-LD datePublished, etc.)

          Return ONLY valid JSON:

          {{
            "publish_date": "...",
            "source_hint": "...",
            "confidence": 0.0-1.0
          }}

          HTML:
          {html[:15000]}
          """

        try:
            # OpenAI-compatible providers
            if self.provider == "openai":
              response = client.chat.completions.create(
                  model=self.model,
                  messages=[{"role": "user", "content": prompt}],
                  max_tokens=200,
                  temperature=0.3,
              )
              llm_response = response.choices[0].message.content.strip()

            elif self.provider == "anthropic":
              response = client.messages.create(
                  model=self.model,
                  max_tokens=200,
                  messages=[{"role": "user", "content": prompt}],
              )
              llm_response = response.content[0].text.strip()

            elif self.provider in ("ollama", "ollama-cloud"):
              llm_response = self._call_ollama(prompt, max_tokens=200)
              

            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            data = parse_llm_json(llm_response)

            return PublishDateResult(
                publish_date=data.get("publish_date"),
                source_hint=data.get("source_hint"),
                confidence=float(data.get("confidence", 0.5)),
                extraction_method=self.provider,
            )

        except Exception:
            return PublishDateResult(
                publish_date=None,
                source_hint="LLM parse error",
                confidence=0.0,
                extraction_method=self.provider,
            )

def parse_llm_json(raw: str) -> Dict[str, Any]:
    """
    Extracts and parses JSON from an LLM response that may include
    markdown code fences like ```json ... ```.
    """

    if not raw:
        raise ValueError("Empty response")

    # Remove triple backtick blocks (```json ... ```)
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    # Strip any leading/trailing whitespace again
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nCleaned string:\n{cleaned}")

        


def get_date_extractor(**kwargs) -> BaseDateExtractor:
    """
    Factory function to retrieve LLM-based date extractor.

    Example:
        extractor = get_date_extractor(
            provider="ollama",
            model="llama3.2"
        )
    """
    return LLMDateExtractor(**kwargs)