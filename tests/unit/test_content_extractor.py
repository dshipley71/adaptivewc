"""
Tests for the ContentExtractor.

Verifies that extraction strategies are correctly applied to HTML
to extract structured content.
"""

import pytest

from crawler.extraction.content_extractor import ContentExtractor
from crawler.models import ExtractionStrategy, SelectorRule


class TestContentExtractor:
    """Test the ContentExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create a ContentExtractor instance."""
        return ContentExtractor()

    @pytest.fixture
    def sample_html(self):
        """Sample HTML for testing."""
        return """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <article>
                <h1 class="headline">Breaking News: Test Article</h1>
                <div class="author">By John Doe</div>
                <time datetime="2024-01-15">January 15, 2024</time>
                <div class="article-body">
                    <p>This is the first paragraph of the article.</p>
                    <p>This is the second paragraph with more content.</p>
                    <p>And a third paragraph to meet the minimum length.</p>
                </div>
                <img src="/image1.jpg" alt="Image 1">
                <img src="/image2.jpg" alt="Image 2">
                <a href="/related-article">Related Article</a>
            </article>
        </body>
        </html>
        """

    @pytest.fixture
    def basic_strategy(self):
        """Basic extraction strategy."""
        return ExtractionStrategy(
            domain="example.com",
            page_type="article",
            version=1,
            title=SelectorRule(
                primary="h1.headline",
                fallbacks=["h1", "title"],
                extraction_method="text",
                confidence=0.9,
            ),
            content=SelectorRule(
                primary=".article-body",
                fallbacks=["article", "main"],
                extraction_method="text",
                confidence=0.85,
            ),
            metadata={
                "author": SelectorRule(
                    primary=".author",
                    confidence=0.8,
                ),
                "date": SelectorRule(
                    primary="time[datetime]",
                    attribute_name="datetime",
                    extraction_method="attribute",
                    confidence=0.95,
                ),
            },
            images=SelectorRule(
                primary="article img",
                confidence=0.8,
            ),
            links=SelectorRule(
                primary="article a",
                confidence=0.7,
            ),
        )

    def test_extract_title(self, extractor, sample_html, basic_strategy):
        """Test extracting title from HTML."""
        result = extractor.extract(
            url="https://example.com/article",
            html=sample_html,
            strategy=basic_strategy,
        )

        assert result.success
        assert result.content is not None
        assert result.content.title == "Breaking News: Test Article"
        assert result.content.confidence > 0.0

    def test_extract_content(self, extractor, sample_html, basic_strategy):
        """Test extracting main content from HTML."""
        result = extractor.extract(
            url="https://example.com/article",
            html=sample_html,
            strategy=basic_strategy,
        )

        assert result.success
        assert result.content is not None
        assert "first paragraph" in result.content.content
        assert "second paragraph" in result.content.content
        assert len(result.content.content) > 100

    def test_extract_metadata(self, extractor, sample_html, basic_strategy):
        """Test extracting metadata fields."""
        result = extractor.extract(
            url="https://example.com/article",
            html=sample_html,
            strategy=basic_strategy,
        )

        assert result.success
        assert result.content is not None
        assert "author" in result.content.metadata
        assert "John Doe" in result.content.metadata["author"]
        assert "date" in result.content.metadata
        assert result.content.metadata["date"] == "2024-01-15"

    def test_extract_images(self, extractor, sample_html, basic_strategy):
        """Test extracting image URLs."""
        result = extractor.extract(
            url="https://example.com/article",
            html=sample_html,
            strategy=basic_strategy,
        )

        assert result.success
        assert result.content is not None
        assert len(result.content.images) == 2
        assert "/image1.jpg" in result.content.images
        assert "/image2.jpg" in result.content.images

    def test_extract_links(self, extractor, sample_html, basic_strategy):
        """Test extracting links."""
        result = extractor.extract(
            url="https://example.com/article",
            html=sample_html,
            strategy=basic_strategy,
        )

        assert result.success
        assert result.content is not None
        assert len(result.content.links) == 1
        assert "/related-article" in result.content.links

    def test_fallback_selectors(self, extractor, sample_html):
        """Test that fallback selectors work when primary fails."""
        strategy = ExtractionStrategy(
            domain="example.com",
            page_type="article",
            title=SelectorRule(
                primary=".nonexistent-title",
                fallbacks=["h1"],
                confidence=0.9,
            ),
            content=SelectorRule(
                primary=".nonexistent-content",
                fallbacks=["article"],
                confidence=0.85,
            ),
        )

        result = extractor.extract(
            url="https://example.com/article",
            html=sample_html,
            strategy=strategy,
        )

        assert result.success
        assert result.content is not None
        assert result.content.title is not None  # Should use fallback
        assert result.content.content is not None  # Should use fallback

    def test_missing_required_field(self, extractor, sample_html):
        """Test that extraction fails when required fields are missing."""
        strategy = ExtractionStrategy(
            domain="example.com",
            page_type="article",
            title=SelectorRule(
                primary=".nonexistent",
                confidence=0.9,
            ),
            content=SelectorRule(
                primary=".article-body",
                confidence=0.85,
            ),
            required_fields=["title", "content"],
        )

        result = extractor.extract(
            url="https://example.com/article",
            html=sample_html,
            strategy=strategy,
        )

        assert not result.success
        assert "Required field 'title' missing" in result.errors

    def test_confidence_scoring(self, extractor, sample_html, basic_strategy):
        """Test that confidence scores are calculated."""
        result = extractor.extract(
            url="https://example.com/article",
            html=sample_html,
            strategy=basic_strategy,
        )

        assert result.content is not None
        assert 0.0 < result.content.confidence <= 1.0

    def test_invalid_html(self, extractor, basic_strategy):
        """Test handling of malformed HTML."""
        invalid_html = "<html><body><broken"

        result = extractor.extract(
            url="https://example.com/article",
            html=invalid_html,
            strategy=basic_strategy,
        )

        # Should still parse (BeautifulSoup is forgiving)
        # but extraction will likely fail
        assert result is not None

    def test_extract_html_method(self, extractor):
        """Test extraction using HTML method instead of text."""
        html = """
        <html>
        <body>
            <article>
                <div class="content">
                    <p>Paragraph with <strong>bold</strong> text.</p>
                </div>
            </article>
        </body>
        </html>
        """

        strategy = ExtractionStrategy(
            domain="example.com",
            page_type="article",
            content=SelectorRule(
                primary=".content",
                extraction_method="html",
                confidence=0.9,
            ),
        )

        result = extractor.extract(
            url="https://example.com/article",
            html=html,
            strategy=strategy,
        )

        assert result.success
        assert result.content is not None
        assert "<strong>" in result.content.content  # HTML tags preserved
        assert "<p>" in result.content.content

    def test_validation(self, extractor, sample_html, basic_strategy):
        """Test extraction validation."""
        result = extractor.extract(
            url="https://example.com/article",
            html=sample_html,
            strategy=basic_strategy,
        )

        assert extractor.validate_extraction(result, min_title_length=5, min_content_length=50)

    def test_validation_fails_short_content(self, extractor):
        """Test that validation fails for too-short content."""
        html = "<html><body><h1>Hi</h1><p>Short.</p></body></html>"

        strategy = ExtractionStrategy(
            domain="example.com",
            page_type="article",
            title=SelectorRule(primary="h1", confidence=0.9),
            content=SelectorRule(primary="p", confidence=0.9),
        )

        result = extractor.extract(url="https://example.com/test", html=html, strategy=strategy)

        assert not extractor.validate_extraction(
            result, min_title_length=5, min_content_length=100
        )
