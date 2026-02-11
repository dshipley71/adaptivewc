#!/usr/bin/env python3
"""
ML-Powered News Website Monitor

A practical example showing how to use the adaptive web crawler's ML features
to monitor news aggregator websites using:
- Sentence-transformer embeddings for semantic structure comparison
- ML classifiers (XGBoost/LightGBM) for page type prediction
- LLM (Ollama Cloud) for rich change descriptions

This is the ML-powered version of news_monitor.py that uses AI/ML
instead of rule-based heuristics for change detection and descriptions.

Use Case:
    An admin monitors news aggregators like Rantingly, Memeorandum, or
    Hacker News for structural changes. When changes are detected via
    embedding similarity, the system uses an LLM to generate rich
    descriptions of what changed and its potential impact.

Usage:
    # Option 1: Start Redis with Docker
    docker run -d -p 6379:6379 redis:7-alpine

    # Run the monitor with Ollama Cloud (default)
    # Note: Ollama Cloud may not require an API key for public endpoints
    python examples/news_monitor_ml.py

    # Monitor specific URL
    python examples/news_monitor_ml.py --url https://rantingly.com

    # Use specific Ollama Cloud model (models use -cloud suffix)
    python examples/news_monitor_ml.py --llm-model gemma3:12b-cloud

    # Use local Ollama instead of cloud
    python examples/news_monitor_ml.py --llm-provider ollama

    # Use OpenAI for descriptions
    export OPENAI_API_KEY=your_key
    python examples/news_monitor_ml.py --llm-provider openai

    # Train classifier on collected data
    python examples/news_monitor_ml.py --train-classifier

Requirements:
    - Redis running on localhost:6379
    - pip install -e ".[ml]"  # Includes sentence-transformers, xgboost, etc.

Ollama Cloud Configuration:
    - Default URL: https://ollama.com/v1
    - Default model: gemma3:12b-cloud (models use -cloud suffix)
    - API key is optional for public endpoints (use OLLAMA_API_KEY if needed)
"""

import argparse
import asyncio
import csv
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def install_redis_local() -> bool:
    """Install Redis locally on Debian/Ubuntu systems."""
    print("Installing Redis locally...")
    print("This requires sudo access on Debian/Ubuntu systems.\n")

    commands = [
        (
            "curl -fsSL https://packages.redis.io/gpg | "
            "sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg --yes"
        ),
        (
            'echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] '
            'https://packages.redis.io/deb $(lsb_release -cs) main" | '
            "sudo tee /etc/apt/sources.list.d/redis.list"
        ),
        "sudo apt-get update",
        "sudo apt-get install -y redis-stack-server",
        "pip install redis",
    ]

    for cmd in commands:
        print(f"Running: {cmd[:60]}...")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  Warning: {result.stderr[:100] if result.stderr else 'Command failed'}")
        except Exception as e:
            print(f"  Error: {e}")
            return False

    print("\nStarting Redis server...")
    try:
        subprocess.run(
            "sudo systemctl start redis-stack-server || redis-server --daemonize yes",
            shell=True,
            capture_output=True,
        )
    except Exception:
        try:
            subprocess.Popen(
                ["redis-server", "--daemonize", "yes"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"Could not start Redis: {e}")
            return False

    print("\nRedis installation complete!")
    return True


def check_redis_installed() -> bool:
    """Check if Redis is available."""
    return shutil.which("redis-server") is not None


import httpx
import redis.asyncio as redis

from crawler.adaptive.change_detector import ChangeDetector, ChangeAnalysis
from crawler.adaptive.strategy_learner import StrategyLearner
from crawler.adaptive.structure_analyzer import StructureAnalyzer
from crawler.compliance.robots_parser import RobotsChecker
from crawler.extraction.content_extractor import ContentExtractor
from crawler.models import ExtractionStrategy, PageStructure, ExtractionResult
from crawler.storage.structure_store import StructureStore
from crawler.utils.logging import CrawlerLogger, setup_logging

# ML components
from crawler.ml import (
    StructureEmbeddingModel,
    StructureClassifier,
    MLChangeDetector,
    LLMDescriptionGenerator,
    get_description_generator,
    ClassifierType,
    DescriptionMode,
    export_training_data,
)


@dataclass
class MLContentChange:
    """Represents a detected content change with ML analysis."""

    url: str
    detected_at: datetime
    change_type: str  # "new_content", "content_updated", "structure_changed"
    embedding_similarity: float  # ML-computed similarity
    llm_description: str | None = None  # LLM-generated description
    predicted_page_type: str | None = None  # ML-predicted page type
    page_type_confidence: float | None = None
    extracted_content: ExtractionResult | None = None
    previous_embedding: list[float] | None = None
    current_embedding: list[float] | None = None
    change_analysis: ChangeAnalysis | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "url": self.url,
            "detected_at": self.detected_at.isoformat(),
            "change_type": self.change_type,
            "embedding_similarity": self.embedding_similarity,
            "llm_description": self.llm_description,
            "predicted_page_type": self.predicted_page_type,
            "page_type_confidence": self.page_type_confidence,
        }

        if self.change_analysis:
            result["change_analysis"] = self.change_analysis.to_dict()

        return result


@dataclass
class MLMonitorConfig:
    """Configuration for the ML-powered monitor."""

    urls: list[str] = field(default_factory=list)
    check_interval: int = 300  # 5 minutes
    redis_url: str = "redis://localhost:6379/0"
    output_dir: str = "./news_monitor_output_ml"
    user_agent: str = "NewsMonitorML/1.0 (ML-Powered Change Detection)"
    respect_robots: bool = True
    max_retries: int = 3
    request_timeout: float = 30.0

    # ML settings
    embedding_model: str = "all-MiniLM-L6-v2"
    classifier_type: str = "xgboost"
    breaking_threshold: float = 0.85

    # LLM settings
    llm_provider: str = "ollama-cloud"
    llm_model: str | None = None
    ollama_api_key: str | None = None
    ollama_base_url: str | None = None

    # Model persistence
    model_dir: str = "./ml_models_news"

    # Verbose output
    verbose: bool = False


class MLNewsMonitor:
    """
    ML-Powered News Website Monitor.

    Uses machine learning for:
    - Semantic structure comparison via sentence-transformer embeddings
    - Page type classification using XGBoost/LightGBM
    - Rich change descriptions via Ollama Cloud LLM

    Features:
    - Learns page structure embeddings on first visit
    - Detects structural changes via embedding similarity
    - Uses LLM to describe changes in natural language
    - Can train classifiers on collected data
    - Stores embeddings for similarity search across sites
    """

    def __init__(
        self,
        config: MLMonitorConfig,
        on_change: Callable[[MLContentChange], None] | None = None,
    ):
        """
        Initialize the ML-powered monitor.

        Args:
            config: Monitor configuration.
            on_change: Callback function when changes detected.
        """
        self.config = config
        self.on_change = on_change
        self.logger = CrawlerLogger("ml_news_monitor")

        # Traditional components
        self.structure_analyzer = StructureAnalyzer(logger=self.logger)
        self.change_detector = ChangeDetector(logger=self.logger)
        self.strategy_learner = StrategyLearner(logger=self.logger)
        self.content_extractor = ContentExtractor(logger=self.logger)
        self.robots_checker = RobotsChecker(user_agent=config.user_agent)

        # ML components
        self.embedding_model = StructureEmbeddingModel(model_name=config.embedding_model)
        self.ml_change_detector = MLChangeDetector(
            embedding_model=self.embedding_model,
            breaking_threshold=config.breaking_threshold,
            classifier_type=ClassifierType(config.classifier_type),
        )

        # LLM description generator
        self.llm_generator: LLMDescriptionGenerator | None = None
        self._init_llm_generator()

        # Classifier (optionally trained)
        self.classifier: StructureClassifier | None = None

        # Storage
        self.redis_client: redis.Redis | None = None
        self.structure_store: StructureStore | None = None

        # HTTP client
        self.http_client: httpx.AsyncClient | None = None

        # State
        self._running = False
        self._structure_embeddings: dict[str, list[float]] = {}
        self._change_history: list[MLContentChange] = []
        self._training_data: list[tuple[PageStructure, str]] = []

    def _init_llm_generator(self) -> None:
        """Initialize the LLM description generator."""
        try:
            llm_kwargs = {
                "provider": self.config.llm_provider,
            }
            if self.config.llm_model:
                llm_kwargs["model"] = self.config.llm_model
            if self.config.ollama_api_key:
                llm_kwargs["api_key"] = self.config.ollama_api_key
            if self.config.ollama_base_url:
                llm_kwargs["ollama_base_url"] = self.config.ollama_base_url

            self.llm_generator = get_description_generator(
                DescriptionMode.LLM,
                **llm_kwargs,
            )
            self.logger.info(
                "LLM generator initialized",
                provider=self.config.llm_provider,
            )
        except Exception as e:
            self.logger.warning(
                "Failed to initialize LLM generator, using rules-based fallback",
                error=str(e),
            )
            self.llm_generator = None

    async def start(self) -> None:
        """Initialize connections and load models."""
        self.logger.info("Starting ML News Monitor")

        # Connect to Redis
        try:
            self.redis_client = redis.from_url(self.config.redis_url)
            await self.redis_client.ping()
            self.structure_store = StructureStore(
                redis_client=self.redis_client,
                logger=self.logger,
            )
            self.logger.info("Connected to Redis")
        except Exception as e:
            self.logger.error("Failed to connect to Redis", error=str(e))
            raise

        # Initialize HTTP client
        self.http_client = httpx.AsyncClient(
            headers={"User-Agent": self.config.user_agent},
            timeout=self.config.request_timeout,
            follow_redirects=True,
        )

        # Create directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)

        # Load ML models if available
        await self._load_ml_models()

        # Load previous embeddings from Redis
        await self._load_embeddings()

    async def stop(self) -> None:
        """Cleanup resources."""
        self._running = False

        # Save ML models
        await self._save_ml_models()

        if self.http_client:
            await self.http_client.aclose()

        if self.redis_client:
            await self.redis_client.aclose()

        self.logger.info("ML Monitor stopped")

    async def crawler_training_data(
      self, 
      redis_client, 
      manual_page_types: Optional[Dict[str, str]] = None
      ):
        """
        Fetch crawler structure records and return training-ready pairs: [stored_structure, page_type].

        Args:
            redis_client: Redis client instance.
            manual_page_types: Optional dictionary mapping domain/page keys to page_type.
                              If provided, Redis will be bypassed for these entries.

        Returns:
            List of [stored_structure, page_type] pairs for training.
        """
        for_training = []

        if manual_page_types:
          # Use manually provided page types
          for url, page_type in manual_page_types.items():

            domain = self._extract_domain(url)

            stored_structure = await self.structure_store.get_structure(domain, page_type, "default")
            if stored_structure:
              for_training.append([stored_structure, page_type])
            else:
              # Grab the structure
              result = await self.fetch_page(url)
              if not result:
                self.logger.error("Failed to fetch page", url=url)
                continue

              html, status_code = result

              if status_code != 200:
                  self.logger.warning("Non-200 status", url=url, status=status_code)
                  continue
              
              stored_structure = self.structure_analyzer.analyze(html, url, page_type)

              # Infer extraction strategy to store to Redis
              learned = self.strategy_learner.infer(html, stored_structure)
              strategy = learned.strategy

              await self.structure_store.save_structure(
                stored_structure, strategy, "default"
              )

              for_training.append([stored_structure, page_type])
        else:
            # Fallback: fetch keys from Redis
            cursor, keys = 0, []
            while True:
                cursor, batch = await redis_client.scan(cursor=cursor, match="crawler:structure:*")
                keys.extend(batch)
                if cursor == 0:
                    break

            if not keys:
                return []

            # Fetch values and prepare training data
            values = await redis_client.mget(keys)
            for raw_value in values:
                if not raw_value:
                  continue

                if isinstance(raw_value, bytes):
                  raw_value = raw_value.decode("utf-8")

                value = json.loads(raw_value)
                stored_structure = await self.structure_store.get_structure(
                    value["domain"], value["page_type"], "default"
                )
                for_training.append([stored_structure, value["page_type"]])

        return for_training

    async def _load_embeddings(self) -> None:
        """Load structure embeddings from Redis."""
        if not self.redis_client:
            return

        for url in self.config.urls:
            key = f"news_ml_embedding:{self._url_to_key(url)}"
            data = await self.redis_client.get(key)
            if data:
                self._structure_embeddings[url] = json.loads(data)

    async def _save_embedding(self, url: str, embedding: list[float]) -> None:
        """Save structure embedding to Redis."""
        if not self.redis_client:
            return

        key = f"news_ml_embedding:{self._url_to_key(url)}"
        await self.redis_client.set(key, json.dumps(embedding))
        self._structure_embeddings[url] = embedding

    def _url_to_key(self, url: str) -> str:
        """Convert URL to a safe Redis key."""
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()

    async def _load_ml_models(self) -> None:
        """Load trained ML models from disk."""
        model_dir = Path(self.config.model_dir)

        change_detector_path = model_dir / "change_detector.pkl"
        if change_detector_path.exists():
            try:
                self.ml_change_detector.load(str(change_detector_path))
                self.logger.info("Loaded change detector model")
            except Exception as e:
                self.logger.warning("Failed to load change detector", error=str(e))

        classifier_path = model_dir / "classifier.pkl"
        if classifier_path.exists():
          print(f"=================> classifier path existed: {classifier_path}")
          try:
              self.classifier = StructureClassifier(
                  embedding_model=self.embedding_model,
                  classifier_type=ClassifierType(self.config.classifier_type),
              )
              self.classifier.load(str(classifier_path))
              self.logger.info("Loaded page type classifier")
          except Exception as e:
              self.logger.warning("Failed to load classifier", error=str(e))
        else:
          print(f"=========> classifier path did not exist")

    async def _save_ml_models(self) -> None:
        """Save trained ML models to disk."""
        model_dir = Path(self.config.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        change_detector_path = model_dir / "change_detector.pkl"
        try:
            self.ml_change_detector.save(str(change_detector_path))
            self.logger.info("Saved change detector model")
        except Exception as e:
            self.logger.warning("Failed to save change detector", error=str(e))

        if self.classifier is not None:
          classifier_path = model_dir / "classifier.pkl"
          try:
              self.classifier.save(str(classifier_path))
              self.logger.info("Saved page type classifier")
          except Exception as e:
              self.logger.warning("Failed to save classifier", error=str(e))

    def _extract_domain(self, url_or_key: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url_or_key):
          url_or_key = "http://" + url_or_key
        parsed = urlparse(url_or_key)
        return parsed.netloc

    def _classify_page_type_rules(self, url: str) -> str:
        """Classify page type based on URL patterns (fallback)."""
        print("===============> Rules based classifier triggered")
        url_lower = url.lower()

        # News aggregator patterns
        if any(p in url_lower for p in ["/politics", "/political"]):
            return "politics"
        elif any(p in url_lower for p in ["/business", "/economy", "/finance", "/markets"]):
            return "business"
        elif any(p in url_lower for p in ["/tech", "/technology", "/science"]):
            return "technology"
        elif any(p in url_lower for p in ["/sport", "/esport", "/football", "/soccer", "/basketball", "/baseball", "/hockey", "/tennis", "/golf", "/athletics", "/cricket", "/fifa", "/olympics", ]):
            return "sports"
        elif any(p in url_lower for p in ["/world", "/international", "/global"]):
            return "world"
        elif any(p in url_lower for p in ["/entertainment", "/celebrity", "/culture"]):
            return "entertainment"
        elif any(p in url_lower for p in ["/health", "/medical"]):
            return "health"
        elif any(p in url_lower for p in ["/opinion", "/editorial", "/commentary"]):
            return "opinion"
        elif any(p in url_lower for p in ["/breaking", "/latest", "/trending"]):
            return "breaking"
        elif any(p in url_lower for p in ["/article/", "/story/", "/news/", "/post/"]):
            return "article"
        elif any(p in url_lower for p in ["/category/", "/section/", "/topic/"]):
            return "category"
        elif any(p in url_lower for p in ["/archive", "/search"]):
            return "archive"
        else:
            return "homepage"

    def _classify_page_type_ml(
        self,
        structure: PageStructure,
        url: str,
    ) -> tuple[str, float]:
        """Classify page type using ML classifier."""
        if self.classifier is None:
            return self._classify_page_type_rules(url), 0.5

        try:
            return self.classifier.predict(structure)
        except Exception as e:
            self.logger.warning("ML classification failed", error=str(e))
            return self._classify_page_type_rules(url), 0.5

    async def fetch_page(self, url: str) -> tuple[str, int] | None:
        """Fetch a page with retry logic."""
        if not self.http_client:
            return None

        if self.config.respect_robots:
            try:
                robots_url = f"{url.split('/')[0]}//{self._extract_domain(url)}/robots.txt"
                robots_response = await self.http_client.get(robots_url)
                if robots_response.status_code == 200:
                    allowed = self.robots_checker.is_allowed(
                        url,
                        robots_response.text,
                        self.config.user_agent,
                    )
                    if not allowed:
                        self.logger.warning("URL blocked by robots.txt", url=url)
                        return None
            except Exception:
                pass

        for attempt in range(self.config.max_retries):
            try:
                response = await self.http_client.get(url)
                return response.text, response.status_code
            except Exception as e:
                self.logger.warning(
                    "Fetch failed",
                    url=url,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        return None

    async def check_url(
      self, 
      url: str, 
      page_type: str = None
    ) -> MLContentChange | None:
        """
        Check a URL for changes using ML-based detection.

        Returns:
            MLContentChange if changes detected, None otherwise.
        """
        self.logger.info("Checking URL (ML)", url=url)

        # Verbose: Start check
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"  CHECKING URL: {url}")
            print(f"{'='*70}")

        result = await self.fetch_page(url)
        if not result:
            self.logger.error("Failed to fetch page", url=url)
            return None

        html, status_code = result

        # Verbose: Fetch results
        if self.config.verbose:
            print(f"\n[1] PAGE FETCH")
            print(f"    Status: {status_code}")
            print(f"    HTML size: {len(html):,} bytes")

        if status_code != 200:
            self.logger.warning("Non-200 status", url=url, status=status_code)
            return None

        domain = self._extract_domain(url)
        if not page_type:
          page_type_rules = self._classify_page_type_rules(url)
        else:
          page_type_rules = page_type
        current_structure = self.structure_analyzer.analyze(html, url, page_type_rules)

        # Verbose: Structure analysis
        if self.config.verbose:
            print(f"\n[2] STRUCTURE ANALYSIS")
            print(f"    Domain: {domain}")
            print(f"    Page type (rules): {page_type_rules}")
            tag_counts = current_structure.tag_hierarchy.get("tag_counts", {}) if current_structure.tag_hierarchy else {}
            total_tags = sum(tag_counts.values()) if tag_counts else 0
            print(f"    Total tags: {total_tags}")
            if tag_counts:
                top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"    Top tags: {', '.join(f'{t}:{c}' for t, c in top_tags)}")
            print(f"    Content regions: {len(current_structure.content_regions) if current_structure.content_regions else 0}")
            if current_structure.content_regions:
                for i, region in enumerate(current_structure.content_regions[:3]):
                    print(f"      [{i+1}] {region.name}: {region.primary_selector}")
            landmarks = list(current_structure.semantic_landmarks.keys()) if current_structure.semantic_landmarks else []
            print(f"    Semantic landmarks: {landmarks}")
            print(f"    Navigation selectors: {len(current_structure.navigation_selectors) if current_structure.navigation_selectors else 0}")
            print(f"    CSS classes tracked: {len(current_structure.css_class_map) if current_structure.css_class_map else 0}")
            print(f"    Content hash: {current_structure.content_hash[:16]}...")

        # ML: Compute structure embedding
        current_embedding_obj = self.embedding_model.embed_structure(current_structure)
        current_embedding = current_embedding_obj.embedding

        # Verbose: Embedding
        if self.config.verbose:
            print(f"\n[3] ML EMBEDDING")
            print(f"    Model: {self.config.embedding_model}")
            print(f"    Embedding dimensions: {len(current_embedding)}")
            print(f"    Embedding norm: {sum(x*x for x in current_embedding)**0.5:.4f}")
            print(f"    First 5 values: [{', '.join(f'{v:.4f}' for v in current_embedding[:5])}]")

        # ML: Get previous embedding
        previous_embedding = self._structure_embeddings.get(url)

        # Verbose: Previous state
        if self.config.verbose:
            print(f"\n[4] COMPARISON WITH PREVIOUS")
            print(f"    Has previous embedding: {previous_embedding is not None}")

        # ML: Compute similarity using embeddings
        if previous_embedding:
            
            similarity = self.embedding_model.compute_similarity(
                previous_embedding, current_embedding
            )
            self.logger.debug(
                "Embedding similarity",
                url=url,
                similarity=f"{similarity:.4f}",
            )

            # Verbose: Similarity
            if self.config.verbose:
                print(f"    Embedding similarity: {similarity:.4f} ({similarity:.2%})")
                print(f"    Threshold for 'no change': > 0.98")
                if similarity > 0.98:
                    print(f"    Decision: NO SIGNIFICANT CHANGE (similarity > 0.98)")
                else:
                    print(f"    Decision: CHANGE DETECTED (similarity <= 0.98)")

            if similarity > 0.98:
                self.logger.debug("No significant structural changes (embedding)", url=url)
                if self.config.verbose:
                    print(f"\n{'='*70}")
                    print(f"  RESULT: No changes detected")
                    print(f"{'='*70}\n")
                return None
        else:
            similarity = 0.0

            if self.config.verbose:
                print(f"    No previous embedding - this is a NEW page")
                print(f"    Similarity set to: 0.0 (new baseline)")

        # ML: Classify page type using ML
        page_type_ml, page_type_confidence = self._classify_page_type_ml(current_structure, url)

        # Verbose: ML Classification
        if self.config.verbose:
            print(f"\n[5] ML PAGE TYPE CLASSIFICATION")
            print(f"    Classifier type: {self.config.classifier_type}")
            print(f"    Predicted type: {page_type_ml}")
            print(f"    Confidence: {page_type_confidence:.2%}")
            print(f"    Rules-based type: {page_type_rules}")
            if page_type_ml != page_type_rules:
                print(f"    ⚠ ML and rules disagree!")

        # Collect training data
        self._training_data.append((current_structure, page_type_rules))

        # Get stored structure and strategy
        assert self.structure_store is not None
        stored_structure = await self.structure_store.get_structure(
            domain, page_type_rules, "default"
        )
        stored_strategy = await self.structure_store.get_strategy(
            domain, page_type_rules, "default"
        )

        # Verbose: Storage lookup
        if self.config.verbose:
            print(f"\n[6] STORAGE LOOKUP")
            print(f"    Has stored structure: {stored_structure is not None}")
            print(f"    Has stored strategy: {stored_strategy is not None}")
            if stored_structure:
                print(f"    Stored version: {stored_structure.version}")
                print(f"    Stored hash: {stored_structure.content_hash[:16]}...")

        change_type = "new_content"
        llm_description: str | None = None
        strategy: ExtractionStrategy
        change_analysis: ChangeAnalysis | None = None

        if stored_structure and stored_strategy:
            # ML: Detect changes using ML change detector
            ml_analysis = self.ml_change_detector.detect_change(
                stored_structure, current_structure
            )

            # Verbose: ML Change Detection
            if self.config.verbose:
                print(f"\n[7] ML CHANGE DETECTION")
                print(f"    Breaking threshold: {self.config.breaking_threshold}")
                print(f"    ML similarity: {ml_analysis['similarity']:.4f} ({ml_analysis['similarity']:.2%})")
                print(f"    Is breaking: {ml_analysis['is_breaking']}")
                print(f"    Change magnitude: {1 - ml_analysis['similarity']:.2%}")

            # Also run rules-based change detection for detailed diff
            change_analysis = self.change_detector.detect_changes(
                stored_structure, current_structure
            )

            # Verbose: Rules-based analysis
            if self.config.verbose:
                print(f"\n[8] RULES-BASED CHANGE ANALYSIS")
                print(f"    Has changes: {change_analysis.has_changes}")
                print(f"    Classification: {change_analysis.classification.value}")
                print(f"    Similarity score: {change_analysis.similarity_score:.2%}")
                print(f"    Requires relearning: {change_analysis.requires_relearning}")
                if change_analysis.changes:
                    print(f"    Changes detected: {len(change_analysis.changes)}")
                    for i, ch in enumerate(change_analysis.changes[:3]):
                        print(f"      [{i+1}] {ch.change_type.value}: {ch.reason[:60]}...")

            # Set baseline for drift detection
            self.ml_change_detector.set_site_baseline(domain, stored_structure)

            if ml_analysis["is_breaking"]:
                change_type = "structure_changed"
                self.logger.info(
                    "ML detected breaking change",
                    url=url,
                    similarity=f"{ml_analysis['similarity']:.2%}",
                )

                if self.config.verbose:
                    print(f"\n[9] BREAKING CHANGE HANDLING")
                    print(f"    Change type: structure_changed")

                # Use LLM to describe changes
                if self.llm_generator:
                    if self.config.verbose:
                        print(f"    Generating LLM description...")
                        print(f"    LLM provider: {self.config.llm_provider}")
                    try:
                        llm_description = self.llm_generator.generate_for_change_detection(
                            stored_structure, current_structure
                        )
                        self.logger.info(
                            "LLM generated change description",
                            description=llm_description[:100] + "...",
                        )
                        if self.config.verbose:
                            print(f"    LLM description: {llm_description[:200]}...")
                    except Exception as e:
                        self.logger.warning("LLM description failed", error=str(e))
                        if self.config.verbose:
                            print(f"    LLM description FAILED: {e}")

                # Adapt strategy
                if self.config.verbose:
                    print(f"    Adapting extraction strategy...")
                adapted = self.strategy_learner.adapt(
                    stored_strategy, current_structure, html
                )
                strategy = adapted.strategy
                strategy.version = stored_strategy.version + 1

                if self.config.verbose:
                    print(f"    New strategy version: {strategy.version}")

                # Save new structure and strategy
                current_structure.version = stored_structure.version + 1
                await self.structure_store.save_structure(
                    current_structure, strategy, "default"
                )
                if self.config.verbose:
                    print(f"    Saved updated structure and strategy")
            else:
                strategy = stored_strategy
                change_type = "content_updated"
                if self.config.verbose:
                    print(f"\n[9] NO BREAKING CHANGE")
                    print(f"    Change type: content_updated")
                    print(f"    Using existing strategy (version {strategy.version})")
        else:
            self.logger.info(
                "Learning new page structure (ML)",
                url=url,
                page_type=page_type_ml,
            )
            change_type = "new_content"

            if self.config.verbose:
                print(f"\n[7] NEW PAGE - LEARNING STRUCTURE")
                print(f"    Change type: new_content")

            # Use LLM to describe structure
            if self.llm_generator:
                if self.config.verbose:
                    print(f"    Generating LLM description of structure...")
                try:
                    llm_description = self.llm_generator.generate(current_structure)
                    if self.config.verbose:
                        print(f"    LLM description: {llm_description[:200] if llm_description else 'None'}...")
                except Exception as e:
                    self.logger.warning("LLM description failed", error=str(e))
                    if self.config.verbose:
                        print(f"    LLM description FAILED: {e}")

            if self.config.verbose:
                print(f"    Inferring extraction strategy...")
            learned = self.strategy_learner.infer(html, current_structure)
            strategy = learned.strategy

            if self.config.verbose:
                print(f"    Strategy version: {strategy.version}")
                print(f"    Extraction rules: {len(strategy.rules)}")
                for field, rule in list(strategy.rules.items())[:3]:
                    print(f"      {field}: {rule.primary} (conf: {rule.confidence:.2f})")

            await self.structure_store.save_structure(
                current_structure, strategy, "default"
            )
            if self.config.verbose:
                print(f"    Saved new structure and strategy")

        # Extract content
        if self.config.verbose:
            print(f"\n[10] CONTENT EXTRACTION")
        extraction_result = self.content_extractor.extract(url, html, strategy)

        if self.config.verbose:
            print(f"    Success: {extraction_result.success}")
            print(f"    Confidence: {extraction_result.confidence:.2%}")
            if extraction_result.errors:
                print(f"    Errors: {extraction_result.errors}")
            if extraction_result.content:
                content = extraction_result.content
                print(f"    Title: {content.title[:60] if content.title else 'None'}...")
                print(f"    Content length: {len(content.content) if content.content else 0} chars")
                print(f"    Links found: {len(content.links) if content.links else 0}")
                print(f"    Images found: {len(content.images) if content.images else 0}")

        if not extraction_result.success:
            self.logger.warning(
                "Extraction failed",
                url=url,
                errors=extraction_result.errors,
            )

        # Save new embedding
        await self._save_embedding(url, current_embedding)

        if self.config.verbose:
            print(f"\n[11] SAVED STATE")
            print(f"    Embedding saved: Yes")
            print(f"    Training samples collected: {len(self._training_data)}")

        # Create ML change record
        change = MLContentChange(
            url=url,
            detected_at=datetime.now(timezone.utc),
            change_type=change_type,
            embedding_similarity=similarity if previous_embedding else 1.0,
            llm_description=llm_description,
            predicted_page_type=page_type_ml,
            page_type_confidence=page_type_confidence,
            extracted_content=extraction_result,
            previous_embedding=previous_embedding,
            current_embedding=current_embedding,
            change_analysis=change_analysis,
        )

        self._change_history.append(change)

        # Verbose: Final summary
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"  RESULT SUMMARY")
            print(f"{'='*70}")
            print(f"    URL: {url}")
            print(f"    Change type: {change_type}")
            print(f"    Embedding similarity: {change.embedding_similarity:.2%}")
            print(f"    Page type (ML): {page_type_ml} ({page_type_confidence:.0%})")
            print(f"    Extraction success: {extraction_result.success}")
            print(f"    LLM description: {'Yes' if llm_description else 'No'}")
            print(f"    Total changes in history: {len(self._change_history)}")
            print(f"{'='*70}\n")

        return change

    async def check_all_urls(
      self, 
      page_type: str | None
      ) -> list[MLContentChange]:
        """Check all configured URLs for changes."""
        changes = []

        for url in self.config.urls:
            change = await self.check_url(url, page_type)
            if change:
                changes.append(change)

                if self.on_change:
                    self.on_change(change)

            await asyncio.sleep(1)

        return changes

    async def run_monitoring_loop(self) -> None:
        """Run continuous monitoring loop."""
        self._running = True
        self.logger.info(
            "Starting ML monitoring loop",
            urls=len(self.config.urls),
            interval=self.config.check_interval,
        )

        while self._running:
            try:
                changes = await self.check_all_urls(None)

                if changes:
                    self.logger.info(
                        "ML changes detected",
                        count=len(changes),
                        types=[c.change_type for c in changes],
                    )
                    await self._save_changes_to_file(changes)
                else:
                    self.logger.debug("No ML changes detected")

            except Exception as e:
                self.logger.error("Error in ML monitoring loop", error=str(e))

            await asyncio.sleep(self.config.check_interval)

    async def _save_changes_to_file(self, changes: list[MLContentChange]) -> None:
        """Save detected changes to a JSON file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = Path(self.config.output_dir) / f"news_ml_changes_{timestamp}.json"

        data = []
        for change in changes:
            record = change.to_dict()

            if change.extracted_content and change.extracted_content.content:
                content = change.extracted_content.content
                record["extracted"] = {
                    "title": content.title,
                    "content": content.content,
                    "content_length": len(content.content) if content.content else 0,
                    "metadata": content.metadata,
                    "links": content.links[:20] if content.links else [],
                }

            data.append(record)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self.logger.info("Saved ML changes to file", filepath=str(filepath))

    def train_classifier(self, min_samples_per_class: int = 5) -> dict[str, Any] | None:
        """Train the page type classifier on collected data."""

        if len(self._training_data) < 10:
            self.logger.warning(
                "Insufficient training data}",
                samples=len(self._training_data),
            )
            return None

        from collections import Counter
        label_counts = Counter(label for _, label in self._training_data)

        if any(count < min_samples_per_class for count in label_counts.values()):
            self.logger.warning(
                "Insufficient samples per class",
                distribution=dict(label_counts),
            )
            return None

        structures = [s for s, _ in self._training_data]
        labels = [l for _, l in self._training_data]

        self.classifier = StructureClassifier(
            embedding_model=self.embedding_model,
            classifier_type=ClassifierType(self.config.classifier_type),
        )

        self.logger.info(
            "Training page type classifier",
            samples=len(structures),
            classifier=self.config.classifier_type,
        )

        metrics = self.classifier.train(structures, labels)

        self.logger.info(
            "Classifier trained",
            accuracy=f"{metrics['accuracy']:.2%}",
            std=f"{metrics['std']:.2%}",
        )

        return metrics

    def export_training_data(self, output_path: str | None = None) -> str:
        """Export collected training data to JSONL file."""
        if not output_path:
            output_path = str(
                Path(self.config.model_dir) / "training_data.jsonl"
            )

        structures = [s for s, _ in self._training_data]
        strategies = []

        for structure, label in self._training_data:
            from crawler.models import ExtractionStrategy
            strategy = ExtractionStrategy(
                domain=structure.domain,
                page_type=label,
            )
            strategies.append(strategy)

        export_training_data(structures, strategies, output_path)
        self.logger.info("Exported training data", path=output_path, samples=len(structures))

        return output_path

    def get_change_history(self) -> list[MLContentChange]:
        """Get the change history."""
        return self._change_history.copy()

    async def run_once(self) -> list[MLContentChange]:
        """Run a single check cycle (useful for testing)."""
        await self.start()
        try:
            return await self.check_all_urls(None)
        finally:
            await self.stop()


def print_ml_change(change: MLContentChange) -> None:
    """Print an ML change notification to console."""
    print(f"\n{'='*70}")
    print(f"  ML CHANGE DETECTED: {change.change_type.upper()}")
    print(f"{'='*70}")
    print(f"URL: {change.url}")
    print(f"Time: {change.detected_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nML Analysis:")
    print(f"  Embedding Similarity: {change.embedding_similarity:.1%}")
    print(f"  Predicted Page Type: {change.predicted_page_type}")
    print(f"  Type Confidence: {change.page_type_confidence:.1%}" if change.page_type_confidence else "  Type Confidence: N/A")

    if change.llm_description:
        print(f"\nLLM Description:")
        print(f"  {change.llm_description}")

    if change.extracted_content and change.extracted_content.content:
        content = change.extracted_content.content
        print(f"\nExtracted Content:")
        print(f"  Title: {content.title}")
        if content.content:
            preview = content.content[:200].replace('\n', ' ')
            print(f"  Preview: {preview}...")
        if content.links:
            print(f"  Links found: {len(content.links)}")

    print(f"{'='*70}\n")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ML-powered news website monitor"
    )
    parser.add_argument(
        "--url",
        type=str,
        action="append",
        help="URL to monitor (can specify multiple)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="CSV of urls to scrape",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./news_monitor_output_ml",
        help="Output directory for change logs",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (don't loop)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output showing detailed processing steps",
    )
    parser.add_argument(
        "--install-redis",
        action="store_true",
        help="Install Redis locally (Debian/Ubuntu) and exit",
    )

    # ML options
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model for embeddings",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="xgboost",
        choices=["logistic_regression", "xgboost", "lightgbm"],
        help="Classifier type for page type prediction",
    )
    parser.add_argument(
        "--breaking-threshold",
        type=float,
        default=0.85,
        help="Embedding similarity threshold for breaking changes (default: 0.85)",
    )

    # LLM options
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="ollama-cloud",
        choices=["openai", "anthropic", "ollama", "ollama-cloud"],
        help="LLM provider for descriptions (default: ollama-cloud)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model name (uses provider default if not specified)",
    )
    parser.add_argument(
        "--ollama-api-key",
        type=str,
        default=None,
        help="Ollama Cloud API key (or use OLLAMA_API_KEY env var)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=None,
        help="Custom Ollama URL (for local/self-hosted)",
    )

    # Training options
    parser.add_argument(
        "--train-classifier",
        action="store_true",
        help="Train classifier on collected data and exit",
    )
    parser.add_argument(
        "--training-data",
        type=str,
        default=None,
        help="JSON of URLs: page type labels for training",
    )
    parser.add_argument(
        "--export-data",
        action="store_true",
        help="Export training data to JSONL and exit",
    )

    args = parser.parse_args()

    if args.install_redis:
        success = install_redis_local()
        sys.exit(0 if success else 1)

    # Setup logging
    setup_logging(
        level="DEBUG" if args.verbose else "INFO",
        format_type="console",
    )

    # Suppress httpx INFO logs (redirect messages, etc.)
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Default URLs for news aggregators
    urls = []
    if args.url:
      urls = args.url
    elif args.csv:
      with open(args.csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)
        urls = [item for sublist in data for item in sublist]
    elif not args.train_classifier:
      raise Exception("Please input URLs to scrape or denote to train the classifier on data in redis")

    print(f"=====> urls has {len(urls)}")

    config = MLMonitorConfig(
        urls=urls,
        check_interval=args.interval,
        output_dir=args.output,
        embedding_model=args.embedding_model,
        classifier_type=args.classifier,
        breaking_threshold=args.breaking_threshold,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        ollama_api_key=args.ollama_api_key,
        ollama_base_url=args.ollama_url,
        verbose=args.verbose,
    )

    print("""
    ===============================================
       ML-POWERED NEWS MONITOR
       Adaptive Web Crawler with AI/ML
    ===============================================

    This ML-powered monitor uses:
    - Sentence-transformer embeddings for semantic comparison
    - XGBoost/LightGBM classifiers for page type prediction
    - Ollama Cloud LLM for rich change descriptions

    Features:
    1. Learn page structure embeddings on first visit
    2. Detect changes via embedding similarity (not hashes)
    3. Predict page types using trained ML models
    4. Generate LLM descriptions of detected changes
    5. Automatically adapt to site redesigns

    """)

    print(f"Monitoring {len(urls)} URL(s):")
    for url in urls:
        print(f"  - {url}")
    print(f"\nCheck interval: {args.interval} seconds")
    print(f"Output directory: {args.output}")
    print(f"\nML Settings:")
    print(f"  Embedding model: {config.embedding_model}")
    print(f"  Classifier: {config.classifier_type}")
    print(f"  Breaking threshold: {config.breaking_threshold}")
    print(f"  LLM provider: {config.llm_provider}")
    print(f"  Verbose mode: {'ENABLED' if config.verbose else 'disabled'}")
    print()

    # Check Redis
    print("Checking prerequisites...")
    try:
        redis_client = redis.from_url(config.redis_url, decode_responses=True)
        await redis_client.ping()
        await redis_client.aclose()
        print("  [OK] Redis is running")
    except Exception as e:
        print(f"  [FAIL] Redis connection failed: {e}")
        print("\nRedis is required. Choose an option:")
        print("\n  Option 1: Docker (recommended)")
        print("    docker run -d -p 6379:6379 redis:7-alpine")
        print("\n  Option 2: Local install (Debian/Ubuntu)")
        print("    python examples/news_monitor_ml.py --install-redis")
        print("\n  Option 3: Use existing Redis installation")
        print("    redis-server --daemonize yes")
        sys.exit(1)

    # Check LLM API key for cloud providers
    if config.llm_provider == "ollama-cloud":
        api_key = config.ollama_api_key or os.environ.get("OLLAMA_API_KEY", "")
        if api_key:
            print("  [OK] Ollama Cloud API key found")
        else:
            print("  [INFO] No OLLAMA_API_KEY set (optional for public Ollama Cloud)")
            print("         Default: https://ollama.com/v1 with gemma3:12b-cloud")
    elif config.llm_provider == "openai":
        if os.environ.get("OPENAI_API_KEY"):
            print("  [OK] OpenAI API key found")
        else:
            print("  [WARN] OPENAI_API_KEY not set - LLM descriptions disabled")
    elif config.llm_provider == "anthropic":
        if os.environ.get("ANTHROPIC_API_KEY"):
            print("  [OK] Anthropic API key found")
        else:
            print("  [WARN] ANTHROPIC_API_KEY not set - LLM descriptions disabled")
    elif config.llm_provider == "ollama":
        print("  [INFO] Using local Ollama - ensure ollama is running")

    # Create monitor
    monitor = MLNewsMonitor(
        config=config,
        on_change=print_ml_change,
    )

    try:
        await monitor.start()

        if args.train_classifier:
          if args.training_data:
            with open(args.training_data, "r") as f:
              training_data = json.load(f)
            monitor._training_data = await monitor.crawler_training_data(redis_client, training_data)
          elif not args.url and not args.csv: # If no URLs are provided, fetch from Redis
              print("No URLs provided for training. Fetching all existing page structures from Redis...")

              monitor._training_data = await monitor.crawler_training_data(redis_client)
          else:
            print("\nTraining classifier on collected data...")
            await monitor.check_all_urls(None)
          
          metrics = monitor.train_classifier()
          if metrics:
              print(f"\nTraining complete!")
              print(f"  Accuracy: {metrics['accuracy']:.2%}")
              print(f"  Samples: {metrics['num_samples']}")
              print(f"  Classes: {metrics['num_classes']}")
          else:
              print("\nInsufficient data for training")
          return

        if args.export_data:
            print("\nCollecting and exporting training data...")
            await monitor.check_all_urls(None)
            path = monitor.export_training_data()
            print(f"\nExported training data to: {path}")
            return

        if args.once:
            print("\nRunning single ML check...")
            changes = await monitor.check_all_urls(None)
            print(f"\nDetected {len(changes)} ML change(s)")
        else:
            print("\nStarting continuous ML monitoring (Ctrl+C to stop)...")
            await monitor.run_monitoring_loop()

    except KeyboardInterrupt:
        print("\n\nStopping ML monitor...")
    finally:
        await monitor.stop()

    print("ML Monitor stopped.")


if __name__ == "__main__":
    asyncio.run(main())



