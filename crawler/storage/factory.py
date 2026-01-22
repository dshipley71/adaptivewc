"""
Factory for creating structure stores based on configuration.

Enables switching between rule-based and LLM-powered structure stores
at runtime based on configuration settings.
"""

from typing import Protocol, Union

import redis.asyncio as redis

from crawler.config import (
    StructureStoreConfig,
    StructureStoreType,
    LLMProviderType,
)
from crawler.storage.structure_store import StructureStore
from crawler.storage.structure_llm_store import (
    LLMStructureStore,
    LLMProvider,
    AnthropicProvider,
    OpenAIProvider,
    OllamaProvider,
)
from crawler.utils.logging import CrawlerLogger


# Type alias for either store type
AnyStructureStore = Union[StructureStore, LLMStructureStore]


class StructureStoreProtocol(Protocol):
    """Protocol defining the common interface for structure stores."""
    
    async def save(self, structure, url: str = "") -> int:
        """Save a structure and return version number."""
        ...
    
    async def get_latest(self, domain: str, page_type: str):
        """Get the latest structure for a domain/page_type."""
        ...
    
    async def get_signature(self, domain: str, page_type: str):
        """Get the signature for quick comparison."""
        ...
    
    async def get_description(self, domain: str, page_type: str) -> str | None:
        """Get the semantic description."""
        ...
    
    async def has_changed(self, structure, url: str = "") -> bool:
        """Check if structure has changed."""
        ...
    
    async def list_domains(self) -> list[tuple[str, str]]:
        """List all tracked domain/page_type pairs."""
        ...
    
    async def get_stats(self) -> dict:
        """Get store statistics."""
        ...


def create_llm_provider(config: StructureStoreConfig) -> LLMProvider:
    """
    Create an LLM provider based on configuration.
    
    Args:
        config: Structure store configuration.
        
    Returns:
        Configured LLM provider.
        
    Raises:
        ValueError: If provider type is unknown.
    """
    api_key = config.llm_api_key or None  # Convert empty string to None
    
    if config.llm_provider == LLMProviderType.ANTHROPIC:
        model = config.llm_model or "claude-sonnet-4-20250514"
        return AnthropicProvider(api_key=api_key, model=model)
    
    elif config.llm_provider == LLMProviderType.OPENAI:
        model = config.llm_model or "gpt-4o-mini"
        return OpenAIProvider(api_key=api_key, model=model)
    
    elif config.llm_provider == LLMProviderType.OLLAMA:
        model = config.llm_model or "llama3.2"
        return OllamaProvider(model=model, base_url=config.ollama_base_url)
    
    else:
        raise ValueError(f"Unknown LLM provider: {config.llm_provider}")


def create_structure_store(
    redis_client: redis.Redis,
    config: StructureStoreConfig,
    logger: CrawlerLogger | None = None,
) -> AnyStructureStore:
    """
    Create a structure store based on configuration.
    
    Args:
        redis_client: Redis async client.
        config: Structure store configuration.
        logger: Logger instance.
        
    Returns:
        Configured structure store (either StructureStore or LLMStructureStore).
        
    Raises:
        ValueError: If store type is unknown.
        
    Example:
        ```python
        from crawler.config import StructureStoreConfig, StructureStoreType
        from crawler.storage.factory import create_structure_store
        
        # Use rule-based descriptions
        config = StructureStoreConfig(
            store_type=StructureStoreType.BASIC,
            enable_embeddings=True,
        )
        store = create_structure_store(redis_client, config)
        
        # Use LLM descriptions with Anthropic
        config = StructureStoreConfig(
            store_type=StructureStoreType.LLM,
            llm_provider=LLMProviderType.ANTHROPIC,
            enable_embeddings=True,
        )
        store = create_structure_store(redis_client, config)
        
        # Use LLM descriptions with local Ollama
        config = StructureStoreConfig(
            store_type=StructureStoreType.LLM,
            llm_provider=LLMProviderType.OLLAMA,
            llm_model="llama3.2",
            ollama_base_url="http://localhost:11434",
        )
        store = create_structure_store(redis_client, config)
        ```
    """
    logger = logger or CrawlerLogger("structure_store_factory")
    
    if config.store_type == StructureStoreType.BASIC:
        logger.info(
            "Creating basic structure store",
            embeddings_enabled=config.enable_embeddings,
            embedding_model=config.embedding_model,
        )
        
        return StructureStore(
            redis_client=redis_client,
            ttl_seconds=config.ttl_seconds,
            max_versions=config.max_versions,
            enable_embeddings=config.enable_embeddings,
            embedding_model=config.embedding_model,
            logger=logger,
        )
    
    elif config.store_type == StructureStoreType.LLM:
        # Create LLM provider
        llm_provider = create_llm_provider(config)
        
        logger.info(
            "Creating LLM structure store",
            llm_provider=config.llm_provider.value,
            llm_model=llm_provider.get_model_name(),
            embeddings_enabled=config.enable_embeddings,
            embedding_model=config.embedding_model,
        )
        
        return LLMStructureStore(
            redis_client=redis_client,
            llm_provider=llm_provider,
            ttl_seconds=config.ttl_seconds,
            max_versions=config.max_versions,
            enable_embeddings=config.enable_embeddings,
            embedding_model=config.embedding_model,
            logger=logger,
        )
    
    else:
        raise ValueError(f"Unknown structure store type: {config.store_type}")


def get_store_info(store: AnyStructureStore) -> dict:
    """
    Get information about a structure store instance.
    
    Args:
        store: Structure store instance.
        
    Returns:
        Dictionary with store type and configuration info.
    """
    if isinstance(store, LLMStructureStore):
        return {
            "type": "llm",
            "llm_model": store._llm_model,
            "embeddings_enabled": store.enable_embeddings,
            "embedding_model": store._embedding_model if store.enable_embeddings else None,
            "ttl_seconds": store.ttl,
            "max_versions": store.max_versions,
        }
    elif isinstance(store, StructureStore):
        return {
            "type": "basic",
            "embeddings_enabled": store.enable_embeddings,
            "embedding_model": store._embedding_model if store.enable_embeddings else None,
            "ttl_seconds": store.ttl,
            "max_versions": store.max_versions,
        }
    else:
        return {"type": "unknown"}
