import asyncio
import os
import json
import datetime
import logging
import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any, Tuple

from crawler.adaptive.strategy_learner import StrategyLearner
from crawler.models import ExtractionStrategy, SelectorRule
from crawler.storage.structure_store import StructureStore



class ManualStrategyManager:
    def __init__(self, redis_client, logger: Optional[logging.Logger] = None):
        self.redis_client = redis_client
        self.logger = logger or logging.getLogger(__name__)
        self.structure_store = StructureStore(
                redis_client=redis_client,
                logger=self.logger,
            )
        self.strategy_learner = StrategyLearner(logger=self.logger)
        self.logger.info("ManualStrategyManager initialized")


    async def update_strategy(
        self, 
        domain: str, 
        page_type: str, 
        new_title: str = None, 
        new_content: str = None, 
        new_date: str = None
      ) -> Optional[ExtractionStrategy]:

        strategy = await self.structure_store.get_strategy(domain, page_type)

        if not strategy:
            self.logger.debug(f"No version found for {domain}:{page_type}")
            # TODO: Prompt for URL to scrape?
            return None
        
        title_selector, content_selector, date_selector = None, None, None

        if new_title:
          title_selector = SelectorRule(primary=new_title,
                                        fallbacks=[strategy.title.primary],
                                        extraction_method='text', # TODO: This probably needs to be updated
                                        attribute_name=None,
                                        post_processors=[],
                                        confidence=1.0)
          strategy.confidence_scores["title"] = title_selector.confidence

        
        if new_content: 
            content_selector =SelectorRule(primary=new_content,
                            fallbacks=[strategy.content.primary],
                            extraction_method='text', 
                            attribute_name=None,
                            post_processors=[],
                            confidence=1.0)
            strategy.confidence_scores["content"] = content_selector.confidence

        if new_date:
          date_selector = SelectorRule(
                primary=new_date,
                fallbacks=[strategy.metadata['date'].primary],
                confidence=1.0,
                extraction_method="str" # TODO: This probably needs to be updated.... 
          )
          strategy.metadata['date'] = date_selector
          strategy.confidence_scores["date"] = date_selector.confidence
         
        return ExtractionStrategy(
            domain=strategy.domain,
            page_type=strategy.page_type,
            version=strategy.version + 1, # Enforce the target version
            title=title_selector if title_selector else strategy.title,
            content=content_selector if content_selector else strategy.content,
            metadata=strategy.metadata,
            images=strategy.images,
            links=strategy.links,
            wait_for_selectors=strategy.wait_for_selectors,
            iframe_extraction=strategy.iframe_extraction,
            required_fields=strategy.required_fields,
            min_content_length=strategy.min_content_length,
            learned_at=datetime.datetime.now(), # set time to now
            learning_source='manual', # Enforce 'manual'
            confidence_scores=strategy.confidence_scores,
            variant_id=strategy.variant_id
        )

      
    def save_new_strategy(
      self, 
      strategy: ExtractionStrategy, 
      validation_html: str = None
    ):
      if validation_html:
        
        overall, results = self.strategy_learner.validate_strategy(strategy, validation_html)
        print(f"=====> overall: {overall}")
        print(f"=====> results: {results}")
        
      


