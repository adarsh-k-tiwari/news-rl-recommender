"""
Category Encoder Module
"""
import numpy as np
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CategoryEncoder:
    def __init__(self, news_df, embedding_dim: int = None):
        self.news_df = news_df
        
        # Build category vocabulary
        self.categories = sorted(news_df['category'].unique().tolist())
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.num_categories = len(self.categories)
        self.embedding_dim = self.num_categories  # One-hot: each category is one dimension
        
        logger.info(f"CategoryEncoder initialized with {self.num_categories} categories (one-hot): {self.categories}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def get_embedding(self, news_id: str) -> np.ndarray:
        """Get one-hot embedding for a single article based on its category."""
        if news_id not in self.news_df.index:
            # Return zero vector for unknown articles
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        category = self.news_df.loc[news_id, 'category']
        cat_idx = self.category_to_idx.get(category, 0)
        
        # Create one-hot vector
        one_hot = np.zeros(self.embedding_dim, dtype=np.float32)
        one_hot[cat_idx] = 1.0
        return one_hot
    
    def get_embeddings(self, news_ids: List[str]) -> np.ndarray:
        """Get embeddings for multiple articles."""
        embeddings = []
        for news_id in news_ids:
            embeddings.append(self.get_embedding(news_id))
        return np.array(embeddings, dtype=np.float32)
    
    def get_category_distribution(self, news_ids: List[str]) -> Dict[str, float]:
        """Get category distribution from a list of article IDs."""
        if not news_ids:
            return {}
        
        cat_counts = {}
        for news_id in news_ids:
            if news_id in self.news_df.index:
                cat = self.news_df.loc[news_id, 'category']
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        
        # Normalize to probabilities
        total = sum(cat_counts.values())
        return {cat: count/total for cat, count in cat_counts.items()}
