"""
MIND Dataset Loader
Loads and parses the Microsoft News Dataset (MIND) for RL-based recommendation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Represents a single news article"""
    news_id: str
    category: str
    subcategory: str
    title: str
    abstract: Optional[str]
    url: str
    title_entities: Optional[str]
    abstract_entities: Optional[str]


@dataclass
class UserBehavior:
    """Represents a single user impression/behavior"""
    impression_id: str
    user_id: str
    timestamp: pd.Timestamp
    history: List[str]  # List of previously clicked news IDs
    impressions: List[Tuple[str, int]]  # List of (news_id, label) where label=1 is click


class MINDDataLoader:
    """
    Loader for MIND dataset with utilities for RL-based recommendation.
    
    The MIND dataset consists of:
    - news.tsv: News articles with metadata
    - behaviors.tsv: User impression logs with click history
    - entity_embedding.vec: Entity embeddings (optional)
    - relation_embedding.vec: Relation embeddings (optional)
    """
    
    def __init__(self, data_dir: str, dataset_type: str = 'train'):
        """
        Initialize MIND data loader.
        
        Args:
            data_dir: Path to MIND data directory
            dataset_type: 'train' or 'dev'
        """
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        
        self.news_df = None
        self.behaviors_df = None
        self.news_dict = {} # news_id -> NewsArticle
        
        logger.info(f"Initialized MINDDataLoader for {dataset_type} data at {data_dir}")
    
    def load_news(self) -> pd.DataFrame:
        """
        Load news articles.
        
        Returns:
            DataFrame with news articles
        """
        news_path = self.data_dir / 'news.tsv'
        
        logger.info(f"Loading news from {news_path}")
        
        news_columns = [
            'news_id', 'category', 'subcategory', 'title', 
            'abstract', 'url', 'title_entities', 'abstract_entities'
        ]
        
        self.news_df = pd.read_csv(
            news_path,
            sep='\t',
            names=news_columns,
            encoding='utf-8'
        )
        
        # Create dictionary for fast lookup
        for _, row in self.news_df.iterrows():
            article = NewsArticle(
                news_id=row['news_id'],
                category=row['category'],
                subcategory=row['subcategory'],
                title=row['title'],
                abstract=row['abstract'] if pd.notna(row['abstract']) else None,
                url=row['url'],
                title_entities=row['title_entities'] if pd.notna(row['title_entities']) else None,
                abstract_entities=row['abstract_entities'] if pd.notna(row['abstract_entities']) else None
            )
            self.news_dict[article.news_id] = article
        
        logger.info(f"Loaded {len(self.news_df)} news articles")
        return self.news_df
    
    def load_behaviors(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load user behaviors (impressions).
        
        Args:
            nrows: Number of rows to load (None for all)
            
        Returns:
            DataFrame with user behaviors
        """
        behaviors_path = self.data_dir / 'behaviors.tsv'
        
        logger.info(f"Loading behaviors from {behaviors_path}")
        
        behaviors_columns = [
            'impression_id', 'user_id', 'time', 'history', 'impressions'
        ]
        
        self.behaviors_df = pd.read_csv(
            behaviors_path,
            sep='\t',
            names=behaviors_columns,
            nrows=nrows
        )
        
        # Parse timestamp
        self.behaviors_df['timestamp'] = pd.to_datetime(
            self.behaviors_df['time'],
            format='%m/%d/%Y %I:%M:%S %p'
        )
        
        logger.info(f"Loaded {len(self.behaviors_df)} behavior records")
        return self.behaviors_df
    
    def parse_history(self, history_str: str) -> List[str]:
        """
        Parse user history string into list of news IDs.
        
        Args:
            history_str: Space-separated news IDs
            
        Returns:
            List of news IDs
        """
        if pd.isna(history_str) or history_str == '':
            return []
        return history_str.split()
    
    def parse_impressions(self, impressions_str: str) -> List[Tuple[str, int]]:
        """
        Parse impressions string into list of (news_id, label) tuples.
        
        Args:
            impressions_str: Space-separated News-Label pairs (e.g., "N123-1 N456-0")
            
        Returns:
            List of (news_id, label) where label=1 is clicked, 0 is not clicked
        """
        if pd.isna(impressions_str):
            return []
        
        impressions = []
        for imp in impressions_str.split():
            news_id, label = imp.rsplit('-', 1)
            impressions.append((news_id, int(label)))
        
        return impressions
    
    def get_user_behaviors(self, user_id: str) -> List[UserBehavior]:
        """
        Get all behaviors for a specific user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of UserBehavior objects
        """
        if self.behaviors_df is None:
            raise ValueError("Behaviors not loaded. Call load_behaviors() first.")
        
        user_behaviors = []
        user_df = self.behaviors_df[self.behaviors_df['user_id'] == user_id]
        
        for _, row in user_df.iterrows():
            behavior = UserBehavior(
                impression_id=row['impression_id'],
                user_id=row['user_id'],
                timestamp=row['timestamp'],
                history=self.parse_history(row['history']),
                impressions=self.parse_impressions(row['impressions'])
            )
            user_behaviors.append(behavior)
        
        return user_behaviors
    
    def get_article(self, news_id: str) -> Optional[NewsArticle]:
        """
        Get a news article by ID.
        
        Args:
            news_id: News ID
            
        Returns:
            NewsArticle object or None if not found
        """
        return self.news_dict.get(news_id)
    
    def get_clicked_articles(self, behavior: UserBehavior) -> List[str]:
        """
        Extract clicked article IDs from a behavior.
        
        Args:
            behavior: UserBehavior object
            
        Returns:
            List of clicked news IDs
        """
        return [news_id for news_id, label in behavior.impressions if label == 1]
    
    def get_unclicked_articles(self, behavior: UserBehavior) -> List[str]:
        """
        Extract unclicked article IDs from a behavior.
        
        Args:
            behavior: UserBehavior object
            
        Returns:
            List of unclicked news IDs
        """
        return [news_id for news_id, label in behavior.impressions if label == 0]
    
    def get_category_distribution(self) -> Dict[str, int]:
        """
        Get distribution of news categories.
        
        Returns:
            Dictionary mapping category to count
        """
        if self.news_df is None:
            raise ValueError("News not loaded. Call load_news() first.")
        
        return self.news_df['category'].value_counts().to_dict()
    
    def get_user_statistics(self) -> Dict:
        """
        Get statistics about users in the dataset.
        
        Returns:
            Dictionary with user statistics
        """
        if self.behaviors_df is None:
            raise ValueError("Behaviors not loaded. Call load_behaviors() first.")
        
        # Parse all behaviors
        self.behaviors_df['history_list'] = self.behaviors_df['history'].apply(
            self.parse_history
        )
        self.behaviors_df['impressions_list'] = self.behaviors_df['impressions'].apply(
            self.parse_impressions
        )
        
        # Calculate statistics
        self.behaviors_df['history_length'] = self.behaviors_df['history_list'].apply(len)
        self.behaviors_df['num_impressions'] = self.behaviors_df['impressions_list'].apply(len)
        self.behaviors_df['num_clicks'] = self.behaviors_df['impressions_list'].apply(
            lambda x: sum(1 for _, label in x if label == 1)
        )
        self.behaviors_df['ctr'] = self.behaviors_df['num_clicks'] / self.behaviors_df['num_impressions']
        
        stats = {
            'num_users': self.behaviors_df['user_id'].nunique(),
            'num_impressions': len(self.behaviors_df),
            'avg_history_length': self.behaviors_df['history_length'].mean(),
            'avg_impressions_per_behavior': self.behaviors_df['num_impressions'].mean(),
            'avg_clicks_per_behavior': self.behaviors_df['num_clicks'].mean(),
            'overall_ctr': self.behaviors_df['ctr'].mean(),
            'cold_start_ratio': (self.behaviors_df['history_length'] == 0).mean()
        }
        
        return stats
    
    def create_session_data(self, min_history_length: int = 1) -> List[Dict]:
        """
        Create session-based data for RL training.
        Each session represents a user's impression sequence.
        
        Args:
            min_history_length: Minimum history length to include
            
        Returns:
            List of session dictionaries
        """
        if self.behaviors_df is None or self.news_df is None:
            raise ValueError("Data not loaded. Call load_news() and load_behaviors() first.")
        
        sessions = []
        
        for user_id in self.behaviors_df['user_id'].unique():
            user_behaviors = self.get_user_behaviors(user_id)
            
            # Sort by timestamp
            user_behaviors.sort(key=lambda x: x.timestamp)
            
            # Filter by minimum history
            if len(user_behaviors[0].history) < min_history_length:
                continue
            
            session = {
                'user_id': user_id,
                'behaviors': user_behaviors,
                'num_impressions': len(user_behaviors),
                'total_clicks': sum(len(self.get_clicked_articles(b)) for b in user_behaviors)
            }
            
            sessions.append(session)
        
        logger.info(f"Created {len(sessions)} sessions with min_history={min_history_length}")
        return sessions
    
    def load_all(self, nrows: Optional[int] = None):
        """
        Load all data (news and behaviors).
        
        Args:
            nrows: Number of behavior rows to load (None for all)
        """
        self.load_news()
        self.load_behaviors(nrows=nrows)
        logger.info("All data loaded successfully")


if __name__ == "__main__":
    # Initialize loader
    loader = MINDDataLoader(
        data_dir='../../data/raw/train',
        dataset_type='train'
    )
    
     # Load first 10k behaviors for testing
    loader.load_all(nrows=10000) 
    
    # Get statistics
    news_stats = {
        'num_articles': len(loader.news_df),
        'num_categories': loader.news_df['category'].nunique(),
        'categories': loader.get_category_distribution()
    }
    
    print("\n=== NEWS STATISTICS ===")
    print(f"Total articles: {news_stats['num_articles']}")
    print(f"Unique categories: {news_stats['num_categories']}")
    print("\nTop 5 categories:")
    for cat, count in list(news_stats['categories'].items())[:5]:
        print(f"  {cat}: {count}")
    
    user_stats = loader.get_user_statistics()
    print("\n=== USER STATISTICS ===")
    for key, value in user_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Create sessions
    sessions = loader.create_session_data(min_history_length=1)
    print(f"\n=== SESSION DATA ===")
    print(f"Total sessions: {len(sessions)}")
    print(f"\nFirst session example:")
    print(f"  User: {sessions[0]['user_id']}")
    print(f"  Impressions: {sessions[0]['num_impressions']}")
    print(f"  Total clicks: {sessions[0]['total_clicks']}")