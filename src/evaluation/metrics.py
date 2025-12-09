"""
Evaluation Metrics for News Recommendation Systems
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Set
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

class RecommenderMetrics:
    def __init__(self, news_df: pd.DataFrame, article_encoder):
        """
        Args:
            news_df: DataFrame containing 'news_id', 'category', 'published_date' (if avail)
            article_encoder: ArticleEncoder instance
        """
        self.news_df = news_df.set_index('news_id') if 'news_id' in news_df.columns else news_df
        self.article_encoder = article_encoder
        
        # Pre-cache all embeddings for the catalog for fast lookup
        all_ids = self.news_df.index.tolist()
        self.all_embeddings = self.article_encoder.get_embeddings(all_ids)
        self.id_to_idx = {news_id: i for i, news_id in enumerate(all_ids)}
        
        self.timestamps = {}
        for news_id, row in self.news_df.iterrows():
            try:
                if 'published_date' in row:
                    ts = pd.to_datetime(row['published_date']).timestamp()
                    self.timestamps[news_id] = ts
                else:
                    self.timestamps[news_id] = 0.0
            except:
                self.timestamps[news_id] = 0.0

    #  ACCURACY 
    def calculate_ctr(self, recommendations: List[str], clicks: List[str]) -> float:
        if not recommendations: return 0.0
        click_set = set(clicks)
        hits = sum(1 for rec in recommendations if rec in click_set)
        return hits / len(recommendations)

    def calculate_ndcg(self, recommendations: List[str], clicks: List[str], k=10) -> float:
        """Normalized Discounted Cumulative Gain @ K"""
        rec_k = recommendations[:k]
        click_set = set(clicks)
        
        # Relevances: 1 if clicked, 0 otherwise
        relevances = [1 if rec in click_set else 0 for rec in rec_k]
        
        if sum(relevances) == 0: return 0.0
        
        # DCG
        dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevances)])
        
        # IDCG (Ideal case: all 1s are at the top)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevances)])
        
        return dcg / idcg if idcg > 0 else 0.0

    #  DIVERSITY 
    def calculate_ild(self, recommendations: List[str]) -> float:
        """Intra-List Diversity (1 - Cosine Similarity)"""
        valid_recs = [r for r in recommendations if r in self.id_to_idx]
        if len(valid_recs) < 2: return 0.0
        
        indices = [self.id_to_idx[r] for r in valid_recs]
        embeddings = self.all_embeddings[indices]
        
        # Cosine Similarity Matrix
        sim_matrix = cosine_similarity(embeddings)
        
        # Get upper triangle (excluding diagonal) to get unique pairs
        n = len(embeddings)
        tri_u = np.triu_indices(n, k=1)
        
        avg_sim = np.mean(sim_matrix[tri_u])
        return 1.0 - avg_sim

    def calculate_category_diversity(self, recommendations: List[str]) -> float:
        """Shannon Entropy of Categories"""
        cats = []
        for r in recommendations:
            if r in self.news_df.index:
                cats.append(self.news_df.loc[r]['category'])
        
        if not cats: return 0.0
        
        counts = Counter(cats)
        total = len(cats)
        probs = np.array([c/total for c in counts.values()])
        
        # Entropy
        return -np.sum(probs * np.log2(probs + 1e-9))

    #  NOVELTY 
    def calculate_novelty(self, recommendations: List[str]) -> float:
        """Mean Inverse Popularity (or Recency)"""
        # Since we don't have live popularity counts in this class, 
        # we will use Recency (Hours since publication)
        
        valid_recs = [r for r in recommendations if r in self.timestamps]
        if not valid_recs: return 0.0
        
        now = datetime.now().timestamp()
        ages = []
        for r in valid_recs:
            pub_ts = self.timestamps[r]
            if pub_ts > 0:
                age_hours = (now - pub_ts) / 3600
                ages.append(age_hours)
                
        # Lower age is "more novel" in news, but usually Novelty = -log(prob). Here we return Mean Age (Lower is fresher/more novel)
        if not ages: return 0.0
        return np.mean(ages)

    def calculate_catalog_coverage(self, all_recommendations_flat: List[str]) -> float:
        unique_recs = set(all_recommendations_flat)
        return len(unique_recs) / len(self.news_df)