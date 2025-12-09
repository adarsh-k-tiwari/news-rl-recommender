"""
Baseline agents for news recommendation.
Includes:
1. RandomAgent: selects articles randomly.
2. PopularityAgent: selects articles based on global popularity.
"""
import numpy as np
import pickle
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RandomAgent:
    """
    Baseline 1: Randomly selects an article from the candidate list.
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, state, env=None):
        """Returns a random action index."""
        return self.action_space.sample()


class PopularityAgent:
    """
    Baseline 2: Recommends the most popular article from the candidates.
    Popularity is calculated based on click counts in the training set.
    """
    def __init__(self, train_session_path: str):
        self.popularity_map = self._compute_popularity(train_session_path)
        logger.info(f"PopularityAgent initialized. Tracked {len(self.popularity_map)} articles.")

    def _compute_popularity(self, session_path: str) -> Dict[str, int]:
        """Counts clicks for every article in the training sessions."""
        logger.info("Computing global article popularity...")
        with open(session_path, 'rb') as f:
            sessions = pickle.load(f)
        
        counts = {}
        for session in sessions:
            for impression in session['impressions']:
                # The labels tell us which candidates were clicked
                # candidates list corresponds to labels list
                candidates = impression['candidates']
                labels = impression['labels']
                
                for news_id, label in zip(candidates, labels):
                    if label == 1: # User clicked
                        counts[news_id] = counts.get(news_id, 0) + 1
        return counts

    def predict(self, state, env):
        """
        Selects the candidate with the highest global click count.
        Requires access to env.current_candidates.
        """
        candidates = env.current_candidates
        
        best_score = -1
        best_action = 0
        
        for idx, news_id in enumerate(candidates):
            score = self.popularity_map.get(news_id, 0)
            if score > best_score:
                best_score = score
                best_action = idx
                
        return best_action