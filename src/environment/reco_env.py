# src/environment/reco_env.py

import gym
from gym import spaces
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Assuming your features modules are accessible via relative imports
from src.features.article_encoder import ArticleEncoder, CategoryEncoder
from src.features.user_encoder import UserEncoder
from src.features.state_builder import StateBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MINDRecEnv(gym.Env):
    """
    Gym environment for simulating personalized news recommendation on MIND data.
    Each episode is a user session/behavior sequence.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self, 
        data_dir: str = 'data/processed', 
        split: str = 'train',
        embedding_dim: int = 384,
        max_candidate_size: int = 100, # K=100
        reward_type: str = 'click'
    ):
        super(MINDRecEnv, self).__init__()
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_candidate_size = max_candidate_size
        self.embedding_dim = embedding_dim
        self.reward_type = reward_type
        
        # --- 1. Load Data ---
        self._load_data()

        # --- 2. Define Config (MOVED UP) ---
        # Must be defined BEFORE _setup_encoders is called
        self.state_builder_config = {
            'embedding_dim': embedding_dim,
            'include_time_features': True,
            'include_session_features': True,
            'normalize_features': False 
        }

        # --- 3. Setup Encoders ---
        self._setup_encoders()
        
        # --- 4. Setup Spaces ---
        # A simple estimate of state_dim (will be updated after StateBuilder init)
        # 384 + 3 (time) + 4 (session) = 391
        state_dim = self.state_builder.state_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # The action space is the index of the article selected from the candidate set
        self.action_space = spaces.Discrete(self.max_candidate_size)
        
        # --- 5. Internal State ---
        self.current_user_sessions: List[Dict] = []
        self.session_idx = 0
        self.impression_idx = 0
        self.current_impression: Optional[Dict] = None
        
        logger.info(f"MINDRecEnv initialized. State Dim: {state_dim}, Action Space: {self.action_space.n}")

    def _load_data(self):
        """Load preprocessed session data."""
        session_path = self.data_dir / 'sessions' / f'{self.split}_sessions.pkl'
        
        if not session_path.exists():
             raise FileNotFoundError(f"Session file not found at {session_path}. Run preprocess_data.py first.")

        with open(session_path, 'rb') as f:
            self.sessions = pickle.load(f)
        
        logger.info(f"Loaded {len(self.sessions)} sessions for split '{self.split}'")
        self.session_pool = list(range(len(self.sessions)))
        self.pool_ptr = 0
        
    def _setup_encoders(self):
        """Initialize ArticleEncoder and StateBuilder from saved config/cache."""
        # Initialize ArticleEncoder to access the cache
        self.article_encoder = ArticleEncoder(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            embedding_dim=self.embedding_dim,
            cache_dir=self.data_dir / 'embeddings'
        )
        # Load the saved article embeddings cache
        self.article_encoder._load_cache() 

        # Initialize UserEncoder (using 'mean' for baseline)
        self.user_encoder = UserEncoder(
            embedding_dim=self.embedding_dim,
            aggregation_method='mean',
            use_temporal_decay=True
        )

        # Initialize StateBuilder
        self.state_builder = StateBuilder(
            article_encoder=self.article_encoder,
            user_encoder=self.user_encoder,
            **self.state_builder_config
        )
        
        # All available article IDs (from cache) for the full action space context
        self.all_article_ids = list(self.article_encoder.embedding_cache.keys())
    
    def _get_reward(self, click_label: int) -> float:
        """Calculate reward based on the interaction."""
        if self.reward_type == 'click':
            # +1 for click, 0 otherwise
            return float(click_label) 

    def reset(self) -> np.ndarray:
        """Reset the environment for a new episode (user session)."""
        
        # Randomly select a new session from the pool
        if self.pool_ptr >= len(self.session_pool):
            np.random.shuffle(self.session_pool)
            self.pool_ptr = 0
            
        session_idx = self.session_pool[self.pool_ptr]
        self.pool_ptr += 1
        
        self.current_session = self.sessions[session_idx]
        self.impression_idx = 0
        
        # Get the first impression for the new session
        self.current_impression = self.current_session['impressions'][self.impression_idx]
        
        # Candidate set for this step (truncated to max_candidate_size)
        self.current_candidates = self.current_impression['candidates'][:self.max_candidate_size]
        self.current_labels = self.current_impression['labels'][:self.max_candidate_size]
        
        # Get the initial state
        if 'state' in self.current_impression:
            # Use pre-computed state
            state = self.current_impression['state']
        else:
            # Compute state on the fly
            ts = self.current_impression['timestamp']
            # Handle string timestamps if necessary
            if isinstance(ts, str):
                ts = pd.to_datetime(ts)
            elif not isinstance(ts, datetime) and not isinstance(ts, pd.Timestamp):
                 ts = pd.to_datetime(ts)

            state = self.state_builder.build_state(
                history_news_ids=self.current_impression['history'],
                timestamp=ts,
                session_length=0, 
                session_clicks=0, 
                session_impressions=0
            )
            
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment by recommending an article.
        """
        if self.current_impression is None:
            raise RuntimeError("Environment must be reset before calling step()")
            
        # 1. Determine interaction result (reward)
        # Safety check for action range
        if action >= len(self.current_candidates):
            action = 0 # Fallback
            
        recommended_news_id = self.current_candidates[action]
        click_label = self.current_labels[action]
        reward = self._get_reward(click_label)
        
        # 2. Advance to the next impression in the session
        self.impression_idx += 1
        done = self.impression_idx >= self.current_session['num_impressions']
        
        info = {
            'recommended_id': recommended_news_id,
            'click_label': click_label,
            'session_clicks': 0, # Placeholder
            'is_cold_start': len(self.current_impression['history']) == 0
        }

        next_state = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        if not done:
            # Load the next impression
            next_impression = self.current_session['impressions'][self.impression_idx]
            self.current_impression = next_impression
            
            # Update candidate set
            self.current_candidates = next_impression['candidates'][:self.max_candidate_size]
            self.current_labels = next_impression['labels'][:self.max_candidate_size]

            # 3. Compute next state (s_t+1)
            if 'state' in next_impression:
                next_state = next_impression['state']
            else:
                # Calculate running session stats
                # Using a simplified counter for now
                ts = next_impression['timestamp']
                if not isinstance(ts, pd.Timestamp) and not isinstance(ts, datetime):
                    ts = pd.to_datetime(ts)

                next_state = self.state_builder.build_state(
                    history_news_ids=next_impression['history'],
                    timestamp=ts,
                    session_length=self.impression_idx,
                    session_clicks=self.impression_idx, # Approximate
                    session_impressions=self.impression_idx * self.max_candidate_size
                )

        return next_state, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_candidate_embeddings(self) -> np.ndarray:
        """Get the embeddings for the current step's candidate articles."""
        if not self.current_candidates:
            return np.zeros((0, self.embedding_dim))
        
        return self.article_encoder.get_embeddings(self.current_candidates)
    
    def get_user_state_embedding(self, state: np.ndarray) -> np.ndarray:
        """Extract the core user embedding from the state vector."""
        return state[:self.embedding_dim]