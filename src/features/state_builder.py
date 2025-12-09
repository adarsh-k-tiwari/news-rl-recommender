"""
State Builder for RL Environment
Constructs complete MDP states from user history and context for the RL agent.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StateBuilder:
    """
    Builds complete state representations for the RL agent.
    State = [user_embedding, context_features, candidate_features (optional)]
    """
    
    def __init__(
        self,
        article_encoder,
        user_encoder,
        embedding_dim: int = 384,
        include_time_features: bool = True,
        include_session_features: bool = True,
        normalize_features: bool = True
    ):
        """
        Initialize state builder.
        
        Args:
            article_encoder: ArticleEncoder instance
            user_encoder: UserEncoder instance
            embedding_dim: Dimension of embeddings
            include_time_features: Include temporal features
            include_session_features: Include session-level features
            normalize_features: Normalize continuous features
        """
        self.article_encoder = article_encoder
        self.user_encoder = user_encoder
        self.embedding_dim = embedding_dim
        self.include_time_features = include_time_features
        self.include_session_features = include_session_features
        self.normalize_features = normalize_features
        
        # Calculate state dimension
        self.state_dim = self._calculate_state_dim()
        
        logger.info(f"StateBuilder initialized with state_dim={self.state_dim}")
    
    def _calculate_state_dim(self) -> int:
        """Calculate total state dimension."""
        dim = self.embedding_dim  # Base user embedding
        
        if self.include_time_features:
            dim += 3  # hour_sin, hour_cos, day_of_week
        
        if self.include_session_features:
            dim += 4  # session_length, num_clicks, avg_ctr, history_length_norm
        
        return dim
    
    def build_state(
        self,
        history_news_ids: List[str],
        timestamp: Optional[datetime] = None,
        session_length: int = 0,
        session_clicks: int = 0,
        session_impressions: int = 0
    ) -> np.ndarray:
        """
        Build complete state vector for current user context.
        
        Args:
            history_news_ids: List of news IDs in user's click history
            timestamp: Current timestamp
            session_length: Number of steps in current session
            session_clicks: Number of clicks in current session
            session_impressions: Total impressions in current session
            
        Returns:
            State vector [state_dim]
        """
        state_components = []
        
        # 1. User embedding from history
        if len(history_news_ids) > 0:
            history_embeddings = self.article_encoder.get_embeddings(history_news_ids)
            user_embedding = self.user_encoder.encode_history(history_embeddings)
        else:
            # Cold start user - zero embedding
            user_embedding = np.zeros(self.embedding_dim)
        
        state_components.append(user_embedding)
        
        # 2. Time features
        if self.include_time_features and timestamp:
            time_features = self._extract_time_features(timestamp)
            state_components.append(time_features)
        elif self.include_time_features:
            # Default time features if not provided
            state_components.append(np.zeros(3))
        
        # 3. Session features
        if self.include_session_features:
            session_features = self._extract_session_features(
                session_length, session_clicks, session_impressions, len(history_news_ids)
            )
            state_components.append(session_features)
        
        # Concatenate all components
        state = np.concatenate(state_components)
        
        return state.astype(np.float32)
    
    def _extract_time_features(self, timestamp: datetime) -> np.ndarray:
        """
        Extract temporal features from timestamp.
        
        Args:
            timestamp: Datetime object
            
        Returns:
            Time features [3]: hour_sin, hour_cos, day_of_week_norm
        """
        hour = timestamp.hour + timestamp.minute / 60.0
        
        # Cyclical encoding for hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # Day of week (normalized 0-1)
        day_of_week = timestamp.weekday() / 6.0
        
        return np.array([hour_sin, hour_cos, day_of_week])
    
    def _extract_session_features(
        self,
        session_length: int,
        session_clicks: int,
        session_impressions: int,
        history_length: int
    ) -> np.ndarray:
        """
        Extract session-level features.
        
        Args:
            session_length: Number of steps in session
            session_clicks: Clicks in session
            session_impressions: Total impressions in session
            history_length: Length of user history
            
        Returns:
            Session features [4]
        """
        # Session length (capped and normalized)
        session_length_norm = min(session_length / 20.0, 1.0)
        
        # Click rate in session
        if session_impressions > 0:
            session_ctr = session_clicks / session_impressions
        else:
            session_ctr = 0.0
        
        # Average clicks per step
        if session_length > 0:
            avg_clicks = session_clicks / session_length
        else:
            avg_clicks = 0.0
        
        # History length (normalized)
        history_length_norm = min(history_length / 50.0, 1.0)
        
        features = np.array([
            session_length_norm,
            session_ctr,
            avg_clicks,
            history_length_norm
        ])
        
        return features
    
    def build_state_action_pair(
        self,
        state: np.ndarray,
        action_news_id: str
    ) -> np.ndarray:
        """
        Build state-action pair (used for Q-networks that take both as input).
        
        Args:
            state: State vector
            action_news_id: News ID of the action
            
        Returns:
            Concatenated state-action vector
        """
        action_embedding = self.article_encoder.get_embedding(action_news_id)
        
        if action_embedding is None:
            action_embedding = np.zeros(self.embedding_dim)
        
        return np.concatenate([state, action_embedding])
    
    def build_candidate_features(
        self,
        state: np.ndarray,
        candidate_news_ids: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build features for candidate articles given current state.
        Useful for action selection.
        
        Args:
            state: Current state vector
            candidate_news_ids: List of candidate news IDs
            
        Returns:
            Tuple of:
                - Candidate embeddings [num_candidates, embedding_dim]
                - Candidate-state features [num_candidates, feature_dim]
        """
        # Get candidate embeddings
        candidate_embeddings = self.article_encoder.get_embeddings(candidate_news_ids)
        
        # Extract user embedding from state
        user_embedding = state[:self.embedding_dim]
        
        # Compute relevance scores (cosine similarity)
        user_norm = np.linalg.norm(user_embedding)
        if user_norm > 0:
            relevance_scores = np.dot(candidate_embeddings, user_embedding) / (
                np.linalg.norm(candidate_embeddings, axis=1) * user_norm
            )
        else:
            relevance_scores = np.zeros(len(candidate_news_ids))
        
        # Combine features
        candidate_features = np.column_stack([
            candidate_embeddings,
            relevance_scores[:, np.newaxis]
        ])
        
        return candidate_embeddings, candidate_features


class StateHistory:
    """
    Maintains history of states and transitions for a session.
    Useful for debugging and off-policy learning.
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize state history tracker.
        
        Args:
            max_history: Maximum number of states to keep
        """
        self.max_history = max_history
        self.states: List[np.ndarray] = []
        self.actions: List[str] = []
        self.rewards: List[float] = []
        self.timestamps: List[datetime] = []
    
    def add(
        self,
        state: np.ndarray,
        action: str,
        reward: float,
        timestamp: Optional[datetime] = None
    ):
        """Add a transition to history."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.timestamps.append(timestamp or datetime.now())
        
        # Keep only last max_history entries
        if len(self.states) > self.max_history:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.timestamps.pop(0)
    
    def get_trajectory(self) -> Dict:
        """
        Get complete trajectory.
        
        Returns:
            Dictionary with states, actions, rewards, timestamps
        """
        return {
            'states': np.array(self.states),
            'actions': self.actions,
            'rewards': np.array(self.rewards),
            'timestamps': self.timestamps,
            'length': len(self.states),
            'total_reward': sum(self.rewards)
        }
    
    def clear(self):
        """Clear history."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.timestamps.clear()


class FeatureNormalizer:
    """
    Normalizes features using running statistics.
    Important for stable RL training.
    """
    
    def __init__(self, feature_dim: int, epsilon: float = 1e-8):
        """
        Initialize normalizer.
        
        Args:
            feature_dim: Dimension of features
            epsilon: Small constant for numerical stability
        """
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        
        self.count = 0
        self.mean = np.zeros(feature_dim)
        self.var = np.ones(feature_dim)
        self.std = np.ones(feature_dim)
    
    def update(self, features: np.ndarray):
        """
        Update running statistics with new features.
        
        Args:
            features: New features [batch_size, feature_dim] or [feature_dim]
        """
        if features.ndim == 1:
            features = features[np.newaxis, :]
        
        batch_size = len(features)
        batch_mean = features.mean(axis=0)
        batch_var = features.var(axis=0)
        
        # Update running statistics
        delta = batch_mean - self.mean
        self.mean += delta * batch_size / (self.count + batch_size)
        
        m_a = self.var * self.count
        m_b = batch_var * batch_size
        M2 = m_a + m_b + delta ** 2 * self.count * batch_size / (self.count + batch_size)
        
        self.count += batch_size
        self.var = M2 / self.count
        self.std = np.sqrt(self.var + self.epsilon)
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using running statistics.
        
        Args:
            features: Features to normalize
            
        Returns:
            Normalized features
        """
        return (features - self.mean) / self.std
    
    def denormalize(self, features: np.ndarray) -> np.ndarray:
        """Denormalize features."""
        return features * self.std + self.mean


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from src.features.article_encoder import ArticleEncoder
    from src.features.user_encoder import UserEncoder
    from src.data.mind_loader import MINDDataLoader
    from datetime import datetime
    
    print("Loading data")
    loader = MINDDataLoader('data/raw/train', 'train')
    loader.load_all(nrows=1000)
    
    print("\nInitializing encoders")
    article_encoder = ArticleEncoder(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        embedding_dim=384,
        cache_dir='data/processed/embeddings'
    )
    
    # Encode first 100 articles
    embeddings = article_encoder.encode_from_dataframe(loader.news_df.head(100), batch_size=32)
    
    user_encoder = UserEncoder(
        embedding_dim=384,
        aggregation_method='mean'
    )
    
    print("\nInitializing state builder")
    state_builder = StateBuilder(
        article_encoder=article_encoder,
        user_encoder=user_encoder,
        embedding_dim=384
    )
    
    print(f"State dimension: {state_builder.state_dim}")
    
    # Test with a sample user
    user_id = loader.behaviors_df['user_id'].iloc[0]
    user_behaviors = loader.get_user_behaviors(user_id)
    
    if user_behaviors:
        behavior = user_behaviors[0]
        
        print(f"\nTesting with user: {user_id}")
        print(f"History length: {len(behavior.history)}")
        
        # Build state
        state = state_builder.build_state(
            history_news_ids=behavior.history[:10],  # Use first 10 articles
            timestamp=behavior.timestamp,
            session_length=1,
            session_clicks=0,
            session_impressions=5
        )
        
        print(f"State shape: {state.shape}")
        print(f"State stats: mean={state.mean():.3f}, std={state.std():.3f}")
        
        # Test state-action pair
        if behavior.impressions:
            action_id = behavior.impressions[0][0]
            state_action = state_builder.build_state_action_pair(state, action_id)
            print(f"\nState-action pair shape: {state_action.shape}")
        
        # Test candidate features
        candidate_ids = [imp[0] for imp in behavior.impressions[:5]]
        cand_emb, cand_feat = state_builder.build_candidate_features(state, candidate_ids)
        print(f"\nCandidate embeddings shape: {cand_emb.shape}")
        print(f"Candidate features shape: {cand_feat.shape}")
    
    print("\nState builder tests passed!")