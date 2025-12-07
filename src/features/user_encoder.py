"""
User Encoder for MIND Dataset
Aggregates user click history into meaningful state representations for RL.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserEncoder:
    """
    Encodes user state from their click history.
    Supports multiple aggregation strategies.
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        max_history_length: int = 50,
        aggregation_method: str = 'attention',
        use_temporal_decay: bool = True,
        decay_factor: float = 0.9
    ):
        """
        Initialize user encoder.
        
        Args:
            embedding_dim: Dimension of article embeddings
            max_history_length: Maximum number of historical articles to consider
            aggregation_method: How to aggregate history ('mean', 'sum', 'attention', 'lstm')
            use_temporal_decay: Whether to apply temporal decay to older articles
            decay_factor: Decay factor for temporal weighting (0-1)
        """
        self.embedding_dim = embedding_dim
        self.max_history_length = max_history_length
        self.aggregation_method = aggregation_method
        self.use_temporal_decay = use_temporal_decay
        self.decay_factor = decay_factor
        
        # Initialize aggregation module if needed
        if aggregation_method == 'attention':
            self.attention = AttentionAggregator(embedding_dim)
        elif aggregation_method == 'lstm':
            self.lstm = LSTMAggregator(embedding_dim)
        
        logger.info(f"UserEncoder initialized with {aggregation_method} aggregation")
    
    def encode_history(
        self,
        history_embeddings: np.ndarray,
        return_tensor: bool = False
    ) -> np.ndarray:
        """
        Encode user history into a single state vector.
        
        Args:
            history_embeddings: History article embeddings [history_length, embedding_dim]
            return_tensor: Return as torch tensor
            
        Returns:
            User state embedding [embedding_dim]
        """
        if len(history_embeddings) == 0:
            # Return zero vector for empty history (cold start)
            state = np.zeros(self.embedding_dim)
            return torch.tensor(state) if return_tensor else state
        
        # Truncate to max_history_length (keep most recent)
        if len(history_embeddings) > self.max_history_length:
            history_embeddings = history_embeddings[-self.max_history_length:]
        
        # Apply temporal decay
        if self.use_temporal_decay:
            history_embeddings = self._apply_temporal_decay(history_embeddings)
        
        # Aggregate based on method
        if self.aggregation_method == 'mean':
            state = np.mean(history_embeddings, axis=0)
        
        elif self.aggregation_method == 'sum':
            state = np.sum(history_embeddings, axis=0)
        
        elif self.aggregation_method == 'attention':
            history_tensor = torch.tensor(history_embeddings, dtype=torch.float32)
            state = self.attention(history_tensor).detach().numpy()
        
        elif self.aggregation_method == 'lstm':
            history_tensor = torch.tensor(history_embeddings, dtype=torch.float32)
            state = self.lstm(history_tensor).detach().numpy()
        
        elif self.aggregation_method == 'last':
            # Just use the most recent article
            state = history_embeddings[-1]
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        return torch.tensor(state) if return_tensor else state
    
    def _apply_temporal_decay(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply exponential temporal decay to embeddings (older = less weight).
        
        Args:
            embeddings: Article embeddings [history_length, embedding_dim]
            
        Returns:
            Weighted embeddings
        """
        history_length = len(embeddings)
        weights = np.array([self.decay_factor ** (history_length - i - 1) 
                           for i in range(history_length)])
        weights = weights / weights.sum()  # Normalize
        
        return embeddings * weights[:, np.newaxis]
    
    def encode_user_with_context(
        self,
        history_embeddings: np.ndarray,
        timestamp: Optional[float] = None,
        session_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Encode user state with additional context features.
        
        Args:
            history_embeddings: History article embeddings
            timestamp: Current timestamp (hour of day as float 0-24)
            session_length: Current session length
            
        Returns:
            Extended user state [embedding_dim + context_features]
        """
        # Get base state
        base_state = self.encode_history(history_embeddings)
        
        # Add context features
        context_features = []
        
        if timestamp is not None:
            # Hour of day (normalized)
            hour_sin = np.sin(2 * np.pi * timestamp / 24)
            hour_cos = np.cos(2 * np.pi * timestamp / 24)
            context_features.extend([hour_sin, hour_cos])
        
        if session_length is not None:
            # Session length (normalized)
            context_features.append(min(session_length / 20.0, 1.0))
        
        # History length indicator
        history_len_norm = len(history_embeddings) / self.max_history_length
        context_features.append(history_len_norm)
        
        # Concatenate
        if context_features:
            return np.concatenate([base_state, np.array(context_features)])
        else:
            return base_state


class AttentionAggregator(nn.Module):
    """
    Attention-based aggregation of history embeddings.
    """
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.scale = np.sqrt(embedding_dim)
    
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: [history_length, embedding_dim]
            
        Returns:
            Aggregated state [embedding_dim]
        """
        # Self-attention
        Q = self.query(history)  # [history_length, embedding_dim]
        K = self.key(history)
        V = self.value(history)
        
        # Attention scores
        scores = torch.matmul(Q, K.T) / self.scale  # [history_length, history_length]
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Weighted sum
        context = torch.matmul(attn_weights, V)  # [history_length, embedding_dim]
        
        # Mean pooling
        return context.mean(dim=0)


class LSTMAggregator(nn.Module):
    """
    LSTM-based aggregation of history embeddings.
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.projection = nn.Linear(hidden_dim, embedding_dim)
    
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: [history_length, embedding_dim]
            
        Returns:
            Final hidden state [embedding_dim]
        """
        # Add batch dimension
        history = history.unsqueeze(0)  # [1, history_length, embedding_dim]
        
        # LSTM forward
        _, (h_n, _) = self.lstm(history)  # h_n: [1, 1, hidden_dim]
        
        # Project to embedding_dim
        output = self.projection(h_n.squeeze())  # [embedding_dim]
        
        return output


class UserProfileBuilder:
    """
    Builds comprehensive user profiles combining multiple features.
    """
    
    def __init__(
        self,
        user_encoder: UserEncoder,
        include_category_preference: bool = True,
        include_diversity_score: bool = True
    ):
        """
        Initialize user profile builder.
        
        Args:
            user_encoder: UserEncoder instance
            include_category_preference: Include category preference features
            include_diversity_score: Include diversity metrics
        """
        self.user_encoder = user_encoder
        self.include_category_preference = include_category_preference
        self.include_diversity_score = include_diversity_score
    
    def build_profile(
        self,
        history_embeddings: np.ndarray,
        history_categories: Optional[List[str]] = None,
        **context
    ) -> Dict[str, np.ndarray]:
        """
        Build comprehensive user profile.
        
        Args:
            history_embeddings: Article embeddings from history
            history_categories: Categories of historical articles
            **context: Additional context (timestamp, session_length, etc.)
            
        Returns:
            Dictionary with profile components
        """
        profile = {}
        
        # Core state representation
        profile['state'] = self.user_encoder.encode_history(history_embeddings)
        
        # Category preferences
        if self.include_category_preference and history_categories:
            cat_counts = {}
            for cat in history_categories:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
            
            # Top 3 preferred categories
            top_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            profile['preferred_categories'] = [cat for cat, _ in top_cats]
            profile['category_counts'] = cat_counts
        
        # Diversity score
        if self.include_diversity_score and len(history_embeddings) > 1:
            # Compute average pairwise cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(history_embeddings)
            # Lower similarity = higher diversity
            avg_similarity = (similarities.sum() - len(similarities)) / (len(similarities) ** 2 - len(similarities))
            diversity_score = 1.0 - avg_similarity
            profile['diversity_score'] = diversity_score
        
        # Context features
        if context:
            profile['context'] = context
        
        return profile
    
    def get_feature_vector(self, profile: Dict) -> np.ndarray:
        """
        Convert profile dict to single feature vector.
        
        Args:
            profile: Profile dictionary from build_profile
            
        Returns:
            Feature vector
        """
        features = [profile['state']]
        
        if 'diversity_score' in profile:
            features.append(np.array([profile['diversity_score']]))
        
        return np.concatenate(features)


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.append('.')
    
    # Create dummy history embeddings
    embedding_dim = 384
    history_length = 10
    
    history_embeddings = np.random.randn(history_length, embedding_dim).astype(np.float32)
    
    print("Testing different aggregation methods...")
    
    # Test mean aggregation
    encoder_mean = UserEncoder(embedding_dim, aggregation_method='mean')
    state_mean = encoder_mean.encode_history(history_embeddings)
    print(f"\nMean aggregation: {state_mean.shape}")
    
    # Test attention aggregation
    encoder_attn = UserEncoder(embedding_dim, aggregation_method='attention')
    state_attn = encoder_attn.encode_history(history_embeddings)
    print(f"Attention aggregation: {state_attn.shape}")
    
    # Test LSTM aggregation
    encoder_lstm = UserEncoder(embedding_dim, aggregation_method='lstm')
    state_lstm = encoder_lstm.encode_history(history_embeddings)
    print(f"LSTM aggregation: {state_lstm.shape}")
    
    # Test with context
    state_context = encoder_mean.encode_user_with_context(
        history_embeddings,
        timestamp=14.5,  # 2:30 PM
        session_length=5
    )
    print(f"\nWith context: {state_context.shape}")
    
    # Test profile builder
    profile_builder = UserProfileBuilder(encoder_mean)
    categories = ['news', 'sports', 'news', 'entertainment', 'news']
    
    profile = profile_builder.build_profile(
        history_embeddings,
        history_categories=categories,
        timestamp=14.5
    )
    
    print(f"\nUser profile keys: {profile.keys()}")
    print(f"Preferred categories: {profile.get('preferred_categories')}")
    print(f"Diversity score: {profile.get('diversity_score', 'N/A')}")
    
    print("\nUser encoder tests passed!")