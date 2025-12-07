# src/models/supervised.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClickPredictor(nn.Module):
    """
    Simple Feed-Forward Network to predict P(click | user_state, article_embedding).
    """
    def __init__(self, state_dim, article_emb_dim, hidden_dim=128):
        super(ClickPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + article_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, state, article_emb):
        x = torch.cat([state, article_emb], dim=1)
        return self.net(x)

class SupervisedDataset(Dataset):
    """Creates (state, article_emb, label) tuples from sessions."""
    def __init__(self, sessions, article_encoder, state_builder=None, max_samples=1000000):
        self.samples = []
        self.article_encoder = article_encoder
        self.state_builder = state_builder
        
        # Flatten sessions into samples
        count = 0
        skipped_no_state = 0
        
        logger.info(f"Building dataset from {len(sessions)} sessions...")
        
        for session in sessions:
            # Track session stats for on-the-fly state building
            running_clicks = 0
            running_impressions = 0
            
            for idx, imp in enumerate(session['impressions']):
                state = imp.get('state')
                
                # If state is missing, build it on the fly
                if state is None:
                    if self.state_builder is None:
                        skipped_no_state += 1
                        continue
                    
                    # Ensure timestamp is datetime
                    ts = imp['timestamp']
                    if not hasattr(ts, 'hour'): 
                        ts = pd.to_datetime(ts)
                        
                    state = self.state_builder.build_state(
                        history_news_ids=imp['history'],
                        timestamp=ts,
                        session_length=idx,
                        session_clicks=running_clicks,
                        session_impressions=running_impressions
                    )

                # For each candidate in the impression
                for news_id, label in zip(imp['candidates'], imp['labels']):
                    self.samples.append((state, news_id, label))
                    count += 1
                    if max_samples and count >= max_samples: break
                
                # Update running stats
                current_clicks = sum(imp['labels'])
                running_clicks += current_clicks
                running_impressions += len(imp['candidates'])
                
            if max_samples and count >= max_samples: break
            
        if count == 0:
            logger.error(f"Dataset is empty! Skipped {skipped_no_state} impressions due to missing state. Pass a StateBuilder to fix this.")
        else:
            logger.info(f"Created dataset with {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        state, news_id, label = self.samples[idx]
        article_emb = self.article_encoder.get_embedding(news_id)
        
        if article_emb is None:
            article_emb = np.zeros(384) # Fallback
            
        return (
            torch.FloatTensor(state), 
            torch.FloatTensor(article_emb), 
            torch.FloatTensor([label])
        )

class SupervisedAgent:
    """
    Wraps the trained ClickPredictor to act as an Agent.
    """
    def __init__(self, model, article_encoder, device='cpu'):
        self.model = model
        self.article_encoder = article_encoder
        self.device = device
        # self.model.to(device)
        self.model.eval()

    def predict(self, state, env):
        """
        Scores all candidates using the network and picks the best one.
        """
        candidates = env.current_candidates
        state_tensor = torch.FloatTensor(state).unsqueeze(0) #.to(self.device) # [1, state_dim]
        
        # Get embeddings for all candidates
        candidate_embs = env.get_candidate_embeddings() # [K, emb_dim]
        candidate_tensor = torch.FloatTensor(candidate_embs) #.to(self.device)
        
        # Repeat state K times to match candidates
        state_repeated = state_tensor.repeat(len(candidates), 1)
        
        with torch.no_grad():
            scores = self.model(state_repeated, candidate_tensor)
            best_action = torch.argmax(scores).item()
            
        return best_action