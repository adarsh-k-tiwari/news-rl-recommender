# src/models/cmab.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RewardPredictor(nn.Module):
    """
    Approximates Reward(s, a).
    Same architecture as the others for fair comparison.
    """
    def __init__(self, state_dim, article_emb_dim, hidden_dim=128):
        super(RewardPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + article_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Output is raw estimated reward
        )
        
    def forward(self, state, article_emb):
        x = torch.cat([state, article_emb], dim=1)
        return self.net(x)

class CMABAgent:
    """
    Neural Contextual Bandit (NeuralUCB / Epsilon-Greedy style).
    Treats every step as an isolated decision.
    """
    def __init__(
        self, 
        state_dim, 
        article_emb_dim, 
        learning_rate=1e-3, 
        epsilon=0.1,         # Exploration rate
        buffer_size=10000,
        device='cpu'
    ):
        self.device = device
        self.epsilon = epsilon
        
        # Single Network (No Target Network needed for Bandits)
        self.net = RewardPredictor(state_dim, article_emb_dim).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        
        # Buffer to store history for mini-batch training
        self.memory = deque(maxlen=buffer_size)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state, candidate_embeddings, eval_mode=False):
        """Epsilon-Greedy Action Selection."""
        k = len(candidate_embeddings)
        
        # Exploration
        if not eval_mode and random.random() < self.epsilon:
            action_idx = random.randint(0, k-1)
            return action_idx, candidate_embeddings[action_idx]
        
        # Exploitation
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        cand_tensor = torch.FloatTensor(candidate_embeddings).to(self.device)
        
        # Repeat state to match candidates
        state_repeated = state_tensor.repeat(k, 1)
        
        with torch.no_grad():
            predicted_rewards = self.net(state_repeated, cand_tensor)
            action_idx = torch.argmax(predicted_rewards).item()
            
        return action_idx, candidate_embeddings[action_idx]

    def update(self, batch_size=32):
        """Train on past interactions."""
        if len(self.memory) < batch_size:
            return 0.0
        
        batch = random.sample(self.memory, batch_size)
        states, action_embs, rewards = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        action_embs = torch.FloatTensor(np.array(action_embs)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        
        # Forward pass
        pred_rewards = self.net(states, action_embs)
        
        # Loss: Minimize difference between predicted and actual reward
        # Note: We only train on the action we actually took (Bandit Feedback)
        loss = self.loss_fn(pred_rewards, rewards)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def push_memory(self, state, action_emb, reward):
        """Store interaction."""
        self.memory.append((state, action_emb, reward))
        
    def predict(self, state, env):
        """Standard interface for evaluation."""
        idx, _ = self.select_action(state, env.get_candidate_embeddings(), eval_mode=True)
        return idx