# src/models/dqn.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QNetwork(nn.Module):

    def __init__(self, state_dim, article_emb_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + article_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Output is raw Q-value (no Sigmoid!)
        )
        
    def forward(self, state, article_emb):
        x = torch.cat([state, article_emb], dim=1)
        return self.net(x)

class ReplayBuffer:
    """Experience Replay Buffer for DQN."""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action_emb, reward, next_state, next_candidates, done):
        """
        Store transition.
        next_candidates: embeddings of ALL candidates available in the next state (for Max Q calc)
        """
        self.buffer.append((state, action_emb, reward, next_state, next_candidates, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(
        self, 
        state_dim, 
        article_emb_dim, 
        learning_rate=1e-4, 
        gamma=0.95,          # Discount factor for future rewards
        epsilon_start=1.0,   # Exploration rate
        epsilon_end=0.20,
        epsilon_decay=0.995,

        buffer_size=50000,
        device='cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-Networks (Policy and Target)
        self.policy_net = QNetwork(state_dim, article_emb_dim).to(device)
        self.target_net = QNetwork(state_dim, article_emb_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(capacity=buffer_size)
        
        self.loss_fn = nn.MSELoss()

    def select_action(self, state, candidate_embeddings, eval_mode=False):
        k = len(candidate_embeddings)
        
        if not eval_mode and random.random() < self.epsilon:
            action_idx = random.randint(0, k-1)
            return action_idx, candidate_embeddings[action_idx]
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        cand_tensor = torch.FloatTensor(candidate_embeddings).to(self.device)
        
        state_repeated = state_tensor.repeat(k, 1)
        
        with torch.no_grad():
            q_values = self.policy_net(state_repeated, cand_tensor)
            action_idx = torch.argmax(q_values).item()
            
        return action_idx, candidate_embeddings[action_idx]

    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return 0.0
        
        transitions = self.memory.sample(batch_size)
        

        states, action_embs, rewards, next_states, next_candidates_list, dones = zip(*transitions)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        action_embs = torch.FloatTensor(np.array(action_embs)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        current_q_values = self.policy_net(states, action_embs)
        

        next_max_q = []
        
        with torch.no_grad():
            for i in range(batch_size):
                if dones[i]:
                    next_max_q.append(0.0)
                    continue
                
                ns = next_states[i].unsqueeze(0) # [1, state_dim]
                nc = torch.FloatTensor(next_candidates_list[i]).to(self.device) # [K, emb_dim]
                ns_repeated = ns.repeat(len(nc), 1)
                
                target_q = self.target_net(ns_repeated, nc)
                max_q = target_q.max().item()
                next_max_q.append(max_q)
        
        next_max_q = torch.FloatTensor(next_max_q).unsqueeze(1).to(self.device)
        
        target_q_values = rewards + (self.gamma * next_max_q * (1 - dones))
        
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def predict(self, state, env):
        """Standard interface for evaluation scripts."""
        idx, _ = self.select_action(state, env.get_candidate_embeddings(), eval_mode=True)
        return idx