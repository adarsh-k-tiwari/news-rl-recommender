# src/models/sac/sac.py (or src/models/sac.py depending on your structure)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SoftQNetwork(nn.Module):
    """
    Critic: Estimates Q(s, a).
    We use two of these (Twin Critic) to reduce overestimation bias.
    """
    def __init__(self, state_dim, article_emb_dim, hidden_dim=128):
        super(SoftQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + article_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state, article_emb):
        x = torch.cat([state, article_emb], dim=1)
        return self.net(x)

class SACAgent:
    def __init__(
        self, 
        state_dim, 
        article_emb_dim, 
        learning_rate=3e-4, 
        gamma=0.99,
        tau=0.005,          
        alpha=0.05,          
        automatic_entropy_tuning=False,
        buffer_size=10000,
        device='cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # 1. Critics (Twin Delayed)
        self.q1 = SoftQNetwork(state_dim, article_emb_dim).to(device)
        self.q2 = SoftQNetwork(state_dim, article_emb_dim).to(device)
        self.q1_target = SoftQNetwork(state_dim, article_emb_dim).to(device)
        self.q2_target = SoftQNetwork(state_dim, article_emb_dim).to(device)
        
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        self.q_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=learning_rate)

        # 2. Policy (Actor)
        self.policy = SoftQNetwork(state_dim, article_emb_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # 3. Automatic Entropy Tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = 2.0 
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha

        self.memory = deque(maxlen=buffer_size)

    def select_action(self, state, candidate_embeddings, eval_mode=False):
        """
        Sample action from the policy distribution.
        """
        k = len(candidate_embeddings)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        cand_tensor = torch.FloatTensor(candidate_embeddings).to(self.device)
        
        # Repeat state
        state_repeated = state_tensor.repeat(k, 1)
        
        # Get logits from Policy Network
        logits = self.policy(state_repeated, cand_tensor).squeeze() # [K]
        
        # Check for NaNs in logits before softmax
        if torch.isnan(logits).any():
            if not eval_mode:
                # Fallback to random action to survive the crash
                idx = random.randint(0, k-1)
                return idx, candidate_embeddings[idx]
            else:
                logits = torch.nan_to_num(logits, 0.0)

        # Softmax to get probabilities
        probs = torch.softmax(logits, dim=0)
        
        if eval_mode:
            action_idx = torch.argmax(probs).item()
        else:
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()
            
        return action_idx, candidate_embeddings[action_idx]

    def update(self, batch_size=64):
        if len(self.memory) < batch_size:
            return 0.0, 0.0
            
        batch = random.sample(self.memory, batch_size)
        states, action_embs, rewards, next_states, next_candidates_list, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        action_embs = torch.FloatTensor(np.array(action_embs)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # --- 1. Train Critics (Q1, Q2) ---
        with torch.no_grad():
            next_q_targets = []
            for i in range(batch_size):
                if dones[i]:
                    next_q_targets.append(0.0)
                    continue

                ns = next_states[i].unsqueeze(0)
                nc = torch.FloatTensor(next_candidates_list[i]).to(self.device)
                ns_repeated = ns.repeat(len(nc), 1)

                next_logits = self.policy(ns_repeated, nc).squeeze()
                next_probs = torch.softmax(next_logits, dim=0)
                next_log_probs = torch.log(next_probs + 1e-8)
                
                target_q1 = self.q1_target(ns_repeated, nc).squeeze()
                target_q2 = self.q2_target(ns_repeated, nc).squeeze()
                min_target_q = torch.min(target_q1, target_q2)
                
                # Soft Value: V(s') = sum( probs * (min_Q - alpha * log_probs) )
                v_next = torch.sum(next_probs * (min_target_q - self.alpha * next_log_probs))
                next_q_targets.append(v_next.item())
                
            next_q_targets = torch.FloatTensor(next_q_targets).unsqueeze(1).to(self.device)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_targets

        current_q1 = self.q1(states, action_embs)
        current_q2 = self.q2(states, action_embs)
        
        q1_loss = nn.MSELoss()(current_q1, target_q_values)
        q2_loss = nn.MSELoss()(current_q2, target_q_values)
        q_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        # [FIX] Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q_optimizer.step()
        
        # --- 2. Train Policy (Actor) ---
        policy_loss_total = 0
        for i in range(batch_size):
            ns = next_states[i].unsqueeze(0)
            nc = torch.FloatTensor(next_candidates_list[i]).to(self.device)
            ns_repeated = ns.repeat(len(nc), 1)
            
            logits = self.policy(ns_repeated, nc).squeeze()
            probs = torch.softmax(logits, dim=0)
            log_probs = torch.log(probs + 1e-8)
            
            with torch.no_grad():
                q1_val = self.q1(ns_repeated, nc).squeeze()
                q2_val = self.q2(ns_repeated, nc).squeeze()
                min_q = torch.min(q1_val, q2_val)
            
            # Loss = alpha * log_probs - Q
            loss = torch.sum(probs * (self.alpha * log_probs - min_q))
            policy_loss_total += loss
            
        policy_loss = policy_loss_total / batch_size
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # [FIX] Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()

        # --- 3. Train Alpha (Auto-tuning) ---
        if self.automatic_entropy_tuning:
            alpha_loss_total = 0
            for i in range(batch_size):
                ns = next_states[i].unsqueeze(0)
                nc = torch.FloatTensor(next_candidates_list[i]).to(self.device)
                ns_repeated = ns.repeat(len(nc), 1)
                
                with torch.no_grad():
                    logits = self.policy(ns_repeated, nc).squeeze()
                    probs = torch.softmax(logits, dim=0)
                    log_probs = torch.log(probs + 1e-8)
                    entropy = -torch.sum(probs * log_probs)
            
                # [FIX] Clamp log_alpha to prevent explosion
                self.log_alpha.data.clamp_(min=-20, max=2)
                
                alpha_loss_total += -(self.log_alpha * (entropy - self.target_entropy).detach())
                
            alpha_loss = alpha_loss_total / batch_size
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            # [FIX] Gradient Clipping
            torch.nn.utils.clip_grad_norm_([self.log_alpha], 1.0)
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()

        # --- 4. Soft Updates ---
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return q_loss.item(), policy_loss.item()

    def push_memory(self, state, action_emb, reward, next_state, next_candidates, done):
        self.memory.append((state, action_emb, reward, next_state, next_candidates, done))
        
    def predict(self, state, env):
        idx, _ = self.select_action(state, env.get_candidate_embeddings(), eval_mode=True)
        return idx