# scripts/train_dqn.py

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from src.environment.reco_env import MINDRecEnv
from src.models.dqn.ddqn import DuelingDQNAgent

def train_ddqn():
    # Setup
    data_dir = 'data/processed'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # Initialize Environment
    # We use 'train' split for training interactions
    env = MINDRecEnv(data_dir=data_dir, split='train', max_candidate_size=100)
    
    # Initialize Agent
    # State dim: 384(user) + 3(time) + 4(session) = 391
    agent = DuelingDQNAgent(
        state_dim=391,
        article_emb_dim=384,
        device=device,
        buffer_size=20000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995 # Slow decay
    )
    
    # Training Parameters
    num_episodes = 100000
    target_update_freq = 10
    batch_size = 64
    
    # Logging
    rewards_history = []
    loss_history = []
    
    print("Starting Dueling DQN Training...")
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 1. Select Action
            candidates = env.get_candidate_embeddings()
            action_idx, action_emb = agent.select_action(state, candidates)
            
            # 2. Step Environment
            next_state, reward, done, info = env.step(action_idx)
            
            # 3. Store Transition
            # Important: We need the NEXT candidates for the Bellman update
            # Since env.step() advances the index, env.get_candidate_embeddings()
            # NOW returns the candidates for next_state.
            next_candidates = env.get_candidate_embeddings()
            
            agent.memory.push(state, action_emb, reward, next_state, next_candidates, done)
            
            # 4. Train Agent
            loss = agent.update(batch_size)
            if loss > 0:
                loss_history.append(loss)
            
            # Update state
            state = next_state
            episode_reward += reward
        
        # End of Episode
        rewards_history.append(episode_reward)
        agent.update_epsilon()
        
        # Update Target Network
        if episode % target_update_freq == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        # Logging
        if episode % 1000 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            tqdm.write(f"Ep {episode} | Avg Reward: {avg_reward:.3f} | Epsilon: {agent.epsilon:.3f}")

    # Save Model
    save_path = Path(data_dir) / 'dueling_dqn_model.pth'
    torch.save(agent.policy_net.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Plot Training Curve
    plt.figure(figsize=(10,5))
    plt.plot(np.convolve(rewards_history, np.ones(100)/100, mode='valid'))
    plt.title('Dueling DQN Training Rewards (Moving Avg)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('ddqn_training.png')
    print("Training plot saved to ddqn_training.png")
if __name__ == "__main__":
    train_ddqn()