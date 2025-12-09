import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from src.environment.reco_env import MINDRecEnv
from src.models.cmab.cmab import CMABAgent

def train_cmab():
    # Setup
    data_dir = 'data/processed'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training CMAB on {device}")
    
    # Initialize Environment
    env = MINDRecEnv(data_dir=data_dir, split='train', max_candidate_size=100)
    
    # Initialize Agent
    # Note: CMAB doesn't need a target network or gamma
    agent = CMABAgent(
        state_dim=391,
        article_emb_dim=384,
        device=device,
        buffer_size=50000,
        epsilon=0.1 # Constant small exploration is common for Bandits
    )
    
    num_episodes = 10000
    batch_size = 64
    save_freq = 10000
    
    rewards_history = []
    loss_history = []
    
    print(f"Starting CMAB Training for {num_episodes} episodes...")
    
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
            
            # 3. Store Transition (Bandit only needs: State, Action, Reward)
            agent.push_memory(state, action_emb, reward)
            
            # 4. Train
            loss = agent.update(batch_size)
            if loss > 0:
                loss_history.append(loss)
            
            state = next_state
            episode_reward += reward
        
        rewards_history.append(episode_reward)
        
        # Logging
        if episode > 0 and episode % 1000 == 0:
            avg_reward = np.mean(rewards_history[-1000:])
            tqdm.write(f"Ep {episode} | Avg Reward: {avg_reward:.3f}")

    # Save Final Model
    save_path = Path(data_dir) / 'cmab_model.pth'
    torch.save(agent.net.state_dict(), save_path)
    print(f"Final model saved to {save_path}")

    # Plot Training Curve
    plt.figure(figsize=(10,5))
    plt.plot(np.convolve(rewards_history, np.ones(100)/100, mode='valid'))
    plt.title('CMAB Training Rewards (Moving Avg)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('result/cmab_training.png')
    print("Training plot saved to result/cmab_training.png")

if __name__ == "__main__":
    train_cmab()