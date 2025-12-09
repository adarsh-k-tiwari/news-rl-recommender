import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from src.environment.reco_env import MINDRecEnv
from src.models.sac.sac import SACAgent

def train_sac():
    # Setup
    data_dir = 'data/processed'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training SAC on {device}")
    
    env = MINDRecEnv(data_dir=data_dir, split='train', max_candidate_size=100)
    
    agent = SACAgent(
        state_dim=391,
        article_emb_dim=384,
        device=device,
        buffer_size=50000,
        learning_rate=3e-4
    )
    
    num_episodes =10000
    batch_size = 64
    save_freq = 5000
    
    rewards_history = []
    
    print(f"Starting SAC Training for {num_episodes} episodes...")
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 1. Select Action
            candidates = env.get_candidate_embeddings()
            action_idx, action_emb = agent.select_action(state, candidates)
            
            # 2. Step
            next_state, reward, done, info = env.step(action_idx)
            
            # 3. Store
            next_candidates = env.get_candidate_embeddings()
            agent.push_memory(state, action_emb, reward, next_state, next_candidates, done)
            
            # 4. Train
            agent.update(batch_size)
            
            state = next_state
            episode_reward += reward
        
        rewards_history.append(episode_reward)
        
        # Logging
        if episode > 0 and episode % 1000 == 0:
            avg_reward = np.mean(rewards_history[-1000:])
            tqdm.write(f"Ep {episode} | Avg Reward: {avg_reward:.3f} | Alpha: {agent.alpha:.3f}")

    # Save Final
    torch.save(agent.policy.state_dict(), Path(data_dir) / 'sac_model.pth')
    print("Saved SAC model.")

    # Plot Training Curve
    plt.figure(figsize=(10,5))
    plt.plot(np.convolve(rewards_history, np.ones(100)/100, mode='valid'))
    plt.title('SAC Training Rewards (Moving Avg)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('result/sac_training.png')
    print("Training plot saved to result/sac_training.png")

if __name__ == "__main__":
    train_sac()