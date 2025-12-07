# run_baselines.py

import numpy as np
from src.models.cmab.cmab import CMABAgent
import torch
import gym
from tqdm import tqdm
from src.environment.reco_env import MINDRecEnv
from src.models.baseline.baselines import RandomAgent, PopularityAgent
from src.models.baseline.supervised import SupervisedAgent, ClickPredictor
from src.models.dqn.dqn import DQNAgent
from src.models.dqn.ddqn import DuelingDQNAgent
from src.models.cmab.cmab import CMABAgent

def evaluate_agent(agent, env, num_episodes=100):
    total_rewards = 0
    total_steps = 0
    
    for _ in tqdm(range(num_episodes), desc=f"Eval {agent.__class__.__name__}"):
        state = env.reset()
        done = False
        while not done:
            # Pass env to predict because Popularity/Supervised need candidate access
            action = agent.predict(state, env)
            next_state, reward, done, info = env.step(action)
            
            total_rewards += reward
            total_steps += 1
            state = next_state
            
    ctr = total_rewards / total_steps if total_steps > 0 else 0
    return ctr, total_rewards

def main():
    # Initialize Env
    env = MINDRecEnv(data_dir='data/processed', split='dev', max_candidate_size=100)
    
    # # 1. Evaluate Random
    # random_agent = RandomAgent(env.action_space)
    # ctr_rand, _ = evaluate_agent(random_agent, env, num_episodes=500)
    # print(f"Random Agent CTR: {ctr_rand:.4f}")
    
    # # 2. Evaluate Popularity
    # pop_agent = PopularityAgent('data/processed/sessions/train_sessions.pkl')
    # ctr_pop, _ = evaluate_agent(pop_agent, env, num_episodes=500)
    # print(f"Popularity Agent CTR: {ctr_pop:.4f}")
    
    # # 3. Evaluate Supervised
    # # Load model
    # model = ClickPredictor(state_dim=391, article_emb_dim=384)
    # model.load_state_dict(torch.load('data/processed/supervised_model.pth'))
    
    # sup_agent = SupervisedAgent(model, env.article_encoder)
    # ctr_sup, _ = evaluate_agent(sup_agent, env, num_episodes=500)
    # print(f"Supervised Agent CTR: {ctr_sup:.4f}")

    # 4. Evaluate DQN
    dqn_agent = DQNAgent(state_dim=391, article_emb_dim=384)
    dqn_agent.policy_net.load_state_dict(torch.load('data/processed/dqn_model.pth'))
    ctr_dqn, _ = evaluate_agent(dqn_agent, env, num_episodes=500)
    print(f"DQN Agent CTR: {ctr_dqn:.4f}")

    # # 5. Evaluate CMAB
    # cmab_agent = CMABAgent(state_dim=391, article_emb_dim=384)
    # cmab_agent.net.load_state_dict(torch.load('data/processed/cmab_model.pth'))
    # ctr_cmab, _ = evaluate_agent(cmab_agent, env, num_episodes=500)
    # print(f"CMAB Agent CTR: {ctr_cmab:.4f}")

    # 6. Deuling DQN
    # ddqn_agent = DuelingDQNAgent(state_dim=391, article_emb_dim=384)
    # ddqn_agent.policy_net.load_state_dict(torch.load('data/processed/dueling_dqn_model.pth'))
    # ctr_ddqn, _ = evaluate_agent(ddqn_agent, env, num_episodes=500)
    # print(f"Dueling DQN Agent CTR: {ctr_ddqn:.4f}")

if __name__ == "__main__":
    main()