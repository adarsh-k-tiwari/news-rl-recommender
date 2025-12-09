"""
Final Evaluation Script for Recommender System Agents
This script evaluates various recommender system agents (baselines and trained models)
on a common evaluation environment using multiple metrics including accuracy and diversity
Results are saved to a CSV file and visualized in a plot
"""
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
from src.environment.reco_env import MINDRecEnv
from src.evaluation.metrics import RecommenderMetrics
from src.models.baseline.baselines import RandomAgent, PopularityAgent
from src.models.baseline.supervised import ClickPredictor, SupervisedAgent
from src.models.dqn.dqn import DQNAgent
from src.models.dqn.ddqn import DuelingDQNAgent
from src.models.cmab.cmab import CMABAgent
from src.models.sac.sac import SACAgent

def run_evaluation_loop(agent, env, metrics_calc, num_episodes=500, name="Agent"):
    """Runs episodes and tracks all detailed metrics."""
    
    # Storage for batch calculation
    all_recs = []   # List of lists of news_ids
    all_clicks = [] # List of lists of news_ids (that were clicked)
    episode_rewards = []
    
    # Disable exploration for RL agents
    if hasattr(agent, 'epsilon'): agent.epsilon = 0.0
    
    for _ in tqdm(range(num_episodes), desc=f"Eval {name}"):
        state = env.reset()
        done = False
        
        # Track session data
        session_recs = []
        session_clicks = []
        total_reward = 0
        
        while not done:
            # 1. Get Action
            candidates = env.get_candidate_embeddings()
            candidate_ids = env.current_candidates # List of news_ids
            
            if name == "Random":
                action = agent.predict(state)
                if action >= len(candidate_ids):
                    action = np.random.randint(0, len(candidate_ids))
            elif name == "Popularity" or name == "Supervised":
                action = agent.predict(state, env)
            else:
                # RL Agents
                action, _ = agent.select_action(state, candidates, eval_mode=True)
            
            if action >= len(candidate_ids):
                action = 0 
            
            # 2. Step
            next_state, reward, done, info = env.step(action)
            
            # 3. Log interactions
            rec_id = candidate_ids[action]
            session_recs.append(rec_id)
            
            if reward > 0: # Assuming reward 1 = click
                session_clicks.append(rec_id)
            
            total_reward += reward
            state = next_state
            
        # Store Episode Data
        all_recs.append(session_recs)
        all_clicks.append(session_clicks)
        episode_rewards.append(total_reward)

    # --- CALCULATE METRICS ---
    # 1. Accuracy
    ctrs = [metrics_calc.calculate_ctr(r, c) for r, c in zip(all_recs, all_clicks)]
    ndcgs = [metrics_calc.calculate_ndcg(r, c) for r, c in zip(all_recs, all_clicks)]
    
    # 2. Diversity
    ilds = [metrics_calc.calculate_ild(r) for r in all_recs]
    cat_divs = [metrics_calc.calculate_category_diversity(r) for r in all_recs]
    
    # 3. Aggregate
    results = {
        'Model': name,
        'CTR': np.mean(ctrs),
        'NDCG@10': np.mean(ndcgs),
        'Diversity (ILD)': np.mean(ilds),
        'Category Entropy': np.mean(cat_divs),
        'Avg Reward': np.mean(episode_rewards)
    }
    
    return results

def main():
    data_dir = 'data/processed'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Final Evaluation on {device}")
    
    # 1. Setup Env & Metrics
    env = MINDRecEnv(data_dir=data_dir, split='dev', max_candidate_size=100)
    from src.data.mind_loader import MINDDataLoader
    loader = MINDDataLoader(data_dir='data/raw/dev', dataset_type='dev')
    loader.load_news() # Populates loader.news_df
    
    metrics_calc = RecommenderMetrics(loader.news_df, env.article_encoder)
    
    # 2. Load Agents
    agents = {}
    
    # Baselines
    agents['Random'] = RandomAgent(env.action_space)
    agents['Popularity'] = PopularityAgent(f'{data_dir}/sessions/train_sessions.pkl')
    
    # Trained Models
    try:
        sup = SupervisedAgent(ClickPredictor(391, 384), env.article_encoder, device=device)
        sup.model.load_state_dict(torch.load(f'{data_dir}/supervised_model.pth', map_location=device))
        agents['Supervised'] = sup
    except: print("Skipping Supervised (not found)")

    try:
        dqn = DQNAgent(391, 384, device=device)
        dqn.policy_net.load_state_dict(torch.load(f'{data_dir}/dqn_model.pth', map_location=device))
        agents['DQN'] = dqn
    except: print("Skipping DQN")

    try:
        cmab = CMABAgent(391, 384, device=device)
        cmab.net.load_state_dict(torch.load(f'{data_dir}/cmab_model.pth', map_location=device))
        agents['CMAB'] = cmab
    except: print("Skipping CMAB")
    
    try:
        ddqn = DuelingDQNAgent(391, 384, device=device)
        ddqn.policy_net.load_state_dict(torch.load(f'{data_dir}/dueling_dqn_model.pth', map_location=device))
        agents['Dueling DQN'] = ddqn
    except: print("Skipping Dueling DQN")

    try:
        sac = SACAgent(391, 384, device=device)
        # Load policy net
        sac.policy.load_state_dict(torch.load(f'{data_dir}/sac_model.pth', map_location=device))
        agents['SAC'] = sac
    except: print("Skipping SAC")

    # 3. Run Loop
    final_results = []
    
    for name, agent in agents.items():
        res = run_evaluation_loop(agent, env, metrics_calc, num_episodes=1000, name=name)
        final_results.append(res)
        
    # 4. Display & Save
    df = pd.DataFrame(final_results)
    df = df.set_index('Model')
    
    print("\n" + "="*50)
    print("FINAL COMPREHENSIVE RESULTS")
    print("="*50)
    print(df)
    
    df.to_csv('result/final_metrics.csv')
    print("\nSaved to result/final_metrics.csv")
    
    # 5. Spider Plot (Radar Chart) for visual comparison
    # Normalize columns for the chart 0-1
    df_norm = (df - df.min()) / (df.max() - df.min())
    
    # (Optional) Simple Bar Chart if Spider is too complex
    df[['CTR', 'Diversity (ILD)', 'NDCG@10']].plot(kind='bar', figsize=(12, 6))
    plt.title("Model Performance across Metrics")
    plt.tight_layout()
    plt.savefig('result/final_metrics_plot.png')

if __name__ == "__main__":
    main()