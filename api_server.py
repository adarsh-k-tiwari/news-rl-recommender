from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import random
from datetime import datetime
from typing import List, Dict, Optional
import logging
from src.features.article_encoder import ArticleEncoder
from src.features.category_encoder import CategoryEncoder
from src.features.user_encoder import UserEncoder
from src.features.state_builder import StateBuilder
from src.data.mind_loader import MINDDataLoader
from src.models.baseline.supervised import ClickPredictor, SupervisedAgent
from src.models.dqn.dqn import DQNAgent
from src.models.dqn.ddqn import DuelingDQNAgent
from src.models.cmab.cmab import CMABAgent
from src.models.sac.sac import SACAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables for loaded resources
article_encoder = None
news_df = None
state_builder = None
agent = None
available_models = {}
user_sessions = {}  # Store user session data
model_usage_stats = {
    'total_requests': 0,
    'model_calls': {},
    'total_rewards': 0,
    'avg_q_values': [],
    'avg_scores': []
}  # Track model usage and performance
def load_resources():
    """Load models and data once at startup."""
    global article_encoder, news_df, state_builder, agent, available_models,CANDIDATE_POOL
    
    logger.info("Loading resources...")
    data_dir = Path('data/processed')
    loader = MINDDataLoader(data_dir='data/raw/dev', dataset_type='dev')
    loader.load_news()
    news_df = loader.news_df.set_index('news_id')
    CANDIDATE_POOL = news_df.index.tolist()[:2000]
    logger.info("Using CategoryEncoder for category-based recommendations (one-hot)")
    article_encoder = CategoryEncoder(news_df=news_df)
    emb_dim = article_encoder.embedding_dim
    logger.info(f"Using embedding dimension: {emb_dim} (one dimension per category)")
    
    # 3. Initialize Encoders & State Builder
    user_encoder = UserEncoder(embedding_dim=emb_dim, aggregation_method='mean')
    state_builder = StateBuilder(
        article_encoder=article_encoder,
        user_encoder=user_encoder,
        embedding_dim=emb_dim
    )
    
    # 4. Load Available Models
    model_paths = {
        'supervised': 'data/processed/supervised_model.pth',
        'dqn': 'data/processed/dqn_model.pth',
        'dueling_dqn': 'data/processed/dueling_dqn_model.pth',
        'cmab': 'data/processed/cmab_model.pth',
        'sac': 'data/processed/sac_model.pth'
    }
    
    state_dim = state_builder.state_dim
    
    article_emb_dim = emb_dim  
    if Path(model_paths['supervised']).exists():
        model = ClickPredictor(state_dim=state_dim, article_emb_dim=article_emb_dim)
        try:
            model.load_state_dict(torch.load(model_paths['supervised'], map_location='cpu'))
            model.train()
            sup_agent = SupervisedAgent(model, article_encoder)
            try:
                sup_agent.optimizer = torch.optim.Adam(sup_agent.model.parameters(), lr=1e-4)
            except Exception:
                sup_agent.optimizer = None
            available_models['supervised'] = sup_agent
            logger.info("Loaded supervised model")
        except Exception as e:
            logger.warning(f"Failed to load supervised model: {e}")
    
    # Load DQN Model (or create fresh for category-based learning)
    if Path(model_paths['dqn']).exists():
        try:
            dqn_agent = DQNAgent(
                state_dim=state_dim,
                article_emb_dim=article_emb_dim,
                device='cpu'
            )
            # Try loading, but if dimensions mismatch, create fresh
            try:
                dqn_agent.policy_net.load_state_dict(torch.load(model_paths['dqn'], map_location='cpu'))
                logger.info("Loaded existing DQN model")
            except Exception:
                logger.warning("DQN dimension mismatch - using fresh model for category-based learning")
            dqn_agent.policy_net.train()
            available_models['dqn'] = dqn_agent
        except Exception as e:
            logger.warning(f"Failed to load DQN model: {e}")
    
    # Load Dueling DQN Model (or create fresh)
    if Path(model_paths['dueling_dqn']).exists():
        try:
            ddqn_agent = DuelingDQNAgent(
                state_dim=state_dim,
                article_emb_dim=article_emb_dim,
                device='cpu'
            )
            try:
                ddqn_agent.policy_net.load_state_dict(torch.load(model_paths['dueling_dqn'], map_location='cpu'))
                logger.info("Loaded existing Dueling DQN model")
            except Exception:
                logger.warning("Dueling DQN dimension mismatch - using fresh model for category-based learning")
            ddqn_agent.policy_net.train()
            available_models['dueling_dqn'] = ddqn_agent
        except Exception as e:
            logger.warning(f"Failed to load Dueling DQN model: {e}")
    
    # Load CMAB Model
    if Path(model_paths['cmab']).exists():
        try:
            cmab_agent = CMABAgent(
                num_categories=emb_dim,
                article_encoder=article_encoder
            )
            try:
                cmab_agent.load(model_paths['cmab'])
                logger.info("Loaded existing CMAB model")
            except Exception:
                logger.warning("CMAB load failed - using fresh model")
            available_models['cmab'] = cmab_agent
        except Exception as e:
            logger.warning(f"Failed to load CMAB model: {e}")
    
    # Load SAC Model
    if Path(model_paths['sac']).exists():
        try:
            sac_agent = SACAgent(
                state_dim=state_dim,
                article_emb_dim=article_emb_dim,
                device='cpu'
            )
            try:
                sac_agent.load(model_paths['sac'])
                logger.info("Loaded existing SAC model")
            except Exception:
                logger.warning("SAC load failed - using fresh model")
            available_models['sac'] = sac_agent
        except Exception as e:
            logger.warning(f"Failed to load SAC model: {e}")
    
    if available_models:
        agent = list(available_models.values())[0]
        logger.info(f"Using default agent: {list(available_models.keys())[0]}")
    else:
        logger.error("No models loaded! Please train at least one model.")
        # Create a dummy agent to prevent crashes (will return random recommendations)
        logger.warning("Creating fallback random agent")
        agent = None
    
    logger.info("Resources loaded successfully")


class MockEnv:
    """Mock environment for agent prediction."""
    def __init__(self, candidates: List[str], article_encoder):
        self.current_candidates = candidates
        self.article_encoder = article_encoder
    
    def get_candidate_embeddings(self):
        return self.article_encoder.get_embeddings(self.current_candidates)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(available_models.keys()),
        'total_articles': len(news_df) if news_df is not None else 0
    })


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models."""
    return jsonify({
        'models': list(available_models.keys()),
        'current': list(available_models.keys())[0] if available_models else None
    })


@app.route('/api/models/<model_name>', methods=['POST'])
def set_model(model_name):
    global agent
    if model_name in available_models:
        agent = available_models[model_name]
        for user in user_sessions:
            user_sessions[user]["last_state"] = None
            user_sessions[user]["last_candidates"] = None
            user_sessions[user]["last_candidate_ids"] = None

        if hasattr(agent, "policy_net"):
            agent.policy_net.train()
        if hasattr(agent, 'model'):
            try:
                agent.model.train()
                if not hasattr(agent, 'optimizer') or agent.optimizer is None:
                    agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=1e-4)
            except Exception:
                logger.warning('Failed to set supervised optimizer or train mode')
        if hasattr(agent, "epsilon"):
            agent.epsilon = 1  

        return jsonify({'status': 'success', 'model': model_name})
    return jsonify({'status': 'error', 'message': 'Model not found'}), 404


@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get recommendations for a user session."""
    global agent, state_builder, article_encoder, news_df

    try:
        if agent is None:
            logger.error("Agent not loaded!")
            return jsonify({'error': 'Agent not loaded. Please check server logs.'}), 500
        
        if news_df is None or len(news_df) == 0:
            logger.error("News dataframe not loaded!")
            return jsonify({'error': 'News data not loaded. Please check server logs.'}), 500
        
        data = request.json
        user_id = data.get('user_id', 'default')
        history = data.get('history', [])  
        count = data.get('count', 10)
        categories = data.get('categories', [])  
        
        logger.info(f"Getting recommendations for user {user_id}, history length: {len(history)}, count: {count}")
        if user_id not in user_sessions:
            user_sessions[user_id] = {
                'history': [],
                'likes': [],
                'dislikes': [],
                'impressions': 0,
                'clicks': 0,
                'interactions': []
            }
        
        session = user_sessions[user_id]
        if history:
            session['history'] = history
            existing_likes = set(session.get('likes', []))
            existing_likes.update(history)
            session['likes'] = list(existing_likes)
        
        logger.info(f"Session for {user_id}: {len(session.get('likes', []))} likes, {len(session.get('interactions', []))} interactions")
        session['impressions'] += count
        
        if categories and len(categories) > 0:
            filtered_pool = news_df[news_df['category'].isin(categories)].index.tolist()
        else:
            filtered_pool = news_df.index.tolist()
        
        shown_articles = set(session.get('shown_articles', []))
        filtered_pool = [nid for nid in filtered_pool if nid not in shown_articles]
        
        if len(filtered_pool) < count:
            shown_articles = set()
            filtered_pool = news_df.index.tolist()
            if categories:
                filtered_pool = news_df[news_df['category'].isin(categories)].index.tolist()
        
        liked_articles = session.get('likes', history)  
        
        if len(filtered_pool) == 0:
            logger.warning("Filtered pool empty, falling back to global CANDIDATE_POOL")
            pool = CANDIDATE_POOL
        else:
            pool = filtered_pool

        if len(pool) == 0:
            logger.error("No candidates available in pool!")
            return jsonify({'error': 'No articles available'}), 500

        user_preferred_cats = []
        if categories and len(categories) > 0:
            user_preferred_cats = categories
            logger.info(f"Using EXPLICIT preferred categories from request: {user_preferred_cats}")
        elif liked_articles and len(liked_articles) >= 2:  
            cat_counts = {}
            for nid in liked_articles:  
                if nid in news_df.index:
                    cat = news_df.loc[nid]['category']
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1
            if cat_counts:
                sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
                total_likes = sum(cat_counts.values())

                threshold_count = max(5, int(total_likes * 0.10))
                user_preferred_cats = [cat for cat, count in sorted_cats if count >= threshold_count]
                
                if len(user_preferred_cats) < 2 and len(sorted_cats) >= 2:
                    user_preferred_cats = [cat for cat, _ in sorted_cats[:3]]
                    logger.info(f"Only 1 category met threshold, including top 3 categories for diversity")
                
                logger.info(f"LEARNED preferred categories from history: {user_preferred_cats} from distribution {dict(sorted_cats[:10])}")
                logger.info(f"Threshold used: {threshold_count} likes (10% of {total_likes} total)")

        k = min(100, len(pool))
        if user_preferred_cats and len(liked_articles) >= 2:
            candidates = []
            per_category = max(15, k // len(user_preferred_cats))  # At least 15 per category
            
            logger.info(f"Sampling {per_category} candidates per category from {user_preferred_cats}")
            for cat in user_preferred_cats:
                cat_pool = [nid for nid in pool if nid in news_df.index and news_df.loc[nid]['category'] == cat]
                logger.info(f"Category '{cat}': {len(cat_pool)} articles available in pool")
                if cat_pool:
                    n_sample = min(per_category, len(cat_pool))
                    candidates.extend(random.sample(cat_pool, n_sample))
                    logger.info(f"Category '{cat}': sampled {n_sample} articles")
                else:
                    logger.warning(f"Category '{cat}': NO articles found in pool!")
            
            other_pool = [nid for nid in pool if nid in news_df.index and news_df.loc[nid]['category'] not in user_preferred_cats]
            if other_pool and len(candidates) < k:
                n_other = min(max(5, k // 10), len(other_pool), k - len(candidates))
                candidates.extend(random.sample(other_pool, n_other))
            
            sampled_cats = {}
            for cid in candidates:
                if cid in news_df.index:
                    cat = news_df.loc[cid]['category']
                    sampled_cats[cat] = sampled_cats.get(cat, 0) + 1
            logger.info(f"Balanced sampling: {sampled_cats} (total: {len(candidates)} candidates)")
        else:
            candidates = random.sample(pool, k)
            logger.debug(f"Random sampling: {len(candidates)} candidates from pool of {len(pool)}")
        try:
            state = state_builder.build_state(
                history_news_ids=liked_articles,  # Use liked articles for user preference
                timestamp=pd.Timestamp.now(),
                session_length=len(session['interactions']),
                session_clicks=session['clicks'],
                session_impressions=session['impressions']
            )
            logger.debug(f"State built: history_length={len(liked_articles)}, clicks={session['clicks']}, impressions={session['impressions']}")
        except Exception as e:
            logger.error(f"Error building state: {e}", exc_info=True)
            return jsonify({'error': f'Failed to build state: {str(e)}'}), 500
        
        #mock_env = MockEnv(candidates, article_encoder)
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            candidate_embs = article_encoder.get_embeddings(candidates)
            if candidate_embs is None or len(candidate_embs) == 0:
                logger.error("Failed to get candidate embeddings!")
                return jsonify({'error': 'Failed to get article embeddings'}), 500
            
            candidate_tensor = torch.FloatTensor(candidate_embs)
            state_repeated = state_tensor.repeat(len(candidates), 1)
        except Exception as e:
            logger.error(f"Error preparing tensors: {e}", exc_info=True)
            return jsonify({'error': f'Failed to prepare data: {str(e)}'}), 500
        
        try:
            with torch.no_grad():
                if isinstance(agent, SupervisedAgent):
                    # SupervisedAgent: model outputs click probability (0-1)
                    scores = agent.model(state_repeated, candidate_tensor).squeeze()
                    if scores.dim() == 0:
                        scores = scores.unsqueeze(0)
                    
                    # For category-based: also add direct similarity bonus
                    # User state is category distribution, article is one-hot
                    # Dot product gives preference score
                    if hasattr(article_encoder, 'num_categories'):
                        similarity_scores = torch.FloatTensor([
                            np.dot(state[:article_encoder.num_categories], emb[:article_encoder.num_categories])
                            for emb in candidate_embs
                        ])
                        # Combine model score with similarity (weighted average)
                        scores = 0.5 * scores + 0.5 * similarity_scores
                        logger.debug(f"Combined scores: model + similarity")
                elif hasattr(agent, 'policy_net'):
                    # For DQN/Dueling DQN agents: use select_action in eval mode to get Q-values
                    # First, get Q-values for all candidates
                    if hasattr(agent, 'select_action'):
                        # Use the agent's select_action method to get Q-values properly
                        # We'll compute Q-values for all candidates
                        k = len(candidates)
                        state_t = torch.FloatTensor(state).unsqueeze(0)
                        cand_t = torch.FloatTensor(candidate_embs)
                        state_rep = state_t.repeat(k, 1)
                        
                        # Get Q-values from policy network (in eval mode, no exploration)
                        scores = agent.policy_net(state_rep, cand_t).squeeze()
                        if scores.dim() == 0:
                            scores = scores.unsqueeze(0)
                    else:
                        # Fallback: direct network call
                        scores = agent.policy_net(state_repeated, candidate_tensor).squeeze()
                        if scores.dim() == 0:
                            scores = scores.unsqueeze(0)
                else:
                    try:
                        scores = []
                        for cand_id in candidates:
                            single_env = MockEnv([cand_id], article_encoder)
                            cand_emb = article_encoder.get_embeddings([cand_id])[0]
                            state_t = torch.FloatTensor(state).unsqueeze(0)
                            cand_t = torch.FloatTensor(cand_emb).unsqueeze(0)
                            state_rep = state_t.repeat(1, 1)
                            
                            # Try to get score from agent's model
                            if hasattr(agent, 'model'):
                                score = agent.model(state_rep, cand_t).item()
                            elif hasattr(agent, 'policy_net'):
                                score = agent.policy_net(state_rep, cand_t).item()
                            else:
                                score = 0.0
                            scores.append(score)
                        scores = torch.FloatTensor(scores)
                    except Exception as e:
                        logger.warning(f"Error scoring candidates: {e}")
                        # Random fallback
                        scores = torch.rand(len(candidates))
        except Exception as e:
            logger.error(f"Error in scoring: {e}", exc_info=True)
            scores = torch.rand(len(candidates))
        
        if isinstance(scores, torch.Tensor):
            scores_np = scores.cpu().numpy() if scores.is_cuda else scores.numpy()
            ranked_indices = np.argsort(scores_np)[::-1]  # Descending order
        else:
            scores_np = np.array(scores)
            ranked_indices = np.argsort(scores_np)[::-1]
        
        if user_preferred_cats and len(user_preferred_cats) >= 2:
            logger.info(f"Enforcing BALANCED distribution for {len(user_preferred_cats)} preferred categories: {user_preferred_cats}")
            diverse_indices = []
            category_counts_so_far = {cat: 0 for cat in user_preferred_cats}
            target_per_category = count // len(user_preferred_cats)  # Equal split
            category_pools = {cat: [] for cat in user_preferred_cats}
            other_pool = []
            
            for idx in ranked_indices:
                cand_id = candidates[idx]
                if cand_id in news_df.index:
                    cand_cat = news_df.loc[cand_id]['category']
                    if cand_cat in user_preferred_cats:
                        category_pools[cand_cat].append(idx)
                    else:
                        other_pool.append(idx)
            
            pool_sizes = {cat: len(pool) for cat, pool in category_pools.items()}
            logger.info(f"Category pool sizes BEFORE selection: {pool_sizes}")     
            max_iterations = count * 2  # Safety limit
            iteration = 0
            while len(diverse_indices) < count and iteration < max_iterations:
                iteration += 1
                added_this_round = False
                for cat in user_preferred_cats:
                    if len(diverse_indices) >= count:
                        break
                    if category_pools[cat]:  
                        diverse_indices.append(category_pools[cat].pop(0))
                        category_counts_so_far[cat] += 1
                        added_this_round = True
                
                if not added_this_round:
                    logger.warning(f"No more candidates in preferred categories after {len(diverse_indices)} selections")
                    break
            while len(diverse_indices) < count and other_pool:
                diverse_indices.append(other_pool.pop(0))
            
            ranked_indices = np.array(diverse_indices)
            logger.info(f"BALANCED recommendations: {category_counts_so_far} (target: {target_per_category} per category)")
        if len(scores_np) > 0:
            top_scores = scores_np[ranked_indices[:5]]
            mean_score = float(scores_np.mean())
            std_score = float(scores_np.std())
            max_score = float(scores_np.max())
            min_score = float(scores_np.min())
            top_ids_for_log = [candidates[i] for i in ranked_indices[:count]]
            top_categories = [news_df.loc[nid]['category'] for nid in top_ids_for_log if nid in news_df.index]
            category_counts = {}
            for cat in top_categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1
            user_cat_dist = {}
            if liked_articles:
                for nid in liked_articles[-50:]:
                    if nid in news_df.index:
                        cat = news_df.loc[nid]['category']
                        user_cat_dist[cat] = user_cat_dist.get(cat, 0) + 1
            
            logger.info(f"Model: {type(agent).__name__} | Top 5 scores: {top_scores} | Mean: {mean_score:.4f} | Std: {std_score:.4f} | Range: [{min_score:.4f}, {max_score:.4f}]")
            logger.info(f"USER HISTORY: {len(liked_articles)} likes → Top categories: {sorted(user_cat_dist.items(), key=lambda x: x[1], reverse=True)[:5]}")
            logger.info(f"RECOMMENDATIONS: {category_counts} ← Should match user history!")
            
            if hasattr(article_encoder, 'categories'):
                user_state_cat_probs = state[:article_encoder.num_categories]
                top_state_indices = np.argsort(user_state_cat_probs)[-5:][::-1]
                top_state_cats = [(article_encoder.categories[idx], user_state_cat_probs[idx]) for idx in top_state_indices if user_state_cat_probs[idx] > 0.01]
                logger.info(f"USER STATE (category distribution): {top_state_cats}")
            
            # Track model usage statistics
            model_usage_stats['total_requests'] += 1
            model_name = type(agent).__name__
            if model_name not in model_usage_stats['model_calls']:
                model_usage_stats['model_calls'][model_name] = 0
            model_usage_stats['model_calls'][model_name] += 1
            model_usage_stats['avg_scores'].append(mean_score)
            model_usage_stats['avg_q_values'].append(max_score)  # Best Q-value
        
        top_ids = [candidates[i] for i in ranked_indices[:count]]
        article_predictions = {}
        for idx, news_id in enumerate(top_ids):
            rank = ranked_indices[idx]
            predicted_score = float(scores_np[rank])
            article_predictions[news_id] = {
                'predicted_score': predicted_score,
                'rank': idx + 1,
                'timestamp': datetime.now().isoformat()
            }
        
        if 'shown_articles' not in session:
            session['shown_articles'] = []
        if 'article_predictions' not in session:
            session['article_predictions'] = {}
        
        session['shown_articles'].extend(top_ids)
        session['article_predictions'].update(article_predictions)
        recommendations = []
        for news_id in top_ids:
            try:
                article = news_df.loc[news_id]
                recommendations.append({
                    'news_id': news_id,
                    'title': str(article['title']),
                    'abstract': str(article.get('abstract', '') or ''),
                    'category': str(article['category']),
                    'subcategory': str(article.get('subcategory', '')),
                    'url': str(article.get('url', ''))
                })
            except Exception as e:
                logger.warning(f"Error formatting article {news_id}: {e}")
                continue
        
        if len(recommendations) == 0:
            logger.error("No recommendations generated!")
            return jsonify({'error': 'No recommendations could be generated'}), 500
        
        logger.info(f"Returning {len(recommendations)} recommendations")
        session['last_state'] = state
        session['last_candidates'] = candidate_embs 
        session['last_candidate_ids'] = candidates   
        user_category_prefs = {}
        if liked_articles:
            for nid in liked_articles[-20:]:  # Last 20 likes
                if nid in news_df.index:
                    cat = news_df.loc[nid]['category']
                    user_category_prefs[cat] = user_category_prefs.get(cat, 0) + 1
        
        return jsonify({
            'recommendations': recommendations,
            'session_id': user_id,
            'debug': {
                'user_likes_count': len(liked_articles),
                'user_top_categories': dict(sorted(user_category_prefs.items(), key=lambda x: x[1], reverse=True)[:3]),
                'recommended_categories': category_counts
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {e}", exc_info=True)
        return jsonify({'error': f'Failed to generate recommendations: {str(e)}'}), 500

@app.route('/api/interaction', methods=['POST'])
def record_interaction():
    """Record user interaction (like/dislike) and compute RL reward."""
    data = request.json
    user_id = data.get('user_id', 'default')
    news_id = data.get('news_id')
    action = data.get('action')  
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            'history': [],
            'likes': [],
            'dislikes': [],
            'impressions': 0,
            'clicks': 0,
            'interactions': [],
            'rewards': [],
            'total_reward': 0.0,
            'prediction_stats': {
                'correct': 0,
                'wrong': 0,
                'missed_opportunities': 0,
                'correct_avoidances': 0
            }
        }

    session = user_sessions[user_id]
    session.setdefault('rewards', [])
    session.setdefault('total_reward', 0.0)
    session.setdefault('prediction_stats', {
        'correct': 0,
        'wrong': 0,
        'missed_opportunities': 0,
        'correct_avoidances': 0
    })
    session.setdefault('likes', [])
    session.setdefault('dislikes', [])
    session.setdefault('history', [])
    session.setdefault('interactions', [])
    
    predicted_score = None
    if 'article_predictions' in session and news_id in session['article_predictions']:
        predicted_score = session['article_predictions'][news_id].get('predicted_score')
    if len(model_usage_stats['avg_scores']) > 0:
        score_threshold = np.median(model_usage_stats['avg_scores'])
    else:
        score_threshold = 0.5 if isinstance(agent, SupervisedAgent) else 0.0
    
    if predicted_score is not None:
        predicted_positive = predicted_score > score_threshold
        
    if action == "like":
        reward = 1.0
    else:
        reward = -1.0  

    prediction_accuracy = "na"  

    
    if action == 'like':
        if news_id not in session['likes']:
            session['likes'].append(news_id)
            session['history'].append(news_id) 
            session['clicks'] += 1
            session['rewards'].append(reward)
            session['total_reward'] += reward
            model_usage_stats['total_rewards'] += reward
            
            pred_info = f"Predicted: {predicted_score:.4f}" if predicted_score is not None else "No prediction"
            logger.info(f"User {user_id} liked article {news_id}. Reward: {reward:+.2f} ({prediction_accuracy}) | {pred_info} | Total reward: {session['total_reward']:.2f}")
    elif action == 'dislike':
        if news_id not in session['dislikes']:
            session['dislikes'].append(news_id)
            if news_id in session['history']:
                session['history'].remove(news_id)
            if news_id in session['likes']:
                session['likes'].remove(news_id)
            session['rewards'].append(reward)
            session['total_reward'] += reward
            pred_info = f"Predicted: {predicted_score:.4f}" if predicted_score is not None else "No prediction"
            logger.info(f"User {user_id} disliked article {news_id}. Reward: {reward:+.2f} ({prediction_accuracy}) | {pred_info} | Total reward: {session['total_reward']:.2f}")
    
    interaction_data = {
        'news_id': news_id,
        'action': action,
        'reward': reward,
        'predicted_score': predicted_score,
        'prediction_accuracy': prediction_accuracy,
        'timestamp': datetime.now().isoformat()
    }
    session['interactions'].append(interaction_data)
    
    # Track prediction accuracy for model evaluation
    if 'prediction_stats' not in session:
        session['prediction_stats'] = {
            'correct': 0,
            'wrong': 0,
            'missed_opportunities': 0,
            'correct_avoidances': 0
        }
    
    if prediction_accuracy == 'correct':
        session['prediction_stats']['correct'] += 1
    elif prediction_accuracy == 'wrong_prediction':
        session['prediction_stats']['wrong'] += 1
    elif prediction_accuracy == 'missed_opportunity':
        session['prediction_stats']['missed_opportunities'] += 1
    elif prediction_accuracy == 'correct_avoidance':
        session['prediction_stats']['correct_avoidances'] += 1
    
    
    if isinstance(agent, DQNAgent):

        prev_state = session.get("last_state")
        prev_candidates = session.get("last_candidates")       
        prev_candidate_ids = session.get("last_candidate_ids")  

        if prev_state is not None and prev_candidates is not None and prev_candidate_ids is not None:
            try:
                idx = prev_candidate_ids.index(news_id)
                action_emb = prev_candidates[idx]
            except ValueError:
                action_emb = article_encoder.get_embeddings([news_id])[0]
            next_state = state_builder.build_state(
                history_news_ids=session['likes'],
                timestamp=pd.Timestamp.now(),
                session_length=len(session['interactions']),
                session_clicks=session['clicks'],
                session_impressions=session['impressions'],
            )
            next_ids = session['last_candidate_ids']
            next_candidate_embs = session['last_candidates']

            done = False  
            agent.memory.push(prev_state, action_emb, reward, next_state, next_candidate_embs, done)
            loss = agent.update(batch_size=32)
            agent.update_epsilon()
            if len(session['interactions']) % 100 == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
                logger.info("Updated Target Network")

            logger.info(f"[RL] Loss = {loss:.4f}, Epsilon = {agent.epsilon:.3f}")
    if isinstance(agent, SupervisedAgent) or hasattr(agent, 'model'):
        try:
            prev_state = session.get('last_state')
            prev_candidates = session.get('last_candidates')
            prev_candidate_ids = session.get('last_candidate_ids')

            if prev_state is not None and (prev_candidates is not None or prev_candidate_ids is not None):
                try:
                    if prev_candidate_ids and news_id in prev_candidate_ids:
                        idx = prev_candidate_ids.index(news_id)
                        cand_emb = prev_candidates[idx]
                    else:
                        cand_emb = article_encoder.get_embeddings([news_id])[0]
                except Exception:
                    cand_emb = article_encoder.get_embeddings([news_id])[0]
                state_t = torch.FloatTensor(prev_state).unsqueeze(0)
                cand_t = torch.FloatTensor(cand_emb).unsqueeze(0)
                label = 1.0 if action == 'like' else 0.0
                label_t = torch.FloatTensor([[label]])
                if hasattr(agent, 'model'):
                    agent.model.train()
                    optim = getattr(agent, 'optimizer', None)
                    if optim is None:
                        try:
                            agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=1e-4)
                            optim = agent.optimizer
                        except Exception:
                            optim = None

                    if optim is not None:
                        optim.zero_grad()
                        pred = agent.model(state_t, cand_t)
                        loss_fn = torch.nn.BCELoss()
                        loss = loss_fn(pred, label_t)
                        loss.backward()
                        optim.step()
                        logger.info(f"[Online Supervised] Updated model for user {user_id} | loss={loss.item():.4f}")
        except Exception as e:
            logger.warning(f"Online supervised update failed: {e}", exc_info=True)
    return jsonify({
        'status': 'success',
        'reward': round(reward, 2),
        'total_reward': round(session['total_reward'], 2),
        'predicted_score': round(predicted_score, 4) if predicted_score is not None else None,
        'prediction_accuracy': prediction_accuracy,
        'model': type(agent).__name__ if agent is not None else None
    })



@app.route('/api/metrics/<user_id>', methods=['GET'])
def get_metrics(user_id):
    """Get metrics for a user session."""
    if user_id not in user_sessions:
        return jsonify({
            'clicks': 0,
            'impressions': 0,
            'ctr': 0.0,
            'likes': 0,
            'dislikes': 0,
            'history_length': 0,
            'category_distribution': {},
            'interactions_over_time': [],
            'total_reward': 0.0,
            'avg_reward': 0.0,
            'recent_rewards': [],
            'prediction_stats': {
                'correct': 0,
                'wrong': 0,
                'missed_opportunities': 0,
                'correct_avoidances': 0,
                'accuracy': 0.0
            }
        })
    
    session = user_sessions[user_id]
    clicks = session['clicks']
    impressions = session['impressions']
    ctr = (clicks / impressions * 100) if impressions > 0 else 0.0
    
    # Category distribution
    category_dist = {}
    if session['history']:
        liked_articles = news_df.loc[session['history']]
        category_dist = liked_articles['category'].value_counts().to_dict()
    
    # Interactions over time (last 20)
    interactions_over_time = session['interactions'][-20:]
    
    # RL Metrics
    total_reward = session.get('total_reward', 0.0)
    rewards = session.get('rewards', [])
    avg_reward = sum(rewards) / len(rewards) if len(rewards) > 0 else 0.0
    recent_rewards = rewards[-10:] if len(rewards) > 0 else []
    
    # Prediction accuracy metrics
    pred_stats = session.get('prediction_stats', {
        'correct': 0,
        'wrong': 0,
        'missed_opportunities': 0,
        'correct_avoidances': 0
    })
    total_predictions = pred_stats['correct'] + pred_stats['wrong'] + pred_stats['missed_opportunities'] + pred_stats['correct_avoidances']
    accuracy = (pred_stats['correct'] + pred_stats['correct_avoidances']) / total_predictions if total_predictions > 0 else 0.0
    
    return jsonify({
        'clicks': clicks,
        'impressions': impressions,
        'ctr': round(ctr, 2),
        'likes': len(session['likes']),
        'dislikes': len(session['dislikes']),
        'history_length': len(session['history']),
        'category_distribution': category_dist,
        'interactions_over_time': interactions_over_time,
        'total_reward': round(total_reward, 2),
        'avg_reward': round(avg_reward, 2),
        'recent_rewards': recent_rewards,
        'reward_history_length': len(rewards),
        'prediction_stats': {
            **pred_stats,
            'accuracy': round(accuracy * 100, 2),  # As percentage
            'total_predictions': total_predictions
        }
    })


@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all available categories."""
    if news_df is None:
        return jsonify({'categories': []})
    
    categories = sorted(news_df['category'].unique().tolist())
    return jsonify({'categories': categories})


@app.route('/api/article/<news_id>', methods=['GET'])
def get_article(news_id):
    """Get full article details."""
    if news_df is None or news_id not in news_df.index:
        return jsonify({'error': 'Article not found'}), 404
    
    article = news_df.loc[news_id]
    return jsonify({
        'news_id': news_id,
        'title': article['title'],
        'abstract': article.get('abstract', '') or '',
        'category': article['category'],
        'subcategory': article.get('subcategory', ''),
        'url': article.get('url', '')
    })


@app.route('/api/model-stats', methods=['GET'])
def get_model_stats():
    """Get model usage statistics."""
    global agent
    
    current_model = type(agent).__name__ if agent else 'None'
    
    # Calculate averages
    avg_score = sum(model_usage_stats['avg_scores']) / len(model_usage_stats['avg_scores']) if model_usage_stats['avg_scores'] else 0.0
    avg_q_value = sum(model_usage_stats['avg_q_values']) / len(model_usage_stats['avg_q_values']) if model_usage_stats['avg_q_values'] else 0.0
    
    return jsonify({
        'current_model': current_model,
        'total_requests': model_usage_stats['total_requests'],
        'model_calls': model_usage_stats['model_calls'],
        'total_rewards': model_usage_stats['total_rewards'],
        'avg_score': round(avg_score, 4),
        'avg_q_value': round(avg_q_value, 4),
        'total_predictions': len(model_usage_stats['avg_scores'])
    })


@app.route('/api/debug/recommendations', methods=['POST'])
def debug_recommendations():
    """Debug endpoint to see what the model is scoring."""
    global agent, state_builder, article_encoder, news_df
    
    data = request.json
    user_id = data.get('user_id', 'default')
    history = data.get('history', [])
    count = data.get('count', 5)
    
    if user_id not in user_sessions:
        return jsonify({'error': 'User session not found'}), 404
    
    session = user_sessions[user_id]
    liked_articles = session.get('likes', history)
    filtered_pool = news_df.index.tolist()
    candidates = random.sample(filtered_pool, min(20, len(filtered_pool)))
    state = state_builder.build_state(
        history_news_ids=liked_articles,
        timestamp=pd.Timestamp.now(),
        session_length=len(session['interactions']),
        session_clicks=session['clicks'],
        session_impressions=session['impressions']
    )
    candidate_embs = article_encoder.get_embeddings(candidates)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    candidate_tensor = torch.FloatTensor(candidate_embs)
    state_repeated = state_tensor.repeat(len(candidates), 1)
    
    with torch.no_grad():
        if isinstance(agent, SupervisedAgent):
            scores = agent.model(state_repeated, candidate_tensor).squeeze()
        elif hasattr(agent, 'policy_net'):
            scores = agent.policy_net(state_repeated, candidate_tensor).squeeze()
        else:
            scores = torch.rand(len(candidates))
    
    scores_np = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores
    ranked_indices = np.argsort(scores_np)[::-1]
    debug_info = {
        'user_history_length': len(liked_articles),
        'num_candidates': len(candidates),
        'model_type': type(agent).__name__,
        'scores': {
            'mean': float(scores_np.mean()),
            'std': float(scores_np.std()),
            'min': float(scores_np.min()),
            'max': float(scores_np.max())
        },
        'top_candidates': []
    }
    
    for i in ranked_indices[:count]:
        article = news_df.loc[candidates[i]]
        debug_info['top_candidates'].append({
            'news_id': candidates[i],
            'title': article['title'][:50],
            'category': article['category'],
            'score': float(scores_np[i])
        })
    
    return jsonify(debug_info)


@app.route('/api/validate-learning/<user_id>', methods=['GET'])
def validate_learning(user_id):
    """Validate that model learned user preferences by comparing scores for different categories."""
    global agent, state_builder, article_encoder, news_df
    
    if user_id not in user_sessions:
        return jsonify({'error': 'User session not found'}), 404
    
    session = user_sessions[user_id]
    liked_articles = session.get('likes', [])
    
    if len(liked_articles) < 10:
        return jsonify({'error': 'Need at least 10 likes to validate learning'}), 400
    user_category_counts = {}
    for nid in liked_articles:
        if nid in news_df.index:
            cat = news_df.loc[nid]['category']
            user_category_counts[cat] = user_category_counts.get(cat, 0) + 1
    
    preferred_cats = sorted(user_category_counts.items(), key=lambda x: x[1], reverse=True)[:2]
    preferred_cat_names = [cat for cat, _ in preferred_cats]
    preferred_articles = []
    other_articles = []
    
    for cat in news_df['category'].unique():
        cat_articles = news_df[news_df['category'] == cat].index.tolist()
        sample_size = min(10, len(cat_articles))
        sampled = random.sample(cat_articles, sample_size)
        
        if cat in preferred_cat_names:
            preferred_articles.extend(sampled)
        else:
            other_articles.extend(sampled[:5])  # fewer from other cats
    
    state = state_builder.build_state(
        history_news_ids=liked_articles,
        timestamp=pd.Timestamp.now(),
        session_length=len(session['interactions']),
        session_clicks=session['clicks'],
        session_impressions=session['impressions']
    )
    
    pref_embs = article_encoder.get_embeddings(preferred_articles)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    pref_tensor = torch.FloatTensor(pref_embs)
    state_rep_pref = state_tensor.repeat(len(preferred_articles), 1)
    
    with torch.no_grad():
        if isinstance(agent, SupervisedAgent):
            pref_scores = agent.model(state_rep_pref, pref_tensor).squeeze()
        elif hasattr(agent, 'policy_net'):
            pref_scores = agent.policy_net(state_rep_pref, pref_tensor).squeeze()
        else:
            pref_scores = torch.rand(len(preferred_articles))
    
    pref_scores_np = pref_scores.cpu().numpy() if isinstance(pref_scores, torch.Tensor) else pref_scores
    other_embs = article_encoder.get_embeddings(other_articles)
    other_tensor = torch.FloatTensor(other_embs)
    state_rep_other = state_tensor.repeat(len(other_articles), 1)
    
    with torch.no_grad():
        if isinstance(agent, SupervisedAgent):
            other_scores = agent.model(state_rep_other, other_tensor).squeeze()
        elif hasattr(agent, 'policy_net'):
            other_scores = agent.policy_net(state_rep_other, other_tensor).squeeze()
        else:
            other_scores = torch.rand(len(other_articles))
    
    other_scores_np = other_scores.cpu().numpy() if isinstance(other_scores, torch.Tensor) else other_scores
    
    return jsonify({
        'user_id': user_id,
        'user_likes': len(liked_articles),
        'preferred_categories': preferred_cat_names,
        'preferred_category_counts': dict(preferred_cats),
        'preferred_articles_scores': {
            'mean': float(pref_scores_np.mean()),
            'median': float(np.median(pref_scores_np)),
            'std': float(pref_scores_np.std()),
            'min': float(pref_scores_np.min()),
            'max': float(pref_scores_np.max())
        },
        'other_articles_scores': {
            'mean': float(other_scores_np.mean()),
            'median': float(np.median(other_scores_np)),
            'std': float(other_scores_np.std()),
            'min': float(other_scores_np.min()),
            'max': float(other_scores_np.max())
        },
        'score_difference': {
            'mean_diff': float(pref_scores_np.mean() - other_scores_np.mean()),
            'median_diff': float(np.median(pref_scores_np) - np.median(other_scores_np))
        },
        'learning_indicator': 'GOOD' if (pref_scores_np.mean() - other_scores_np.mean()) > 0.1 else 'WEAK' if (pref_scores_np.mean() - other_scores_np.mean()) > 0 else 'NOT LEARNING'
    })


if __name__ == '__main__':
    load_resources()
    app.run(debug=True, port=5000, host='0.0.0.0')

