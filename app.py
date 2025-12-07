# app.py
import streamlit as st
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import random

# Import your existing modules
from src.features.article_encoder import ArticleEncoder, CategoryEncoder
from src.features.user_encoder import UserEncoder
from src.features.state_builder import StateBuilder
from src.models.baseline.supervised import ClickPredictor, SupervisedAgent
# from src.models.dqn import DQNAgent # (Uncomment when you have the DQN agent)

# --- 1. CONFIGURATION & CACHING ---
st.set_page_config(page_title="Personalized News RecSys", layout="wide")

@st.cache_resource
def load_resources():
    """Load models and data once to improve performance."""
    data_dir = Path('data/processed')
    
    # 1. Load Article Encoder (Cached embeddings)
    st.write("Loading resources... (this may take a minute)")
    article_encoder = ArticleEncoder(
        embedding_dim=384, 
        cache_dir=data_dir / 'embeddings'
    )
    article_encoder._load_cache()
    
    # 2. Load Metadata (News Dataframe) to display titles/abstracts
    # We need the raw text to show to the user
    # Assuming you have a way to get the original dataframe, or reload it here:
    from src.data.mind_loader import MINDDataLoader
    loader = MINDDataLoader(data_dir='data/raw/dev', dataset_type='dev')
    loader.load_news()
    news_df = loader.news_df.set_index('news_id')
    
    # 3. Load Models
    # Initialize User Encoder & State Builder
    user_encoder = UserEncoder(embedding_dim=384, aggregation_method='mean')
    state_builder = StateBuilder(article_encoder, user_encoder, embedding_dim=384)
    
    # Load Trained Agent (Supervised for now)
    model = ClickPredictor(state_dim=391, article_emb_dim=384)
    try:
        model.load_state_dict(torch.load('data/processed/supervised_model.pth', map_location='cpu'))
        model.eval()
    except FileNotFoundError:
        st.error("Model file not found! Run train_supervised.py first.")
    
    agent = SupervisedAgent(model, article_encoder)
    
    return article_encoder, news_df, state_builder, agent

# Load everything
article_encoder, news_df, state_builder, agent = load_resources()

# --- 2. SESSION STATE MANAGEMENT ---
# This persists variables across button clicks
if 'history' not in st.session_state:
    st.session_state.history = [] # List of news_ids
if 'clicks' not in st.session_state:
    st.session_state.clicks = 0
if 'impressions' not in st.session_state:
    st.session_state.impressions = 0
if 'current_candidates' not in st.session_state:
    st.session_state.current_candidates = []

# --- 3. HELPER FUNCTIONS ---
def get_recommendation():
    """
    1. Sample random candidates from the news pool.
    2. Use the Agent to rank them based on current user history.
    3. Return the best one.
    """
    # 1. Sample 50 random articles as candidates
    all_news_ids = list(news_df.index)
    candidates = random.sample(all_news_ids, 50)
    
    # 2. Build Current State
    # Note: We are mocking the session features for this demo
    state = state_builder.build_state(
        history_news_ids=st.session_state.history,
        timestamp=pd.Timestamp.now(),
        session_length=st.session_state.impressions,
        session_clicks=st.session_state.clicks,
        session_impressions=st.session_state.impressions
    )
    
    # 3. Agent Prediction
    # Mocking the env object just to pass candidates to the agent
    class MockEnv:
        def __init__(self, cands): self.current_candidates = cands
        def get_candidate_embeddings(self): return article_encoder.get_embeddings(cands)
    
    mock_env = MockEnv(candidates)
    best_idx = agent.predict(state, mock_env)
    recommended_id = candidates[best_idx]
    
    return recommended_id, state

# --- 4. MAIN UI LAYOUT ---
st.title("🤖 Personal News Feed")
st.markdown("This system learns from your clicks using **Deep Learning**.")

# Sidebar for Stats
with st.sidebar:
    st.header("User Profile")
    st.metric("Articles Read", st.session_state.clicks)
    st.metric("Total Impressions", st.session_state.impressions)
    ctr = (st.session_state.clicks / st.session_state.impressions * 100) if st.session_state.impressions > 0 else 0
    st.metric("CTR", f"{ctr:.1f}%")
    
    st.subheader("Reading History")
    for news_id in reversed(st.session_state.history[-5:]): # Show last 5
        try:
            title = news_df.loc[news_id]['title']
            st.caption(f"📄 {title[:40]}...")
        except:
            pass

# Main Interaction Loop
if 'current_rec' not in st.session_state:
    # First run initialization
    rec_id, _ = get_recommendation()
    st.session_state.current_rec = rec_id

# Display the Current Article
rec_id = st.session_state.current_rec
try:
    article = news_df.loc[rec_id]
    
    st.divider()
    st.subheader(article['title'])
    st.info(f"Category: {article['category']}")
    st.write(article['abstract'])
    st.divider()
    
    col1, col2 = st.columns(2)
    
    def on_click(clicked):
        st.session_state.impressions += 1
        if clicked:
            st.session_state.clicks += 1
            st.session_state.history.append(rec_id)
            st.toast("Updated User Profile! 🧠")
        
        # Get NEXT recommendation
        next_id, _ = get_recommendation()
        st.session_state.current_rec = next_id

    with col1:
        st.button("📖 Read Article", on_click=on_click, args=(True,), use_container_width=True, type="primary")
    with col2:
        st.button("⏭️ Skip", on_click=on_click, args=(False,), use_container_width=True)

except KeyError:
    st.error("Error loading article metadata. Try refreshing.")
    # Reset if stuck
    next_id, _ = get_recommendation()
    st.session_state.current_rec = next_id