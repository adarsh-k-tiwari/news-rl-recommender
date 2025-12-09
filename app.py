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
st.title("📰 Personalized News Recommendation System")
st.caption("Powered by your RL-trained agent.")

st.markdown("### 🔎 Choose Categories You're Interested In")
all_categories = sorted(news_df["category"].unique())

default_cats = ["sports", "finance"]

selected_categories = st.multiselect(
    "Filter news by categories",
    options=all_categories,
    default=default_cats,
    help="This filter affects the candidate pool before the RL agent ranks articles."
)

# Build pool AND fallback if filter empty
if len(selected_categories) > 0:
    filtered_pool = news_df[news_df["category"].isin(selected_categories)].index.tolist()
else:
    filtered_pool = news_df.index.tolist()

#=============================
# 2. GRID-BASED NEWS CARDS
#=============================

st.markdown("## 🧠 Recommended For You")

# --- Replace sampling logic ---
def get_recommendations(count=6):
    """
    Return top-k recommended article IDs using the RL agent.
    """
    # Step 1: sample 50 candidates from filtered pool
    candidates = random.sample(filtered_pool, min(50, len(filtered_pool)))

    # Step 2: build state
    state = state_builder.build_state(
        history_news_ids=st.session_state.history,
        timestamp=pd.Timestamp.now(),
        session_length=st.session_state.impressions,
        session_clicks=st.session_state.clicks,
        session_impressions=st.session_state.impressions
    )

    # Step 3: RL agent ranks them
    class MockEnv:
        def __init__(self, cands): 
            self.current_candidates = cands
        def get_candidate_embeddings(self):
            return article_encoder.get_embeddings(self.current_candidates)

    mock_env = MockEnv(candidates)
    scores = agent.predict(state, mock_env)  # You may use agent.predict OR full scoring
    ranked = np.argsort(scores)[::-1]  # best to worst
    
    top_ids = [candidates[i] for i in ranked[:count]]
    return top_ids


#=============================
# 3. DISPLAY RECOMMENDATIONS
#=============================
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = get_recommendations()

cols = st.columns(3)  # 3 cards per row

for idx, news_id in enumerate(st.session_state.recommendations):
    article = news_df.loc[news_id]

    with cols[idx % 3]:
        st.markdown(
            f"""
            <div style='padding:15px; border-radius:10px; background:#f8f9fa;
                        border:1px solid #ddd; height:310px; display:flex; flex-direction:column; justify-content:space-between;'>
                <h4 style='margin-bottom:8px'>{article['title'][:80]}</h4>
                <span style='font-size:12px; color:#555;'>Category: <b>{article['category']}</b></span>
                <p style='font-size:13px; color:#333; height:80px; overflow:hidden;'>{article['abstract'][:150]}...</p>
            """,
            unsafe_allow_html=True
        )

        colA, colB = st.columns(2)

        def handle_click(clicked, news_id=news_id):
            st.session_state.impressions += 1
            if clicked:
                st.session_state.clicks += 1
                st.session_state.history.append(news_id)

            # refresh recommendations
            st.session_state.recommendations = get_recommendations()

        with colA:
            st.button("Read", key=f"read_{news_id}", on_click=handle_click, args=(True,))
        with colB:
            st.button("Skip", key=f"skip_{news_id}", on_click=handle_click, args=(False,))

        st.markdown("</div>", unsafe_allow_html=True)

#=============================
# 4. SIDEBAR PROFILE
#=============================

with st.sidebar:
    st.header("📊 Your Reader Profile")
    st.metric("Articles Read", st.session_state.clicks)
    st.metric("Total Impressions", st.session_state.impressions)
    ctr = (st.session_state.clicks / st.session_state.impressions * 100) if st.session_state.impressions > 0 else 0
    st.metric("CTR", f"{ctr:.2f}%")

    st.subheader("Recent Articles")
    for nid in reversed(st.session_state.history[-5:]):
        st.caption("• " + news_df.loc[nid]["title"][:50])

    # Optional small chart
    st.subheader("Category Distribution")
    if len(st.session_state.history) > 0:
        hist_df = news_df.loc[st.session_state.history]
        pie = hist_df["category"].value_counts()
        st.pyplot(pie.plot(kind='pie', autopct='%1.1f%%').figure)