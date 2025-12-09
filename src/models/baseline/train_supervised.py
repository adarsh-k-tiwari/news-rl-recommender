# src/models/train_supervised.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
from pathlib import Path
import logging

from src.models.baseline.supervised import ClickPredictor, SupervisedDataset
from src.features.article_encoder import ArticleEncoder
from src.features.user_encoder import UserEncoder
from src.features.state_builder import StateBuilder

# Add logging
logging.basicConfig(level=logging.INFO)

def train_supervised_baseline():
    # Setup
    data_dir = Path('data/processed')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading Encoders...")
    # 1. Article Encoder
    article_encoder = ArticleEncoder(
        embedding_dim=384, 
        cache_dir=data_dir / 'embeddings'
    )
    article_encoder._load_cache()

    # 2. User Encoder
    user_encoder = UserEncoder(
        embedding_dim=384,
        aggregation_method='mean'
    )

    # 3. State Builder (Required since preprocess skipped state computation)
    state_builder = StateBuilder(
        article_encoder=article_encoder,
        user_encoder=user_encoder,
        embedding_dim=384
    )
    
    print("Loading Training Sessions...")
    session_path = data_dir / 'sessions/train_sessions.pkl'
    if not session_path.exists():
        raise FileNotFoundError(f"Could not find {session_path}")
        
    with open(session_path, 'rb') as f:
        sessions = pickle.load(f)
    
    # Create Dataset 
    # Pass state_builder to calculate states on-the-fly
    print("Creating Dataset...")
    dataset = SupervisedDataset(
        sessions, 
        article_encoder, 
        state_builder=state_builder,
        max_samples=10000000
    )
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty! Check if sessions loaded correctly.")

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize Model
    # State dim = 384 (user) + 3 (time) + 4 (session) = 391
    model = ClickPredictor(state_dim=391, article_emb_dim=384).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    # Training Loop
    print(f"Starting Training on {len(dataset)} samples...")
    model.train()
    for epoch in range(5):
        total_loss = 0
        batch_count = 0
        for states, articles, labels in dataloader:
            states, articles, labels = states.to(device), articles.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(states, articles)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    # Save Model
    save_path = data_dir / 'supervised_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_supervised_baseline()