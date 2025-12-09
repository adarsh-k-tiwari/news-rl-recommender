"""
Preprocessing Pipeline for MIND Dataset
Generates article embeddings and prepares data for RL training.
"""

import sys
sys.path.append('.')

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import yaml
from tqdm import tqdm
import logging

from src.data.mind_loader import MINDDataLoader
from src.features.article_encoder import ArticleEncoder, CategoryEncoder
from src.features.user_encoder import UserEncoder
from src.features.state_builder import StateBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess MIND dataset')
    parser.add_argument(
        '--data_root',
        type=str,
        default='data/raw'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        choices=['train', 'dev', 'test'],
        default=['train', 'dev']
    )
    parser.add_argument(
        '--nrows',
        type=int,
        default=None
    )
    parser.add_argument(
        '--embedding_model',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2'
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=384
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32
    )
    parser.add_argument(
        '--skip_state_computation',
        action='store_true'
    )
    parser.add_argument(
        '--skip_embeddings',
        action='store_true'
    )
    
    return parser.parse_args()


def create_directory_structure(output_dir: Path):
    """Create output directory structure."""
    dirs = [
        output_dir,
        output_dir / 'embeddings',
        output_dir / 'sessions',
        output_dir / 'metadata'
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created directory structure in {output_dir}")


def encode_all_articles(loaders, encoder, batch_size=32):
    """
    Encode all news articles from all splits (avoiding duplicates).
    
    Args:
        loaders: Dictionary of {split_name: MINDDataLoader}
        encoder: ArticleEncoder instance
        batch_size: Batch size
        
    Returns:
        Dictionary mapping news_id to embedding
    """
    logger.info("Collecting unique articles across all splits")
    
    # Combine all news articles (remove duplicates based on news_id)
    all_news = []
    seen_ids = set()
    
    for split_name, loader in loaders.items():
        for _, row in loader.news_df.iterrows():
            if row['news_id'] not in seen_ids:
                all_news.append(row.to_dict())
                seen_ids.add(row['news_id'])
    
    logger.info(f"Found {len(all_news)} unique articles across all splits")
    
    # Encode all articles
    embeddings = encoder.encode_articles_batch(
        all_news,
        batch_size=batch_size
    )
    
    logger.info(f"Encoded {len(embeddings)} articles")
    
    return embeddings


def create_category_encoder(loaders, output_dir):
    """Create and save category encoder from all splits."""
    logger.info("Creating category encoder")
    
    # Collect all unique categories
    all_categories = set()
    for loader in loaders.values():
        all_categories.update(loader.news_df['category'].unique())
    
    categories = sorted(list(all_categories))
    cat_encoder = CategoryEncoder(categories, embedding_dim=16)
    
    # Save
    save_path = output_dir / 'metadata' / 'category_encoder.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(cat_encoder, f)
    
    logger.info(f"Saved category encoder with {len(categories)} categories")
    
    return cat_encoder


def process_sessions_for_split(
    loader, 
    split_name, 
    article_encoder, 
    user_encoder, 
    state_builder, 
    output_dir,
    skip_state_computation=False
):
    """
    Process user behaviors into session data for a specific split.
    OPTIMIZED VERSION: Processes by impression rather than by user.
    
    Args:
        loader: MINDDataLoader for this split
        split_name: 'train', 'dev', or 'test'
        article_encoder: ArticleEncoder instance
        user_encoder: UserEncoder instance
        state_builder: StateBuilder instance
        output_dir: Output directory
        skip_state_computation: If True, don't compute states (much faster)
        
    Returns:
        List of processed sessions
    """
    logger.info(f"Processing {split_name} sessions (optimized)")
    logger.info(f"Total impressions to process: {len(loader.behaviors_df)}")
    
    if skip_state_computation:
        logger.info("Fast mode: Skipping state computation")
    
    # Parse all behaviors at once
    logger.info("Parsing impression data")
    loader.behaviors_df['history_list'] = loader.behaviors_df['history'].apply(
        loader.parse_history
    )
    loader.behaviors_df['impressions_list'] = loader.behaviors_df['impressions'].apply(
        loader.parse_impressions
    )
    
    # Group by user
    logger.info("Grouping by user")
    grouped = loader.behaviors_df.groupby('user_id')
    
    # Process each user's impressions
    sessions_data = []
    
    for user_id, user_df in tqdm(grouped, desc=f"Processing {split_name} users"):
        # Sort by timestamp
        user_df = user_df.sort_values('timestamp')
        
        session = {
            'user_id': user_id,
            'num_impressions': len(user_df),
            'impressions': []
        }
        
        session_clicks = 0
        session_impressions = 0
        
        for idx, row in user_df.iterrows():
            # Extract impression data
            candidates = [imp[0] for imp in row['impressions_list']]
            labels = [imp[1] for imp in row['impressions_list']]
            
            impression_data = {
                'impression_id': row['impression_id'],
                'timestamp': row['timestamp'],
                'history': row['history_list'],
                'candidates': candidates,
                'labels': labels,
                'num_candidates': len(candidates)
            }
            
            # Optionally build state
            if not skip_state_computation:
                state = state_builder.build_state(
                    history_news_ids=row['history_list'],
                    timestamp=row['timestamp'],
                    session_length=len(session['impressions']),
                    session_clicks=session_clicks,
                    session_impressions=session_impressions
                )
                impression_data['state'] = state
            
            session['impressions'].append(impression_data)
            
            # Update session stats
            session_clicks += sum(labels)
            session_impressions += len(candidates)
        
        sessions_data.append(session)
    
    logger.info(f"Processed {len(sessions_data)} {split_name} sessions")
    
    # Save sessions
    save_path = output_dir / 'sessions' / f'{split_name}_sessions.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(sessions_data, f)
    
    logger.info(f"Saved {split_name} sessions to {save_path}")
    
    return sessions_data


def generate_statistics(loaders, all_sessions, output_dir):
    """Generate and save dataset statistics for all splits."""
    logger.info("Generating statistics")
    
    stats = {}
    
    # Aggregate news statistics across all splits
    total_articles = sum(len(loader.news_df) for loader in loaders.values())
    all_categories = set()
    category_counts = {}
    
    for loader in loaders.values():
        all_categories.update(loader.news_df['category'].unique())
        for cat, count in loader.news_df['category'].value_counts().items():
            category_counts[cat] = category_counts.get(cat, 0) + count
    
    stats['news'] = {
        'num_articles': total_articles,
        'num_categories': len(all_categories),
        'category_distribution': category_counts
    }
    
    # Per-split statistics
    stats['splits'] = {}
    
    for split_name, loader in loaders.items():
        user_stats = loader.get_user_statistics()
        
        # Session statistics for this split
        split_sessions = all_sessions.get(split_name, [])
        session_lengths = [s['num_impressions'] for s in split_sessions]
        total_clicks = []
        
        for session in split_sessions:
            clicks = sum(sum(imp['labels']) for imp in session['impressions'])
            total_clicks.append(clicks)
        
        stats['splits'][split_name] = {
            'users': user_stats,
            'sessions': {
                'num_sessions': len(split_sessions),
                'avg_length': np.mean(session_lengths) if session_lengths else 0,
                'median_length': np.median(session_lengths) if session_lengths else 0,
                'avg_clicks': np.mean(total_clicks) if total_clicks else 0,
                'median_clicks': np.median(total_clicks) if total_clicks else 0
            }
        }
    
    # Save statistics
    save_path = output_dir / 'metadata' / 'statistics.yaml'
    with open(save_path, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    
    logger.info(f"Saved statistics to {save_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    print(f"\nNews Articles (total): {stats['news']['num_articles']:,}")
    print(f"Categories: {stats['news']['num_categories']}")
    
    for split_name in sorted(stats['splits'].keys()):
        split_stats = stats['splits'][split_name]
        print(f"\n--- {split_name.upper()} Split ---")
        print(f"Users: {split_stats['users']['num_users']:,}")
        print(f"Impressions: {split_stats['users']['num_impressions']:,}")
        print(f"CTR: {split_stats['users']['overall_ctr']*100:.2f}%")
        print(f"Sessions: {split_stats['sessions']['num_sessions']:,}")
        print(f"Avg session length: {split_stats['sessions']['avg_length']:.1f}")
        print(f"Avg clicks/session: {split_stats['sessions']['avg_clicks']:.1f}")
    
    print("="*80)
    
    return stats


def main():
    args = parse_args()
    
    print("="*80)
    print("MIND DATASET PREPROCESSING")
    print("="*80)
    print(f"Processing splits: {', '.join(args.splits)}")
    
    # Setup
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    create_directory_structure(output_dir)
    
    # Load data for each split
    logger.info("Loading data from all splits")
    loaders = {}
    
    for split_name in args.splits:
        split_dir = data_root / split_name
        if not split_dir.exists():
            logger.warning(f"Directory not found: {split_dir}, skipping")
            continue
        
        logger.info(f"Loading {split_name} data from {split_dir}")
        loader = MINDDataLoader(data_dir=str(split_dir), dataset_type=split_name)
        loader.load_all(nrows=args.nrows)
        loaders[split_name] = loader
    
    if not loaders:
        raise ValueError("No data loaded! Check your data directories.")
    
    # Initialize encoders
    logger.info("Initializing encoders")
    
    article_encoder = ArticleEncoder(
        model_name=args.embedding_model,
        embedding_dim=args.embedding_dim,
        cache_dir=output_dir / 'embeddings'
    )
    
    # Encode articles from all splits
    if not args.skip_embeddings:
        embeddings = encode_all_articles(loaders, article_encoder, args.batch_size)
        article_encoder.save_cache()
    else:
        logger.info("Skipping embedding generation (using cache)")
    
    # Create category encoder
    cat_encoder = create_category_encoder(loaders, output_dir)
    
    # Initialize user encoder and state builder
    user_encoder = UserEncoder(
        embedding_dim=args.embedding_dim,
        aggregation_method='mean',
        use_temporal_decay=True
    )
    
    state_builder = StateBuilder(
        article_encoder=article_encoder,
        user_encoder=user_encoder,
        embedding_dim=args.embedding_dim
    )
    
    # Process sessions for each split
    all_sessions = {}
    
    for split_name, loader in loaders.items():
        sessions = process_sessions_for_split(
            loader, split_name, article_encoder, user_encoder, state_builder, 
            output_dir, skip_state_computation=args.skip_state_computation
        )
        all_sessions[split_name] = sessions
    
    # Generate statistics
    stats = generate_statistics(loaders, all_sessions, output_dir)
    
    # Save config
    config = {
        'embedding_model': args.embedding_model,
        'embedding_dim': args.embedding_dim,
        'state_dim': state_builder.state_dim,
        'data_root': str(data_root),
        'output_dir': str(output_dir),
        'processed_splits': list(loaders.keys()),
        'preprocessing_date': pd.Timestamp.now().isoformat()
    }
    
    config_path = output_dir / 'metadata' / 'preprocessing_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Saved preprocessing config to {config_path}")
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Article embeddings: {output_dir / 'embeddings' / 'article_embeddings.pkl'}")
    print(f"\nProcessed splits:")
    for split_name, sessions in all_sessions.items():
        print(f"  {split_name}: {len(sessions):,} sessions")
        print(f"    -> {output_dir / 'sessions' / f'{split_name}_sessions.pkl'}")
    print(f"\nStatistics: {output_dir / 'metadata' / 'statistics.yaml'}")
    print("\nReady for RL training!")


if __name__ == "__main__":
    main()