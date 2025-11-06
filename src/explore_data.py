"""
Quick data exploration script for MIND dataset.
Run this after downloading the dataset to get initial insights.
"""

import sys
sys.path.append('.')

from src.data.mind_loader import MINDDataLoader
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def plot_category_distribution(loader):
    """Plot distribution of news categories"""
    cat_dist = loader.get_category_distribution()
    
    # Get top 15 categories
    top_cats = dict(sorted(cat_dist.items(), key=lambda x: x[1], reverse=True)[:15])
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_cats)), list(top_cats.values()))
    plt.xticks(range(len(top_cats)), list(top_cats.keys()), rotation=45, ha='right')
    plt.xlabel('Category')
    plt.ylabel('Number of Articles')
    plt.title('Top 15 News Categories')
    plt.tight_layout()
    plt.savefig('results/figures/category_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/category_distribution.png")

def plot_user_engagement(loader):
    """Plot user engagement metrics"""
    if loader.behaviors_df is None:
        return
    
    # Parse behaviors
    loader.behaviors_df['history_list'] = loader.behaviors_df['history'].apply(
        loader.parse_history
    )
    loader.behaviors_df['history_length'] = loader.behaviors_df['history_list'].apply(len)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # History length distribution
    hist_lengths = loader.behaviors_df[loader.behaviors_df['history_length'] > 0]['history_length']
    axes[0, 0].hist(hist_lengths, bins=50, edgecolor='black')
    axes[0, 0].set_xlabel('History Length')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of User History Lengths')
    axes[0, 0].set_xlim(0, 100)
    
    # Impressions per user
    user_impressions = loader.behaviors_df.groupby('user_id').size()
    axes[0, 1].hist(user_impressions, bins=50, edgecolor='black')
    axes[0, 1].set_xlabel('Number of Impressions')
    axes[0, 1].set_ylabel('Number of Users')
    axes[0, 1].set_title('Impressions per User')
    
    # Parse impressions
    loader.behaviors_df['impressions_list'] = loader.behaviors_df['impressions'].apply(
        loader.parse_impressions
    )
    loader.behaviors_df['num_clicks'] = loader.behaviors_df['impressions_list'].apply(
        lambda x: sum(1 for _, label in x if label == 1)
    )
    
    # Clicks distribution
    axes[1, 0].hist(loader.behaviors_df['num_clicks'], bins=20, edgecolor='black')
    axes[1, 0].set_xlabel('Clicks per Impression')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Clicks per Impression')
    
    # Cold start analysis
    cold_start = (loader.behaviors_df['history_length'] == 0).sum()
    warm_start = (loader.behaviors_df['history_length'] > 0).sum()
    axes[1, 1].bar(['Cold Start\n(No History)', 'Warm Start\n(Has History)'], 
                   [cold_start, warm_start])
    axes[1, 1].set_ylabel('Number of Impressions')
    axes[1, 1].set_title('Cold Start vs Warm Start Users')
    
    plt.tight_layout()
    plt.savefig('results/figures/user_engagement.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/user_engagement.png")

def plot_temporal_patterns(loader):
    """Plot temporal patterns in the data"""
    if loader.behaviors_df is None:
        return
    
    loader.behaviors_df['hour'] = loader.behaviors_df['timestamp'].dt.hour
    loader.behaviors_df['day_of_week'] = loader.behaviors_df['timestamp'].dt.dayofweek
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Hourly distribution
    hourly_counts = loader.behaviors_df['hour'].value_counts().sort_index()
    axes[0].plot(hourly_counts.index, hourly_counts.values, marker='o')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Number of Impressions')
    axes[0].set_title('Impressions by Hour of Day')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(0, 24, 2))
    
    # Day of week distribution
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_counts = loader.behaviors_df['day_of_week'].value_counts().sort_index()
    axes[1].bar(range(7), dow_counts.values)
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(day_names)
    axes[1].set_xlabel('Day of Week')
    axes[1].set_ylabel('Number of Impressions')
    axes[1].set_title('Impressions by Day of Week')
    
    plt.tight_layout()
    plt.savefig('results/figures/temporal_patterns.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/temporal_patterns.png")

def analyze_click_patterns(loader):
    """Analyze and visualize click patterns"""
    if loader.behaviors_df is None:
        return
    
    # Get all clicked articles
    all_clicked = []
    for imp_list in loader.behaviors_df['impressions_list']:
        clicked = [news_id for news_id, label in imp_list if label == 1]
        all_clicked.extend(clicked)
    
    click_counts = Counter(all_clicked)
    
    # Get categories of clicked articles
    clicked_categories = []
    for news_id in all_clicked:
        article = loader.get_article(news_id)
        if article:
            clicked_categories.append(article.category)
    
    cat_counts = Counter(clicked_categories)
    top_cats = dict(sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_cats)), list(top_cats.values()))
    plt.xticks(range(len(top_cats)), list(top_cats.keys()), rotation=45, ha='right')
    plt.xlabel('Category')
    plt.ylabel('Number of Clicks')
    plt.title('Top 10 Most Clicked Categories')
    plt.tight_layout()
    plt.savefig('results/figures/clicked_categories.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/clicked_categories.png")
    
    return click_counts, cat_counts

def main():
    parser = argparse.ArgumentParser(description='Explore MIND dataset')
    parser.add_argument('--data_dir', type=str, default='data/raw/train',
                        help='Path to MIND data directory')
    parser.add_argument('--nrows', type=int, default=50000,
                        help='Number of behavior rows to load (None for all)')
    parser.add_argument('--plots', action='store_true',
                        help='Generate visualization plots')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MIND DATASET EXPLORATION")
    print("="*80)
    
    # Initialize loader
    loader = MINDDataLoader(data_dir=args.data_dir, dataset_type='train')
    
    # Load data
    print("\nLoading data...")
    loader.load_all(nrows=args.nrows)
    
    # Get statistics
    print("\n" + "="*80)
    print("NEWS STATISTICS")
    print("="*80)
    
    news_stats = {
        'num_articles': len(loader.news_df),
        'num_categories': loader.news_df['category'].nunique(),
        'num_subcategories': loader.news_df['subcategory'].nunique()
    }
    
    for key, value in news_stats.items():
        print(f"{key}: {value}")
    
    print("\nTop 10 categories:")
    cat_dist = loader.get_category_distribution()
    for i, (cat, count) in enumerate(list(cat_dist.items())[:10], 1):
        print(f"  {i}. {cat:>20}: {count:>6} articles ({count/news_stats['num_articles']*100:>5.1f}%)")
    
    # User statistics
    print("\n" + "="*80)
    print("USER STATISTICS")
    print("="*80)
    
    user_stats = loader.get_user_statistics()
    for key, value in user_stats.items():
        if isinstance(value, float):
            if 'ratio' in key or 'ctr' in key:
                print(f"{key}: {value*100:.2f}%")
            else:
                print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value:,}")
    
    # Session data
    print("\n" + "="*80)
    print("SESSION DATA")
    print("="*80)
    
    sessions = loader.create_session_data(min_history_length=1)
    print(f"Total sessions (min_history=1): {len(sessions):,}")
    
    if sessions:
        avg_impressions = sum(s['num_impressions'] for s in sessions) / len(sessions)
        avg_clicks = sum(s['total_clicks'] for s in sessions) / len(sessions)
        print(f"Average impressions per session: {avg_impressions:.2f}")
        print(f"Average clicks per session: {avg_clicks:.2f}")
    
    # Generate plots if requested
    if args.plots:
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        import os
        os.makedirs('results/figures', exist_ok=True)
        
        print("\nGenerating plots...")
        plot_category_distribution(loader)
        plot_user_engagement(loader)
        plot_temporal_patterns(loader)
        click_counts, cat_counts = analyze_click_patterns(loader)
        
        print("\n✓ All plots generated successfully!")
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS FOR RL SYSTEM")
    print("="*80)
    print(f"""
1. Dataset Scale:
   - {news_stats['num_articles']:,} unique news articles
   - {user_stats['num_users']:,} unique users
   - {user_stats['num_impressions']:,} total impressions
   
2. Sparsity Challenges:
   - Overall CTR: {user_stats['overall_ctr']*100:.2f}% (very sparse rewards!)
   - Cold-start users: {user_stats['cold_start_ratio']*100:.1f}%
   - Need reward shaping and exploration strategies
   
3. State Representation:
   - Average history length: {user_stats['avg_history_length']:.1f} articles
   - Variable history lengths require padding/truncation
   - Consider recency weighting for state features
   
4. Action Space:
   - {news_stats['num_categories']} categories to consider
   - Start with K=100 candidates as proposed
   - Implement hierarchical action selection if needed
   
5. Evaluation Considerations:
   - Need to handle position bias in logged data
   - Implement proper off-policy evaluation
   - Use chronological splits (not random)
    """)

if __name__ == "__main__":
    main()