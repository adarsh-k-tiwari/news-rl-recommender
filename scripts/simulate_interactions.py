"""
Simulation script to generate many interactions against the local recommendation API.

Usage examples (PowerShell):
  python scripts\simulate_interactions.py --user-id test_user --n 1000 --genres politics,sports --match-prob 0.9 --other-prob 0.02

What it does:
- Repeatedly requests recommendations for `user_id` (passes current liked history)
- For each recommended article, decides "like" or "dislike" based on whether the article's
  category is in the preferred genres and the configured probabilities.
- Sends `/api/interaction` for every shown article (like or dislike) so the backend receives
  feedback and can train online.
- Stops after `n` interactions (likes + dislikes) have been recorded.
- Prints summary metrics at the end and optionally writes a CSV of interactions.

Notes:
- Requires the backend server running at http://localhost:5000
- Install dependencies: `pip install requests tqdm pandas`

"""
import argparse
import time
import random
import requests
import csv
from tqdm import tqdm
import sys

API_BASE = "http://localhost:5000/api"


def get_recommendations(user_id, history, count=10, categories=None):
    payload = {
        'user_id': user_id,
        'history': history,
        'count': count,
        'categories': categories or []
    }
    resp = requests.post(f"{API_BASE}/recommendations", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def record_interaction(user_id, news_id, action):
    payload = {
        'user_id': user_id,
        'news_id': news_id,
        'action': action
    }
    resp = requests.post(f"{API_BASE}/interaction", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_metrics(user_id):
    resp = requests.get(f"{API_BASE}/metrics/{user_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def simulate(user_id, n_interactions=1000, genres=None, match_prob=0.9, other_prob=0.02, per_request=10, delay=0.05, out_csv=None):
    """Run the simulation until n_interactions are performed."""
    history = []
    interactions_done = 0
    interactions = []  # list of dicts for CSV

    genres = [g.strip().lower() for g in (genres or []) if g.strip()]
    print(f"Simulating {n_interactions} interactions for user '{user_id}' | preferred genres: {genres} | match_prob={match_prob} | other_prob={other_prob}")

    pbar = tqdm(total=n_interactions)
    try:
        while interactions_done < n_interactions:
            # Request recommendations
            try:
                data = get_recommendations(user_id, history, count=per_request, categories=genres if genres else [])
            except Exception as e:
                print("Error fetching recommendations:", e)
                time.sleep(1)
                continue

            recs = data.get('recommendations', [])

            if not recs:
                print("No recommendations returned; stopping.")
                break

            # For each recommended article, decide like/dislike and record
            for art in recs:
                if interactions_done >= n_interactions:
                    break

                news_id = art.get('news_id')
                category = (art.get('category') or '').lower()

                # Decide whether user likes it
                if genres and category in genres:
                    prob = match_prob
                else:
                    prob = other_prob

                liked = random.random() < prob
                action = 'like' if liked else 'dislike'

                # Send interaction
                try:
                    resp = record_interaction(user_id, news_id, action)
                except Exception as e:
                    print(f"Failed to record interaction for {news_id}: {e}")
                    # don't count this interaction; try next
                    continue

                interactions_done += 1
                pbar.update(1)

                # Update local history only on likes (frontend behavior uses liked history)
                if action == 'like':
                    history.append(news_id)

                # Save interaction info
                interactions.append({
                    'index': interactions_done,
                    'news_id': news_id,
                    'category': category,
                    'action': action,
                    'reward': resp.get('reward'),
                    'total_reward': resp.get('total_reward'),
                    'model': resp.get('model')
                })

                # Small delay to avoid spamming the server too fast
                time.sleep(delay)

    except KeyboardInterrupt:
        print('\nInterrupted by user')
    finally:
        pbar.close()

    # Print summary
    print(f"Completed {interactions_done} interactions")
    try:
        metrics = get_metrics(user_id)
        print("Final metrics:")
        for k, v in metrics.items():
            if k in ('interactions_over_time', 'category_distribution'):
                continue
            print(f"  {k}: {v}")
    except Exception as e:
        print("Could not fetch metrics:", e)

    # Optionally write CSV
    if out_csv:
        keys = interactions[0].keys() if interactions else ['index', 'news_id', 'category', 'action', 'reward', 'total_reward', 'model']
        try:
            with open(out_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(interactions)
            print(f"Wrote interactions to {out_csv}")
        except Exception as e:
            print("Failed to write CSV:", e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate user interactions with the recommendation API')
    parser.add_argument('--user-id', type=str, default='sim_user', help='User id to simulate')
    parser.add_argument('--n', type=int, default=1000, help='Number of interactions to perform')
    parser.add_argument('--genres', type=str, default='', help='Comma-separated list of preferred genres (categories)')
    parser.add_argument('--match-prob', type=float, default=0.9, help='Probability of liking an article in preferred genres')
    parser.add_argument('--other-prob', type=float, default=0.02, help='Probability of liking a non-preferred article')
    parser.add_argument('--per-request', type=int, default=10, help='Number of recommendations to request per API call')
    parser.add_argument('--delay', type=float, default=0.05, help='Delay (seconds) between interactions')
    parser.add_argument('--out-csv', type=str, default=None, help='Optional CSV file to write interactions')

    args = parser.parse_args()

    genres = [g.strip() for g in args.genres.split(',')] if args.genres else []

    try:
        simulate(
            user_id=args.user_id,
            n_interactions=args.n,
            genres=genres,
            match_prob=args.match_prob,
            other_prob=args.other_prob,
            per_request=args.per_request,
            delay=args.delay,
            out_csv=args.out_csv
        )
    except Exception as e:
        print('Fatal error during simulation:', e)
        sys.exit(1)
