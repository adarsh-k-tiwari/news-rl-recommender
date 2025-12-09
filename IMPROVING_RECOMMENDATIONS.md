# Improving Recommendation Quality

## Current Issues

The recommendations might feel random because:

1. **Cold Start Problem**: When a user has no history, the model has no preferences to learn from
2. **Model Not Properly Trained**: The RL models might not be well-trained
3. **State Not Updating**: User preferences might not be properly reflected in the state
4. **Score Similarity**: All candidates might have similar Q-values, making ranking appear random

## Solutions Implemented

### 1. Proper State Building
- Now uses `liked_articles` (from user's likes) instead of just `history`
- State includes session statistics (clicks, impressions, session length)
- User embedding is built from liked articles

### 2. Better Scoring Logic
- Fixed scoring for DQN agents to properly compute Q-values
- Added proper handling for different agent types
- Scores are now properly ranked in descending order

### 3. Debug Endpoint
- Added `/api/debug/recommendations` to inspect what the model is doing
- Shows score distributions and top candidates with their scores

## How to Improve Recommendations

### 1. Check Model Quality
```bash
# Test the debug endpoint
curl -X POST http://localhost:5000/api/debug/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "history": [], "count": 10}'
```

Look for:
- **Score variance**: If std is very low (< 0.01), all articles score similarly
- **Score range**: If max - min is small, the model isn't discriminating well

### 2. Retrain Models
If scores are too similar, the models might need more training:
```bash
# Retrain supervised model
python src/models/baseline/train_supervised.py

# Retrain DQN
python src/models/dqn/train_dqn.py

# Retrain Dueling DQN
python src/models/dqn/train_ddqn.py
```

### 3. Increase Candidate Pool Diversity
Currently we sample 50 candidates. You can:
- Increase to 100-200 candidates for better diversity
- Use category-based filtering to ensure variety
- Add diversity penalty to scores

### 4. Add Diversity to Recommendations
You can modify the ranking to include diversity:

```python
# After computing scores, add diversity bonus
for i, news_id in enumerate(candidates):
    # Check if similar articles already recommended
    article = news_df.loc[news_id]
    category = article['category']
    
    # Penalize if too many from same category already shown
    category_count = sum(1 for id in top_ids if news_df.loc[id]['category'] == category)
    if category_count > 2:
        scores_np[i] *= 0.8  # Reduce score for diversity
```

### 5. Use User Feedback More Effectively
- Track which categories user likes/dislikes
- Weight recommendations by category preference
- Update state more aggressively based on recent interactions

## Testing Recommendations

1. **Start with no history**: See if recommendations are diverse
2. **Like 5-10 articles**: Check if recommendations change
3. **Like articles from one category**: See if recommendations focus on that category
4. **Use debug endpoint**: Check score distributions

## Expected Behavior

- **Cold start (0 likes)**: Recommendations should be diverse across categories
- **After 5 likes**: Should start showing similar articles/categories
- **After 10+ likes**: Should strongly prefer liked categories
- **Score variance**: Should increase as user history grows

## If Recommendations Still Feel Random

1. Check if models are actually loaded:
   ```python
   # In api_server.py, check logs for "Loaded X model"
   ```

2. Verify models are trained:
   ```bash
   ls -lh data/processed/*.pth
   # Should show model files with reasonable sizes (> 1MB)
   ```

3. Check state building:
   ```python
   # Add logging in state_builder.build_state()
   logger.info(f"User embedding norm: {np.linalg.norm(user_embedding)}")
   ```

4. Test with a simple baseline:
   - Try switching to "supervised" model
   - If that works better, DQN models might need retraining

