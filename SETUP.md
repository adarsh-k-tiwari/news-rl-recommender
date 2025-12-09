# Setup Guide for News RL Recommender Interface

This guide will help you set up and run the interactive React.js interface for your news recommendation system.

## Prerequisites

- Python 3.8+
- Node.js 18+ and npm
- Trained RL models (at least one of: supervised_model.pth, dqn_model.pth, dueling_dqn_model.pth, cmab_model.pth, sac_model.pth)

## Backend Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have trained models in `data/processed/`:
   - `supervised_model.pth`
   - `dqn_model.pth` 
   - `dueling_dqn_model.pth` 
   - `cmab_model.pth`
   - `sac_model.pth`

3. Start the Flask API server:
```bash
python api_server.py
```

The API will be available at `http://localhost:5000`

## Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Usage

1. Open your browser and go to `http://localhost:3000`
2. The interface will start with random news articles
3. Scroll or swipe to see more articles
4. Click the button to like an article (adds to your preference history)
5. Click the button to dislike an article
6. Watch the metrics dashboard update in real-time on the right side
7. Switch between different RL models using the model selector at the top
8. Optionally filter by categories using the category buttons

## Features

### Infinite Scroll Feed
- Smooth animations between articles
- Mouse wheel for desktop
- Auto-loads more articles as you scroll

### Metrics Dashboard
- Real-time CTR (Click-Through Rate)
- Total clicks and impressions
- Like/Dislike counts
- Category preference distribution (pie chart)
- Category distribution bar chart
- Session statistics

### Model Selection
- Switch between available RL models
- Compare performance across models
- Models are loaded dynamically from `data/processed/`

## API Endpoints

The backend provides the following endpoints:

- `GET /api/health` - Health check
- `GET /api/models` - List available models
- `POST /api/models/<model_name>` - Switch model
- `POST /api/recommendations` - Get recommendations
- `POST /api/interaction` - Record like/dislike
- `GET /api/metrics/<user_id>` - Get user metrics
- `GET /api/categories` - Get all categories
- `GET /api/article/<news_id>` - Get article details