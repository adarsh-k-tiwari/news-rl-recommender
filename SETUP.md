# Setup Guide for News RL Recommender Interface

This guide will help you set up and run the interactive React.js interface for your news recommendation system.

## Prerequisites

- Python 3.8+
- Node.js 18+ and npm
- Trained RL models (at least one of: supervised_model.pth, dqn_model.pth, dueling_dqn_model.pth)

## Backend Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have trained models in `data/processed/`:
   - `supervised_model.pth`
   - `dqn_model.pth` (optional)
   - `dueling_dqn_model.pth` (optional)

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
4. Click the ❤️ button to like an article (adds to your preference history)
5. Click the ❌ button to dislike an article
6. Watch the metrics dashboard update in real-time on the right side
7. Switch between different RL models using the model selector at the top
8. Optionally filter by categories using the category buttons

## Features

### Infinite Scroll Feed
- TikTok-style vertical scrolling
- Smooth animations between articles
- Touch/swipe support for mobile
- Mouse wheel support for desktop
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

## Troubleshooting

### Backend Issues

1. **Model not found errors**: Make sure you have at least one trained model in `data/processed/`
2. **Import errors**: Ensure all Python dependencies are installed
3. **Data loading errors**: Check that `data/raw/dev/news.tsv` exists

### Frontend Issues

1. **CORS errors**: Make sure the backend is running on port 5000
2. **API connection errors**: Check that `api_server.py` is running
3. **Build errors**: Make sure Node.js 18+ is installed

## Development

### Backend Development
- The API server uses Flask with CORS enabled
- Models are loaded at startup
- User sessions are stored in memory (resets on server restart)

### Frontend Development
- Uses Vite for fast development
- Hot module replacement enabled
- Tailwind CSS for styling
- Framer Motion for animations

## Production Deployment

### Backend
1. Use a production WSGI server like Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
```

### Frontend
1. Build the production bundle:
```bash
cd frontend
npm run build
```

2. Serve the `dist` directory with a web server (nginx, Apache, etc.)

## Notes

- User sessions are stored in memory and will reset when the server restarts
- The interface starts with random news articles
- Recommendations improve as you interact (like/dislike articles)
- Metrics update every 2 seconds automatically

