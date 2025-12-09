# News RL Recommender - Frontend

Modern React.js frontend for the News Recommendation System with TikTok-like infinite scroll interface.

## Features

- Flash card scroll interface
- Like/Dislike interactions
- Real-time metrics dashboard
- Multiple RL model support
- Responsive design

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Build for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations
- **Recharts** - Data visualization
- **Axios** - HTTP client
- **Lucide React** - Icons

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── NewsFeed.jsx       # Main infinite scroll feed
│   │   ├── MetricsDashboard.jsx # Metrics and charts
│   │   └── ModelSelector.jsx   # RL model switcher
│   ├── services/
│   │   └── api.js              # API client
│   ├── App.jsx                # Main app component
│   ├── main.jsx               # Entry point
│   └── index.css              # Global styles
├── package.json
└── vite.config.js
```

