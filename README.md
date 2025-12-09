# Personalized News Recommendation System using Reinforcement Learning

A RL-based news recommendation system with an interactive web interface to optimize long-term user engagement using the MIND dataset. Features real-time training, multiple RL algorithms (CMAB, DQN, Dueling DQN, SAC), and a TAMU AUX themed interface with live performance metrics.

## Authors
- Adarsh Kumar (635009465)
- Neil Roy (436001049)

## Project Overview
This project formulates personalized news recommendation as a Markov Decision Process (MDP) and implements various reinforcement learning algorithms to balance user preferences, diversity, and novelty for sustained engagement. The system includes:

- **Multiple RL Models**: CMAB, DQN, Dueling DQN, SAC, and Supervised Learning baseline
- **Interactive Web Interface**: Modern React-based UI with real-time metrics
- **Automatic Simulation**: Train models with automated user interactions
- **Category-Based Learning**: Uses one-hot encoding for news categories
- **Performance Tracking**: Live dashboards showing rewards, Q-values, and category preferences

## Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- Node.js 16+ and npm (for frontend) ( Very Important , Dashboard is made with React)
- Git

### 1. Clone Repository
```bash
git clone https://github.com/adarsh-k-tiwari/news-rl-recommender.git
cd news-rl-recommender
```

### 2. Setup Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Download MIND Dataset
The MIND dataset is required for the recommendation system:

1. Visit [MSNews GitHub](https://msnews.github.io/index.html#getting-start)
2. Accept the License Terms
3. Download the dataset (MINDsmall recommended for testing)
4. Extract to `data/raw/` directory:
   ```
   data/raw/dev/
   data/raw/train/
   data/raw/test/
   ```

### 4. Preprocess Data
```bash
python -m scripts.preprocess_data
```

This will generate:
- Category embeddings in `data/processed/embeddings/`
- Preprocessing metadata in `data/processed/metadata/`
- Train and Dev Sessions in `data/processed/sessions/`
- Model checkpoints (if any exist) in `data/processed/*.pth`

### 5. Model Training
You can either train the Reinforcement Learning agents from scratch or use our pre-trained models to skip the training time.
#### Option 1: Training from scratch
Run the following commands to train individual agents. Each script will save the best model checkpoint to `data/processed/`.
```bash
# 1. Supervised Baseline: Trains a standard classifier to predict clicks
python -m src.models.train_supervised

# 2. Deep Q-Network (DQN): Trains an agent using Value-Based RL with Experience Replay
python -m src.models.dqn.train_dqn

# 3. Dueling DQN: An improvement over DQN that separates State-Value and Action-Advantage
python -m src.models.dqn.train_ddqn

# 4. Contextual Bandit (CMAB): A "greedy" agent that optimizes for immediate reward (no future planning)
python -m src.models.cmab.train_cmab

# 5. Soft Actor-Critic (SAC): The advanced agent that balances Accuracy vs. Diversity (Entropy)
python -m src.models.sac.train_sac
```

#### Option 2: Using pre-trained models
If you wish to test the web interface or evaluation scripts without waiting for training:
1. Download the full `data.zip` folder from our [Google Drive](https://drive.google.com/drive/folders/1hkcqKRMCE3AxIjfW7kCSCftYD3PdLvYy?usp=sharing)
2. Delete your existing data/ folder in the project root
3. Extract the downloaded zip file so that the new data/ folder replaces it
4. This folder contains all processed embeddings and trained .pth model files

### 6. Model Evaluation
Generate a comparative report of all trained models (Random, Popularity, Supervised, DQN, CMAB, SAC).
```bash
python -m scripts.evaluate_final
```
Output:
- This script calculates key metrics: Click-Through Rate (CTR), Average Reward, and Diversity Scores
- It generates comparison plots saved in the experiments/ or results/ directory
- Console output will display a table ranking the models by performance

## Project Structure
```
news-rl-recommender/
├── config/                 # Configuration files
├── data/                   # MIND dataset (raw and processed)
├── results/                # Contains graphs
├── src/                    # Source code
│   ├── data/               # MIND download and parse script for sessions
│   ├── environment/        # RL environment (RecoGym adaptation)
│   ├── features/           # Feature engineering (embeddings)
│   ├── models/             # Baselines and RL agents training script
│   ├── training/           # Training loops
│   ├── evaluation/         # Evaluation metrics
├── frontend/               # React.js web interface (NEW!)
│   ├── src/
│   │   ├── components/     # UI components
│   │   └── services/       # API client
│   └── package.json
├── api_server.py           # Flask API server
├── notebooks/              # Jupyter notebooks for data exploration
├── scripts/                # Utility scripts
├── LICENSE                 # License file
├── requirements.txt        # Project's module dependencies
├── SETUP.md                # Setup file for the web app
├── Readme.md               # This file
```

## Running the Interactive Application
1. **Start the Backend API:**
   ```bash
   # Make sure virtual environment is activated
   python api_server.py
   ```
   Server will run on `http://localhost:5000`

2. **Start the Frontend (new terminal):**
   ```bash
   cd frontend
   npm install  # First time only
   npm run dev
   ```
   UI will be available at `http://localhost:5173`

3. **Verify Connection:**
   - Frontend should show available models
   - If you see "Backend API not connected" warning, ensure backend is running

## Using the Interface

### Interactive Features

1. **News Feed:**
   - Swipe or scroll through news articles
   - Click (Like) to add to preference history
   - Click (Dislike) to provide negative feedback
   - Progress bar shows reading time

2. **Model Selection:**
   - Switch between RL models (Supervised, DQN, Dueling DQN, CMAB, SAC)
   - Selected model shown in dashboard

3. **Simulation Panel:**
   - **Preferred Genres:** Enter comma-separated categories (e.g., `sports,news` or `foodanddrink,tv`)
   - **Match Probability:** Likelihood of liking preferred categories (0.0-1.0)
   - **Other Probability:** Likelihood of liking other categories
   - **Total Interactions:** Number of automated interactions to simulate
   - Click `Start Simulation` to train model automatically

4. **Real-Time Metrics:**
   - **RL Performance:** Shows total reward and recommendations over time
   - **Performance Metrics:** Current reward, Q-values, and prediction accuracy
   - **Session Statistics:** Click-through rate, engagement metrics
   - **Category Distribution:** Bar chart of article categories
   - **Category Preferences:** Pie chart of user preferences
   - **Recent Rewards:** Timeline of model performance


### Understanding the Metrics

- **Total Reward:** Cumulative reward from all interactions
- **Avg Q-Value:** Average predicted value of recommended articles
- **CTR (Click-Through Rate):** Percentage of recommendations clicked
- **Prediction Accuracy:** How well model predicts user preferences
- **Category Distribution:** Shows diversity of recommendations
- **Recent Rewards:** Tracks model performance over last 20 interactions

## Technical Architecture

## Interactive RL Environment
### Backend (Flask API)
- **Framework:** Flask with CORS enabled
- **Models:** PyTorch-based RL agents (DQN, Dueling DQN, CMAB, SAC, Supervised)
- **Features:**
  - Category-based encoding (one-hot per category)
  - Real-time model training with online updates
  - Session management for multi-user support
  - Balanced category distribution for diverse recommendations
  - Automatic state building from user history

### Frontend (React + Vite)
- **Framework:** React 18 with Vite
- **Styling:** Tailwind CSS with Texas A&M branding
- **Visualization:** Recharts for real-time metrics
- **Animation:** Framer Motion for smooth transitions
- **Features:**
  - Scroll interface
  - Real-time metrics dashboard
  - Automated simulation control
  - Model switching

### Key Components

1. **Article Encoder:** Converts news articles to category-based embeddings
2. **User Encoder:** Aggregates user history into preference state
3. **State Builder:** Constructs MDP state from user interactions
4. **RL Agents:** Various algorithms with online learning capability
5. **Reward System:** Computed from user feedback (like/dislike)


## Configuration Files

- `config/model_config.yaml` - Model hyperparameters
- `config/training_config.yaml` - Training settings
- `config/environment_config.yaml` - Environment parameters
- `data/processed/metadata/preprocessing_config.yaml` - Preprocessing settings

## Data Files

After preprocessing, you should have:
```
data/processed/
├── embeddings/              # Category embeddings
├── metadata/
│   ├── preprocessing_config.yaml
│   └── statistics.yaml
├── sessions/                # Saved user sessions
├── supervised_model.pth     # Supervised learning model
├── dqn_model.pth           # DQN model
├── dueling_dqn_model.pth   # Dueling DQN model
├── cmab_model.pth          # CMAB model
└── sac_model.pth           # SAC model
```

## Features

### Category-Based Recommendations
- Uses one-hot encoding for news categories
- Learns category preferences from user interactions
- Ensures balanced distribution across preferred categories
- Supports multiple simultaneous category preferences

### Multi-Category Support
- Automatically detects user's preferred categories
- Round-robin selection ensures equal representation
- Configurable through simulation panel
- Works with any combination of categories (e.g., `sports,news`, `foodanddrink,tv`)

### Real-Time Learning
- Models update after each interaction
- Epsilon-greedy exploration for RL agents
- Adaptive learning rates
- Session-based state management

## Evaluation Metrics
- **Short-term**: CTR, average reward per session, prediction accuracy
- **Long-term**: Total reward accumulation, user engagement patterns
- **Diversity**: Category distribution, intra-list diversity
- **Model Performance**: Q-values, score distributions, convergence

## Project Timeline
- **Week 1-2**: Environment setup, data preprocessing, initial exploration
- **Week 3**: Baseline implementation (supervised learning)
- **Week 4-5**: Deep RL algorithms (DQN, Dueling DQN)
- **Week 6**: Advanced algorithms (CMAB, SAC)
- **Week 7**: Interactive web interface development
- **Week 8**: Final testing, evaluation, and documentation

## References
- MIND Dataset: [Microsoft News Dataset](https://msnews.github.io/)
- DQN Paper: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- Dueling DQN: [Dueling Network Architectures](https://arxiv.org/abs/1511.06581)
- CMAB: [Combinatorial Multi-Armed Bandit](https://proceedings.mlr.press/v28/chen13a.html)
- SAC: [Soft Actor Critic](https://arxiv.org/abs/1812.05905)

## License
This project is for educational purposes as part of CSCE 642 - Deep Reinforcement Learning course at Texas A&M University.

## Contact
For questions or issues:
- Adarsh Kumar: adarsh0801@tamu.edu
- Neil Roy: neilroy@tamu.edu
