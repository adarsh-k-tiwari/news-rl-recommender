# Personalized News Recommendation System using Reinforcement Learning

A RL-based news recommendation to optimize long-term user engagement using the MIND dataset.

## Authors
- Adarsh Kumar (635009465)
- Neil Roy (436001049)

## Project Overview
This project formulates personalized news recommendation as a Markov Decision Process (MDP) and implements various reinforcement learning algorithms including Contextual Multi-Armed Bandits (CMAB), Deep Q-Networks (DQN), and Dueling DQN to balance user preferences, diversity, and novelty for sustained engagement.

## Setup Instructions

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install RecoGym
pip install git+https://github.com/criteo-research/reco-gym.git
```

### 2. Download MIND Dataset
Go to MSNews GitHub page and agree to the License Terms to download the dataset
[https://msnews.github.io/index.html#getting-start](https://msnews.github.io/index.html#getting-start)

### 3. Preprocess Data
```bash
python scripts/preprocess_data.py --subset small
```

## Project Structure
```
news-rl-recommender/
├── config/              # Configuration files
├── data/               # MIND dataset (raw and processed)
├── src/                # Source code
│   ├── data/          # Data loading and preprocessing
│   ├── environment/   # RL environment (RecoGym adaptation)
│   ├── features/      # Feature engineering (embeddings)
│   ├── models/        # Baselines and RL agents
│   ├── training/      # Training loops
│   ├── evaluation/    # Evaluation metrics
│   └── utils/         # Utilities
├── notebooks/         # Jupyter notebooks for exploration
├── experiments/       # Experiment results
├── scripts/          # Utility scripts
└── tests/            # Unit tests
```

## Usage

### Train Baseline Models
```bash
python scripts/train_baseline.py --model collaborative_filtering
```

### Train RL Agent
```bash
python scripts/train_rl_agent.py --algorithm dqn --episodes 10000
```

### Evaluate Model
```bash
python scripts/run_evaluation.py --model_path results/checkpoints/dqn_best.pt
```

## Evaluation Metrics
- **Short-term**: CTR, average reward per session
- **Long-term**: Session length, user return rate
- **Diversity**: Intra-list diversity, topic coverage, novelty

## Timeline
- **Week 1-2**: Environment setup and data preprocessing
- **Week 3**: Baseline implementation
- **Week 4-6**: Deep RL algorithms (DQN, Dueling DQN, CMAB)
- **Week 7-8**: Advanced techniques and final evaluation

## References
See `literature_survey.md` for detailed related work.

## License
This project is for educational purposes as part of a graduate-level reinforcement learning course.