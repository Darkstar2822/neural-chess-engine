# 🧠 Neural Chess Engine

[![Test Status](https://github.com/yourusername/neural-chess-engine/workflows/Test%20Neural%20Chess%20Engine/badge.svg)](https://github.com/yourusername/neural-chess-engine/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A revolutionary pure neural network chess engine that learns through self-play without any tree search algorithms. The engine discovers unique strategies and creative play styles through generational learning and neural network intuition.

🎯 **No Decision Trees** • 🧠 **Pure Neural Network** • 🚀 **Self-Learning** • 🎮 **Web Interface**

## Features

- **Pure Neural Network**: No decision trees or Monte Carlo Tree Search - just neural network intuition
- **Self-Play Learning**: Engine learns by playing against itself and discovering new strategies
- **Generational Evolution**: Newer models must beat older ones to become the champion
- **Random Starting Positions**: Explores diverse positions for creative strategy development
- **Adaptive Learning**: Engine adapts to different opponents and playing styles
- **Multiple Interfaces**: Both console and web-based interfaces available

## Architecture

- **Neural Network**: AlphaZero-style architecture with policy and value heads
- **Training**: Direct neural network training from self-play games
- **No Search**: Moves are selected directly from neural network policy outputs
- **Exploration**: Built-in exploration mechanisms for discovering novel strategies

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create initial model:
```bash
python main.py init
```

## 🚀 Quick Start

### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neural-chess-engine.git
cd neural-chess-engine

# Install dependencies
pip install -r requirements.txt

# Create initial model
python main.py init
```

### 🏋️ Training the Engine
```bash
# Start training with default settings (uses GPU acceleration)
python main.py train

# High-performance training for powerful PCs
python main.py train --iterations 50 --games 200

# Quick test training
python main.py train --iterations 5 --games 25
```

### 🎮 Playing Against the Engine

#### 🌐 Web Interface (Recommended)
```bash
# Start beautiful web interface with drag & drop
python main.py web

# Use trained model with learning enabled
python main.py web --model models/best_model.pt

# Disable learning mode
python main.py web --no-learn
```

#### 💻 Console Interface
```bash
# Play in terminal with AI learning from your games
python main.py play --learn

# Play against specific trained model
python main.py play --model models/checkpoint_epoch_10_*.pt --learn
```

## How It Works

### Training Process

1. **Self-Play**: The engine plays games against itself using neural network move predictions
2. **Data Collection**: Training examples are collected from these games (positions, policies, outcomes)
3. **Neural Network Training**: The network learns to predict good moves and position evaluations
4. **Generational Testing**: New models must beat previous champions to become the new champion
5. **Iteration**: This process repeats, with each generation potentially discovering new strategies

### Unique Features

- **No Tree Search**: Unlike traditional engines, this uses pure neural network intuition
- **Creative Play**: Without search constraints, the engine can develop unique playing styles
- **Adaptive Learning**: The engine adjusts its play based on opponent patterns
- **Random Exploration**: Starting from random positions helps discover unconventional strategies

### Neural Network Architecture

- **Input**: 16-plane board representation (pieces, castling rights, turn)
- **Backbone**: Residual neural network with 19 blocks
- **Policy Head**: Predicts move probabilities for all legal moves
- **Value Head**: Evaluates position strength (-1 to +1)

## Configuration

Key settings can be modified in `config.py`:

- `NEURAL_NET_RESIDUAL_BLOCKS`: Network depth (default: 19)
- `SELF_PLAY_GAMES_PER_ITERATION`: Games per training iteration (default: 100)
- `TRAINING_LEARNING_RATE`: Learning rate for neural network (default: 0.001)
- `WIN_RATE_THRESHOLD`: Required win rate for model promotion (default: 0.55)

## Advanced Usage

### Analyzing Games
```python
from src.utils.data_manager import DataManager, GameAnalyzer

data_manager = DataManager()
analyzer = GameAnalyzer(data_manager)

# Analyze a completed game
analysis = analyzer.analyze_game(game)
print(analysis)
```

### Custom Training
```python
from src.training.direct_selfplay import CreativeTrainingManager
from src.neural_network.chess_net import ChessNet

model = ChessNet()
training_manager = CreativeTrainingManager(model)

# Generate training games with high exploration
training_data = training_manager.train_iteration(games_per_iteration=200)
```

## Project Structure

```
├── src/
│   ├── engine/          # Chess game logic and neural player
│   ├── neural_network/  # Neural network architecture and training
│   ├── training/        # Self-play and tournament systems
│   ├── ui/             # User interfaces (console and web)
│   └── utils/          # Utilities and data management
├── models/             # Saved model checkpoints
├── games/              # Game records
├── logs/               # Training logs
├── config.py           # Configuration settings
└── main.py            # Main entry point
```

## Development

The engine is designed to be modular and extensible:

- Add new neural network architectures in `src/neural_network/`
- Implement new training strategies in `src/training/`
- Create custom interfaces in `src/ui/`
- Modify game logic in `src/engine/`

## Performance Notes

- Training is GPU-accelerated when CUDA is available
- Initial training will be random play until the network learns
- Expect interesting strategies to emerge after several training iterations
- Web interface provides real-time play against the current best model

## Philosophy

This engine prioritizes creative and unique play over traditional chess engine strength. By removing tree search and relying purely on neural network intuition, it can develop unconventional strategies that might surprise both human and computer opponents.