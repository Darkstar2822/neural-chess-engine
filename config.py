import os
import torch

class Config:
    # Prioritize M1/M2 MPS, then CUDA, then CPU
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda" 
    else:
        DEVICE = "cpu"
    
    BOARD_SIZE = 8
    MAX_MOVES = 512
    
    MCTS_SIMULATIONS = 800
    MCTS_C_PUCT = 1.0
    MCTS_TEMP_THRESHOLD = 30
    
    NEURAL_NET_HIDDEN_SIZE = 256
    NEURAL_NET_RESIDUAL_BLOCKS = 19
    NEURAL_NET_FILTERS = 256
    
    TRAINING_BATCH_SIZE = 64  # Increased for M2 efficiency
    TRAINING_LEARNING_RATE = 0.001
    TRAINING_WEIGHT_DECAY = 1e-4
    TRAINING_EPOCHS_PER_ITERATION = 10
    
    SELF_PLAY_GAMES_PER_ITERATION = 100
    MEMORY_SIZE = 100000
    
    TOURNAMENT_GAMES = 50
    WIN_RATE_THRESHOLD = 0.55
    
    MODEL_DIR = "models"
    GAMES_DIR = "games"
    LOGS_DIR = "logs"
    
    @classmethod
    def create_dirs(cls):
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.GAMES_DIR, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)