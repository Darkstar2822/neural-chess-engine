import os
import torch

class Config:
    @classmethod
    def get_device(cls):
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    DEVICE = None
    
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
    
    # Evolutionary Training Settings
    EVOLUTION_POPULATION_SIZE = 20
    EVOLUTION_MAX_GENERATIONS = 10
    EVOLUTION_MUTATION_RATE = 0.1
    EVOLUTION_CROSSOVER_RATE = 0.8
    EVOLUTION_TOURNAMENT_SIZE = 3
    EVOLUTION_ELITE_SIZE = 2
    
    # Neuroevolution Settings
    NEURO_ADD_NODE_RATE = 0.03
    NEURO_ADD_CONNECTION_RATE = 0.05
    NEURO_WEIGHT_MUTATION_RATE = 0.8
    NEURO_DISABLE_CONNECTION_RATE = 0.01
    NEURO_COMPATIBILITY_THRESHOLD = 3.0
    NEURO_SPECIES_ELITISM = 0.2
    
    # Multi-Objective Optimization
    PARETO_OBJECTIVES = [
        'win_rate', 'game_length', 'piece_efficiency', 
        'positional_strength', 'tactical_sharpness'
    ]
    NOVELTY_ARCHIVE_SIZE = 100
    NOVELTY_K_NEAREST = 15
    
    MODEL_DIR = "models"
    GAMES_DIR = "games" 
    LOGS_DIR = "logs"
    EVOLUTION_DIR = "evolution_data"
    
    @classmethod
    def create_dirs(cls):
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.GAMES_DIR, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)
        os.makedirs(cls.EVOLUTION_DIR, exist_ok=True)
        cls.summary()

    @classmethod
    def summary(cls):
        for key in dir(cls):
            if not key.startswith("__") and key.isupper():
                print(f"{key}: {getattr(cls, key)}")

Config.DEVICE = Config.get_device()