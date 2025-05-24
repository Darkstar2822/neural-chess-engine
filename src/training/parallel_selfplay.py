import multiprocessing as mp
import numpy as np
import torch
from typing import List, Tuple
import pickle
import tempfile
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.engine.chess_game import ChessGame
from src.engine.neural_player import DirectNeuralTraining
from src.neural_network.chess_net import ChessNet
from src.utils.error_handler import safe_execute, handle_errors
from src.utils.memory_manager import memory_manager
from config import Config

@safe_execute(default_return=([], [], []))
def play_single_game(model_path: str, game_id: int, use_exploration: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """Play a single self-play game in a separate process"""
    # Load model in worker process
    model = ChessNet.load_model(model_path)
    model.eval()
    
    trainer = DirectNeuralTraining(model)
    
    if use_exploration:
        from src.engine.neural_player import AdaptiveNeuralPlayer
        exploration_player = AdaptiveNeuralPlayer(model, exploration_rate=0.15)
        states, policies, values = trainer.play_training_game(
            opponent_player=exploration_player,
            use_random_start=True
        )
    else:
        states, policies, values = trainer.play_training_game(use_random_start=True)
    
    return states, policies, values

class ParallelSelfPlay:
    def __init__(self, model: ChessNet, num_workers: int = None):
        self.model = model
        self.num_workers = num_workers or min(mp.cpu_count(), 8)  # Don't overwhelm system
        
    @handle_errors
    def generate_training_games(self, num_games: int = 100) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """Generate training games using parallel processing"""
        print(f"🚀 Generating {num_games} games using {self.num_workers} parallel workers...")
        cleanup = memory_manager.monitor_memory("parallel_selfplay")
        
        try:
            # Save model to temporary file for worker processes
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                model_path = tmp_file.name
                self.model.save_model(model_path)
            
            all_states = []
            all_policies = []
            all_values = []
            
            # Use ProcessPoolExecutor for parallel game generation
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all games
                futures = []
                for game_id in range(num_games):
                    use_exploration = game_id % 3 == 0  # 1/3 of games use exploration
                    future = executor.submit(play_single_game, model_path, game_id, use_exploration)
                    futures.append(future)
                
                # Collect results as they complete
                completed_games = 0
                successful_games = 0
                for future in as_completed(futures):
                    try:
                        states, policies, values = future.result(timeout=120)  # Increased timeout
                        if states:  # Only add if game completed successfully
                            all_states.extend(states)
                            all_policies.extend(policies)
                            all_values.extend(values)
                            successful_games += 1
                        
                        completed_games += 1
                        if completed_games % 10 == 0:
                            print(f"  ✅ Completed {completed_games}/{num_games} games ({successful_games} successful)")
                            
                    except Exception as e:
                        print(f"  ❌ Game failed: {e}")
                        completed_games += 1
            
            # Clean up temporary model file
            try:
                os.unlink(model_path)
            except:
                pass
            
            print(f"🎯 Generated {len(all_states)} training examples from {successful_games}/{completed_games} games")
            return all_states, all_policies, all_values
        
        finally:
            cleanup()

class FastTrainingManager:
    def __init__(self, model: ChessNet):
        self.model = model
        self.parallel_selfplay = ParallelSelfPlay(model)
        self.training_data = {
            'states': [],
            'policies': [],
            'values': []
        }
        
    def train_iteration(self, games_per_iteration: int = 100):
        print("🚀 Fast parallel training iteration starting...")
        
        # Generate games in parallel
        states, policies, values = self.parallel_selfplay.generate_training_games(games_per_iteration)
        
        # Add to training data
        self.add_training_data(states, policies, values)
        
        return self.get_training_batch()
    
    def add_training_data(self, states: List[np.ndarray], policies: List[np.ndarray], values: List[float]):
        self.training_data['states'].extend(states)
        self.training_data['policies'].extend(policies)
        self.training_data['values'].extend(values)
        
        # Memory management
        if len(self.training_data['states']) > Config.MEMORY_SIZE:
            excess = len(self.training_data['states']) - Config.MEMORY_SIZE
            self.training_data['states'] = self.training_data['states'][excess:]
            self.training_data['policies'] = self.training_data['policies'][excess:]
            self.training_data['values'] = self.training_data['values'][excess:]
    
    def get_training_batch(self):
        return (
            np.array(self.training_data['states']),
            np.array(self.training_data['policies']),
            np.array(self.training_data['values'])
        )