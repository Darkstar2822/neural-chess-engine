import numpy as np
import torch
from typing import List, Tuple, Optional
import random
from src.engine.chess_game import ChessGame
from src.engine.neural_player import NeuralPlayer, DirectNeuralTraining, AdaptiveNeuralPlayer
from src.neural_network.chess_net import ChessNet
from src.utils.game_utils import create_random_position, get_opening_positions
from config import Config

class DirectSelfPlay:
    def __init__(self, model: ChessNet):
        self.model = model
        self.neural_trainer = DirectNeuralTraining(model)
        
    def generate_training_games(self, num_games: int = 50, use_exploration: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        all_states = []
        all_policies = []
        all_values = []
        
        for game_idx in range(num_games):
            if game_idx % 10 == 0:
                print(f"Playing training game {game_idx + 1}/{num_games}")
            
            if use_exploration and random.random() < 0.3:
                exploration_player = AdaptiveNeuralPlayer(self.model, exploration_rate=0.2)
                states, policies, values = self.neural_trainer.play_training_game(
                    opponent_player=exploration_player, 
                    use_random_start=True
                )
            else:
                states, policies, values = self.neural_trainer.play_training_game(
                    use_random_start=random.random() < 0.4
                )
            
            all_states.extend(states)
            all_policies.extend(policies)
            all_values.extend(values)
        
        print(f"Generated {len(all_states)} training examples from {num_games} games")
        return all_states, all_policies, all_values

class CreativeTrainingManager:
    def __init__(self, model: ChessNet):
        self.model = model
        self.training_data = {
            'states': [],
            'policies': [],
            'values': []
        }
        self.self_play = DirectSelfPlay(model)
        
    def train_iteration(self, games_per_iteration: int = 100):
        print("Generating creative self-play games...")
        
        states, policies, values = self.self_play.generate_training_games(
            num_games=games_per_iteration,
            use_exploration=True
        )
        
        self.add_training_data(states, policies, values)
        
        return self.get_training_batch()
    
    def add_training_data(self, states: List[np.ndarray], policies: List[np.ndarray], values: List[float]):
        self.training_data['states'].extend(states)
        self.training_data['policies'].extend(policies)
        self.training_data['values'].extend(values)
        
        if len(self.training_data['states']) > Config.MEMORY_SIZE:
            excess = len(self.training_data['states']) - Config.MEMORY_SIZE
            self.training_data['states'] = self.training_data['states'][excess:]
            self.training_data['policies'] = self.training_data['policies'][excess:]
            self.training_data['values'] = self.training_data['values'][excess:]
    
    def get_training_batch(self, batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if batch_size is None:
            return (
                np.array(self.training_data['states']),
                np.array(self.training_data['policies']),
                np.array(self.training_data['values'])
            )
        
        indices = np.random.choice(
            len(self.training_data['states']), 
            size=min(batch_size, len(self.training_data['states'])), 
            replace=False
        )
        
        return (
            np.array([self.training_data['states'][i] for i in indices]),
            np.array([self.training_data['policies'][i] for i in indices]),
            np.array([self.training_data['values'][i] for i in indices])
        )
    
    def get_data_size(self) -> int:
        return len(self.training_data['states'])

class PositionalLearning:
    def __init__(self, model: ChessNet):
        self.model = model
        self.player = NeuralPlayer(model)
        self.tactical_positions = []
        self.strategic_patterns = {}
        
    def learn_from_position(self, game: ChessGame, target_result: float) -> Tuple[np.ndarray, np.ndarray, float]:
        state = game.get_state_planes().permute(2, 0, 1).cpu().numpy()
        policy_dist = self.player.get_policy_distribution(game)
        
        return state, policy_dist, target_result
    
    def generate_tactical_training(self, num_positions: int = 200) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        states = []
        policies = []
        values = []
        
        for _ in range(num_positions):
            game = create_random_position(max_moves=random.randint(10, 40))
            
            if not game.is_game_over():
                state, policy, value = self.learn_from_position(game, random.uniform(-1, 1))
                states.append(state)
                policies.append(policy)
                values.append(value)
        
        return states, policies, values

class OpponentAdaptation:
    def __init__(self, model: ChessNet):
        self.model = model
        self.player = AdaptiveNeuralPlayer(model)
        self.opponent_patterns = {}
        
    def learn_from_opponent_game(self, moves_history: List[str], result: float) -> None:
        pattern_key = "_".join(moves_history[:10])
        
        if pattern_key not in self.opponent_patterns:
            self.opponent_patterns[pattern_key] = {'games': 0, 'total_result': 0.0}
        
        self.opponent_patterns[pattern_key]['games'] += 1
        self.opponent_patterns[pattern_key]['total_result'] += result
    
    def get_adaptation_strategy(self, opponent_opening: List[str]) -> float:
        pattern_key = "_".join(opponent_opening[:5])
        
        if pattern_key in self.opponent_patterns:
            pattern_data = self.opponent_patterns[pattern_key]
            avg_result = pattern_data['total_result'] / pattern_data['games']
            
            if avg_result < -0.3:
                return 1.5
            elif avg_result > 0.3:
                return 0.5
        
        return 1.0