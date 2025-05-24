import chess
import numpy as np
import torch
from typing import List, Tuple, Optional
import random
from src.engine.chess_game import ChessGame
from src.mcts.mcts import MCTSPlayer
from src.neural_network.chess_net import ChessNet
from src.utils.game_utils import create_random_position, get_opening_positions
from config import Config

class SelfPlayGame:
    def __init__(self, model: ChessNet, use_random_start: bool = False):
        self.player = MCTSPlayer(model)
        self.use_random_start = use_random_start
        
    def play_game(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        if self.use_random_start and random.random() < 0.3:
            game = create_random_position(max_moves=20)
        elif random.random() < 0.1:
            opening_fens = get_opening_positions()
            fen = random.choice(opening_fens)
            game = ChessGame(fen)
        else:
            game = ChessGame()
        
        states = []
        policies = []
        move_count = 0
        
        while not game.is_game_over() and move_count < Config.MAX_MOVES:
            temperature = 1.0 if move_count < Config.MCTS_TEMP_THRESHOLD else 0.1
            
            state = game.get_state_planes().permute(2, 0, 1).cpu().numpy()
            policy = self.player.get_policy(game, temperature)
            
            states.append(state)
            policies.append(policy)
            
            move = self.player.get_move(game, temperature)
            if move is None:
                break
                
            game.make_move(move)
            move_count += 1
        
        game_result = game.get_game_result()
        if game_result is None:
            game_result = 0.0
        
        values = []
        for i in range(len(states)):
            player_to_move = (i % 2 == 0)
            if player_to_move:
                values.append(game_result)
            else:
                values.append(-game_result)
        
        return states, policies, values

class SelfPlayManager:
    def __init__(self, model: ChessNet):
        self.model = model
        self.training_data = {
            'states': [],
            'policies': [],
            'values': []
        }
    
    def generate_games(self, num_games: int = Config.SELF_PLAY_GAMES_PER_ITERATION, 
                      use_random_starts: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        all_states = []
        all_policies = []
        all_values = []
        
        for game_idx in range(num_games):
            if game_idx % 10 == 0:
                print(f"Playing game {game_idx + 1}/{num_games}")
            
            self_play_game = SelfPlayGame(self.model, use_random_starts)
            states, policies, values = self_play_game.play_game()
            
            all_states.extend(states)
            all_policies.extend(policies)
            all_values.extend(values)
        
        print(f"Generated {len(all_states)} training examples from {num_games} games")
        
        return all_states, all_policies, all_values
    
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
    
    def clear_data(self):
        self.training_data = {
            'states': [],
            'policies': [],
            'values': []
        }
    
    def get_data_size(self) -> int:
        return len(self.training_data['states'])