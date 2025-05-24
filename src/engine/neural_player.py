import chess
import numpy as np
import torch
from typing import Optional, List, Tuple
from src.engine.chess_game import ChessGame
from src.neural_network.chess_net import ChessNet

class NeuralPlayer:
    def __init__(self, model: ChessNet, name: str = "NeuralEngine"):
        self.model = model
        self.name = name
        
    def get_move(self, game: ChessGame, temperature: float = 1.0) -> Optional[chess.Move]:
        if game.is_game_over():
            return None
        
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None
        
        state_tensor = game.get_state_planes().permute(2, 0, 1).unsqueeze(0)
        policy_probs, value = self.model.predict(state_tensor)
        
        legal_mask = game.get_move_probabilities_mask().numpy()
        masked_probs = policy_probs * legal_mask
        
        if np.sum(masked_probs) == 0:
            return np.random.choice(legal_moves)
        
        masked_probs = masked_probs / np.sum(masked_probs)
        
        if temperature == 0:
            best_move_idx = np.argmax(masked_probs)
            move = ChessGame.index_to_move(best_move_idx)
            return move if move in legal_moves else np.random.choice(legal_moves)
        
        move_indices = np.where(legal_mask > 0)[0]
        move_probs = masked_probs[move_indices]
        
        if temperature != 1.0:
            move_probs = move_probs ** (1.0 / temperature)
            move_probs = move_probs / np.sum(move_probs)
        
        chosen_idx = np.random.choice(move_indices, p=move_probs)
        move = ChessGame.index_to_move(chosen_idx)
        
        return move if move in legal_moves else np.random.choice(legal_moves)
    
    def get_policy_distribution(self, game: ChessGame) -> np.ndarray:
        state_tensor = game.get_state_planes().permute(2, 0, 1).unsqueeze(0)
        policy_probs, _ = self.model.predict(state_tensor)
        
        legal_mask = game.get_move_probabilities_mask().numpy()
        masked_probs = policy_probs * legal_mask
        
        if np.sum(masked_probs) > 0:
            masked_probs = masked_probs / np.sum(masked_probs)
        
        return masked_probs
    
    def evaluate_position(self, game: ChessGame) -> float:
        state_tensor = game.get_state_planes().permute(2, 0, 1).unsqueeze(0)
        _, value = self.model.predict(state_tensor)
        return value

class DirectNeuralTraining:
    def __init__(self, model: ChessNet):
        self.model = model
        self.player = NeuralPlayer(model)
        
    def play_training_game(self, opponent_player=None, use_random_start: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        if use_random_start and np.random.random() < 0.3:
            from src.utils.game_utils import create_random_position
            game = create_random_position(max_moves=15)
        else:
            game = ChessGame()
        
        states = []
        policies = []
        move_count = 0
        max_moves = 200
        
        while not game.is_game_over() and move_count < max_moves:
            temperature = 1.2 if move_count < 10 else 0.8 if move_count < 30 else 0.3
            
            state = game.get_state_planes().permute(2, 0, 1).cpu().numpy()
            policy_dist = self.player.get_policy_distribution(game)
            
            states.append(state)
            policies.append(policy_dist)
            
            current_player = self.player if move_count % 2 == 0 else (opponent_player or self.player)
            move = current_player.get_move(game, temperature)
            
            if move is None:
                break
                
            game.make_move(move)
            move_count += 1
        
        game_result = game.get_game_result()
        if game_result is None:
            game_result = 0.0
        
        values = []
        for i in range(len(states)):
            player_perspective = 1 if i % 2 == 0 else -1
            values.append(game_result * player_perspective)
        
        return states, policies, values

class AdaptiveNeuralPlayer(NeuralPlayer):
    def __init__(self, model: ChessNet, exploration_rate: float = 0.1):
        super().__init__(model)
        self.exploration_rate = exploration_rate
        self.position_memory = {}
        
    def get_move(self, game: ChessGame, temperature: float = 1.0) -> Optional[chess.Move]:
        if game.is_game_over():
            return None
        
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None
        
        fen = game.get_fen()
        
        if np.random.random() < self.exploration_rate:
            return np.random.choice(legal_moves)
        
        state_tensor = game.get_state_planes().permute(2, 0, 1).unsqueeze(0)
        policy_probs, value = self.model.predict(state_tensor)
        
        legal_mask = game.get_move_probabilities_mask().numpy()
        masked_probs = policy_probs * legal_mask
        
        if np.sum(masked_probs) == 0:
            return np.random.choice(legal_moves)
        
        if fen in self.position_memory:
            previous_moves = self.position_memory[fen]
            for move_idx in previous_moves:
                if move_idx < len(masked_probs):
                    masked_probs[move_idx] *= 0.7
        
        masked_probs = masked_probs / np.sum(masked_probs)
        
        if temperature == 0:
            best_move_idx = np.argmax(masked_probs)
        else:
            move_indices = np.where(legal_mask > 0)[0]
            move_probs = masked_probs[move_indices]
            
            if temperature != 1.0:
                move_probs = move_probs ** (1.0 / temperature)
                move_probs = move_probs / np.sum(move_probs)
            
            best_move_idx = np.random.choice(move_indices, p=move_probs)
        
        move = ChessGame.index_to_move(best_move_idx)
        
        if move in legal_moves:
            if fen not in self.position_memory:
                self.position_memory[fen] = []
            self.position_memory[fen].append(best_move_idx)
            return move
        
        return np.random.choice(legal_moves)