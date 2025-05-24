import chess
import numpy as np
import torch
from typing import Optional, Tuple
from src.engine.chess_game import ChessGame
from src.mcts.mcts_node import MCTSNode
from src.neural_network.chess_net import ChessNet
from config import Config

class MCTS:
    def __init__(self, model: ChessNet, simulations: int = Config.MCTS_SIMULATIONS, 
                 c_puct: float = Config.MCTS_C_PUCT):
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        
    def search(self, game: ChessGame, temperature: float = 1.0) -> Tuple[chess.Move, np.ndarray]:
        root = MCTSNode(game)
        
        for _ in range(self.simulations):
            self._simulate(root)
        
        policy_target = root.get_policy_target()
        best_move = root.get_best_move(temperature)
        
        return best_move, policy_target
    
    def _simulate(self, root: MCTSNode):
        node = root
        path = [node]
        
        while not node.is_leaf() and not node.is_terminal():
            node = node.select_child(self.c_puct)
            path.append(node)
        
        if node.is_terminal():
            value = self._get_terminal_value(node)
        else:
            value = self._expand_and_evaluate(node)
        
        for node in reversed(path):
            node.backup(value)
            value = -value
    
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        state_tensor = node.game.get_state_planes()
        state_tensor = state_tensor.permute(2, 0, 1).unsqueeze(0)
        
        policy_probs, value = self.model.predict(state_tensor)
        
        legal_mask = node.game.get_move_probabilities_mask().cpu().numpy()
        policy_probs = policy_probs * legal_mask
        
        if np.sum(policy_probs) > 0:
            policy_probs = policy_probs / np.sum(policy_probs)
        else:
            policy_probs = legal_mask / np.sum(legal_mask) if np.sum(legal_mask) > 0 else policy_probs
        
        node.expand(policy_probs)
        
        return value
    
    def _get_terminal_value(self, node: MCTSNode) -> float:
        result = node.game.get_game_result()
        
        if result is None:
            return 0.0
        
        current_player = node.game.current_player()
        
        if result == 1.0:
            return 1.0 if current_player == chess.WHITE else -1.0
        elif result == -1.0:
            return -1.0 if current_player == chess.WHITE else 1.0
        else:
            return 0.0
    
    def get_move_probabilities(self, game: ChessGame, temperature: float = 1.0) -> np.ndarray:
        _, policy = self.search(game, temperature)
        return policy
    
    def get_best_move(self, game: ChessGame, temperature: float = 1.0) -> Optional[chess.Move]:
        move, _ = self.search(game, temperature)
        return move

class MCTSPlayer:
    def __init__(self, model: ChessNet, simulations: int = Config.MCTS_SIMULATIONS):
        self.mcts = MCTS(model, simulations)
        self.name = f"MCTS_{simulations}"
    
    def get_move(self, game: ChessGame, temperature: float = 1.0) -> Optional[chess.Move]:
        return self.mcts.get_best_move(game, temperature)
    
    def get_policy(self, game: ChessGame, temperature: float = 1.0) -> np.ndarray:
        return self.mcts.get_move_probabilities(game, temperature)