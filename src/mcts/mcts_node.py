import chess
import numpy as np
from typing import Optional, Dict, List
from src.engine.chess_game import ChessGame

class MCTSNode:
    def __init__(self, game: ChessGame, parent: Optional['MCTSNode'] = None, 
                 move: Optional[chess.Move] = None, prior_prob: float = 0.0):
        self.game = game.copy()
        self.parent = parent
        self.move = move
        self.prior_prob = prior_prob
        
        self.children: Dict[chess.Move, 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        
    def is_leaf(self) -> bool:
        return not self.is_expanded
    
    def is_terminal(self) -> bool:
        return self.game.is_game_over()
    
    def get_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def get_ucb_score(self, c_puct: float = 1.0) -> float:
        if self.visit_count == 0:
            return float('inf')
        
        exploration = c_puct * self.prior_prob * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.get_value() + exploration
    
    def select_child(self, c_puct: float = 1.0) -> Optional['MCTSNode']:
        if not self.children:
            return None
        
        best_move = max(self.children.keys(), 
                       key=lambda move: self.children[move].get_ucb_score(c_puct))
        return self.children[best_move]
    
    def expand(self, policy_probs: np.ndarray):
        if self.is_expanded or self.is_terminal():
            return
        
        legal_moves = self.game.get_legal_moves()
        if not legal_moves:
            return
        
        for move in legal_moves:
            move_index = ChessGame.move_to_index(move)
            prior_prob = policy_probs[move_index] if move_index < len(policy_probs) else 0.0
            
            child_game = self.game.copy()
            child_game.make_move(move)
            
            child_node = MCTSNode(child_game, parent=self, move=move, prior_prob=prior_prob)
            self.children[move] = child_node
        
        self.is_expanded = True
    
    def backup(self, value: float):
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent:
            self.parent.backup(-value)
    
    def get_visit_distribution(self, temperature: float = 1.0) -> Dict[chess.Move, float]:
        if not self.children:
            return {}
        
        visits = np.array([child.visit_count for child in self.children.values()])
        
        if temperature == 0:
            probs = np.zeros_like(visits, dtype=float)
            best_idx = np.argmax(visits)
            probs[best_idx] = 1.0
        else:
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / np.sum(visits_temp)
        
        return {move: prob for move, prob in zip(self.children.keys(), probs)}
    
    def get_best_move(self, temperature: float = 1.0) -> Optional[chess.Move]:
        if not self.children:
            return None
        
        visit_dist = self.get_visit_distribution(temperature)
        return max(visit_dist.keys(), key=lambda move: visit_dist[move])
    
    def get_policy_target(self) -> np.ndarray:
        policy = np.zeros(4096 + 4 * 4096)
        
        if not self.children:
            return policy
        
        total_visits = sum(child.visit_count for child in self.children.values())
        if total_visits == 0:
            return policy
        
        for move, child in self.children.items():
            move_index = ChessGame.move_to_index(move)
            policy[move_index] = child.visit_count / total_visits
        
        return policy