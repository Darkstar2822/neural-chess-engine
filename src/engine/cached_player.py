import hashlib
import numpy as np
import torch
from typing import Optional, Dict, Tuple
from src.engine.chess_game import ChessGame
from src.engine.neural_player import NeuralPlayer
from src.neural_network.chess_net import ChessNet

class CachedNeuralPlayer(NeuralPlayer):
    def __init__(self, model: ChessNet, name: str = "CachedNeuralEngine", cache_size: int = 10000):
        super().__init__(model, name)
        self.position_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
    def get_position_hash(self, game: ChessGame) -> str:
        """Generate a unique hash for the current position"""
        fen = game.get_fen()
        # Only use position part of FEN (not move clocks) for caching
        position_fen = ' '.join(fen.split()[:4])
        return hashlib.md5(position_fen.encode()).hexdigest()
    
    def get_policy_distribution(self, game: ChessGame) -> np.ndarray:
        position_hash = self.get_position_hash(game)
        
        # Check cache first
        if position_hash in self.position_cache:
            self.cache_hits += 1
            cached_policy, _ = self.position_cache[position_hash]
            return cached_policy
        
        # Cache miss - compute normally
        self.cache_misses += 1
        state_tensor = game.get_state_planes().permute(2, 0, 1).unsqueeze(0)
        policy_probs, value = self.model.predict(state_tensor)
        
        legal_mask = game.get_move_probabilities_mask().numpy()
        masked_probs = policy_probs * legal_mask
        
        if np.sum(masked_probs) > 0:
            masked_probs = masked_probs / np.sum(masked_probs)
        
        # Add to cache
        self.add_to_cache(position_hash, masked_probs, value)
        
        return masked_probs
    
    def evaluate_position(self, game: ChessGame) -> float:
        position_hash = self.get_position_hash(game)
        
        # Check cache first
        if position_hash in self.position_cache:
            self.cache_hits += 1
            _, cached_value = self.position_cache[position_hash]
            return cached_value
        
        # Cache miss - compute normally
        self.cache_misses += 1
        state_tensor = game.get_state_planes().permute(2, 0, 1).unsqueeze(0)
        _, value = self.model.predict(state_tensor)
        
        # Add to cache (with dummy policy)
        dummy_policy = np.zeros(4096 + 4 * 4096)
        self.add_to_cache(position_hash, dummy_policy, value)
        
        return value
    
    def add_to_cache(self, position_hash: str, policy: np.ndarray, value: float):
        """Add position to cache with LRU eviction"""
        if len(self.position_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self.position_cache))
            del self.position_cache[oldest_key]
        
        self.position_cache[position_hash] = (policy.copy(), value)
    
    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_size': len(self.position_cache),
            'max_cache_size': self.cache_size
        }
    
    def clear_cache(self):
        """Clear the position cache"""
        self.position_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

class SmartCachedPlayer(CachedNeuralPlayer):
    """Enhanced cached player with smart cache management"""
    
    def __init__(self, model: ChessNet, name: str = "SmartCachedEngine", cache_size: int = 50000):
        super().__init__(model, name, cache_size)
        self.access_count: Dict[str, int] = {}
        self.position_complexity: Dict[str, float] = {}
    
    def get_position_complexity(self, game: ChessGame) -> float:
        """Estimate position complexity for cache prioritization"""
        legal_moves = game.get_legal_moves()
        piece_count = len([sq for sq in game.board.piece_map()])
        
        # More complex positions = more legal moves, fewer pieces (endgame)
        complexity = len(legal_moves) * (1.0 + (32 - piece_count) / 32.0)
        return complexity
    
    def add_to_cache(self, position_hash: str, policy: np.ndarray, value: float):
        """Smart cache management with complexity-based prioritization"""
        if position_hash in self.position_cache:
            # Update access count
            self.access_count[position_hash] = self.access_count.get(position_hash, 0) + 1
            self.position_cache[position_hash] = (policy.copy(), value)
            return
        
        if len(self.position_cache) >= self.cache_size:
            # Evict least valuable position (low access count + low complexity)
            min_score = float('inf')
            evict_key = None
            
            for key in list(self.position_cache.keys())[:100]:  # Check first 100 for efficiency
                access_score = self.access_count.get(key, 1)
                complexity_score = self.position_complexity.get(key, 1.0)
                combined_score = access_score * complexity_score
                
                if combined_score < min_score:
                    min_score = combined_score
                    evict_key = key
            
            if evict_key:
                del self.position_cache[evict_key]
                self.access_count.pop(evict_key, None)
                self.position_complexity.pop(evict_key, None)
        
        # Add new position
        self.position_cache[position_hash] = (policy.copy(), value)
        self.access_count[position_hash] = 1