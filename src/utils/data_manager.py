import os
import pickle
import json
import datetime
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from src.engine.chess_game import ChessGame
from config import Config

class DataManager:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.games_dir = os.path.join(base_dir, "games")
        self.training_dir = os.path.join(base_dir, "training")
        self.models_dir = os.path.join(base_dir, "models")
        self.analysis_dir = os.path.join(base_dir, "analysis")
        
        self._create_directories()
    
    def _create_directories(self):
        os.makedirs(self.games_dir, exist_ok=True)
        os.makedirs(self.training_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)
    
    def save_game(self, game: ChessGame, metadata: Dict[str, Any] = None) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"game_{timestamp}.pkl"
        filepath = os.path.join(self.games_dir, filename)
        
        game_data = {
            'moves': [str(move) for move in game.move_history],
            'final_fen': game.get_fen(),
            'result': game.get_game_result(),
            'move_count': game.move_count(),
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(game_data, f)
        
        return filepath
    
    def load_game(self, filepath: str) -> Tuple[ChessGame, Dict[str, Any]]:
        with open(filepath, 'rb') as f:
            game_data = pickle.load(f)
        
        game = ChessGame()
        for move_str in game_data['moves']:
            move = game.board.parse_san(move_str) if move_str else None
            if move:
                game.make_move(move)
        
        return game, game_data['metadata']
    
    def save_training_data(self, states: np.ndarray, policies: np.ndarray, 
                          values: np.ndarray, epoch: int) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_data_epoch_{epoch}_{timestamp}.npz"
        filepath = os.path.join(self.training_dir, filename)
        
        np.savez_compressed(
            filepath,
            states=states,
            policies=policies,
            values=values,
            epoch=epoch,
            timestamp=timestamp
        )
        
        return filepath
    
    def load_training_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = np.load(filepath)
        return data['states'], data['policies'], data['values']
    
    def save_model_checkpoint(self, model, optimizer, epoch: int, 
                            training_history: List[Dict], metadata: Dict = None) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_epoch_{epoch}_{timestamp}.pt"
        filepath = os.path.join(self.models_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_history': training_history,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        torch.save(checkpoint, filepath)
        return filepath
    
    def load_model_checkpoint(self, filepath: str) -> Dict[str, Any]:
        return torch.load(filepath, map_location=Config.DEVICE)
    
    def save_tournament_results(self, results: Dict[str, Any]) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tournament_{timestamp}.json"
        filepath = os.path.join(self.analysis_dir, filename)
        
        results['timestamp'] = timestamp
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return filepath
    
    def get_latest_model(self) -> Optional[str]:
        if not os.path.exists(self.models_dir):
            return None
        
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pt')]
        if not model_files:
            return None
        
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.models_dir, x)))
        return os.path.join(self.models_dir, model_files[-1])
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        history = []
        
        if not os.path.exists(self.analysis_dir):
            return history
        
        for filename in os.listdir(self.analysis_dir):
            if filename.startswith('tournament_') and filename.endswith('.json'):
                filepath = os.path.join(self.analysis_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    history.append(data)
        
        history.sort(key=lambda x: x.get('timestamp', ''))
        return history
    
    def cleanup_old_files(self, days_to_keep: int = 30):
        cutoff_time = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
        
        for directory in [self.games_dir, self.training_dir]:
            if not os.path.exists(directory):
                continue
                
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                file_time = datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
                
                if file_time < cutoff_time:
                    os.remove(filepath)
                    print(f"Removed old file: {filepath}")

class GameAnalyzer:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
    
    def analyze_game(self, game: ChessGame) -> Dict[str, Any]:
        analysis = {
            'move_count': game.move_count(),
            'final_result': game.get_game_result(),
            'game_length_category': self._categorize_game_length(game.move_count()),
            'opening_moves': [str(move) for move in game.move_history[:10]],
            'game_phases': self._analyze_game_phases(game)
        }
        
        return analysis
    
    def _categorize_game_length(self, move_count: int) -> str:
        if move_count < 20:
            return "very_short"
        elif move_count < 40:
            return "short"
        elif move_count < 80:
            return "medium"
        elif move_count < 120:
            return "long"
        else:
            return "very_long"
    
    def _analyze_game_phases(self, game: ChessGame) -> Dict[str, int]:
        move_count = game.move_count()
        
        return {
            'opening': min(15, move_count),
            'middlegame': max(0, min(move_count - 15, 40)),
            'endgame': max(0, move_count - 55)
        }