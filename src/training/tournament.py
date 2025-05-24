import os
import random
import chess
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from src.engine.chess_game import ChessGame
from src.engine.neural_player import NeuralPlayer, AdaptiveNeuralPlayer
from src.neural_network.chess_net import ChessNet
from src.utils.data_manager import DataManager
from src.utils.game_utils import create_random_position, get_opening_positions
from config import Config

class Tournament:
    def __init__(self, models: List[ChessNet], model_names: List[str] = None, use_adaptive: bool = False):
        self.models = models
        self.model_names = model_names or [f"Model_{i}" for i in range(len(models))]
        if use_adaptive:
            self.players = [AdaptiveNeuralPlayer(model, exploration_rate=0.05) for model in models]
        else:
            self.players = [NeuralPlayer(model) for model in models]
        
    def play_game(self, player1_idx: int, player2_idx: int, 
                  use_random_start: bool = True) -> Tuple[int, ChessGame]:
        if use_random_start and random.random() < 0.3:
            game = create_random_position(max_moves=15)
        elif random.random() < 0.1:
            opening_fens = get_opening_positions()
            fen = random.choice(opening_fens)
            game = ChessGame(fen)
        else:
            game = ChessGame()
        
        current_player_idx = player1_idx
        move_count = 0
        
        while not game.is_game_over() and move_count < Config.MAX_MOVES:
            temperature = 0.2 if move_count < 15 else 0.05
            
            current_player = self.players[current_player_idx]
            move = current_player.get_move(game, temperature)
            
            if move is None:
                break
            
            game.make_move(move)
            current_player_idx = player2_idx if current_player_idx == player1_idx else player1_idx
            move_count += 1
        
        result = game.get_game_result()
        if result is None:
            return -1, game  # Draw
        elif result == 1.0:
            return 0, game  # Player 1 (White) wins
        elif result == -1.0:
            return 1, game  # Player 2 (Black) wins
        else:
            return -1, game  # Draw
    
    def play_match(self, player1_idx: int, player2_idx: int, 
                   num_games: int = Config.TOURNAMENT_GAMES) -> Dict[str, int]:
        results = {'player1_wins': 0, 'player2_wins': 0, 'draws': 0}
        
        for game_num in range(num_games):
            if game_num % 2 == 0:
                white_idx, black_idx = player1_idx, player2_idx
            else:
                white_idx, black_idx = player2_idx, player1_idx
            
            winner, _ = self.play_game(white_idx, black_idx)
            
            if winner == -1:
                results['draws'] += 1
            elif (winner == 0 and white_idx == player1_idx) or (winner == 1 and black_idx == player1_idx):
                results['player1_wins'] += 1
            else:
                results['player2_wins'] += 1
        
        return results
    
    def run_round_robin(self, games_per_match: int = Config.TOURNAMENT_GAMES) -> Dict[str, Any]:
        num_models = len(self.models)
        results_matrix = {}
        
        print(f"Running round-robin tournament with {num_models} models...")
        
        for i in range(num_models):
            results_matrix[i] = {}
            for j in range(num_models):
                if i == j:
                    results_matrix[i][j] = {'player1_wins': 0, 'player2_wins': 0, 'draws': 0}
                    continue
                
                print(f"Match: {self.model_names[i]} vs {self.model_names[j]}")
                match_results = self.play_match(i, j, games_per_match)
                results_matrix[i][j] = match_results
        
        tournament_stats = self._calculate_tournament_stats(results_matrix)
        
        return {
            'results_matrix': results_matrix,
            'tournament_stats': tournament_stats,
            'model_names': self.model_names
        }
    
    def _calculate_tournament_stats(self, results_matrix: Dict) -> Dict[str, Any]:
        stats = {}
        num_models = len(self.models)
        
        for i in range(num_models):
            wins = 0
            losses = 0
            draws = 0
            total_games = 0
            
            for j in range(num_models):
                if i == j:
                    continue
                
                wins += results_matrix[i][j]['player1_wins']
                losses += results_matrix[i][j]['player2_wins']
                draws += results_matrix[i][j]['draws']
                total_games += sum(results_matrix[i][j].values())
            
            win_rate = wins / total_games if total_games > 0 else 0
            
            stats[self.model_names[i]] = {
                'wins': wins,
                'losses': losses,
                'draws': draws,
                'total_games': total_games,
                'win_rate': win_rate,
                'score': wins + 0.5 * draws
            }
        
        return stats

class GenerationalTraining:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.current_champion = None
        self.generation = 0
        
    def evaluate_candidate(self, candidate_model: ChessNet, 
                         champion_model: ChessNet = None) -> bool:
        if champion_model is None and self.current_champion is None:
            self.current_champion = candidate_model
            self.generation += 1
            return True
        
        champion = champion_model or self.current_champion
        
        tournament = Tournament([candidate_model, champion], 
                              [f"Candidate_Gen_{self.generation + 1}", f"Champion_Gen_{self.generation}"],
                              use_adaptive=True)
        
        print(f"Evaluating generation {self.generation + 1} candidate against current champion...")
        
        match_results = tournament.play_match(0, 1, Config.TOURNAMENT_GAMES)
        
        candidate_score = match_results['player1_wins'] + 0.5 * match_results['draws']
        total_games = sum(match_results.values())
        win_rate = candidate_score / total_games
        
        print(f"Candidate win rate: {win_rate:.3f}")
        print(f"Results: {match_results}")
        
        tournament_data = {
            'generation': self.generation + 1,
            'candidate_wins': match_results['player1_wins'],
            'champion_wins': match_results['player2_wins'],
            'draws': match_results['draws'],
            'candidate_win_rate': win_rate,
            'promoted': win_rate >= Config.WIN_RATE_THRESHOLD
        }
        
        self.data_manager.save_tournament_results(tournament_data)
        
        if win_rate >= Config.WIN_RATE_THRESHOLD:
            print(f"New champion! Generation {self.generation + 1} promoted.")
            self.current_champion = candidate_model
            self.generation += 1
            return True
        else:
            print(f"Candidate failed to beat champion. Win rate: {win_rate:.3f} < {Config.WIN_RATE_THRESHOLD}")
            return False
    
    def get_current_champion(self) -> Optional[ChessNet]:
        return self.current_champion
    
    def get_generation(self) -> int:
        return self.generation