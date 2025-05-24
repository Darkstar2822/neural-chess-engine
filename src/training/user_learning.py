import numpy as np
from typing import List, Tuple
from src.engine.chess_game import ChessGame
from src.neural_network.chess_net import ChessNet
from src.neural_network.trainer import ChessNetTrainer
from src.utils.data_manager import DataManager

class UserGameLearning:
    def __init__(self, model: ChessNet, data_manager: DataManager):
        self.model = model
        self.trainer = ChessNetTrainer(model, learning_rate=0.0001)  # Lower learning rate
        self.data_manager = data_manager
        self.user_games = []
        
    def record_user_game(self, game: ChessGame, user_was_white: bool, result: float):
        """Record a game against a user for learning"""
        states = []
        policies = []
        values = []
        
        # Replay the game to get training data
        temp_game = ChessGame()
        for i, move in enumerate(game.move_history):
            # Get current state
            state = temp_game.get_state_planes().permute(2, 0, 1).cpu().numpy()
            
            # Get model's policy for this position
            policy_dist = self.model.predict(
                temp_game.get_state_planes().permute(2, 0, 1).unsqueeze(0)
            )[0]
            
            # Apply legal move mask
            legal_mask = temp_game.get_move_probabilities_mask().numpy()
            policy_dist = policy_dist * legal_mask
            if np.sum(policy_dist) > 0:
                policy_dist = policy_dist / np.sum(policy_dist)
            
            # Determine if this was the AI's move
            ai_move = (i % 2 == 0 and not user_was_white) or (i % 2 == 1 and user_was_white)
            
            if ai_move:  # Only learn from AI's positions
                states.append(state)
                policies.append(policy_dist)
                
                # Value from AI's perspective
                ai_result = result if not user_was_white else -result
                values.append(ai_result)
            
            temp_game.make_move(move)
        
        if states:  # Only add if we have AI positions
            self.user_games.append({
                'states': states,
                'policies': policies,
                'values': values,
                'user_was_white': user_was_white,
                'result': result
            })
            
            print(f"Recorded user game: {len(states)} AI positions, result: {result}")
    
    def learn_from_user_games(self, min_games: int = 5):
        """Learn from accumulated user games"""
        if len(self.user_games) < min_games:
            print(f"Need at least {min_games} user games to learn. Currently have {len(self.user_games)}")
            return
        
        # Combine all user game data
        all_states = []
        all_policies = []
        all_values = []
        
        for game_data in self.user_games:
            all_states.extend(game_data['states'])
            all_policies.extend(game_data['policies'])
            all_values.extend(game_data['values'])
        
        if not all_states:
            print("No training data from user games")
            return
        
        print(f"Learning from {len(all_states)} positions from {len(self.user_games)} user games...")
        
        # Train on user game data with small learning rate
        training_data = (
            np.array(all_states),
            np.array(all_policies),
            np.array(all_values)
        )
        
        # Train for just 1-2 epochs to avoid overfitting
        self.trainer.train(training_data, epochs=2)
        
        # Clear old games to prevent overfitting
        self.user_games = []
        
        print("Completed learning from user games!")
    
    def get_user_stats(self) -> dict:
        """Get statistics about games against users"""
        if not self.user_games:
            return {'total_games': 0}
        
        wins = sum(1 for game in self.user_games if game['result'] > 0)
        losses = sum(1 for game in self.user_games if game['result'] < 0)
        draws = sum(1 for game in self.user_games if game['result'] == 0)
        
        return {
            'total_games': len(self.user_games),
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / len(self.user_games) if self.user_games else 0
        }