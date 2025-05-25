import chess
import chess.svg
from typing import Optional, List
from src.engine.chess_game import ChessGame
from src.engine.neural_player import NeuralPlayer
from src.neural_network.chess_net import ChessNet
from src.training.user_learning import UserGameLearning
from src.utils.data_manager import DataManager

class GameInterface:
    def __init__(self, model: ChessNet, enable_learning: bool = False):
        self.ai_player = NeuralPlayer(model, "ChessEngine")
        self.game = ChessGame()
        self.game_history = []
        self.enable_learning = enable_learning
        
        if enable_learning:
            self.data_manager = DataManager()
            self.user_learning = UserGameLearning(model, self.data_manager)
        else:
            self.user_learning = None
        
    def reset_game(self, fen: Optional[str] = None):
        self.game.reset(fen)
        self.game_history = []
    
    def record_move(self, move_str: str, player: str):
        self.game_history.append({
            'move': move_str,
            'player': player,
            'fen': self.get_board_fen()
        })
    
    def make_move(self, move_str: str) -> bool:
        try:
            # Handle promotion moves
            if len(move_str) == 4 and self.is_promotion_move(move_str):
                # Auto-promote to Queen (most common choice)
                move_str += 'q'
            
            move = chess.Move.from_uci(move_str)
            if move in self.game.get_legal_moves():
                success = self.game.make_move(move)
                if success:
                    self.record_move(move_str, 'human')
                return success
        except Exception:
            try:
                move = self.game.board.parse_san(move_str)
                if move in self.game.get_legal_moves():
                    success = self.game.make_move(move)
                    if success:
                        self.record_move(move_str, 'human')
                    return success
            except Exception:
                return False
        return False
    
    def is_promotion_move(self, move_str: str) -> bool:
        """Check if a move is a pawn promotion"""
        if len(move_str) != 4:
            return False
        
        from_square = move_str[:2]
        to_square = move_str[2:4]
        
        # Check if it's a pawn moving to the 8th or 1st rank
        piece = self.game.board.piece_at(chess.parse_square(from_square))
        if piece and piece.piece_type == chess.PAWN:
            to_rank = int(to_square[1])
            return to_rank == 8 or to_rank == 1
        
        return False
    
    def get_ai_move(self, temperature: float = 0.1) -> Optional[str]:
        """Get and play the AI's move using the neural model."""
        if self.game.is_game_over():
            return None
        
        move = self.ai_player.get_move(self.game, temperature)
        if move:
            move_str = move.uci()
            self.game.make_move(move)
            self.record_move(move_str, 'ai')
            return move_str
        return None
    
    def get_legal_moves(self) -> List[str]:
        return [move.uci() for move in self.game.get_legal_moves()]
    
    def get_board_fen(self) -> str:
        return self.game.get_fen()
    
    def is_game_over(self) -> bool:
        return self.game.is_game_over()
    
    def get_game_result(self) -> Optional[str]:
        """Get a human-readable result if the game has ended."""
        if not self.is_game_over():
            return None
        
        result = self.game.get_game_result()
        if result == 1.0:
            return "White wins"
        elif result == -1.0:
            return "Black wins"
        else:
            return "Draw"
    
    def finish_game(self, user_was_white: bool):
        """Call this when a game finishes to enable learning"""
        if self.enable_learning and self.user_learning and self.is_game_over():
            result = self.game.get_game_result()
            if result is not None:
                # Convert result to user perspective
                user_result = result if user_was_white else -result
                ai_result = -user_result
                
                self.user_learning.record_user_game(self.game, user_was_white, ai_result)
                
                # Learn after every 5 games
                if len(self.user_learning.user_games) >= 5:
                    self.user_learning.learn_from_user_games()
    
    def get_game_status(self) -> dict:
        """Return current game state for external use."""
        status = {
            'fen': self.get_board_fen(),
            'legal_moves': self.get_legal_moves(),
            'is_game_over': self.is_game_over(),
            'game_result': self.get_game_result(),
            'move_count': self.game.move_count(),
            'current_player': 'white' if self.game.current_player() else 'black'
        }
        
        if self.enable_learning and self.user_learning:
            status['user_stats'] = self.user_learning.get_user_stats()
        
        return status

class ConsoleInterface:
    def __init__(self, model: ChessNet):
        self.game_interface = GameInterface(model)
        
    def display_board(self):
        print("\n" + str(self.game_interface.game.board))
        print(f"FEN: {self.game_interface.get_board_fen()}")
        
    def play_game(self):
        print("Welcome to Chess Engine!")
        print("Enter moves in UCI format (e.g., 'e2e4') or algebraic notation (e.g., 'e4')")
        print("Type 'quit' to exit, 'reset' to start a new game")
        
        human_color = input("Do you want to play as White or Black? (w/b): ").lower()
        human_is_white = human_color.startswith('w')
        
        if not human_is_white:
            print("AI plays first move...")
            ai_move = self.game_interface.get_ai_move()
            print(f"AI plays: {ai_move}")
        
        while True:
            self.display_board()
            
            if self.game_interface.is_game_over():
                result = self.game_interface.get_game_result()
                print(f"\nGame Over! Result: {result}")
                break
            
            current_turn = self.game_interface.game.current_player()
            is_human_turn = (current_turn == chess.WHITE and human_is_white) or \
                           (current_turn == chess.BLACK and not human_is_white)
            
            if is_human_turn:
                move_input = input(f"\nYour move ({'White' if current_turn else 'Black'}): ").strip()
                
                if move_input.lower() == 'quit':
                    break
                elif move_input.lower() == 'reset':
                    self.game_interface.reset_game()
                    print("Game reset!")
                    continue
                
                if self.game_interface.make_move(move_input):
                    print(f"You played: {move_input}")
                else:
                    print("Invalid move! Try again.")
                    continue
            else:
                print("AI is thinking...")
                ai_move = self.game_interface.get_ai_move()
                if ai_move:
                    print(f"AI plays: {ai_move}")
                else:
                    print("AI couldn't find a move!")
                    break