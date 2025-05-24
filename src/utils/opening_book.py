import chess
import random
from typing import Dict, List, Optional, Tuple
from src.engine.chess_game import ChessGame

class OpeningBook:
    def __init__(self):
        self.openings = {
            # Italian Game
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": ["e7e5"],
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2": ["g1f3"],
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2": ["b8c6"],
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3": ["f1c4"],
            
            # Spanish Opening (Ruy Lopez)
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3": ["f1b5", "f1c4"],
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3": ["a7a6", "f7f5", "g8f6"],
            
            # Queen's Gambit
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1": ["d7d5"],
            "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2": ["c2c4"],
            "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2": ["e7e6", "c7c6", "d5c4"],
            
            # King's Indian Defense
            "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2": ["g8f6"],
            "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 1 3": ["b1c3"],
            
            # Sicilian Defense
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": ["c7c5"],
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2": ["g1f3"],
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2": ["d7d6", "b8c6", "g7g6"],
            
            # French Defense
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": ["e7e6"],
            "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": ["d2d4"],
            "rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2": ["d7d5"],
            
            # Caro-Kann Defense  
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": ["c7c6"],
            "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": ["d2d4"],
            "rnbqkbnr/pp1ppppp/2p5/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2": ["d7d5"],
            
            # English Opening
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": ["c2c4"],
            "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1": ["e7e5", "g8f6", "c7c5"],
            
            # Nimzo-Indian Defense
            "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2": ["g8f6"],
            "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 1 3": ["b1c3"],
            "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 2 3": ["f8b4"],
        }
        
        # Add some creative and aggressive variations
        self.creative_openings = {
            # King's Gambit
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2": ["f2f4"],
            
            # Bird's Opening
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": ["f2f4"],
            
            # Scandinavian Defense
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": ["d7d5"],
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2": ["e4d5"],
            
            # Alekhine's Defense
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": ["g8f6"],
            "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2": ["e4e5"],
        }
    
    def get_opening_move(self, game: ChessGame, creativity: float = 0.3) -> Optional[str]:
        """Get an opening book move if available"""
        fen = game.get_fen()
        position_key = ' '.join(fen.split()[:4])  # Position without clocks
        
        # Check main openings first
        if position_key in self.openings:
            moves = self.openings[position_key]
            
            # Sometimes use creative alternatives
            if random.random() < creativity and position_key in self.creative_openings:
                creative_moves = self.creative_openings[position_key]
                moves = moves + creative_moves
            
            if moves:
                move_str = random.choice(moves)
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in game.get_legal_moves():
                        return move_str
                except:
                    pass
        
        # Check creative openings
        if position_key in self.creative_openings and random.random() < creativity:
            moves = self.creative_openings[position_key]
            if moves:
                move_str = random.choice(moves)
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in game.get_legal_moves():
                        return move_str
                except:
                    pass
        
        return None
    
    def is_in_opening(self, game: ChessGame, max_moves: int = 15) -> bool:
        """Check if we're still in the opening phase"""
        return game.move_count() <= max_moves
    
    def add_opening_line(self, fen: str, moves: List[str]):
        """Add a new opening line to the book"""
        position_key = ' '.join(fen.split()[:4])
        if position_key in self.openings:
            self.openings[position_key].extend(moves)
        else:
            self.openings[position_key] = moves
    
    def get_book_stats(self) -> Dict[str, int]:
        """Get statistics about the opening book"""
        total_positions = len(self.openings) + len(self.creative_openings)
        total_moves = sum(len(moves) for moves in self.openings.values())
        total_moves += sum(len(moves) for moves in self.creative_openings.values())
        
        return {
            'total_positions': total_positions,
            'total_moves': total_moves,
            'main_openings': len(self.openings),
            'creative_openings': len(self.creative_openings)
        }

class BookPlayer:
    """Player that uses opening book when available"""
    
    def __init__(self, base_player, opening_book: OpeningBook, book_usage: float = 0.8):
        self.base_player = base_player
        self.opening_book = opening_book
        self.book_usage = book_usage  # Probability of using book moves
        self.book_moves_played = 0
        
    def get_move(self, game: ChessGame, temperature: float = 1.0):
        # Try opening book first if in opening phase
        if (self.opening_book.is_in_opening(game) and 
            random.random() < self.book_usage):
            
            creativity = min(temperature, 0.5)  # More creative with higher temperature
            book_move = self.opening_book.get_opening_move(game, creativity)
            
            if book_move:
                self.book_moves_played += 1
                try:
                    return chess.Move.from_uci(book_move)
                except:
                    pass
        
        # Fall back to base player
        return self.base_player.get_move(game, temperature)
    
    def get_stats(self) -> Dict[str, any]:
        return {
            'book_moves_played': self.book_moves_played,
            'book_stats': self.opening_book.get_book_stats(),
            'base_player': type(self.base_player).__name__
        }