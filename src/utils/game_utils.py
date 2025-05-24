import chess
import random
from typing import List, Optional
from src.engine.chess_game import ChessGame

def create_random_position(max_moves: int = 20) -> ChessGame:
    game = ChessGame()
    moves_made = random.randint(0, max_moves)
    
    for _ in range(moves_made):
        if game.is_game_over():
            break
        
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            break
            
        move = random.choice(legal_moves)
        game.make_move(move)
    
    return game

def get_opening_positions() -> List[str]:
    return [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    ]

def save_game_pgn(game: ChessGame, filename: str):
    import chess.pgn
    
    pgn_game = chess.pgn.Game()
    node = pgn_game
    
    temp_board = chess.Board()
    for move in game.move_history:
        node = node.add_variation(move)
        temp_board.push(move)
    
    with open(filename, 'w') as f:
        f.write(str(pgn_game))

def load_game_from_pgn(filename: str) -> List[ChessGame]:
    import chess.pgn
    
    games = []
    with open(filename) as f:
        while True:
            pgn_game = chess.pgn.read_game(f)
            if pgn_game is None:
                break
                
            game = ChessGame()
            for move in pgn_game.mainline_moves():
                game.make_move(move)
            games.append(game)
    
    return games