import chess
import chess.engine
import numpy as np
import torch
from typing import List, Tuple, Optional

class ChessGame:
    def __init__(self, fen: Optional[str] = None):
        self.board = chess.Board(fen) if fen else chess.Board()
        self.move_history = []
        
    def reset(self, fen: Optional[str] = None):
        self.board = chess.Board(fen) if fen else chess.Board()
        self.move_history = []
    
    def copy(self):
        game = ChessGame()
        game.board = self.board.copy()
        game.move_history = self.move_history.copy()
        return game
    
    def get_legal_moves(self) -> List[chess.Move]:
        return list(self.board.legal_moves)
    
    def make_move(self, move: chess.Move) -> bool:
        if move in self.board.legal_moves:
            self.move_history.append(move)
            self.board.push(move)
            return True
        return False
    
    def unmake_move(self):
        if self.move_history:
            self.board.pop()
            self.move_history.pop()
    
    def is_game_over(self) -> bool:
        return self.board.is_game_over()
    
    def get_game_result(self) -> Optional[float]:
        if not self.is_game_over():
            return None
        
        result = self.board.result()
        if result == "1-0":
            return 1.0
        elif result == "0-1": 
            return -1.0
        else:
            return 0.0
    
    def current_player(self) -> bool:
        return self.board.turn
    
    def get_fen(self) -> str:
        return self.board.fen()
    
    def move_count(self) -> int:
        return len(self.move_history)
    
    def get_board_tensor(self) -> torch.Tensor:
        tensor = torch.zeros(8, 8, 12)
        
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row = square // 8
                col = square % 8
                piece_type = piece_map[piece.piece_type]
                plane = piece_type + (6 if piece.color == chess.BLACK else 0)
                tensor[row, col, plane] = 1
        
        return tensor
    
    def get_state_planes(self) -> torch.Tensor:
        planes = torch.zeros(8, 8, 16)
        
        planes[:, :, :12] = self.get_board_tensor()
        
        if self.board.turn == chess.WHITE:
            planes[:, :, 12] = 1
        else:
            planes[:, :, 13] = 1
            
        planes[:, :, 14] = self.board.has_kingside_castling_rights(chess.WHITE)
        planes[:, :, 15] = self.board.has_queenside_castling_rights(chess.WHITE)
        
        return planes
    
    @staticmethod
    def move_to_index(move: chess.Move) -> int:
        from_square = move.from_square
        to_square = move.to_square
        
        if move.promotion:
            promotion_offset = {
                chess.QUEEN: 0, chess.ROOK: 1, 
                chess.BISHOP: 2, chess.KNIGHT: 3
            }
            return 4096 + from_square * 64 + to_square + promotion_offset[move.promotion] * 4096
        
        return from_square * 64 + to_square
    
    @staticmethod
    def index_to_move(index: int) -> chess.Move:
        if index >= 4096:
            index -= 4096
            promotion_type = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT][index // 4096]
            index %= 4096
            from_square = index // 64
            to_square = index % 64
            return chess.Move(from_square, to_square, promotion=promotion_type)
        
        from_square = index // 64
        to_square = index % 64
        return chess.Move(from_square, to_square)
    
    def get_move_probabilities_mask(self) -> torch.Tensor:
        mask = torch.zeros(4096 + 4 * 4096)
        legal_moves = self.get_legal_moves()
        
        for move in legal_moves:
            index = self.move_to_index(move)
            mask[index] = 1
            
        return mask