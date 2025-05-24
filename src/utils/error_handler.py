import logging
import traceback
import functools
from typing import Any, Callable, Optional
import torch

class ChessEngineError(Exception):
    """Base exception for chess engine errors"""
    pass

class ModelError(ChessEngineError):
    """Neural network model errors"""
    pass

class GameError(ChessEngineError):
    """Chess game logic errors"""
    pass

class TrainingError(ChessEngineError):
    """Training process errors"""
    pass

def safe_execute(default_return=None, log_errors=True):
    """Decorator for safe function execution with error handling"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logging.error(f"Error in {func.__name__}: {str(e)}")
                    logging.debug(traceback.format_exc())
                
                # Return safe default
                if default_return is not None:
                    return default_return
                elif hasattr(func, '__annotations__') and func.__annotations__.get('return'):
                    # Try to return appropriate default based on return type
                    return_type = func.__annotations__['return']
                    if return_type == bool:
                        return False
                    elif return_type == list:
                        return []
                    elif return_type == dict:
                        return {}
                
                return None
        return wrapper
    return decorator

def handle_errors(func: Callable) -> Callable:
    """Decorator for handling errors in class methods"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            logging.debug(traceback.format_exc())
            return None
    return wrapper

def validate_move(move_str: str) -> bool:
    """Validate move string format"""
    if not isinstance(move_str, str):
        return False
    
    # Basic UCI format validation
    if len(move_str) < 4 or len(move_str) > 5:
        return False
    
    # Check file/rank format
    if not (move_str[0].isalpha() and move_str[1].isdigit() and 
            move_str[2].isalpha() and move_str[3].isdigit()):
        return False
    
    # Check valid squares
    if not ('a' <= move_str[0] <= 'h' and '1' <= move_str[1] <= '8' and
            'a' <= move_str[2] <= 'h' and '1' <= move_str[3] <= '8'):
        return False
    
    # Check promotion piece if present
    if len(move_str) == 5:
        if move_str[4] not in 'qrbn':
            return False
    
    return True

def validate_fen(fen: str) -> bool:
    """Validate FEN string format"""
    if not isinstance(fen, str):
        return False
    
    parts = fen.split()
    if len(parts) != 6:
        return False
    
    # Basic position validation
    ranks = parts[0].split('/')
    if len(ranks) != 8:
        return False
    
    return True

def safe_tensor_operation(operation: Callable, *args, **kwargs) -> Optional[torch.Tensor]:
    """Safely execute tensor operations with memory cleanup"""
    try:
        result = operation(*args, **kwargs)
        return result
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logging.error("GPU out of memory - clearing cache and retrying")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            try:
                return operation(*args, **kwargs)
            except:
                logging.error("Retry failed - operation aborted")
                return None
        else:
            logging.error(f"Tensor operation failed: {e}")
            return None
    except Exception as e:
        logging.error(f"Unexpected error in tensor operation: {e}")
        return None

class ErrorLogger:
    def __init__(self):
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/chess_engine.log'),
                logging.StreamHandler()
            ]
        )
    
    def log_training_error(self, iteration: int, error: Exception):
        logging.error(f"Training failed at iteration {iteration}: {error}")
    
    def log_game_error(self, game_id: int, error: Exception):
        logging.error(f"Game {game_id} failed: {error}")
    
    def log_model_error(self, operation: str, error: Exception):
        logging.error(f"Model operation '{operation}' failed: {error}")

# Global error logger
error_logger = ErrorLogger()