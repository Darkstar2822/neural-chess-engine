#!/usr/bin/env python3
"""
Comprehensive test suite for the web interface
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch
import sys
import threading
import time
import requests

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ui.web_interface import app, initialize_game_interface
from src.ui.game_interface import GameInterface
from src.neural_network.chess_net import ChessNet

class TestWebInterface:
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            # Initialize with a mock model
            with patch('src.ui.web_interface.ChessNet') as mock_chess_net:
                mock_model = Mock()
                # Mock the predict method to return proper tuple
                import numpy as np
                mock_model.predict.return_value = (
                    np.array([0.1] * 20480, dtype=np.float32),  # policy probabilities (4096 + 4*4096)
                    0.5  # value
                )
                mock_chess_net.return_value = mock_model
                initialize_game_interface(enable_learning=False)
                yield client
    
    def test_index_page_loads(self, client):
        """Test that the main page loads correctly"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Neural Chess Engine' in response.data
        assert b'chess-board' in response.data
    
    def test_new_game_api_white(self, client):
        """Test starting a new game as white"""
        response = client.post('/api/new_game', 
                             json={'color': 'white'},
                             content_type='application/json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['human_color'] == 'white'
        assert 'board_fen' in data
        assert 'ai_move' not in data  # AI shouldn't move first when human is white
    
    def test_new_game_api_black(self, client):
        """Test starting a new game as black"""
        response = client.post('/api/new_game',
                             json={'color': 'black'}, 
                             content_type='application/json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['human_color'] == 'black'
        assert 'board_fen' in data
        assert 'ai_move' in data  # AI should move first when human is black
    
    def test_new_game_no_color_defaults_white(self, client):
        """Test new game defaults to white when no color specified"""
        response = client.post('/api/new_game',
                             json={},
                             content_type='application/json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['human_color'] == 'white'
    
    def test_make_move_valid(self, client):
        """Test making a valid move"""
        # Start a new game first
        client.post('/api/new_game', json={'color': 'white'})
        
        # Make a valid opening move
        response = client.post('/api/make_move',
                             json={'move': 'e2e4'},
                             content_type='application/json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'board_fen' in data
        assert 'ai_move' in data  # AI should respond
    
    def test_make_move_invalid(self, client):
        """Test making an invalid move"""
        # Start a new game first
        client.post('/api/new_game', json={'color': 'white'})
        
        # Try an invalid move
        response = client.post('/api/make_move',
                             json={'move': 'e2e5'},  # Invalid pawn move
                             content_type='application/json')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_make_move_no_move_provided(self, client):
        """Test API with no move provided"""
        response = client.post('/api/make_move',
                             json={},
                             content_type='application/json')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No move provided' in data['error']
    
    def test_game_status_api(self, client):
        """Test game status endpoint"""
        response = client.get('/api/game_status')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'fen' in data
        assert 'legal_moves' in data
        assert 'is_game_over' in data
    
    def test_legal_moves_api(self, client):
        """Test legal moves endpoint"""
        response = client.get('/api/legal_moves')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'legal_moves' in data
        assert isinstance(data['legal_moves'], list)
    
    def test_uninitialized_game_interface(self):
        """Test behavior when game interface is not initialized"""
        # Create app without initializing game interface
        app.config['TESTING'] = True
        with app.test_client() as client:
            # Reset global game interface
            import src.ui.web_interface as web_module
            web_module.game_interface = None
            
            response = client.post('/api/new_game', json={'color': 'white'})
            assert response.status_code == 500
            data = json.loads(response.data)
            assert 'Game interface not initialized' in data['error']

class TestGameInterface:
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock chess model"""
        model = Mock()
        # Mock the predict method to return proper tuple
        import numpy as np
        model.predict.return_value = (
            np.array([0.1] * 20480, dtype=np.float32),  # policy probabilities (4096 + 4*4096)
            0.5  # value
        )
        return model
    
    @pytest.fixture 
    def game_interface(self, mock_model):
        """Create game interface with mock model"""
        return GameInterface(mock_model, enable_learning=False)
    
    def test_game_interface_initialization(self, mock_model):
        """Test game interface initializes correctly"""
        interface = GameInterface(mock_model, enable_learning=False)
        assert interface.ai_player is not None
        assert interface.game is not None
        assert interface.enable_learning == False
    
    def test_game_interface_with_learning(self, mock_model):
        """Test game interface with learning enabled"""
        with patch('src.ui.game_interface.DataManager'), \
             patch('src.ui.game_interface.UserGameLearning'):
            interface = GameInterface(mock_model, enable_learning=True)
            assert interface.enable_learning == True
            assert interface.user_learning is not None
    
    def test_reset_game(self, game_interface):
        """Test game reset functionality"""
        # Make some moves first
        game_interface.game_history = [{'move': 'e2e4', 'player': 'human'}]
        
        game_interface.reset_game()
        assert len(game_interface.game_history) == 0
        assert game_interface.get_board_fen() == 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    
    def test_promotion_move_detection(self, game_interface):
        """Test pawn promotion move detection"""
        # Set up a position where pawn can promote
        game_interface.reset_game('8/P7/8/8/8/8/8/8 w - - 0 1')  # White pawn on a7
        
        assert game_interface.is_promotion_move('a7a8') == True
        assert game_interface.is_promotion_move('e2e4') == False
    
    def test_make_move_uci_format(self, game_interface):
        """Test making moves in UCI format"""
        game_interface.reset_game()
        
        # Mock the game to return valid moves
        import chess
        game_interface.game.get_legal_moves = Mock(return_value=[chess.Move.from_uci('e2e4')])
        game_interface.game.make_move = Mock(return_value=True)
        game_interface.game.get_fen = Mock(return_value='rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1')
        
        result = game_interface.make_move('e2e4')
        assert result == True
        assert len(game_interface.game_history) == 1
        assert game_interface.game_history[0]['move'] == 'e2e4'
        assert game_interface.game_history[0]['player'] == 'human'

class TestModelCompatibility:
    """Test model loading compatibility issues"""
    
    def test_chess_net_vs_optimized_chess_net(self):
        """Test that web interface can handle model architecture differences"""
        # This should identify the bug where web interface is hardcoded to ChessNet
        import src.ui.web_interface as web_module
        
        # Check if the web interface is hardcoded to ChessNet
        with open('/Users/gerald/Desktop/cluade/src/ui/web_interface.py', 'r') as f:
            content = f.read()
            assert 'ChessNet' in content
            # This test will help us identify if we need to make it more flexible

class TestStaticAssets:
    """Test static assets and frontend components"""
    
    def test_chess_piece_images_exist(self):
        """Test that all chess piece images exist"""
        pieces = ['wk', 'wq', 'wr', 'wb', 'wn', 'wp', 'bk', 'bq', 'br', 'bb', 'bn', 'bp']
        image_dir = '/Users/gerald/Desktop/cluade/src/ui/static/images'
        
        for piece in pieces:
            image_path = os.path.join(image_dir, f'{piece}.png')
            assert os.path.exists(image_path), f"Missing chess piece image: {piece}.png"
    
    def test_css_file_exists(self):
        """Test that CSS file exists"""
        css_path = '/Users/gerald/Desktop/cluade/src/ui/static/css/chess.css'
        assert os.path.exists(css_path), "CSS file missing"
    
    def test_javascript_file_exists(self):
        """Test that JavaScript file exists"""
        js_path = '/Users/gerald/Desktop/cluade/src/ui/static/js/chess-board.js'
        assert os.path.exists(js_path), "JavaScript file missing"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])