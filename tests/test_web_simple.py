#!/usr/bin/env python3
"""
Simple integration tests for web interface without Puppeteer
"""

import requests
import json
import subprocess
import time
import signal
import os
import sys
import threading
from urllib.parse import urljoin

class WebIntegrationTest:
    def __init__(self):
        self.base_url = 'http://127.0.0.1:5666'
        self.server_process = None
        
    def start_server(self):
        """Start the Flask server"""
        print("Starting Flask server...")
        
        # Change to project directory
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.server_process = subprocess.Popen([
            sys.executable, 'main.py', 'web', '--no-learn', '--port', '5666'
        ], cwd=project_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        for _ in range(10):
            time.sleep(1)
            try:
                response = requests.get(self.base_url, timeout=2)
                if response.status_code == 200:
                    print("✅ Server started successfully")
                    return True
            except:
                continue
        
        print("❌ Failed to start server")
        return False
    
    def stop_server(self):
        """Stop the Flask server"""
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            print("🛑 Server stopped")
    
    def test_home_page(self):
        """Test that home page loads"""
        print("Testing home page...")
        response = requests.get(self.base_url)
        assert response.status_code == 200
        assert 'Neural Chess Engine' in response.text
        assert 'chess-board' in response.text
        print("✅ Home page loads correctly")
    
    def test_static_assets(self):
        """Test that static assets are accessible"""
        print("Testing static assets...")
        
        # Test CSS
        css_url = urljoin(self.base_url, '/static/css/chess.css')
        response = requests.get(css_url)
        assert response.status_code == 200
        assert 'chess-container' in response.text
        print("✅ CSS loads correctly")
        
        # Test JavaScript
        js_url = urljoin(self.base_url, '/static/js/chess-board.js')
        response = requests.get(js_url)
        assert response.status_code == 200
        assert 'ChessBoard' in response.text
        print("✅ JavaScript loads correctly")
        
        # Test chess piece images
        piece_url = urljoin(self.base_url, '/static/images/wk.png')
        response = requests.get(piece_url)
        assert response.status_code == 200
        assert response.headers.get('Content-Type', '').startswith('image/')
        print("✅ Chess piece images load correctly")
    
    def test_api_new_game(self):
        """Test new game API"""
        print("Testing new game API...")
        
        # Test as white
        response = requests.post(
            urljoin(self.base_url, '/api/new_game'),
            json={'color': 'white'},
            headers={'Content-Type': 'application/json'}
        )
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert data['human_color'] == 'white'
        assert 'board_fen' in data
        print("✅ New game as white works")
        
        # Test as black
        try:
            response = requests.post(
                urljoin(self.base_url, '/api/new_game'),
                json={'color': 'black'},
                headers={'Content-Type': 'application/json'}
            )
            print(f"Response status: {response.status_code}")
            print(f"Response text: {response.text}")
            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'success'
            assert data['human_color'] == 'black'
            assert 'ai_move' in data  # AI should move first as black
            print("✅ New game as black works (AI moves first)")
        except Exception as e:
            print(f"Error in new game as black: {e}")
            raise
    
    def test_api_make_move(self):
        """Test making moves via API"""
        print("Testing make move API...")
        
        # Start new game as white
        requests.post(
            urljoin(self.base_url, '/api/new_game'),
            json={'color': 'white'}
        )
        
        # Make a valid move
        response = requests.post(
            urljoin(self.base_url, '/api/make_move'),
            json={'move': 'e2e4'},
            headers={'Content-Type': 'application/json'}
        )
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert 'ai_move' in data
        print("✅ Valid move works")
        
        # Try invalid move
        response = requests.post(
            urljoin(self.base_url, '/api/make_move'),
            json={'move': 'e2e5'},  # Invalid pawn move
            headers={'Content-Type': 'application/json'}
        )
        assert response.status_code == 400
        data = response.json()
        assert 'error' in data
        print("✅ Invalid move properly rejected")
    
    def test_api_game_status(self):
        """Test game status API"""
        print("Testing game status API...")
        
        response = requests.get(urljoin(self.base_url, '/api/game_status'))
        assert response.status_code == 200
        data = response.json()
        assert 'fen' in data
        assert 'legal_moves' in data
        assert 'is_game_over' in data
        assert 'current_player' in data
        print("✅ Game status API works")
    
    def test_api_legal_moves(self):
        """Test legal moves API"""
        print("Testing legal moves API...")
        
        response = requests.get(urljoin(self.base_url, '/api/legal_moves'))
        assert response.status_code == 200
        data = response.json()
        assert 'legal_moves' in data
        assert isinstance(data['legal_moves'], list)
        assert len(data['legal_moves']) > 0  # Should have legal moves at start
        print("✅ Legal moves API works")
    
    def run_all_tests(self):
        """Run all integration tests"""
        print("🚀 Starting Web Integration Tests")
        print("=" * 50)
        
        try:
            if not self.start_server():
                print("❌ Failed to start server, aborting tests")
                return False
            
            self.test_home_page()
            self.test_static_assets()
            self.test_api_new_game()
            self.test_api_make_move()
            self.test_api_game_status()
            self.test_api_legal_moves()
            
            print("=" * 50)
            print("🎉 All integration tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
        finally:
            self.stop_server()

if __name__ == '__main__':
    test = WebIntegrationTest()
    success = test.run_all_tests()
    sys.exit(0 if success else 1)