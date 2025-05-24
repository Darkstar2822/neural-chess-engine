#!/usr/bin/env python3
"""
Complete workflow test for the chess engine web interface
"""

import requests
import subprocess
import time
import sys
import os
import json

def test_complete_chess_workflow():
    """Test a complete chess game workflow"""
    print("🚀 Testing Complete Chess Workflow")
    print("=" * 50)
    
    base_url = 'http://127.0.0.1:5888'
    
    # Start server
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    server = subprocess.Popen([
        sys.executable, 'main.py', 'web', '--no-learn', '--port', '5888'
    ], cwd=project_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        # Wait for server
        print("Starting server...")
        for _ in range(10):
            time.sleep(1)
            try:
                response = requests.get(base_url, timeout=2)
                if response.status_code == 200:
                    print("✅ Server started")
                    break
            except:
                continue
        else:
            print("❌ Server failed to start")
            return False
        
        # Test 1: Start game as white
        print("\n1. Starting new game as white...")
        response = requests.post(f'{base_url}/api/new_game', json={'color': 'white'})
        assert response.status_code == 200
        data = response.json()
        assert data['human_color'] == 'white'
        print(f"✅ Game started, FEN: {data['board_fen'][:20]}...")
        
        # Test 2: Make opening move
        print("\n2. Making opening move e2e4...")
        response = requests.post(f'{base_url}/api/make_move', json={'move': 'e2e4'})
        assert response.status_code == 200
        data = response.json()
        assert 'ai_move' in data
        print(f"✅ Move made, AI responded with: {data['ai_move']}")
        
        # Test 3: Make another move
        print("\n3. Making second move d2d4...")
        response = requests.post(f'{base_url}/api/make_move', json={'move': 'd2d4'})
        assert response.status_code == 200
        data = response.json()
        print(f"✅ Second move made, AI responded with: {data['ai_move']}")
        
        # Test 4: Check game status
        print("\n4. Checking game status...")
        response = requests.get(f'{base_url}/api/game_status')
        assert response.status_code == 200
        data = response.json()
        assert 'current_player' in data
        assert 'move_count' in data
        print(f"✅ Game status: {data['current_player']} to move, {data.get('move_count', 0)} moves played")
        
        # Test 5: Get legal moves
        print("\n5. Getting legal moves...")
        response = requests.get(f'{base_url}/api/legal_moves')
        assert response.status_code == 200
        data = response.json()
        legal_moves = data['legal_moves']
        assert len(legal_moves) > 0
        print(f"✅ Found {len(legal_moves)} legal moves")
        
        # Test 6: Start new game as black
        print("\n6. Starting new game as black...")
        response = requests.post(f'{base_url}/api/new_game', json={'color': 'black'})
        assert response.status_code == 200
        data = response.json()
        assert data['human_color'] == 'black'
        assert 'ai_move' in data  # AI should move first
        print(f"✅ Game as black started, AI opened with: {data['ai_move']}")
        
        # Test 7: Respond to AI move
        print("\n7. Responding to AI opening...")
        response = requests.post(f'{base_url}/api/make_move', json={'move': 'e7e5'})
        assert response.status_code == 200
        data = response.json()
        print(f"✅ Responded with e7e5, AI played: {data['ai_move']}")
        
        # Test 8: Try invalid move
        print("\n8. Testing invalid move handling...")
        response = requests.post(f'{base_url}/api/make_move', json={'move': 'z9z9'})
        assert response.status_code == 400
        data = response.json()
        assert 'error' in data
        print("✅ Invalid move properly rejected")
        
        print("\n" + "=" * 50)
        print("🎉 Complete workflow test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False
        
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
        print("🛑 Server stopped")

if __name__ == '__main__':
    success = test_complete_chess_workflow()
    sys.exit(0 if success else 1)