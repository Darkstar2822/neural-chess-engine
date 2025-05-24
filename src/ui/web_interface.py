from flask import Flask, render_template, request, jsonify, session
import chess
import chess.svg
import os
from src.ui.game_interface import GameInterface
from src.neural_network.chess_net import ChessNet
from config import Config

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = 'chess_engine_secret_key'

game_interface = None

def initialize_game_interface(model_path: str = None, enable_learning: bool = True):
    global game_interface
    
    if model_path and os.path.exists(model_path):
        model = ChessNet.load_model(model_path)
    else:
        model = ChessNet()
    
    game_interface = GameInterface(model, enable_learning=enable_learning)

@app.route('/')
def index():
    return render_template('chess_game.html')

@app.route('/api/new_game', methods=['POST'])
def new_game():
    global game_interface
    if game_interface is None:
        return jsonify({'error': 'Game interface not initialized'}), 500
    
    data = request.get_json()
    human_color = data.get('color', 'white')
    
    # Finish previous game if it was completed (for learning)
    if game_interface.is_game_over() and 'human_color' in session:
        previous_human_was_white = session['human_color'] == 'white'
        game_interface.finish_game(previous_human_was_white)
    
    game_interface.reset_game()
    session['human_color'] = human_color
    
    response = {
        'status': 'success',
        'board_fen': game_interface.get_board_fen(),
        'human_color': human_color,
        'learning_enabled': game_interface.enable_learning
    }
    
    if human_color == 'black':
        ai_move = game_interface.get_ai_move()
        response['ai_move'] = ai_move
        response['board_fen'] = game_interface.get_board_fen()
    
    return jsonify(response)

@app.route('/api/make_move', methods=['POST'])
def make_move():
    global game_interface
    if game_interface is None:
        return jsonify({'error': 'Game interface not initialized'}), 500
    
    data = request.get_json()
    move = data.get('move')
    
    if not move:
        return jsonify({'error': 'No move provided'}), 400
    
    if game_interface.make_move(move):
        response = {
            'status': 'success',
            'board_fen': game_interface.get_board_fen(),
            'is_game_over': game_interface.is_game_over()
        }
        
        if game_interface.is_game_over():
            response['game_result'] = game_interface.get_game_result()
            # Trigger learning when game ends
            if 'human_color' in session:
                human_was_white = session['human_color'] == 'white'
                game_interface.finish_game(human_was_white)
                if game_interface.enable_learning:
                    response['learning_message'] = "Game recorded for AI learning!"
        else:
            ai_move = game_interface.get_ai_move()
            response['ai_move'] = ai_move
            response['board_fen'] = game_interface.get_board_fen()
            response['is_game_over'] = game_interface.is_game_over()
            
            if game_interface.is_game_over():
                response['game_result'] = game_interface.get_game_result()
                # Trigger learning when game ends
                if 'human_color' in session:
                    human_was_white = session['human_color'] == 'white'
                    game_interface.finish_game(human_was_white)
                    if game_interface.enable_learning:
                        response['learning_message'] = "Game recorded for AI learning!"
        
        return jsonify(response)
    else:
        return jsonify({'error': 'Invalid move'}), 400

@app.route('/api/game_status')
def game_status():
    global game_interface
    if game_interface is None:
        return jsonify({'error': 'Game interface not initialized'}), 500
    
    return jsonify(game_interface.get_game_status())

@app.route('/api/legal_moves')
def legal_moves():
    global game_interface
    if game_interface is None:
        return jsonify({'error': 'Game interface not initialized'}), 500
    
    return jsonify({'legal_moves': game_interface.get_legal_moves()})

def run_web_interface(model_path: str = None, host: str = '127.0.0.1', port: int = 5000, enable_learning: bool = True):
    initialize_game_interface(model_path, enable_learning)
    if enable_learning:
        print("🧠 Learning mode enabled - AI will learn from your games!")
    app.run(host=host, port=port, debug=True)

if __name__ == '__main__':
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    enable_learning = '--learn' in sys.argv
    run_web_interface(model_path, enable_learning=enable_learning)