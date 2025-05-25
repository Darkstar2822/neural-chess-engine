import logging
import os
from flask import Flask, render_template, request, jsonify, session
import chess
import chess.svg
import glob
import torch
from datetime import datetime
from src.ui.game_interface import GameInterface
from src.neural_network.chess_net import ChessNet
from src.neural_network.ultra_fast_chess_net import ModelFactory
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = os.getenv('CHESS_ENGINE_SECRET_KEY', 'default_dev_secret')

game_interface = None
available_models = {}

def discover_available_models():
    """Discover all available trained models"""
    global available_models
    available_models = {}
    
    model_dirs = [Config.MODEL_DIR, "models", "data/models"]
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            # Standard models
            for pattern in ["*.pt", "*.pth"]:
                for model_path in glob.glob(os.path.join(model_dir, pattern)):
                    model_name = os.path.basename(model_path)
                    model_info = analyze_model(model_path)
                    available_models[model_name] = {
                        'path': model_path,
                        'info': model_info,
                        'mtime': os.path.getmtime(model_path)
                    }
            
            # Evolved models
            for pattern in ["evolved_*.pth", "*champion*.pth"]:
                for model_path in glob.glob(os.path.join(model_dir, pattern)):
                    model_name = os.path.basename(model_path)
                    model_info = analyze_model(model_path)
                    model_info['type'] = 'evolved'
                    available_models[model_name] = {
                        'path': model_path,
                        'info': model_info,
                        'mtime': os.path.getmtime(model_path)
                    }
    
    # Sort by modification time (newest first)
    available_models = dict(sorted(available_models.items(), 
                                 key=lambda x: x[1]['mtime'], reverse=True))
    
    return available_models

def analyze_model(model_path: str):
    """Analyze a model file to extract metadata"""
    info = {
        'type': 'standard',
        'architecture': 'ChessNet',
        'size_mb': round(os.path.getsize(model_path) / (1024*1024), 2),
        'created': datetime.fromtimestamp(os.path.getctime(model_path)).strftime('%Y-%m-%d %H:%M'),
        'description': 'Neural chess model'
    }
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Check for evolutionary metadata
        if 'playing_style' in checkpoint:
            info['type'] = 'evolved'
            info['playing_style'] = checkpoint['playing_style']
            info['architecture'] = 'EvolvableChessNet'
            info['description'] = f"Evolved {checkpoint['playing_style']} specialist"
            
            if 'glicko_rating' in checkpoint:
                info['rating'] = round(checkpoint['glicko_rating'], 1)
                info['description'] += f" (Rating: {info['rating']})"
            
            if 'generation' in checkpoint:
                info['generation'] = checkpoint['generation']
                info['description'] += f" Gen-{info['generation']}"
        
        # Check for optimized architecture
        elif 'config' in checkpoint:
            config = checkpoint['config']
            if config.get('optimized', False):
                info['architecture'] = 'OptimizedChessNet'
                info['description'] = 'Optimized neural chess model'
        
        # Standard model metadata
        if 'epoch' in checkpoint or 'iteration' in checkpoint:
            epoch = checkpoint.get('epoch', checkpoint.get('iteration', 0))
            info['epoch'] = epoch
            info['description'] += f" (Epoch {epoch})"
            
    except Exception as e:
        info['description'] = f'Model file (analysis failed: {str(e)[:30]}...)'
    
    return info

def get_best_available_model():
    """Get the best available model (prioritize evolved champions)"""
    models = discover_available_models()
    
    if not models:
        return None
    
    # Prioritize evolved models with high ratings
    evolved_models = {k: v for k, v in models.items() 
                     if v['info'].get('type') == 'evolved'}
    
    if evolved_models:
        # Sort by rating if available, otherwise by modification time
        best_evolved = max(evolved_models.items(), 
                          key=lambda x: (x[1]['info'].get('rating', 0), x[1]['mtime']))
        return best_evolved[1]['path']
    
    # Fall back to newest standard model
    return list(models.values())[0]['path']

def initialize_game_interface(model_path: str = None, enable_learning: bool = True):
    global game_interface
    
    # Auto-discover best model if none specified
    if model_path is None:
        model_path = get_best_available_model()
        if model_path:
            logger.info(f"🏆 Auto-selected best model: {os.path.basename(model_path)}")
    
    if model_path and os.path.exists(model_path):
        # Try to load with architecture detection
        try:
            checkpoint = torch.load(model_path, map_location=Config.DEVICE, weights_only=False)
            config = checkpoint.get('config', {})
            
            # Check if it's an evolved model
            if 'playing_style' in checkpoint:
                logger.info(f"🧬 Loading evolved model: {checkpoint.get('playing_style', 'unknown')} specialist")
                from src.evolution.neuroevolution import EvolvableChessNet, NetworkGenome
                
                # Load the genome if available
                if 'genome' in checkpoint:
                    genome = checkpoint['genome']
                    model = EvolvableChessNet(genome)
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # Fallback to standard model
                    model = ChessNet()
                    model.load_state_dict(checkpoint)
                    
            # Check if it's an optimized model
            elif config.get('optimized', False) or config.get('architecture_version', '1.0') != '1.0':
                from src.neural_network.optimized_chess_net import OptimizedChessNet
                model = OptimizedChessNet.load_model(model_path)
            else:
                # Try standard ChessNet loading, with fallback for missing config
                try:
                    model = ChessNet.load_model(model_path)
                except KeyError as ke:
                    if "'config'" in str(ke):
                        logger.info("🔧 Model missing config, creating standard model and loading weights...")
                        model = ChessNet()
                        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
                    else:
                        raise ke
                
            logger.info(f"✅ Model loaded successfully: {os.path.basename(model_path)}")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load model from {model_path}: {e}")
            logger.info("🆕 Creating ultra-fast model instead")
            model = ModelFactory.create_ultra_fast('medium')
    else:
        logger.info("🆕 No model specified or found, creating ultra-fast model")
        model = ModelFactory.create_ultra_fast('medium')  # Use optimized model by default
    
    game_interface = GameInterface(model, enable_learning=enable_learning)


# --- Add maybe_finish_game function before new_game ---
def maybe_finish_game(response):
    global game_interface
    if game_interface.is_game_over():
        response['game_result'] = game_interface.get_game_result()
        if 'human_color' in session:
            human_was_white = session['human_color'] == 'white'
            game_interface.finish_game(human_was_white)
            if game_interface.enable_learning:
                response['learning_message'] = "Game recorded for AI learning!"

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
            maybe_finish_game(response)
        else:
            ai_move = game_interface.get_ai_move()
            response['ai_move'] = ai_move
            response['board_fen'] = game_interface.get_board_fen()
            response['is_game_over'] = game_interface.is_game_over()
            if game_interface.is_game_over():
                maybe_finish_game(response)
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

@app.route('/api/models')
def list_models():
    """Get list of available models"""
    models = discover_available_models()
    
    # Format for frontend
    model_list = []
    for name, data in models.items():
        info = data['info']
        model_list.append({
            'name': name,
            'path': data['path'],
            'type': info.get('type', 'standard'),
            'architecture': info.get('architecture', 'ChessNet'),
            'description': info.get('description', 'Neural chess model'),
            'rating': info.get('rating'),
            'playing_style': info.get('playing_style'),
            'generation': info.get('generation'),
            'size_mb': info.get('size_mb'),
            'created': info.get('created'),
            'is_evolved': info.get('type') == 'evolved'
        })
    
    return jsonify({'models': model_list})

@app.route('/api/current_model')
def current_model():
    """Get information about currently loaded model"""
    global game_interface
    if game_interface is None:
        return jsonify({'error': 'Game interface not initialized'}), 500
    
    # Try to determine model info from the loaded model
    model_info = {
        'name': 'Current Model',
        'type': 'unknown',
        'architecture': type(game_interface.ai_player.model).__name__,
        'description': 'Currently loaded neural chess model',
        'learning_enabled': game_interface.enable_learning
    }
    
    # Check if it's an evolved model
    if hasattr(game_interface.ai_player.model, 'genome'):
        model_info['type'] = 'evolved'
        model_info['architecture'] = 'EvolvableChessNet'
        model_info['description'] = 'Evolved neural chess model'
    
    return jsonify(model_info)

@app.route('/api/switch_model', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    global game_interface
    
    data = request.get_json()
    model_path = data.get('model_path')
    
    if not model_path or not os.path.exists(model_path):
        return jsonify({'error': 'Invalid model path'}), 400
    
    try:
        # Reinitialize with new model
        enable_learning = game_interface.enable_learning if game_interface else True
        initialize_game_interface(model_path, enable_learning)
        
        return jsonify({
            'success': True,
            'message': f'Switched to model: {os.path.basename(model_path)}',
            'info': analyze_model(model_path)
        })
    except Exception as e:
        return jsonify({'error': f'Failed to load model: {str(e)}'}), 500

@app.route('/api/best_model')
def get_best_model():
    """Get the best available model path"""
    best_path = get_best_available_model()
    
    if best_path:
        models = discover_available_models()
        model_name = os.path.basename(best_path)
        model_info = models.get(model_name, {}).get('info', {})
        
        return jsonify({
            'path': best_path,
            'name': model_name,
            'info': model_info
        })
    else:
        return jsonify({'error': 'No models found'}), 404

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})


def run_web_interface(model_path: str = None, host: str = '127.0.0.1', port: int = 5000, enable_learning: bool = True):
    initialize_game_interface(model_path, enable_learning)
    if enable_learning:
        logger.info("🧠 Learning mode enabled - AI will learn from your games!")
    app.run(host=host, port=port, debug=True)

if __name__ == '__main__':
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    enable_learning = '--learn' in sys.argv
    run_web_interface(model_path, enable_learning=enable_learning)