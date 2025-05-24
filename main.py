#!/usr/bin/env python3

import os
import sys
import argparse
import torch
from config import Config
from src.neural_network.chess_net import ChessNet
from src.neural_network.trainer import ChessNetTrainer
from src.training.direct_selfplay import CreativeTrainingManager
from src.training.tournament import GenerationalTraining
from src.utils.data_manager import DataManager
from src.ui.game_interface import ConsoleInterface
from src.ui.web_interface import run_web_interface

def train_engine(iterations: int = 10, games_per_iteration: int = 50, use_parallel: bool = False, num_workers: int = None):
    print("Starting Neural Chess Engine Training")
    print("=====================================")
    
    Config.create_dirs()
    data_manager = DataManager()
    
    model = ChessNet()
    model.to(Config.DEVICE)
    
    trainer = ChessNetTrainer(model)
    
    if use_parallel:
        from src.training.parallel_selfplay import FastTrainingManager
        training_manager = FastTrainingManager(model)
        if num_workers:
            training_manager.parallel_selfplay.num_workers = num_workers
        print(f"🚀 Using parallel training with {training_manager.parallel_selfplay.num_workers} workers")
    else:
        from src.training.direct_selfplay import CreativeTrainingManager
        training_manager = CreativeTrainingManager(model)
    
    generational_trainer = GenerationalTraining(data_manager)
    
    for iteration in range(iterations):
        print(f"\n--- Training Iteration {iteration + 1}/{iterations} ---")
        
        training_data = training_manager.train_iteration(games_per_iteration)
        
        print(f"Training on {len(training_data[0])} examples...")
        training_history = trainer.train(training_data)
        
        model_path = data_manager.save_model_checkpoint(
            model, trainer.optimizer, iteration,
            training_history, {'iteration': iteration}
        )
        print(f"Model saved: {model_path}")
        
        if iteration > 0 and iteration % 3 == 0:
            print("Running generational evaluation...")
            was_promoted = generational_trainer.evaluate_candidate(model)
            
            if was_promoted:
                print(f"New champion model! Generation {generational_trainer.get_generation()}")
            else:
                print("Model did not beat current champion")
    
    print("\nTraining completed!")
    return model

def play_against_engine(model_path: str = None, enable_learning: bool = False):
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = ChessNet.load_model(model_path)
    else:
        print("Using untrained model (will play randomly)")
        model = ChessNet()
    
    if enable_learning:
        print("🧠 Learning mode enabled - AI will learn from your games!")
    
    console_interface = ConsoleInterface(model)
    console_interface.play_game()

def run_web_ui(model_path: str = None, host: str = '127.0.0.1', port: int = 5000, enable_learning: bool = True):
    print(f"Starting web interface on http://{host}:{port}")
    run_web_interface(model_path, host, port, enable_learning)

def create_initial_model():
    model = ChessNet()
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    model_path = os.path.join(Config.MODEL_DIR, "initial_model.pt")
    model.save_model(model_path)
    print(f"Created initial model: {model_path}")
    return model_path

def main():
    parser = argparse.ArgumentParser(description="Neural Chess Engine")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    train_parser = subparsers.add_parser('train', help='Train the neural chess engine')
    train_parser.add_argument('--iterations', type=int, default=10, help='Number of training iterations')
    train_parser.add_argument('--games', type=int, default=50, help='Games per iteration')
    train_parser.add_argument('--parallel', action='store_true', help='Use parallel training for faster speed')
    train_parser.add_argument('--workers', type=int, help='Number of parallel workers (default: auto-detect)')
    train_parser.add_argument('--optimized', action='store_true', help='Use optimized training pipeline')
    train_parser.add_argument('--standard', action='store_true', help='Force standard training (override optimized)')
    
    play_parser = subparsers.add_parser('play', help='Play against the engine in console')
    play_parser.add_argument('--model', type=str, help='Path to trained model')
    play_parser.add_argument('--learn', action='store_true', help='Enable learning from user games')
    
    web_parser = subparsers.add_parser('web', help='Start web interface')
    web_parser.add_argument('--model', type=str, help='Path to trained model')
    web_parser.add_argument('--host', type=str, default='127.0.0.1', help='Host address')
    web_parser.add_argument('--port', type=int, default=5000, help='Port number')
    web_parser.add_argument('--no-learn', action='store_true', help='Disable learning from user games')
    
    init_parser = subparsers.add_parser('init', help='Create initial model')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        if args.optimized and not args.standard:
            # Use optimized training pipeline
            print("🚀 Using optimized training pipeline")
            from optimized_training import OptimizedTrainingPipeline
            pipeline = OptimizedTrainingPipeline(use_optimized_model=True)
            pipeline.run_training(
                max_generations=args.iterations,
                games_per_generation=args.games
            )
        else:
            # Use standard training
            train_engine(args.iterations, args.games, args.parallel, args.workers)
    
    elif args.command == 'play':
        play_against_engine(args.model, args.learn)
    
    elif args.command == 'web':
        enable_learning = not args.no_learn
        run_web_ui(args.model, args.host, args.port, enable_learning)
    
    elif args.command == 'init':
        create_initial_model()
    
    else:
        parser.print_help()
        print("\n🚀 Quick start:")
        print("1. python main.py init                    # Create initial model")
        print("2. python main.py train --parallel        # Train with parallel processing")  
        print("3. python main.py web                     # Beautiful web interface")
        print("4. python main.py play --learn            # Console with AI learning")
        print("")
        print("🎯 Performance options:")
        print("   --parallel                             # Use all CPU cores for training")
        print("   --workers N                            # Specify number of parallel workers")
        print("   --games N                              # Games per training iteration")

if __name__ == "__main__":
    main()