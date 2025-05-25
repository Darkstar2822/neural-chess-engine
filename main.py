#!/usr/bin/env python3

import os
import sys
import argparse
import torch
from typing import Optional
from config import Config
from src.neural_network.chess_net import ChessNet
from src.neural_network.trainer import ChessNetTrainer
from src.training.direct_selfplay import CreativeTrainingManager
from src.training.tournament import GenerationalTraining
from src.utils.data_manager import DataManager
from src.ui.game_interface import ConsoleInterface
from src.ui.web_interface import run_web_interface
def load_model_safe(path: str) -> ChessNet:
    if os.path.exists(path):
        try:
            return ChessNet.load_model(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model at {path}: {e}")
    raise FileNotFoundError(f"Model path not found: {path}")

def train_engine(iterations: int = 10, games_per_iteration: int = 50, use_parallel: bool = False, num_workers: int = None, 
                use_evolution: bool = False, population_size: int = 20, use_neuroevolution: bool = False):
    if use_evolution or use_neuroevolution:
        print("🧬 Starting Evolutionary Chess Engine Training")
        print("=" * 50)
        
        if use_neuroevolution:
            print("🔬 Using Advanced Neuroevolution with Topology Evolution")
            from src.evolution.enhanced_evolutionary_engine import EnhancedEvolutionaryEngine
            engine = EnhancedEvolutionaryEngine(
                population_size=population_size,
                max_generations=iterations,
                use_topology_evolution=True
            )
        else:
            print("🏆 Using Multi-Objective Evolutionary Training")
            from src.evolution.evolutionary_engine import EvolutionaryEngine
            engine = EvolutionaryEngine(
                population_size=population_size,
                max_generations=iterations
            )
        
        print(f"Population size: {population_size}")
        print(f"Generations: {iterations}")
        print("=" * 50)
        
        # Initialize and evolve
        engine.initialize_population()
        final_population = engine.evolve()
        
        # Save best individual
        best_individual = max(final_population, key=lambda x: x.glicko_rating)
        
        Config.create_dirs()
        model_path = os.path.join(Config.MODEL_DIR, f"evolved_champion_gen_{iterations}.pth")
        torch.save(best_individual.model.state_dict(), model_path)
        
        print(f"\n🏆 Evolution Complete!")
        print(f"Best Individual Rating: {best_individual.glicko_rating:.1f}")
        print(f"Champion saved: {model_path}")
        
        return best_individual.model
    
    else:
        print("🧠 Starting Standard Neural Chess Engine Training")
        print("=" * 50)
        
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
        
        print("\nStandard training completed!")
        return model

def play_against_engine(model_path: Optional[str] = None, enable_learning: bool = False):
    if model_path:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        try:
            model = load_model_safe(model_path)
        except Exception as e:
            print(f"❌ {e}")
            return
    else:
        print("Using untrained model (will play randomly)")
        model = ChessNet()
    
    if enable_learning:
        print("🧠 Learning mode enabled - AI will learn from your games!")
    
    console_interface = ConsoleInterface(model)
    console_interface.play_game()

def run_web_ui(model_path: Optional[str] = None, host: str = '127.0.0.1', port: int = 5000, enable_learning: bool = True):
    print(f"Starting web interface on http://{host}:{port}")
    # import logging
    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger(__name__)
    # logger.info("Starting web interface...")
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
    train_parser.add_argument('--iterations', type=int, default=10, help='Number of training iterations/generations')
    train_parser.add_argument('--games', type=int, default=50, help='Games per iteration')
    train_parser.add_argument('--parallel', action='store_true', help='Use parallel training for faster speed')
    train_parser.add_argument('--workers', type=int, help='Number of parallel workers (default: auto-detect)')
    train_parser.add_argument('--optimized', action='store_true', help='Use optimized training pipeline')
    train_parser.add_argument('--standard', action='store_true', help='Force standard training (override optimized)')
    
    # Evolutionary training options
    train_parser.add_argument('--evolution', action='store_true', help='🧬 Use evolutionary training with population')
    train_parser.add_argument('--neuroevolution', action='store_true', help='🔬 Use advanced neuroevolution with topology evolution')
    train_parser.add_argument('--population-size', type=int, default=20, help='Population size for evolutionary training')
    
    play_parser = subparsers.add_parser('play', help='Play against the engine in console')
    play_parser.add_argument('--model', type=str, help='Path to trained model')
    play_parser.add_argument('--learn', action='store_true', help='Enable learning from user games')
    
    web_parser = subparsers.add_parser('web', help='Start web interface')
    web_parser.add_argument('--model', type=str, help='Path to trained model')
    web_parser.add_argument('--host', type=str, default='127.0.0.1', help='Host address')
    web_parser.add_argument('--port', type=int, default=5000, help='Port number')
    web_parser.add_argument('--no-learn', action='store_true', help='Disable learning from user games')
    web_parser.add_argument('--champion', action='store_true', help='🏆 Auto-load best evolved champion')
    web_parser.add_argument('--evolved', action='store_true', help='🧬 Prefer evolved models over standard models')
    
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
            # Use standard or evolutionary training
            train_engine(
                iterations=args.iterations, 
                games_per_iteration=args.games, 
                use_parallel=args.parallel, 
                num_workers=args.workers,
                use_evolution=args.evolution,
                population_size=args.population_size,
                use_neuroevolution=args.neuroevolution
            )
    
    elif args.command == 'play':
        play_against_engine(args.model, args.learn)
    
    elif args.command == 'web':
        enable_learning = not args.no_learn
        model_path = args.model
        
        # Handle special model selection flags
        if args.champion or args.evolved:
            from src.ui.web_interface import get_best_available_model, discover_available_models
            
            if args.champion:
                print("🏆 Auto-selecting best evolved champion...")
                model_path = get_best_available_model()
            elif args.evolved:
                print("🧬 Searching for evolved models...")
                models = discover_available_models()
                evolved_models = {k: v for k, v in models.items() 
                                if v['info'].get('type') == 'evolved'}
                if evolved_models:
                    # Get the newest evolved model
                    model_path = list(evolved_models.values())[0]['path']
                    print(f"🧬 Selected evolved model: {os.path.basename(model_path)}")
                else:
                    print("⚠️ No evolved models found, using best available")
                    model_path = get_best_available_model()
        
        run_web_ui(model_path, args.host, args.port, enable_learning)
    
    elif args.command == 'init':
        create_initial_model()
    
    else:
        parser.print_help()
        print("\n🚀 Quick start:")
        print("1. python main.py init                    # Create initial model")
        print("2. python main.py train --parallel        # Train with parallel processing")
        print("3. python main.py train --evolution       # 🧬 Evolutionary training")
        print("4. python main.py train --neuroevolution  # 🔬 Advanced topology evolution")
        print("5. python main.py web                     # Beautiful web interface")
        print("6. python main.py play --learn            # Console with AI learning")
        print("")
        print("🎯 Training options:")
        print("   --parallel                             # Use all CPU cores for training")
        print("   --workers N                            # Specify number of parallel workers")
        print("   --games N                              # Games per training iteration")
        print("   --evolution                            # 🧬 Multi-objective evolutionary training")
        print("   --neuroevolution                       # 🔬 Topology evolution with NEAT")
        print("   --population-size N                    # Population size for evolution (default: 20)")

if __name__ == "__main__":
    main()