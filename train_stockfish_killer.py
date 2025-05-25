#!/usr/bin/env python3
"""
Stockfish-Killer Training Script
Main training script for the co-evolutionary chess engine designed to beat Stockfish
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path
import time

# Add src to path
sys.path.append('.')

from src.evolution.stockfish_killer import StockfishKillerEngine


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # File handler for detailed logs
    file_handler = logging.FileHandler('stockfish_killer_training.log')
    file_handler.setLevel(logging.DEBUG)
    
    # Formatters
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


async def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description='Train the Stockfish-Killer Co-Evolutionary Chess Engine',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training parameters
    parser.add_argument('--population-size', type=int, default=150,
                       help='Total population size across all specialists')
    parser.add_argument('--cycles', type=int, default=20,
                       help='Number of training cycles')
    parser.add_argument('--specialists', type=int, default=5,
                       help='Number of specialist populations')
    parser.add_argument('--stockfish-path', type=str, default='/opt/homebrew/bin/stockfish',
                       help='Path to Stockfish executable')
    
    # Stockfish testing
    parser.add_argument('--test-frequency', type=int, default=3,
                       help='Test against Stockfish every N cycles')
    parser.add_argument('--max-stockfish-level', type=int, default=8,
                       help='Maximum Stockfish level to test against')
    
    # Performance options
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with small populations')
    parser.add_argument('--intensive', action='store_true',
                       help='Intensive training with large populations')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode - minimal output')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose and not args.quiet)
    logger = logging.getLogger(__name__)
    
    # Adjust parameters for quick test or intensive mode
    if args.quick_test:
        args.population_size = 30
        args.cycles = 5
        args.test_frequency = 2
        logger.info("Quick test mode: Small populations, short training")
    elif args.intensive:
        args.population_size = 300
        args.cycles = 100
        args.test_frequency = 5
        logger.info("Intensive mode: Large populations, extended training")
    
    # Validate Stockfish path
    if not Path(args.stockfish_path).exists():
        logger.error(f"Stockfish not found at: {args.stockfish_path}")
        logger.info("Please install Stockfish:")
        logger.info("  macOS: brew install stockfish")
        logger.info("  Linux: sudo apt-get install stockfish")
        logger.info("  Windows: Download from https://stockfishchess.org/download/")
        return 1
    
    # Print training configuration
    if not args.quiet:
        print("🏆 STOCKFISH-KILLER TRAINING")
        print("=" * 50)
        print(f"Population Size: {args.population_size}")
        print(f"Training Cycles: {args.cycles}")
        print(f"Specialists: {args.specialists}")
        print(f"Stockfish Path: {args.stockfish_path}")
        print(f"Test Frequency: Every {args.test_frequency} cycles")
        print(f"Max Stockfish Level: {args.max_stockfish_level}")
        print("=" * 50)
        print()
    
    # Initialize the Stockfish-Killer engine
    logger.info("Initializing Stockfish-Killer Engine...")
    
    try:
        engine = StockfishKillerEngine(
            total_population_size=args.population_size,
            num_specialists=args.specialists,
            generations_per_cycle=5,  # Fixed for now
            stockfish_path=args.stockfish_path
        )
        
        logger.info("Engine initialized successfully")
        
        # Start training
        logger.info(f"Starting {args.cycles} training cycles...")
        start_time = time.time()
        
        champion_file = await engine.training_cycle(
            cycles=args.cycles,
            stockfish_test_frequency=args.test_frequency
        )
        
        end_time = time.time()
        training_duration = end_time - start_time
        
        # Training complete
        if not args.quiet:
            print("\n🎉 TRAINING COMPLETE!")
            print("=" * 50)
            print(f"Duration: {training_duration/3600:.1f} hours")
            print(f"Champion: {champion_file}")
            print()
            
            # Get final champion stats
            champion = engine.get_best_individual()
            print(f"🏆 Final Champion Stats:")
            print(f"  Style: {champion.playing_style.value}")
            print(f"  Rating: {champion.glicko_rating:.1f}")
            print(f"  Generation: {champion.generation}")
            
            # Stockfish performance summary
            stockfish_wins = 0
            for name, rating in champion.engine_performance.items():
                if "Stockfish" in name and rating.score_rate > 0.5:
                    level = name.split("-")[1] if "-" in name else "Unknown"
                    print(f"  ✅ Beats Stockfish Level {level}: {rating.score_rate:.1%}")
                    stockfish_wins += 1
            
            if stockfish_wins == 0:
                print("  ⚠️ No Stockfish levels consistently beaten yet")
                print("  💡 Try more training cycles or larger populations")
            else:
                print(f"  🎯 Total Stockfish levels beaten: {stockfish_wins}")
        
        logger.info(f"Training completed successfully in {training_duration:.1f} seconds")
        
        # Final benchmark if requested
        if not args.quick_test:
            logger.info("Running final Stockfish gauntlet...")
            final_results = await engine.stockfish_gauntlet(
                num_levels=min(args.max_stockfish_level, 8)
            )
            
            if not args.quiet:
                print("\n📊 FINAL STOCKFISH GAUNTLET")
                print("-" * 30)
                for level, score in final_results.items():
                    status = "✅" if score > 0.5 else "❌" if score < 0.3 else "⚠️"
                    print(f"Level {level}: {score:.1%} {status}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)