"""
Quick test of the evolutionary engine with minimal setup
"""

import sys
sys.path.append('.')

import logging
import torch
from pathlib import Path
from src.evolution.evolutionary_engine import EvolutionaryEngine

def test_evolution():
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("Starting evolutionary engine test...")
    
    # Initialize evolutionary engine with very small parameters
    engine = EvolutionaryEngine(
        population_size=4,  # Very small for quick test
        max_generations=2,  # Just 2 generations
        tournament_size=2,
        mutation_rate=0.2
    )
    
    print("Initializing population...")
    # Initialize population with random models (no base model)
    engine.initialize_population()
    
    print(f"Population initialized with {len(engine.population)} individuals")
    
    # Run evolution
    print("Running evolution...")
    final_population = engine.evolve()
    
    print(f"Evolution complete! Final population size: {len(final_population)}")
    
    # Show results
    for i, individual in enumerate(final_population):
        print(f"Individual {i}: Rating={individual.glicko_rating:.1f}, "
              f"Win Rate={individual.fitness_scores.get('win_rate', 0):.3f}")

if __name__ == "__main__":
    test_evolution()