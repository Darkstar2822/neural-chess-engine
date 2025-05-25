"""
Example script to run the evolutionary engine
"""

import logging
import torch
from pathlib import Path
from evolutionary_engine import EvolutionaryEngine

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('evolution.log'),
            logging.StreamHandler()
        ]
    )
    
    # Initialize evolutionary engine
    engine = EvolutionaryEngine(
        population_size=20,  # Start small for testing
        max_generations=10,
        tournament_size=6,
        mutation_rate=0.15
    )
    
    # Initialize population with base model
    base_model_path = "models/best_model.pth"
    if Path(base_model_path).exists():
        engine.initialize_population(base_model_path)
    else:
        engine.initialize_population()  # Random initialization
    
    # Run evolution
    final_population = engine.evolve()
    
    # Save best individuals
    ranks = engine.fitness_evaluator.pareto_dominance_rank(final_population)
    pareto_front = [ind for ind, rank in zip(final_population, ranks) if rank == 0]
    
    for i, individual in enumerate(pareto_front):
        torch.save(individual.model.state_dict(), f"evolved_model_{i}.pth")
        print(f"Saved evolved model {i}: Rating={individual.glicko_rating:.1f}")

if __name__ == "__main__":
    main()