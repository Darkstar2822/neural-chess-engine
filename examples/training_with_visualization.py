"""
Example: How to integrate training visualization with actual chess training
This demonstrates how to add monitoring to existing training scripts
"""

import sys
import os
import time
import torch
import numpy as np

# Add src and root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.training_monitor import MonitoredTraining, StandardTrainingMonitor
from neural_network.ultra_fast_chess_net import ModelFactory


def example_standard_training_with_visualization():
    """Example of standard training with integrated visualization"""
    print("🧠 Example: Standard Training with Visualization")
    print("="*60)
    
    # Training parameters
    training_params = {
        'epochs': 20,
        'batch_size': 32,
        'learning_rate': 0.001,
        'model_type': 'ultra_fast_medium'
    }
    
    # Start monitored training session
    with MonitoredTraining('standard', training_params, auto_save=True, update_interval=2.0) as monitor:
        
        # Create a simple model for demonstration
        model = ModelFactory.create_ultra_fast('medium')
        optimizer = torch.optim.Adam(model.parameters(), lr=training_params['learning_rate'])
        
        print(f"🚀 Started training session: {monitor.session_id}")
        print("📊 Visualizations will be updated every 2 seconds")
        
        for epoch in range(training_params['epochs']):
            print(f"Epoch {epoch+1}/{training_params['epochs']}")
            
            # Simulate training step
            train_loss = simulate_training_step(model, optimizer, epoch)
            
            # Simulate validation
            val_loss = train_loss + np.random.normal(0, 0.1)
            
            # Simulate ELO calculation (would be real evaluation)
            elo_rating = 1200 + epoch * 25 + np.random.normal(0, 15)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Simulate win rate (would be from actual games)
            win_rate = min(0.85, 0.3 + epoch * 0.025 + np.random.normal(0, 0.05))
            
            # Log all metrics to the monitor
            monitor.log_epoch_data(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                elo_rating=elo_rating,
                learning_rate=current_lr,
                win_rate=win_rate
            )
            
            # Optional: Learning rate scheduling
            if epoch > 0 and epoch % 5 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.9
            
            # Simulate training time
            time.sleep(0.5)
            
            print(f"  Loss: {train_loss:.6f}, ELO: {elo_rating:.0f}, Win Rate: {win_rate:.3f}")
        
        print(f"✅ Training completed! Session: {monitor.session_id}")
        print(f"📊 Check training_logs/ for visualizations")


def simulate_training_step(model, optimizer, epoch):
    """Simulate a training step with realistic loss progression"""
    # Create dummy input (chess position)
    batch_size = 32
    input_tensor = torch.randn(batch_size, 16, 8, 8)  # 16 channels, 8x8 board
    
    # Forward pass to get actual output shapes
    with torch.no_grad():
        policy_output, value_output = model(input_tensor)
    
    # Create dummy targets with correct shapes
    policy_target = torch.randn_like(policy_output)  # Policy targets
    value_target = torch.randn_like(value_output)  # Value targets
    
    optimizer.zero_grad()
    
    # Forward pass
    policy_output, value_output = model(input_tensor)
    
    # Calculate losses (simplified)
    policy_loss = torch.nn.functional.mse_loss(policy_output, policy_target)
    value_loss = torch.nn.functional.mse_loss(value_output, value_target)
    total_loss = policy_loss + value_loss
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    # Simulate decreasing loss over time
    base_loss = 2.0 * np.exp(-epoch/8) + 0.2
    noise = np.random.normal(0, 0.1)
    
    return max(0.1, base_loss + noise)


def example_evolutionary_training_with_visualization():
    """Example of evolutionary training with visualization"""
    print("\n🧬 Example: Evolutionary Training with Visualization")
    print("="*60)
    
    training_params = {
        'population_size': 50,
        'generations': 25,
        'mutation_rate': 0.1,
        'crossover_rate': 0.7
    }
    
    with MonitoredTraining('evolutionary', training_params, auto_save=True, update_interval=1.0) as monitor:
        
        print(f"🚀 Started evolutionary session: {monitor.session_id}")
        
        # Initialize population
        population_size = training_params['population_size']
        
        for generation in range(training_params['generations']):
            print(f"Generation {generation+1}/{training_params['generations']}")
            
            # Simulate population fitness evaluation
            base_fitness = generation * 0.1
            population_fitness = []
            
            for individual in range(population_size):
                # Simulate individual fitness (increasing over generations)
                fitness = base_fitness + np.random.normal(0, 0.3)
                fitness = max(0, fitness)  # Ensure non-negative
                population_fitness.append(fitness)
            
            # Simulate species count (decreasing as population converges)
            species_count = max(1, 12 - generation // 3 + np.random.randint(-1, 2))
            
            # Get champion (best individual)
            champion_fitness = max(population_fitness)
            
            # Log evolutionary data
            monitor.log_generation_data(
                generation=generation,
                population_fitness=population_fitness,
                species_count=species_count,
                champion_fitness=champion_fitness
            )
            
            time.sleep(0.3)  # Simulate evolution time
            
            print(f"  Best: {champion_fitness:.4f}, Avg: {np.mean(population_fitness):.4f}, Species: {species_count}")
        
        print(f"✅ Evolution completed! Session: {monitor.session_id}")


def example_custom_monitoring():
    """Example of custom monitoring with callbacks"""
    print("\n🎯 Example: Custom Monitoring with Callbacks")
    print("="*60)
    
    # Create custom monitor
    monitor = StandardTrainingMonitor(auto_save=False, update_interval=1.0)
    
    # Register callbacks for specific metrics
    def on_loss_improvement(loss_value, step):
        if step > 0 and loss_value < 0.5:
            print(f"🎉 Great progress! Loss dropped to {loss_value:.6f} at step {step}")
    
    def on_elo_milestone(elo_value, step):
        if elo_value > 1500:
            print(f"🏆 ELO milestone reached: {elo_value:.0f} at step {step}")
    
    monitor.register_callback('train_loss', on_loss_improvement)
    monitor.register_callback('elo_rating', on_elo_milestone)
    
    # Start monitoring
    monitor.start_monitoring({'custom_training': True})
    
    try:
        for step in range(15):
            # Simulate improving metrics
            loss = 2.0 * (0.85 ** step) + 0.1
            elo = 1200 + step * 30
            
            monitor.log_metric('train_loss', loss, step)
            monitor.log_metric('elo_rating', elo, step)
            
            time.sleep(0.2)
            
    finally:
        monitor.stop_monitoring()
    
    print(f"✅ Custom monitoring completed!")


def run_all_examples():
    """Run all visualization examples"""
    print("🎨 Training Visualization Examples")
    print("="*80)
    
    # Example 1: Standard training
    example_standard_training_with_visualization()
    
    # Example 2: Evolutionary training  
    example_evolutionary_training_with_visualization()
    
    # Example 3: Custom monitoring
    example_custom_monitoring()
    
    print("\n" + "="*80)
    print("✅ All examples completed!")
    print("📊 Generated visualization files in training_logs/")
    print("🌐 Start the dashboard to view: python src/ui/training_dashboard.py")
    print("📈 Or view HTML files directly in your browser")


if __name__ == "__main__":
    # Create examples directory and training_logs if they don't exist
    os.makedirs('training_logs', exist_ok=True)
    
    run_all_examples()