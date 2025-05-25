"""
Training Monitor Integration
Provides hooks for real-time training visualization across different training types
"""

import os
import json
import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import threading
try:
    from .training_visualizer import TrainingVisualizer
except ImportError:
    from training_visualizer import TrainingVisualizer

class TrainingMonitor:
    """Real-time training monitor that integrates with different training systems"""
    
    def __init__(self, training_type: str, save_dir: str = "training_logs", 
                 update_interval: float = 1.0, auto_save: bool = True):
        self.training_type = training_type
        self.save_dir = save_dir
        self.update_interval = update_interval
        self.auto_save = auto_save
        
        # Create visualizer
        self.visualizer = TrainingVisualizer(training_type, save_dir)
        
        # Monitoring state
        self.is_monitoring = False
        self.start_time = None
        self.callbacks = {}
        
        # Auto-save thread
        self._save_thread = None
        self._stop_saving = threading.Event()
        
        # Training session metadata
        self.session_id = f"{training_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.metadata = {
            'session_id': self.session_id,
            'training_type': training_type,
            'start_time': None,
            'end_time': None,
            'total_duration': None,
            'parameters': {}
        }
        
        os.makedirs(save_dir, exist_ok=True)
    
    def start_monitoring(self, training_params: Dict[str, Any] = None):
        """Start monitoring a training session"""
        self.is_monitoring = True
        self.start_time = time.time()
        self.metadata['start_time'] = datetime.now().isoformat()
        self.metadata['parameters'] = training_params or {}
        
        print(f"🔍 Started monitoring {self.training_type} training session: {self.session_id}")
        
        # Start auto-save thread if enabled
        if self.auto_save:
            self._start_auto_save()
    
    def stop_monitoring(self):
        """Stop monitoring and finalize session"""
        self.is_monitoring = False
        
        if self.start_time:
            duration = time.time() - self.start_time
            self.metadata['end_time'] = datetime.now().isoformat()
            self.metadata['total_duration'] = duration
        
        # Stop auto-save thread
        if self._save_thread:
            self._stop_saving.set()
            self._save_thread.join()
        
        # Save final visualization and metadata
        self._save_session_data()
        
        print(f"✅ Stopped monitoring. Session duration: {duration/60:.1f} minutes")
    
    def log_metric(self, metric_name: str, value: Any, step: Optional[int] = None):
        """Log a training metric"""
        if not self.is_monitoring:
            return
        
        self.visualizer.add_data_point(metric_name, value, step)
        
        # Execute any registered callbacks
        if metric_name in self.callbacks:
            self.callbacks[metric_name](value, step)
    
    def register_callback(self, metric_name: str, callback: Callable):
        """Register a callback for when a specific metric is logged"""
        self.callbacks[metric_name] = callback
    
    def _start_auto_save(self):
        """Start auto-save thread for periodic visualization updates"""
        def auto_save_loop():
            while not self._stop_saving.wait(self.update_interval):
                if self.is_monitoring:
                    try:
                        self._generate_current_visualization()
                    except Exception as e:
                        print(f"⚠️ Auto-save error: {e}")
        
        self._save_thread = threading.Thread(target=auto_save_loop, daemon=True)
        self._save_thread.start()
    
    def _generate_current_visualization(self):
        """Generate current visualization based on training type"""
        try:
            if self.training_type == 'standard':
                fig = self.visualizer.create_standard_training_viz()
            elif self.training_type == 'evolutionary':
                fig = self.visualizer.create_evolutionary_viz()
            elif self.training_type == 'neuroevolution':
                fig, _ = self.visualizer.create_neuroevolution_viz()
            elif self.training_type == 'anti_stockfish':
                fig = self.visualizer.create_anti_stockfish_viz()
            else:
                return
            
            # Save with session ID
            filename = f"{self.session_id}_live"
            self.visualizer.save_visualization(fig, filename)
            
        except Exception as e:
            print(f"⚠️ Visualization generation error: {e}")
    
    def _save_session_data(self):
        """Save session metadata and final visualizations"""
        # Save metadata
        metadata_file = os.path.join(self.save_dir, f"{self.session_id}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Generate final visualization
        self._generate_current_visualization()
        
        # Generate training report
        report = self.visualizer.generate_training_report()
        report_file = os.path.join(self.save_dir, f"{self.session_id}_report.md")
        with open(report_file, 'w') as f:
            f.write(report)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current training metrics summary"""
        metrics = {}
        
        for metric_name, data_buffer in self.visualizer.data_buffer.items():
            if len(data_buffer) > 0:
                latest_value = data_buffer[-1][1]
                metrics[metric_name] = {
                    'latest': latest_value,
                    'count': len(data_buffer)
                }
        
        return metrics


class StandardTrainingMonitor(TrainingMonitor):
    """Specialized monitor for standard neural training"""
    
    def __init__(self, **kwargs):
        super().__init__('standard', **kwargs)
    
    def log_epoch_data(self, epoch: int, train_loss: float, val_loss: float = None, 
                      elo_rating: float = None, learning_rate: float = None, 
                      win_rate: float = None):
        """Log epoch-level training data"""
        self.log_metric('train_loss', train_loss, epoch)
        if val_loss is not None:
            self.log_metric('val_loss', val_loss, epoch)
        if elo_rating is not None:
            self.log_metric('elo_rating', elo_rating, epoch)
        if learning_rate is not None:
            self.log_metric('learning_rate', learning_rate, epoch)
        if win_rate is not None:
            self.log_metric('win_rate', win_rate, epoch)


class EvolutionaryTrainingMonitor(TrainingMonitor):
    """Specialized monitor for evolutionary training"""
    
    def __init__(self, **kwargs):
        super().__init__('evolutionary', **kwargs)
    
    def log_generation_data(self, generation: int, population_fitness: list, 
                           species_count: int = None, champion_fitness: float = None):
        """Log generation-level evolutionary data"""
        self.log_metric('population_fitness', population_fitness, generation)
        if species_count is not None:
            self.log_metric('species_count', species_count, generation)
        if champion_fitness is not None:
            self.log_metric('champion_fitness', champion_fitness, generation)


class NeuroevolutionMonitor(TrainingMonitor):
    """Specialized monitor for neuroevolution training"""
    
    def __init__(self, **kwargs):
        super().__init__('neuroevolution', **kwargs)
    
    def log_neuroevolution_data(self, generation: int, network_nodes: int, 
                               network_connections: int, innovations: int = None,
                               best_fitness: float = None, complexity_dist: list = None):
        """Log neuroevolution-specific data"""
        self.log_metric('network_nodes', network_nodes, generation)
        self.log_metric('network_connections', network_connections, generation)
        if innovations is not None:
            self.log_metric('innovations_per_gen', innovations, generation)
        if best_fitness is not None:
            self.log_metric('best_fitness', best_fitness, generation)
        if complexity_dist is not None:
            self.log_metric('complexity_distribution', complexity_dist, generation)


class AntiStockfishMonitor(TrainingMonitor):
    """Specialized monitor for anti-Stockfish training"""
    
    def __init__(self, **kwargs):
        super().__init__('anti_stockfish', **kwargs)
    
    def log_stockfish_data(self, games_played: int, win_rates_by_level: list,
                          avg_performance: float = None, position_analysis: dict = None):
        """Log anti-Stockfish training data"""
        self.log_metric('games_played', games_played)
        self.log_metric('stockfish_win_rates', win_rates_by_level)
        if avg_performance is not None:
            self.log_metric('avg_performance', avg_performance)
        if position_analysis:
            self.log_metric('position_complexity', position_analysis.get('complexity', []))
            self.log_metric('position_win_rate', position_analysis.get('win_rates', []))


# Integration helpers
def create_monitor(training_type: str, **kwargs) -> TrainingMonitor:
    """Factory function to create appropriate monitor"""
    monitors = {
        'standard': StandardTrainingMonitor,
        'evolutionary': EvolutionaryTrainingMonitor,
        'neuroevolution': NeuroevolutionMonitor,
        'anti_stockfish': AntiStockfishMonitor
    }
    
    monitor_class = monitors.get(training_type, TrainingMonitor)
    return monitor_class(**kwargs)


# Context manager for easy usage
class MonitoredTraining:
    """Context manager for monitored training sessions"""
    
    def __init__(self, training_type: str, training_params: Dict[str, Any] = None, **kwargs):
        self.monitor = create_monitor(training_type, **kwargs)
        self.training_params = training_params
    
    def __enter__(self):
        self.monitor.start_monitoring(self.training_params)
        return self.monitor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop_monitoring()


# Example usage demonstrations
def demo_standard_training():
    """Demonstrate standard training monitoring"""
    print("📊 Demo: Standard Training Monitoring")
    
    with MonitoredTraining('standard', {'epochs': 10, 'batch_size': 32}) as monitor:
        for epoch in range(10):
            # Simulate training
            train_loss = 2.0 * (0.9 ** epoch) + 0.1
            val_loss = train_loss + 0.1
            elo = 1200 + epoch * 50
            lr = 0.001 * (0.95 ** epoch)
            win_rate = min(0.8, 0.3 + epoch * 0.05)
            
            monitor.log_epoch_data(epoch, train_loss, val_loss, elo, lr, win_rate)
            time.sleep(0.1)  # Simulate training time
    
    print("✅ Standard training demo completed")


def demo_evolutionary_training():
    """Demonstrate evolutionary training monitoring"""
    print("🧬 Demo: Evolutionary Training Monitoring")
    
    with MonitoredTraining('evolutionary', {'population_size': 50, 'generations': 20}) as monitor:
        for generation in range(20):
            # Simulate evolution
            import numpy as np
            population_fitness = np.random.normal(generation * 0.1, 0.3, 50)
            population_fitness = np.maximum(population_fitness, 0)
            species_count = max(1, 8 - generation // 5)
            champion_fitness = max(population_fitness)
            
            monitor.log_generation_data(generation, population_fitness.tolist(), 
                                      species_count, champion_fitness)
            time.sleep(0.1)
    
    print("✅ Evolutionary training demo completed")


if __name__ == "__main__":
    print("🎯 Training Monitor System Demo")
    print("="*50)
    
    # Run demos
    demo_standard_training()
    print()
    demo_evolutionary_training()
    
    print("\n🎉 All demos completed! Check training_logs/ for results.")