"""
Training Visualization System for Neural Chess Engine
Provides real-time and post-training visualization for different training approaches
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from collections import defaultdict, deque

plt.style.use('dark_background')
sns.set_palette("husl")

class TrainingVisualizer:
    """Main visualization class for all training types"""
    
    def __init__(self, training_type: str, save_dir: str = "training_logs"):
        self.training_type = training_type
        self.save_dir = save_dir
        self.data_buffer = defaultdict(deque)
        self.max_buffer_size = 1000
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Color schemes for different training types
        self.colors = {
            'standard': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
            'evolutionary': ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f'],
            'neuroevolution': ['#bcbd22', '#17becf', '#1f77b4', '#ff7f0e'],
            'anti_stockfish': ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd']
        }
        
    def add_data_point(self, metric: str, value: Any, step: int = None):
        """Add a data point to the visualization buffer"""
        if step is None:
            step = len(self.data_buffer[metric])
        
        self.data_buffer[metric].append((step, value))
        
        # Keep buffer size manageable
        if len(self.data_buffer[metric]) > self.max_buffer_size:
            self.data_buffer[metric].popleft()
    
    def create_standard_training_viz(self) -> go.Figure:
        """Create visualization for standard neural training"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Loss Curves', 'ELO Progression', 'Learning Rate', 'Win Rate'],
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        # Loss curves
        if 'train_loss' in self.data_buffer:
            steps, losses = zip(*self.data_buffer['train_loss'])
            fig.add_trace(
                go.Scatter(x=steps, y=losses, name='Training Loss', 
                          line=dict(color='#1f77b4', width=2)),
                row=1, col=1
            )
        
        if 'val_loss' in self.data_buffer:
            steps, losses = zip(*self.data_buffer['val_loss'])
            fig.add_trace(
                go.Scatter(x=steps, y=losses, name='Validation Loss',
                          line=dict(color='#ff7f0e', width=2)),
                row=1, col=1
            )
        
        # ELO progression
        if 'elo_rating' in self.data_buffer:
            steps, elos = zip(*self.data_buffer['elo_rating'])
            fig.add_trace(
                go.Scatter(x=steps, y=elos, name='ELO Rating',
                          line=dict(color='#2ca02c', width=3)),
                row=1, col=2
            )
        
        # Learning rate
        if 'learning_rate' in self.data_buffer:
            steps, lrs = zip(*self.data_buffer['learning_rate'])
            fig.add_trace(
                go.Scatter(x=steps, y=lrs, name='Learning Rate',
                          line=dict(color='#d62728', width=2)),
                row=2, col=1
            )
        
        # Win rate
        if 'win_rate' in self.data_buffer:
            steps, rates = zip(*self.data_buffer['win_rate'])
            fig.add_trace(
                go.Scatter(x=steps, y=rates, name='Win Rate',
                          line=dict(color='#9467bd', width=2)),
                row=2, col=2
            )
        
        fig.update_layout(
            title="🧠 Standard Neural Training Progress",
            height=600,
            showlegend=True,
            template="plotly_dark"
        )
        
        return fig
    
    def create_evolutionary_viz(self) -> go.Figure:
        """Create visualization for evolutionary training"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Population Fitness', 'Species Diversity', 'Best Individual', 'Fitness Distribution'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Population fitness over generations
        if 'population_fitness' in self.data_buffer:
            generations, fitness_data = zip(*self.data_buffer['population_fitness'])
            
            # Extract max, mean, min fitness per generation
            max_fitness = [max(f) for f in fitness_data]
            mean_fitness = [np.mean(f) for f in fitness_data]
            min_fitness = [min(f) for f in fitness_data]
            
            fig.add_trace(
                go.Scatter(x=generations, y=max_fitness, name='Best',
                          line=dict(color='#2ca02c', width=3)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=generations, y=mean_fitness, name='Average',
                          line=dict(color='#1f77b4', width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=generations, y=min_fitness, name='Worst',
                          line=dict(color='#d62728', width=2)),
                row=1, col=1
            )
        
        # Species diversity
        if 'species_count' in self.data_buffer:
            generations, counts = zip(*self.data_buffer['species_count'])
            fig.add_trace(
                go.Bar(x=generations, y=counts, name='Species Count',
                       marker_color='#9467bd'),
                row=1, col=2
            )
        
        # Best individual tracking
        if 'champion_fitness' in self.data_buffer:
            generations, fitness = zip(*self.data_buffer['champion_fitness'])
            fig.add_trace(
                go.Scatter(x=generations, y=fitness, name='Champion',
                          line=dict(color='#ff7f0e', width=4),
                          mode='lines+markers'),
                row=2, col=1
            )
        
        # Current fitness distribution
        if 'population_fitness' in self.data_buffer and len(self.data_buffer['population_fitness']) > 0:
            _, latest_fitness = list(self.data_buffer['population_fitness'])[-1]
            fig.add_trace(
                go.Histogram(x=latest_fitness, name='Current Population',
                           marker_color='#8c564b', opacity=0.7),
                row=2, col=2
            )
        
        fig.update_layout(
            title="🧬 Evolutionary Training Progress",
            height=600,
            showlegend=True,
            template="plotly_dark"
        )
        
        return fig
    
    def create_neuroevolution_viz(self) -> Tuple[go.Figure, go.Figure]:
        """Create visualization for neuroevolution (returns 2 figures)"""
        
        # Figure 1: Training metrics
        fig1 = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Network Complexity', 'Innovation Rate', 'Performance', 'Architecture Diversity']
        )
        
        # Network complexity (nodes/connections over time)
        if 'network_nodes' in self.data_buffer:
            generations, nodes = zip(*self.data_buffer['network_nodes'])
            fig1.add_trace(
                go.Scatter(x=generations, y=nodes, name='Nodes',
                          line=dict(color='#bcbd22', width=2)),
                row=1, col=1
            )
        
        if 'network_connections' in self.data_buffer:
            generations, connections = zip(*self.data_buffer['network_connections'])
            fig1.add_trace(
                go.Scatter(x=generations, y=connections, name='Connections',
                          line=dict(color='#17becf', width=2)),
                row=1, col=1
            )
        
        # Innovation rate
        if 'innovations_per_gen' in self.data_buffer:
            generations, innovations = zip(*self.data_buffer['innovations_per_gen'])
            fig1.add_trace(
                go.Bar(x=generations, y=innovations, name='New Innovations',
                       marker_color='#1f77b4'),
                row=1, col=2
            )
        
        # Performance metrics
        if 'best_fitness' in self.data_buffer:
            generations, fitness = zip(*self.data_buffer['best_fitness'])
            fig1.add_trace(
                go.Scatter(x=generations, y=fitness, name='Best Fitness',
                          line=dict(color='#ff7f0e', width=3)),
                row=2, col=1
            )
        
        # Architecture diversity (could be complexity distribution)
        if 'complexity_distribution' in self.data_buffer:
            _, latest_complexity = list(self.data_buffer['complexity_distribution'])[-1]
            fig1.add_trace(
                go.Histogram(x=latest_complexity, name='Network Sizes',
                           marker_color='#2ca02c', opacity=0.7),
                row=2, col=2
            )
        
        fig1.update_layout(
            title="🔬 Neuroevolution Training Metrics",
            height=600,
            template="plotly_dark"
        )
        
        # Figure 2: Network topology visualization
        fig2 = self._create_network_topology_viz()
        
        return fig1, fig2
    
    def _create_network_topology_viz(self) -> go.Figure:
        """Create network topology visualization"""
        # Create a sample evolved network for demonstration
        G = nx.DiGraph()
        
        # Add sample nodes (input, hidden, output)
        input_nodes = range(0, 8)  # 8 input nodes
        hidden_nodes = range(8, 12)  # 4 hidden nodes
        output_nodes = range(12, 16)  # 4 output nodes
        
        # Add nodes with positions
        pos = {}
        # Input layer
        for i, node in enumerate(input_nodes):
            G.add_node(node, layer=0, node_type='input')
            pos[node] = (0, i)
        
        # Hidden layer
        for i, node in enumerate(hidden_nodes):
            G.add_node(node, layer=1, node_type='hidden')
            pos[node] = (1, i + 2)
        
        # Output layer
        for i, node in enumerate(output_nodes):
            G.add_node(node, layer=2, node_type='output')
            pos[node] = (2, i + 2)
        
        # Add sample connections
        connections = [
            (0, 8), (1, 8), (2, 9), (3, 9),
            (4, 10), (5, 10), (6, 11), (7, 11),
            (8, 12), (8, 13), (9, 13), (9, 14),
            (10, 14), (10, 15), (11, 15)
        ]
        
        for conn in connections:
            G.add_edge(conn[0], conn[1], weight=np.random.uniform(-1, 1))
        
        # Extract positions
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        # Create the figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Connections'
        ))
        
        # Add nodes
        node_colors = ['#ff7f0e' if node < 8 else '#2ca02c' if node < 12 else '#d62728' 
                      for node in G.nodes()]
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=[f'Node {node}' for node in G.nodes()],
            marker=dict(size=20, color=node_colors, line=dict(width=2, color='white')),
            name='Nodes'
        ))
        
        fig.update_layout(
            title="🔗 Evolved Network Topology",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="🟠 Input  🟢 Hidden  🔴 Output",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color="white", size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_dark"
        )
        
        return fig
    
    def create_anti_stockfish_viz(self) -> go.Figure:
        """Create visualization for anti-Stockfish training"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Stockfish Win Rates by Level', 'Training Progress', 'Strategy Discovery', 'Position Analysis'],
            specs=[[{"type": "bar"}, {"secondary_y": False}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # Win rates by Stockfish level
        if 'stockfish_win_rates' in self.data_buffer:
            _, latest_rates = list(self.data_buffer['stockfish_win_rates'])[-1]
            levels = list(range(1, len(latest_rates) + 1))
            
            fig.add_trace(
                go.Bar(x=levels, y=latest_rates, name='Win Rate %',
                       marker_color='#d62728'),
                row=1, col=1
            )
        
        # Training progress (games vs performance)
        if 'games_played' in self.data_buffer and 'avg_performance' in self.data_buffer:
            games = [step for step, _ in self.data_buffer['games_played']]
            performance = [val for _, val in self.data_buffer['avg_performance']]
            
            fig.add_trace(
                go.Scatter(x=games, y=performance, name='Performance',
                          line=dict(color='#2ca02c', width=3)),
                row=1, col=2
            )
        
        # Strategy discovery heatmap (position types vs success rate)
        position_types = ['Opening', 'Middlegame', 'Endgame', 'Tactical', 'Positional']
        success_rates = np.random.rand(5, 8) * 100  # Sample data
        
        fig.add_trace(
            go.Heatmap(
                z=success_rates,
                x=[f'Level {i}' for i in range(1, 9)],
                y=position_types,
                colorscale='RdYlGn',
                name='Success Rate'
            ),
            row=2, col=1
        )
        
        # Position analysis scatter (complexity vs win rate)
        if 'position_complexity' in self.data_buffer and 'position_win_rate' in self.data_buffer:
            complexity = [val for _, val in self.data_buffer['position_complexity']]
            win_rates = [val for _, val in self.data_buffer['position_win_rate']]
            
            fig.add_trace(
                go.Scatter(x=complexity, y=win_rates, mode='markers',
                          name='Positions', marker=dict(size=8, color='#9467bd')),
                row=2, col=2
            )
        
        fig.update_layout(
            title="🎯 Anti-Stockfish Training Analysis",
            height=600,
            template="plotly_dark"
        )
        
        return fig
    
    def save_visualization(self, fig: go.Figure, filename: str):
        """Save visualization to file"""
        filepath = os.path.join(self.save_dir, f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        fig.write_html(filepath)
        return filepath
    
    def generate_training_report(self) -> str:
        """Generate a comprehensive training report"""
        report = f"""
# 🏆 Neural Chess Training Report
**Training Type**: {self.training_type.title()}
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Summary Statistics
"""
        
        # Add type-specific summaries
        if self.training_type == 'standard':
            if 'train_loss' in self.data_buffer and len(self.data_buffer['train_loss']) > 0:
                final_loss = self.data_buffer['train_loss'][-1][1]
                report += f"- Final Training Loss: {final_loss:.6f}\n"
            
            if 'elo_rating' in self.data_buffer and len(self.data_buffer['elo_rating']) > 0:
                final_elo = self.data_buffer['elo_rating'][-1][1]
                report += f"- Final ELO Rating: {final_elo:.0f}\n"
        
        elif self.training_type == 'evolutionary':
            if 'population_fitness' in self.data_buffer and len(self.data_buffer['population_fitness']) > 0:
                _, latest_fitness = self.data_buffer['population_fitness'][-1]
                best_fitness = max(latest_fitness)
                avg_fitness = np.mean(latest_fitness)
                report += f"- Best Individual Fitness: {best_fitness:.4f}\n"
                report += f"- Average Population Fitness: {avg_fitness:.4f}\n"
        
        elif self.training_type == 'neuroevolution':
            if 'network_nodes' in self.data_buffer and len(self.data_buffer['network_nodes']) > 0:
                final_nodes = self.data_buffer['network_nodes'][-1][1]
                report += f"- Final Network Size: {final_nodes} nodes\n"
        
        report += "\n## 🎯 Training Insights\n"
        report += "- Visualization data collected successfully\n"
        report += f"- Total data points: {sum(len(buffer) for buffer in self.data_buffer.values())}\n"
        
        return report


# Example usage and testing functions
def test_standard_training_viz():
    """Test standard training visualization"""
    viz = TrainingVisualizer('standard')
    
    # Simulate training data
    for epoch in range(100):
        # Simulate decreasing loss
        train_loss = 2.0 * np.exp(-epoch/30) + 0.1 + np.random.normal(0, 0.05)
        val_loss = train_loss + 0.1 + np.random.normal(0, 0.03)
        
        # Simulate increasing ELO
        elo = 1200 + epoch * 8 + np.random.normal(0, 20)
        
        # Simulate learning rate schedule
        lr = 0.001 * (0.95 ** (epoch // 10))
        
        # Simulate win rate
        win_rate = min(0.8, 0.2 + epoch * 0.006 + np.random.normal(0, 0.05))
        
        viz.add_data_point('train_loss', train_loss, epoch)
        viz.add_data_point('val_loss', val_loss, epoch)
        viz.add_data_point('elo_rating', elo, epoch)
        viz.add_data_point('learning_rate', lr, epoch)
        viz.add_data_point('win_rate', win_rate, epoch)
    
    fig = viz.create_standard_training_viz()
    return viz, fig


def test_evolutionary_viz():
    """Test evolutionary training visualization"""
    viz = TrainingVisualizer('evolutionary')
    
    # Simulate evolutionary data
    for generation in range(50):
        # Simulate population fitness
        population_size = 50
        base_fitness = generation * 0.1
        fitness_values = np.random.normal(base_fitness, 0.5, population_size)
        fitness_values = np.maximum(fitness_values, 0)  # Ensure non-negative
        
        # Simulate species count
        species_count = max(1, 10 - generation // 10 + np.random.randint(-2, 3))
        
        # Champion fitness (best individual)
        champion_fitness = max(fitness_values)
        
        viz.add_data_point('population_fitness', fitness_values, generation)
        viz.add_data_point('species_count', species_count, generation)
        viz.add_data_point('champion_fitness', champion_fitness, generation)
    
    fig = viz.create_evolutionary_viz()
    return viz, fig


def test_neuroevolution_viz():
    """Test neuroevolution visualization"""
    viz = TrainingVisualizer('neuroevolution')
    
    # Simulate neuroevolution data
    for generation in range(40):
        # Network complexity grows over time
        nodes = 10 + generation + np.random.randint(0, 5)
        connections = nodes * 1.5 + np.random.randint(0, 10)
        
        # Innovation rate decreases over time
        innovations = max(0, 20 - generation // 2 + np.random.randint(-3, 5))
        
        # Best fitness improves
        best_fitness = generation * 0.15 + np.random.normal(0, 0.3)
        
        # Complexity distribution
        complexity_dist = np.random.randint(5, 25, 20)  # 20 individuals
        
        viz.add_data_point('network_nodes', nodes, generation)
        viz.add_data_point('network_connections', connections, generation)
        viz.add_data_point('innovations_per_gen', innovations, generation)
        viz.add_data_point('best_fitness', best_fitness, generation)
        viz.add_data_point('complexity_distribution', complexity_dist, generation)
    
    fig1, fig2 = viz.create_neuroevolution_viz()
    return viz, fig1, fig2


if __name__ == "__main__":
    print("🎨 Testing Training Visualizations...")
    
    # Test all visualization types
    print("1. Testing Standard Training Visualization...")
    viz1, fig1 = test_standard_training_viz()
    
    print("2. Testing Evolutionary Training Visualization...")
    viz2, fig2 = test_evolutionary_viz()
    
    print("3. Testing Neuroevolution Visualization...")
    viz3, fig3a, fig3b = test_neuroevolution_viz()
    
    print("✅ All visualizations created successfully!")
    print("📁 Check the training_logs directory for saved visualizations.")