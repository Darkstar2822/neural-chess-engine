import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from config import Config
from src.neural_network.chess_net import ChessNet

class ChessNetTrainer:
    def __init__(self, model: ChessNet, learning_rate=Config.TRAINING_LEARNING_RATE):
        self.model = model
        self.device = Config.DEVICE
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=Config.TRAINING_WEIGHT_DECAY
        )
        
        self.policy_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.value_loss_fn = nn.MSELoss()
        
    def train_batch(self, states, target_policies, target_values):
        self.model.train()
        
        states = states.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values = target_values.to(self.device)
        
        self.optimizer.zero_grad()
        
        pred_policies, pred_values = self.model(states)
        
        policy_loss = self.policy_loss_fn(pred_policies, target_policies)
        value_loss = self.value_loss_fn(pred_values.squeeze(), target_values)
        
        total_loss = policy_loss + value_loss
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_losses = {'total_loss': 0, 'policy_loss': 0, 'value_loss': 0}
        num_batches = len(dataloader)
        
        for states, policies, values in tqdm(dataloader, desc="Training"):
            losses = self.train_batch(states, policies, values)
            
            for key in total_losses:
                total_losses[key] += losses[key]
        
        for key in total_losses:
            total_losses[key] /= num_batches
            
        return total_losses
    
    def train(self, training_data, epochs=Config.TRAINING_EPOCHS_PER_ITERATION):
        states, policies, values = training_data
        
        dataset = TensorDataset(
            torch.FloatTensor(states),
            torch.FloatTensor(policies),
            torch.FloatTensor(values)
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=Config.TRAINING_BATCH_SIZE, 
            shuffle=True
        )
        
        training_history = []
        
        for epoch in range(epochs):
            epoch_losses = self.train_epoch(dataloader)
            training_history.append(epoch_losses)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Loss: {epoch_losses['total_loss']:.4f} "
                  f"(Policy: {epoch_losses['policy_loss']:.4f}, "
                  f"Value: {epoch_losses['value_loss']:.4f})")
        
        return training_history
    
    def evaluate(self, test_data):
        self.model.eval()
        states, policies, values = test_data
        
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            target_policies = torch.FloatTensor(policies).to(self.device)
            target_values = torch.FloatTensor(values).to(self.device)
            
            pred_policies, pred_values = self.model(states)
            
            policy_loss = self.policy_loss_fn(pred_policies, target_policies)
            value_loss = self.value_loss_fn(pred_values.squeeze(), target_values)
            total_loss = policy_loss + value_loss
            
            return {
                'total_loss': total_loss.item(),
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item()
            }