import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from config import Config
from src.neural_network.chess_net import ChessNet
from src.utils.memory_manager import memory_manager
from src.utils.error_handler import safe_execute, handle_errors

class ChessNetTrainer:
    def __init__(self, model: ChessNet, learning_rate=Config.TRAINING_LEARNING_RATE):
        self.model = model
        self.device = Config.DEVICE
        self.model.to(self.device)
        
        # Use AdamW for better weight decay
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=Config.TRAINING_WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )
        
        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
        
        self.policy_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.value_loss_fn = nn.MSELoss()
        
        # Mixed precision training
        self.use_amp = Config.DEVICE == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
    @handle_errors
    def train_batch(self, states, target_policies, target_values):
        self.model.train()
        cleanup = memory_manager.monitor_memory("train_batch")
        
        try:
            states = memory_manager.optimize_tensor(states.to(self.device))
            target_policies = target_policies.to(self.device)
            target_values = target_values.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    pred_policies, pred_values = self.model(states)
                    policy_loss = self.policy_loss_fn(pred_policies, target_policies)
                    value_loss = self.value_loss_fn(pred_values.squeeze(), target_values)
                    total_loss = policy_loss + value_loss
                
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred_policies, pred_values = self.model(states)
                policy_loss = self.policy_loss_fn(pred_policies, target_policies)
                value_loss = self.value_loss_fn(pred_values.squeeze(), target_values)
                total_loss = policy_loss + value_loss
                
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
            
            return {
                'total_loss': total_loss.item(),
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item()
            }
        finally:
            cleanup()
    
    @handle_errors
    def train_epoch(self, dataloader):
        self.model.train()
        total_losses = {'total_loss': 0, 'policy_loss': 0, 'value_loss': 0}
        num_batches = len(dataloader)
        
        for states, policies, values in tqdm(dataloader, desc="Training"):
            losses = self.train_batch(states, policies, values)
            
            if losses:  # Check if batch training was successful
                for key in total_losses:
                    total_losses[key] += losses[key]
            
            # Step learning rate scheduler
            self.scheduler.step()
            
            # Periodic memory cleanup
            if num_batches % 50 == 0:
                memory_manager.force_cleanup()
        
        for key in total_losses:
            total_losses[key] /= num_batches
        
        # Add learning rate to losses
        total_losses['learning_rate'] = self.scheduler.get_last_lr()[0]
            
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