import torch 
import torch.nn as nn 
import torch.optim as optim
from utils import save_hyperparameters

import numpy as np

import matplotlib.pyplot as plt

class Trainer(nn.Module):
    def __init__(self, vae_model, train_loader, val_loader, vae_loss, optimizer, scheduler, device):
        save_hyperparameters(self)
    
    def train_one_epoch(self, train_dataloader):
        running_loss = 0.0
        
        for signal in train_dataloader:
            signal = signal[0].to(self.device)
            
            self.optimizer.zero_grad()
            
            signal_reconstructed, z_mean, z_log_var = self.vae_model(signal)

            loss = self.vae_loss(signal_reconstructed, signal, z_mean, z_log_var)
            loss.backward()
            self.optimizer.step()
                        
            running_loss += loss
                   
        return running_loss / len(train_dataloader)
    
    def validate_one_epoch(self, val_dataloader):
        running_loss = 0.0
        
        for vsignal in val_dataloader:
            vsignal = vsignal[0].to(self.device)
            vsignal_reconstructed, z_mean, z_log_var = self.vae_model(vsignal)
            vloss = self.vae_loss(vsignal_reconstructed, vsignal, z_mean, z_log_var)
            
            running_loss += vloss 
            
        return running_loss / len(val_dataloader)
    
    def train(self, train_dataloader, val_dataloader, n_epochs):
        training_loss = []
        validation_loss = []
        
        for epoch in range(n_epochs):
            # train one epoch
            self.vae_model.train()
            avg_loss = self.train_one_epoch(train_dataloader)
            print(f'Training Loss on epoch {epoch} has avg value {avg_loss}')
            training_loss.append(avg_loss.cpu().detach().numpy())
            
            # validate 
            self.vae_model.eval()
            
            with torch.no_grad():
                avg_vloss = self.validate_one_epoch(val_dataloader)
                print(f'Validation Loss on epoch {epoch} has avg value {avg_vloss}')
                validation_loss.append(avg_vloss.cpu().detach().numpy())
                
            # adjust scheduler
            self.scheduler.step(avg_vloss)

        return np.array(training_loss), np.array(validation_loss)
    
    def plot_losses(self, training_loss, validation_loss):
        epochs = range(1 ,len(training_loss) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, training_loss, label='Training Loss', marker='o', linestyle='-', color='b')
        plt.plot(epochs, validation_loss, label='Validation Loss', marker='s', linestyle='--', color='r')
        
        plt.title('Training vs Validation loss', fontsize=16, fontweight='bold')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.xticks(epochs)
        plt.tight_layout()
        plt.show()
        plt.savefig('loss_plot.png', dpi=300)
        
        
    def convert(self, dataloader, X_len, latent_len):
        converted_signals = torch.empty((X_len, latent_len), device=self.device)
        for i, signal in enumerate(dataloader):
            signal = signal[0]
            enc = self.vae_model.encoder(signal)
            z = self.vae_model.reparametrize(*self.vae_model.latent(enc))
            converted_signals[i] = z
        return converted_signals.cpu().detach().numpy()
            
            
        
         
            
        
        