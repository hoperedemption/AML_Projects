import torch 
import torch.nn as nn
import torch.nn.functional as F 

from utils import save_hyperparameters

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layer_depth, dropout):
        super().__init__()
        save_hyperparameters(self)
        
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim, num_layers=hidden_layer_depth, batch_first=True, dropout=dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.reshape(batch_size, seq_len, 1)
        h0 = torch.zeros(self.hidden_layer_depth, batch_size, self.hidden_dim)
        output, h_n = self.gru(x, h0)
        return h_n[-1, :, :]

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layer_depth, dropout):
        super().__init__()
        save_hyperparameters(self)
        
        self.gru = nn.GRU(input_size=1, hidden_size=input_dim, num_layers=hidden_layer_depth, batch_first=True, dropout=dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(-1)
        h0 = torch.zeros(self.hidden_layer_depth, batch_size, self.input_dim)
        output, h_n = self.gru(x, h0)
        return h_n[-1, :, :]
    
class Latent(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        save_hyperparameters(self)
        
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        return self.fc_mean(x), self.fc_log_var(x)
    
class VAE(nn.Module):
    def __init__(self, encoder, latent, decoder):
        super().__init__()
        self.encoder = encoder 
        self.latent = latent
        self.decoder = decoder 
    
    def reparametrize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        epsilon = torch.rand_like(std)
        return z_mean + epsilon * std 
    
    def forward(self, x):
        e = self.encoder(x)
        z_mean, z_log_var = self.latent(e)
        z = self.reparametrize(z_mean, z_log_var)
        return self.decoder(z), z_mean, z_log_var

def vae_loss(x_reconstructed, x, z_mean, z_log_var):
    # Reconstruction Loss (MSE)
    reconstruction_loss = nn.functional.mse_loss(x_reconstructed, x, reduction='mean')
    
    # KL divergence
    kl_divergence = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    
    return reconstruction_loss + kl_divergence



