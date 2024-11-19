import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

import pandas as pd 
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN

from vae import *
from trainer import *

df = pd.read_csv('transformed_data.csv', header=0, index_col='id')
X = df.drop(columns=['y']).to_numpy()
y = df['y']

oversampler = ADASYN()
X, y = oversampler.fit_resample(X, y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

print(f'DEVICE: {device}')

x_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
x_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)

print(f'x_train, x_val are on {device}')

train_dataset = torch.utils.data.TensorDataset(x_train_tensor)
val_dataset = torch.utils.data.TensorDataset(x_val_tensor)

batch_size = 10
input_dim = X_train.shape[1]
hidden_dim = 100
latent_dim = 25
hidden_layer_depth = 5

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))

model_encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, hidden_layer_depth=hidden_layer_depth, dropout=0.1)
model_latent = Latent(hidden_dim=hidden_dim, latent_dim=latent_dim)
model_decoder = Decoder(input_dim=input_dim, hidden_dim=hidden_dim, hidden_layer_depth=hidden_layer_depth, dropout=0.1)
model_vae = VAE(model_encoder, model_latent, model_decoder)

optimizer = torch.optim.Adam(model_vae.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

trainer = Trainer(model_vae, train_dataloader, val_dataloader, vae_loss, optimizer, scheduler, device=device)

training_loss, validation_loss = trainer.train(train_dataloader, val_dataloader, n_epochs=30)
trainer.plot_losses(training_loss, validation_loss)

x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
test_dataset = torch.utils.data.TensorDataset(x_tensor)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

converted_signals = trainer.convert(test_dataloader, X.shape[0], latent_dim)
converted_signals_df = pd.DataFrame(converted_signals)
concat_convert = pd.concat([y, converted_signals_df], axis=1)
concat_convert.to_csv('features.csv', index=True)