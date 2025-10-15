import json
import numpy as np
from torch.utils import data
import h5py
import torch
from example.trainer import TrainerWassersteinNormalizedAutoEncoder
from example.loader import Loader
from example.architectures import Encoder, Decoder
from wnae._logger import log
from pathlib import Path
import os
import shutil


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset parameters
n_train = 1000000
n_test = 100000
n_ood = 1000000
D = 10     # number of dimensions
N = 3       # scaling factor for correlated features
noise_std = 0.4

# Training data
x_train = np.zeros((n_train, D))
x_train[:,0] = np.random.normal(0, 1, n_train)  # first dimension
for i in range(1, D):
    x_train[:,i] = N * np.random.normal(0, 1, n_train) + np.random.normal(0, noise_std, n_train)

# Validation data
x_test = np.zeros((n_test, D))
x_test[:,0] = np.random.normal(0, 1, n_test)
for i in range(1, D):
    x_test[:,i] = N * np.random.normal(0, 1, n_test) + np.random.normal(0, noise_std, n_test)

# OOD data / signal
x_sig = np.zeros((n_ood, D))
x_sig[:,0] = np.random.normal(2, 1, n_ood)  # shifted mean
for i in range(1, D):
    x_sig[:,i] = N * np.random.normal(2, 1, n_ood) + np.random.normal(0, noise_std, n_ood)

# Convert to torch tensors and move to device
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
x_sig = torch.tensor(x_sig, dtype=torch.float32).to(device)

print("Train shape:", x_train.shape)
print("OOD shape:", x_sig.shape)

# DataLoaders
batch_size = 1024
train_loader = data.DataLoader(data.TensorDataset(x_train), batch_size=batch_size, shuffle=True)
val_loader   = data.DataLoader(data.TensorDataset(x_test),  batch_size=batch_size)
val_loader_no_batch = data.DataLoader(data.TensorDataset(x_test), batch_size=len(x_train))
sig_loader   = data.DataLoader(data.TensorDataset(x_sig), batch_size=batch_size)

# Wrap in MyLoader format
class MyLoader():
    def __init__(self, train_loader, val_loader, val_loader_no_batch, ood_loader):
        self.training_loader = train_loader
        self.validation_loader = val_loader
        self.validation_loader_no_batch = val_loader_no_batch
        self.ood_loader = ood_loader

loaders = MyLoader(train_loader, val_loader, val_loader_no_batch, sig_loader)

config = import_module("example.config")

# config.training_params["learning_rate"]
config.training_params["batch_size"] = batch_size
config.training_params['n_epochs'] = 5

input_size = x_train.shape[-1]
intermediate_architecture_encoder = (28,15)
intermediate_architecture_decoder = (57, 128, 64, 32, 24)
bottleneck_size = 8
output_path = "./toy_test2D"

encoder = Encoder(
    input_size=input_size,
    intermediate_architecture=intermediate_architecture_encoder,
    bottleneck_size=bottleneck_size,
    drop_out=None,
)
decoder = Decoder(
    output_size=input_size,
    intermediate_architecture=intermediate_architecture_decoder,
    bottleneck_size=bottleneck_size,
    drop_out=None,
)

trainer = TrainerWassersteinNormalizedAutoEncoder(
    config=config,
    loader=loaders,
    encoder=encoder,
    decoder=decoder,
    device=device,
    output_path=output_path,
    loss_function="wnae",  # can change to "ae" or "nae"
)

trainer.train()
log.info("Saving...")
trainer.save_train_plot()
log.info("Done.")
