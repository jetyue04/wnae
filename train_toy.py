import os
import json
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from torch.utils import data
from example.trainer import TrainerWassersteinNormalizedAutoEncoder
from example.architectures import Encoder, Decoder
from example.pdf_generation import create_report
from wnae._logger import log

# ------------------------
# Helper functions
# ------------------------

def load_config(config_path, overrides=None):
    """Load YAML config and apply CLI overrides."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if overrides:
        for key, value in overrides.items():
            keys = key.split(".")
            sub = config
            for k in keys[:-1]:
                sub = sub.setdefault(k, {})
            sub[keys[-1]] = value
    return config

def prepare_dataloaders(data_config, device):
    """Generate toy dataset from config and prepare DataLoaders."""

    # Read parameters from config
    n_train = data_config.get("n_train", 10000)
    n_test = data_config.get("n_test", 1000)
    n_ood = data_config.get("n_ood", 1000)
    D = data_config.get("D", 10)
    N = data_config.get("N", 3)
    noise_std = data_config.get("noise_std", 0.4)
    batch_size = data_config.get("batch_size", 1024)

    # Training data
    x_train = np.zeros((n_train, D))
    x_train[:, 0] = np.random.normal(0, 1, n_train)

    for i in range(1, D):
        x_train[:, i] = N * np.random.normal(0, 1, n_train) + np.random.normal(0, noise_std, n_train)
    if data_config.get("correlated", False):
        correlated_idx = data_config.get("correlated_idx", [])
        rho = 0.5                      # target correlation strength (moderate)
        for i in correlated_idx:
            # moderate correlation: mixture of base feature and noise
            x_train[:, i] = (
                rho * x_train[:, 0] +
                (1 - rho) * np.random.normal(0, 1, n_train)
            )

    # Validation data
    x_test = np.zeros((n_test, D))
    x_test[:, 0] = np.random.normal(0, 1, n_test)
    for i in range(1, D):
        x_test[:, i] = N * np.random.normal(0, 1, n_test) + np.random.normal(0, noise_std, n_test)

    # OOD data / signal
    x_sig = np.zeros((n_ood, D))
    x_sig[:, 0] = np.random.normal(2, 1, n_ood)
    for i in range(1, D):
        x_sig[:, i] = N * np.random.normal(2, 1, n_ood) + np.random.normal(0, noise_std, n_ood)

    # Convert to torch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    x_sig = torch.tensor(x_sig, dtype=torch.float32).to(device)

    if data_config.get("min_max", False):
        data_min = torch.min(x_train, dim=0).values
        data_max = torch.max(x_train, dim=0).values

        x_train = (x_train - data_min) / (data_max - data_min + 1e-8)
        x_test = (x_test - data_min) / (data_max - data_min + 1e-8)
        x_sig = (x_sig - data_min) / (data_max - data_min + 1e-8)

    # DataLoaders
    train_loader = data.DataLoader(data.TensorDataset(x_train), batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(data.TensorDataset(x_test), batch_size=batch_size)
    val_loader_no_batch = data.DataLoader(data.TensorDataset(x_test), batch_size=len(x_train))
    sig_loader = data.DataLoader(data.TensorDataset(x_sig), batch_size=batch_size)

    class MyLoader:
        def __init__(self, train_loader, val_loader, val_loader_no_batch, ood_loader):
            self.training_loader = train_loader
            self.validation_loader = val_loader
            self.validation_loader_no_batch = val_loader_no_batch
            self.ood_loader = ood_loader

    return MyLoader(train_loader, val_loader, val_loader_no_batch, sig_loader), x_train.shape[-1]


def save_config(output_path, config):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(f"{output_path}/config.json", "w") as file:
        json.dump(config, file, indent=4)

# ------------------------
# Main training function
# ------------------------

def main(args):
    config = load_config(args.config, overrides=args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]

    loaders, input_size = prepare_dataloaders(
        {**data_cfg, "batch_size": train_cfg["batch_size"]}, device
    )

    # Create unique output path
    output_path_base = Path(data_cfg["output"])
    counter = 1
    output_path = output_path_base
    while output_path.exists():
        output_path = Path(f"{output_path_base}_{counter}")
        counter += 1
    Path(output_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to {output_path}")

    save_config(output_path, config)

    encoder = Encoder(
        input_size=input_size,
        intermediate_architecture=tuple(model_cfg["encoder"]["intermediate_architecture"]),
        bottleneck_size=model_cfg["encoder"]["bottleneck_size"],
        drop_out=model_cfg["encoder"]["drop_out"],
    )
    decoder = Decoder(
        output_size=input_size,
        intermediate_architecture=tuple(model_cfg["decoder"]["intermediate_architecture"]),
        bottleneck_size=model_cfg["decoder"]["bottleneck_size"],
        drop_out=model_cfg["decoder"]["drop_out"],
    )

    trainer = TrainerWassersteinNormalizedAutoEncoder(
        config=config,
        loader=loaders,
        encoder=encoder,
        decoder=decoder,
        device=device,
        output_path=output_path,
        loss_function="wnae",
    )

    trainer.train()
    log.info("Saving...")
    trainer.save_train_plot()
    log.info("Done.")

    log.info('Creating PDF Report...')
    create_report(output_path, config)

# ------------------------
# CLI Entry Point
# ------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WNAE on toy dataset")

    parser.add_argument("--config", type=str, default="config/toy_config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--override", nargs="*", default=[],
                        help='Override config, e.g. train.batch_size=512 model.encoder.bottleneck_size=6')

    args = parser.parse_args()

    # Parse CLI overrides into dict
    override_dict = {}
    for item in args.override:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        override_dict[key] = value
    args.override = override_dict

    main(args)
