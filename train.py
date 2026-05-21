#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import lightning as L

from torch.utils.data import DataLoader, random_split
from data.synthetic_streamlines import SyntheticTractographyDataset
from lightning_modules.tractography_module import TractographyLightningModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train Tractography Representation Model with PyTorch Lightning.")
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of synthetic streamlines")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="GRU hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of GRU layers")
    parser.add_argument("--seq-len", type=int, default=64, help="Number of streamline points")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():

    args = parse_args()

    # Dataset
    dataset = SyntheticTractographyDataset(
        n_samples=args.n_samples,
        num_points=args.seq_len
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Model
    model = TractographyLightningModule(
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        seq_len=args.seq_len
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10
    )

    # Train
    trainer.fit(
        model,
        train_loader,
        val_loader
    )


if __name__ == "__main__":
    main()