#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import lightning as L
import torch
import torch.nn.functional as F
from models.autoencoder import StreamlineAutoencoder


class TractographyLightningModule(L.LightningModule):

    def __init__(self, lr=1e-3, hidden_dim=128, latent_dim=128, seq_len=64):
        super().__init__()

        self.save_hyperparameters()

        self.model = StreamlineAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            seq_len=seq_len
        )

    def forward(self, x):
        reconstructed, z = self.model(x)
        return reconstructed, z


    def training_step(self, batch, batch_idx):
        streamlines, _ = batch
        reconstructed, _ = self(streamlines)
        loss = F.mse_loss(reconstructed, streamlines)

        self.log("train_loss", loss, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        streamlines, _ = batch
        reconstructed, _ = self(streamlines)
        val_loss = F.mse_loss(reconstructed, streamlines)

        self.log("val_loss", val_loss, prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        
        return optimizer