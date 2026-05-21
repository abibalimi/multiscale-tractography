#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class StreamlineEncoderGRU(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=128, latent_dim=128, num_layers=2):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers, # num stacked GRUs
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            latent_dim
        )

    def forward(self, x):

        # x shape: [B, T, 3] 

        _, hidden = self.gru(x)

        # last GRU layer hidden state
        hidden = hidden[-1]

        z = self.fc(hidden)

        return z


class StreamlineDecoderGRU(nn.Module):

    def __init__(self, latent_dim=128, hidden_dim=128, output_dim=3, num_layers=2, seq_len=64):
        super().__init__()

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.latent_to_hidden = nn.Linear(
            latent_dim,
            hidden_dim
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            output_dim
        )

    def forward(self, z):

        batch_size = z.size(0)

        hidden = self.latent_to_hidden(z)

        # initialize GRU hidden states
        hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)

        # repeated latent token sequence
        decoder_input = z.unsqueeze(1).repeat(1, self.seq_len, 1)

        outputs, _ = self.gru(decoder_input, hidden)

        reconstructed = self.output_layer(outputs)

        return reconstructed


class StreamlineAutoencoder(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=128, latent_dim=128, num_layers=2, seq_len=64):
        super().__init__()

        self.encoder = StreamlineEncoderGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers
        )

        self.decoder = StreamlineDecoderGRU(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=num_layers,
            seq_len=seq_len
        )

    def forward(self, x):

        z = self.encoder(x)
        reconstructed = self.decoder(z)

        return reconstructed, z
    
    
if __name__ == "__main__":

    model = StreamlineAutoencoder()

    x = torch.randn(8, 64, 3)

    reconstructed, z = model(x)

    print("Input shape:", x.shape)
    print("Latent shape:", z.shape)
    print("Reconstruction shape:", reconstructed.shape)