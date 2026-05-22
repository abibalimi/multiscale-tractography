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


class StreamlineEncoderCNN(nn.Module):

    def __init__(self, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=3,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                padding=2
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(
            128,
            latent_dim
        )

    def forward(self, x):

        # [B, T, 3]
        x = x.permute(0, 2, 1) # [B, 3, T]

        x = self.encoder(x)

        x = x.squeeze(-1)

        z = self.fc(x)

        return z


class StreamlineDecoderGRU(nn.Module):

    def __init__(self, latent_dim=128, hidden_dim=128, output_dim=3, num_layers=2, seq_len=64, encoder_type="gru"):
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

    def __init__(self, input_dim=3, hidden_dim=128, latent_dim=128, num_layers=2, seq_len=64, encoder_type="gru"):
        super().__init__()
        
        self.encoder_type = (
            encoder_type
        )
        
        # GRU encoder
        self.gru_encoder = StreamlineEncoderGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers
        )
        
        # CNN encoder
        self.cnn_encoder = StreamlineEncoderCNN(
            latent_dim=latent_dim
        )
        
        # Shared decoder
        self.decoder = StreamlineDecoderGRU(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=num_layers,
            seq_len=seq_len
        )

    def forward(self, x):

        if self.encoder_type == "gru":
            z = self.gru_encoder(x)

        elif self.encoder_type == "cnn":
            z = self.cnn_encoder(x)

        else:
            raise ValueError("encoder_type must be 'gru' or 'cnn'")
        
        reconstructed = self.decoder(z)

        return reconstructed, z
    
    
if __name__ == "__main__":

    model = StreamlineAutoencoder(encoder_type="cnn")

    x = torch.randn(8, 64, 3)

    reconstructed, z = model(x)

    print("Input shape:", x.shape)
    print("Latent shape:", z.shape)
    print("Reconstruction shape:", reconstructed.shape)