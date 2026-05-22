#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticStreamlineGenerator:
    """
    Generate biologically-inspired synthetic tractography streamlines.
    """

    def __init__(self, num_points=64, noise_std=0.01, seed=42):
        self.num_points = num_points
        self.noise_std = noise_std
        np.random.seed(seed)

    def _add_noise(self, streamline):
        noise = np.random.normal(0, self.noise_std, streamline.shape)
        return streamline + noise

    def straight_bundle(self):
        """
        Corticospinal-like
        """
        t = np.linspace(0, 1, self.num_points)

        x = t
        y = np.zeros_like(t)
        z = np.zeros_like(t)

        streamline = np.stack([x, y, z], axis=1)
        streamline = self._add_noise(streamline)

        return streamline

    def curved_bundle(self):
        """
        Arcuate fasciculus-like
        """
        t = np.linspace(0, 1, self.num_points)

        amplitude = np.random.uniform(0.3, 0.6)

        x = t
        y = amplitude * np.sin(np.pi * t)
        z = 0.2 * np.sin(2 * np.pi * t)

        streamline = np.stack([x, y, z], axis=1)
        streamline = self._add_noise(streamline)

        return streamline

    def fan_bundle(self):
        """
        Corona radiata-like
        """
        t = np.linspace(0, 1, self.num_points)

        alpha = np.random.uniform(-1.0, 1.0)
        beta = np.random.uniform(-0.5, 0.5)

        x = t
        y = alpha * (t ** 2)
        z = beta * (t ** 2)

        streamline = np.stack([x, y, z], axis=1)
        streamline = self._add_noise(streamline)

        return streamline

    def crossing_bundle(self):
        """
        Crossing-fiber geometry
        """
        t = np.linspace(0, 1, self.num_points)

        if np.random.rand() < 0.5:
            x = t
            y = t
            z = np.zeros_like(t)
        else:
            x = t
            y = -t +1 # crossing at coordonate (0.5, 0.5, 0)
            z = np.zeros_like(t)

        streamline = np.stack([x, y, z], axis=1)
        streamline = self._add_noise(streamline)

        return streamline

    def generate_streamline(self, tract_type):

        if tract_type == 0:
            return self.straight_bundle()

        elif tract_type == 1:
            return self.curved_bundle()

        elif tract_type == 2:
            return self.fan_bundle()

        elif tract_type == 3:
            return self.crossing_bundle()

        else:
            raise ValueError("Unknown tract type")


class SyntheticTractographyDataset(Dataset):

    def __init__(self, n_samples=5000, num_points=64, noise_std=0.01):
        self.n_samples = n_samples
        self.generator = SyntheticStreamlineGenerator(num_points=num_points, noise_std=noise_std)

        self.data = []
        self.labels = []

        for _ in range(n_samples):

            tract_type = np.random.randint(0, 4)

            streamline = self.generator.generate_streamline(tract_type)

            self.data.append(streamline)
            self.labels.append(tract_type)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        streamline = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return streamline, label