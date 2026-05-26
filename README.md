# Multiscale Tractography Representation Learning

A PyTorch Lightning prototype for learning latent representations of white matter tractography across scales.

## Overview

This project explores representation learning approaches for streamline-based white matter tractography using synthetic tractography data. The long-term goal is to investigate learning-based methods for understanding white matter organization across scales and bridging diffusion MRI (dMRI) with higher-resolution imaging modalities such as Three-Dimensional Polarized Light Imaging (3D-PLI).

The current prototype focuses on:

- Learning latent representations of streamline geometries
- Comparing recurrent and convolutional inductive biases
- Visualizing tractography organization in latent space

## Current Approach

Synthetic streamlines are generated using parametric curve models representing simplified tractography geometries:

- Straight streamlines
- Curved streamlines
- Fanning configurations
- Crossing configurations

Autoencoder-based representation learning is implemented using:

- **GRU encoder**
- **1D-CNN encoder**

Latent representations are visualized using **UMAP** to investigate the organization of streamline geometries.

## Preliminary Findings

Initial experiments suggest that both GRU and CNN encoders learn meaningful geometric representations of streamlines. Latent spaces consistently organize according to streamline geometry (e.g., straight, curved, fanning, crossing patterns).

Reconstruction analysis shows high-fidelity streamline reconstruction for synthetic tractography data. Visual inspection indicates that reconstructed streamlines closely match original trajectories, suggesting that the current prototype primarily captures streamline-level geometric information rather than higher-order tract semantics.

These observations motivate future evaluation on more realistic tractography phantoms (e.g., FiberCup) and contrastive/self-supervised objectives.

## Next Steps

- Evaluate on more realistic tractography phantoms (e.g., FiberCup)
- Explore contrastive/self-supervised representation learning
- Extend toward real dMRI and 3D-PLI tractography data
- Investigate multiscale and cross-modal tractography representations

## Tech Stack

- PyTorch
- PyTorch Lightning
- UMAP
- NumPy
- Matplotlib

## Repository Structure

```text
data/                  # synthetic streamline generation
models/                # GRU/CNN autoencoders
lightning_modules/     # Lightning training module
train.py               # training entry point
visualize_latent_space.py
evaluate_reconstruction.py
```

## Motivation

This prototype was developed as part of an independent research effort at the intersection of neuroimaging, diffusion MRI, tractography, and representation learning.