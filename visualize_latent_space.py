#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap

from torch.utils.data import DataLoader
from data.synthetic_streamlines import SyntheticTractographyDataset
from lightning_modules.tractography_module import TractographyLightningModule

from sklearn.cluster import KMeans


def extract_latents(model, dataloader, device="mps"):
    """
    Extract latent representations from the trained model for visualization.
    """
    model.eval()

    latent_vectors = []
    labels = []

    with torch.no_grad():

        for batch in dataloader:

            streamlines, y = batch
            streamlines = streamlines.to(device)

            reconstructed, z = model(streamlines)

            latent_vectors.append(z.cpu().numpy())

            labels.append(y.numpy())

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)

    return latent_vectors, labels


def plot_streamlines(streamlines, title):
    """
    Visualize streamlines in 3D space.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for streamline in streamlines:
        ax.plot(
            streamline[:, 0],
            streamline[:, 1],
            streamline[:, 2],
            alpha=0.7
        )

    ax.set_title(title)

    plt.show()    
    
    
def crossing_sanitycheck(dataset, labels, embedding):
    """
    Sanity check: visualize crossing fibers in latent space
    """
    crossing_idx = labels == 3
    crossing_embedding = embedding[crossing_idx]
    
    all_streamlines = []
    for i in range(len(dataset)):
        streamline, _ = dataset[i]
        all_streamlines.append(streamline.numpy())

    all_streamlines = np.array(all_streamlines)
    crossing_streamlines = (all_streamlines[crossing_idx])
    
    # clustering in latent space to separate the two crossing bundles
    kmeans = KMeans(n_clusters=2, random_state=42)
    crossing_clusters = (kmeans.fit_predict(crossing_embedding))

    cluster_0 = (crossing_streamlines[crossing_clusters == 0])
    cluster_1 = (crossing_streamlines[crossing_clusters == 1])

    plot_streamlines(cluster_0[:10], "Crossing Cluster A")
    plot_streamlines(cluster_1[:10], "Crossing Cluster B")
    
        
def main():

    checkpoint_path = (
        "lightning_logs/"
        "version_4/"
        "checkpoints/"
        "epoch=9-step=1250.ckpt"
    )

    model = (
        TractographyLightningModule
        .load_from_checkpoint(
            checkpoint_path
        )
    )

    dataset = (
        SyntheticTractographyDataset(
            n_samples=2000
        )
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False
    )

    latent_vectors, labels = (
        extract_latents(
            model,
            dataloader
        )
    )

    reducer = umap.UMAP(
        n_neighbors=50,
        min_dist=0.3,
        random_state=42
    )

    embedding = reducer.fit_transform(
        latent_vectors
    )
    
    # Sanity check: visualize crossing fibers in latent space
    crossing_sanitycheck(dataset, labels, embedding)
    
    tract_names = [
        "Straight",
        "Curved",
        "Fan",
        "Crossing"
    ]

    plt.figure(figsize=(10, 8))

    for i in range(4):

        idx = labels == i

        plt.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            label=tract_names[i],
            alpha=0.7
        )

    plt.legend()

    plt.title(
        "Latent Space of "
        "Synthetic Streamlines"
    )

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    plt.show()


if __name__ == "__main__":
    main()