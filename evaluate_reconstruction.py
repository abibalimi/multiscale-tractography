import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.synthetic_streamlines import SyntheticTractographyDataset
from lightning_modules.tractography_module import TractographyLightningModule


def reconstruct_batch(model, batch, device="mps"):
    """
    Reconstruct a batch of streamlines using the trained model.
    """
    streamlines, _ = batch
    streamlines = streamlines.to(device)
    
    with torch.no_grad():
        reconstructed, _ = model.model(streamlines)
    
    return streamlines.cpu().numpy(), reconstructed.cpu().numpy()


def plot_reconstructions(originals, reconstructions, n_examples=8):
    """
    Plot original vs reconstructed streamlines for a few samples in 2D.
    """ 
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    axes = axes.flatten()

    idxs = random.sample(range(len(originals)), n_examples)

    for ax, idx in zip(axes, idxs):
        orig = originals[idx]
        recon = reconstructions[idx]

        ax.plot(orig[:, 0], orig[:, 1], label="Original", color="blue")
        ax.plot(recon[:, 0], recon[:, 1], linestyle="--", label="Reconstructed", color="red")

        ax.axis("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend()
    
    fig.suptitle("Original vs Reconstructed Streamlines (2D Projection)")
    plt.tight_layout()
    plt.show()


def plot_reconstructions_5(original, reconstructed, num_samples=5):
    """
    Plot original vs reconstructed streamlines for a few samples.
    """
    fig = plt.figure(figsize=(12, 6))
    
    for i in range(num_samples):
        ax = fig.add_subplot(1, num_samples, i + 1, projection='3d')
        ax.plot(original[i][:, 0], original[i][:, 1], original[i][:, 2], label='Original', color='blue')
        ax.plot(reconstructed[i][:, 0], reconstructed[i][:, 1], reconstructed[i][:, 2], label='Reconstructed', color='red', linestyle='dashed')
        ax.set_title(f'Sample {i + 1}')
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    

def main():
    
    # deveice handling
    device = ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # load checkpoint
    checkpoint_path = "lightning_logs/version_5/checkpoints/epoch=9-step=1250.ckpt"
    model = TractographyLightningModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    
    # dataset and dataloader
    dataset = SyntheticTractographyDataset(n_samples=1000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    for batch in dataloader:
        originals, reconstructions = (
            reconstruct_batch(model, batch, device)
        )

        plot_reconstructions(originals, reconstructions)

        break # only plot for the first batch
    

if __name__ == "__main__":
    main()