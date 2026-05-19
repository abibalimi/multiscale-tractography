# Multi-Scale White Matter Representation Learning

PyTorch Lightning prototype for learning latent representations of white matter tractography across scales.

This project explores **self-supervised representation learning for white matter streamlines**, inspired by tractography, diffusion MRI (dMRI), and 3D Polarized Light Imaging (3D-PLI). The long-term objective is to learn biologically meaningful latent spaces capable of capturing white matter organization across multiple spatial scales.

---

## Motivation

Diffusion MRI tractography provides a macro-scale representation of white matter pathways, while techniques such as **3D-PLI** reveal microstructural fiber organization at much higher resolution.

A central challenge in neuroimaging is learning representations that remain meaningful **across scales**.

This project investigates whether latent representations can be learned from streamline geometries in a self-supervised manner using:

- sequence autoencoders
- multi-scale consistency learning
- contrastive representation learning (SimCLR-inspired)

The long-term vision is to bridge:

```text
microstructure (3D-PLI)
        ↕
 mesostructure
        ↕
macro-scale tractography (dMRI)
```

---

## Research Question

Can we learn **scale-aware latent representations of white matter organization** from streamline geometry alone?

More specifically:

1. Can streamline autoencoders learn meaningful geometric embeddings?
2. Can representations remain stable across streamline resolutions?
3. Can self-supervised contrastive learning improve anatomical consistency?

---

## Current Prototype Scope

This repository currently focuses on an **M1-friendly proof of concept**.

### Phase 1 — Synthetic tractography

Generate biologically inspired synthetic streamlines:

- Straight bundles (corticospinal-like)
- Curved bundles (association fibers)
- Fanning bundles (corona radiata-like)
- Crossing fibers

### Phase 2 — Sequence autoencoder

Train a lightweight **GRU-based autoencoder** to learn latent streamline embeddings.

### Phase 3 — Multi-scale representation learning

Encourage latent consistency across streamline resolutions.

### Phase 4 — SimCLR extension (planned)

Learn contrastive representations from augmented streamline views.

---

## Repository Structure

```text
linc_multiscale_tractography/

├── data/
│   ├── synthetic_streamlines.py
│   ├── augmentations.py
│
├── models/
│   ├── autoencoder.py
│   ├── simclr_head.py
│
├── lightning/
│   ├── tractography_module.py
│
├── visualize_streamlines.py
├── train.py
├── configs.py
└── README.md
```

---

## Method Overview

Each streamline is represented as a sequence of 3D coordinates:

```text
[(x₁, y₁, z₁), ..., (xₙ, yₙ, zₙ)]
```

where:

- `n = 64` sampled points
- coordinates represent streamline geometry

A recurrent encoder maps streamlines into a latent representation:

```text
streamline → encoder → latent vector z
```

A decoder reconstructs streamline geometry:

```text
z → decoder → reconstructed streamline
```

---

## Synthetic Tractography

Before using real dMRI tractography, we validate the learning framework on controlled synthetic geometries.

The synthetic streamlines mimic common white matter configurations:

| Bundle Type | Biological Inspiration |
|-------------|-------------------------|
| Straight | Corticospinal tract |
| Curved | Arcuate fasciculus |
| Fan | Corona radiata |
| Crossing | Centrum semiovale |

This allows controlled experiments and interpretable latent evaluation.

---

## Training Objective

### Reconstruction objective

The first prototype optimizes streamline reconstruction:

L = ||s - ŝ||²

where:

- `s` = input streamline
- `ŝ` = reconstructed streamline

### Multi-scale consistency (planned)

Representations of the same streamline at different resolutions should remain close in latent space.

### Contrastive learning (planned)

Positive pairs:

- augmented versions of same streamline

Negative pairs:

- unrelated streamlines

---

## Installation

Clone repository:

```bash
git clone <repo_url>
cd linc_multiscale_tractography
```

Create environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Experiments

### Visualize synthetic streamlines

```bash
python visualize_streamlines.py
```

### Train autoencoder

```bash
python train.py
```

---

## Planned Evaluation

Latent representations will be evaluated using:

- t-SNE
- UMAP
- clustering behavior
- reconstruction fidelity
- robustness across scales

A successful representation should naturally organize streamline families in latent space.

---

## Future Directions

- Real dMRI tractography (HCP)
- Cross-scale dMRI ↔ 3D-PLI representation learning
- SimCLR-style self-supervision
- Transformer encoders for streamline modeling
- Anatomical priors from white matter connectivity graphs
- Domain adaptation across acquisition modalities

---

## Why This Matters

This prototype explores a practical path toward **multi-scale white matter representation learning**, combining ideas from:

- diffusion MRI
- tractography
- self-supervised learning
- geometric representation learning
- computational neuroimaging

The broader goal is to improve how structural brain organization is modeled across spatial scales.

---

## Citation

Work in progress.