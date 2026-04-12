# FFNOML — 3D NLTE Radiative Transfer with Fourier Neural Operators

A machine learning framework for predicting NLTE departure coefficients in 3-D stellar atmospheres using 3D Fourier Neural Operators (FFNO).

This repository provides an end-to-end pipeline to:

- Load MULTI3D / Bifrost simulation outputs
- Construct training datasets on a column-mass grid
- Train a 3-D operator network mapping atmosphere → NLTE departure coefficients
- Predict NLTE populations on unseen simulations

The model is designed for large-scale solar atmosphere simulations and incorporates physics-informed loss functions derived from radiative transfer constraints.

---

# Overview

The network learns the operator

Atmospheric cube → NLTE departure coefficients

Input features per grid cell:

log10(T)
vx
vy
vz
log10(ne)
log10(rho)

Model output:

log10(n_NLTE / n_LTE)

for each atomic level.

The architecture combines:

- Global spectral mixing (FNO) in horizontal directions (x–y)
- Local vertical coupling (1-D depth mixing in z)
- Physics-informed loss terms

---

# Key Features

## Physics-aware training

Loss includes:

- Weighted MSE + L1 on departure coefficients
- Source function consistency penalty
- Multi-level NLTE constraints

---

## Operator learning

The model learns a 3-D operator:

[B, Cin, D, H, W] → [B, Cout, D, H, W]

This enables:

- patch-based training
- full-domain inference
- tiled prediction for large simulations

---

## Multi-scale supervision

Training samples are generated across multiple spatial scales:

scales = (1, 2, 3, 4)

with anti-alias filtering to improve generalization.

---

## Distributed training

Supports large-scale training using:

- PyTorch FSDP (Fully Sharded Data Parallel)
- multi-GPU training
- sharded checkpointing

---

# Repository Structure

.
├── config.py
├── pipeline.py
├── FFNONet.py
├── model_builder.py
├── train_utils.py
├── interp_utils.py
├── normalize_utils.py
├── data/
│   └── dataset.py
├── models/
│   └── ffno_model.py
├── loss/
│   ├── nlte_composite_loss.py
│   └── weighted_mse_loss.py
└── Forward.jl

---

# Model Architecture

The network is a 3D Factorized Fourier Neural Operator adapted for radiative transfer.

Main structure:

Input lifting layer
↓
FFNO blocks (spectral mixing in x–y)
↓
Vertical mixer (depth-wise coupling in z)
↓
Pointwise MLP
↓
Projection to output channels

Each FFNO block includes:

- spectral convolution (Fourier space)
- vertical coupling layer
- residual channel MLP

---

# Dataset Generation

Training data is generated from MULTI3D simulation outputs.

Pipeline:

1. Load Bifrost atmosphere
2. Interpolate to column-mass grid
3. Compute NLTE departure coefficients
4. Extract spatial patches
5. Store in HDF5 format

Dataset structure:

inputs  : [N, Cin, D, P, P]
targets : [N, Cout, D, P, P]

---

# Installation

Recommended environment:

Python >= 3.10
PyTorch >= 2.1
NumPy
SciPy
h5py
matplotlib
helita

Install dependencies:

pip install torch numpy scipy h5py matplotlib

For MULTI3D / Bifrost I/O:

pip install helita

---

# Configuration

All parameters are defined in:

config.py

Example model configuration:

MODEL_CONFIG = dict(
    width=64,
    modes_x=16,
    modes_y=16,
    n_layers=6,
    z_kernel=9,
    dropout=0.1,
    mlp_expansion=2
)

Dataset parameters:

PATCH = 32
STRIDE = 16
NDEP = 400
SCALES = (1, 2, 3, 4)

---

# Running the Pipeline

The full workflow is executed via:

python pipeline.py

Stages:

1. Load MULTI3D simulations
2. Build training dataset
3. Build validation dataset
4. Train FFNO model
5. Predict departure coefficients

---

# Training

Training uses:

- AdamW optimizer
- gradient clipping
- validation-based checkpointing

Example parameters:

NUM_EPOCHS = 50
BATCH_SIZE = 1
LEARNING_RATE = 2e-4

---

# Prediction

Supports two modes.

Full cube inference:

model(x, dx, dy)

Tiled inference:

tiled=True
patch=32
stride=16

Predictions are blended using a window function to avoid edge artifacts.

---

# Output

Predictions are stored as:

departure_coefficients

with dimensions:

[nx, ny, Ndep, Nlevels]

in HDF5 files.

---

# Physics Loss

The composite loss is:

L = L_data + λ L_phys

where:

- L_data = weighted MSE + L1
- L_phys = source function consistency

---

# Forward Synthesis

Forward.jl provides:

- LTE population computation
- NLTE population reconstruction
- spectral synthesis using Muspel

Run:

julia Forward.jl

---

# Applications

This framework enables:

- fast NLTE population prediction
- surrogate radiative transfer modeling
- acceleration of spectral synthesis
- large-scale solar atmosphere simulations

---

# License

MIT

---

# Author

Harsh Mathur
Computational Astrophysics / Radiative Transfer