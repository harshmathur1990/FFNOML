# NLTE Radiative Transfer with Factorized Fourier Neural Operators (FFNO)

Physics-informed machine learning framework for predicting **NLTE departure coefficients** in 3-D stellar atmospheres using **Factorized Fourier Neural Operators (FFNO)**.

This repository provides an end-to-end pipeline to:

- Load **MULTI3D / Bifrost simulation outputs**
- Construct training datasets on a **column-mass grid**
- Train a **3-D FFNO operator** mapping atmosphere → NLTE departure coefficients
- Predict NLTE populations on unseen simulations

The model is designed for **large-scale solar atmosphere simulations** and integrates **physics-based losses** derived from radiative transfer source functions.

---

# Overview

The network learns the operator

```
Atmospheric cube → NLTE departure coefficients
```

Input features per grid cell:

```
log10(T)
vx / 100
vy / 100
vz / 100
log10(ne)
log10(rho)
```

Model output:

```
log10(n_nlte / n_lte)
```

for each atomic level.

The architecture combines:

- **Global spectral mixing (FNO) in x–y**
- **Local vertical coupling (1-D convolution in z)**
- **Physics-informed loss terms**

---

# Key Features

## Physics-aware training

Loss includes:

- Weighted MSE on departure coefficients
- Source function consistency penalty
- Multi-atom NLTE constraints

---

## Operator learning

Unlike CNN approaches predicting a single column, this model learns a **full 3-D operator**

```
[B, Cin, D, H, W] → [B, Cout, D, H, W]
```

which enables:

- patch training
- full cube inference
- tiled prediction for large domains

---

## Multi-resolution training

Training samples are generated using

```
scales = (1, 2, 3, 4)
```

with anti-alias filtering to improve generalization across spatial resolutions.

---

# Repository Structure

```
.
├── config.py
│   Global experiment configuration
│
├── pipeline.py
│   End-to-end pipeline:
│   data generation → training → prediction
│
├── ffnonet.py
│   Dataset construction and FFNO training utilities
│
├── model_builder.py
│   Model + optimizer + loss builder
│
├── train_utils.py
│   Training loops and validation
│
├── interp_utils.py
│   Column-mass interpolation utilities
│
├── data/
│   ├── dataset.py
│   └── dataloader_builder.py
│
├── models/
│   └── ffno_model.py
│
└── loss/
    ├── nlte_composite_loss.py
    └── weighted_mse_loss.py
```

---

# Model Architecture

The network is a **Factorized Fourier Neural Operator** adapted for radiative transfer.

Main components:

```
Lift layer
   ↓
FFNO Blocks (spectral mixing in x,y)
   ↓
Vertical mixer (1-D conv in z)
   ↓
Pointwise MLP
   ↓
Projection to output channels
```

Each FFNO block contains:

- spectral convolution
- vertical depth mixing
- residual channel MLP

---

# Dataset Generation

Training data is created directly from **MULTI3D simulation outputs**.

Steps:

1. Load Bifrost atmosphere  
2. Interpolate to column-mass grid  
3. Compute departure coefficients  
4. Extract spatial patches  
5. Store in HDF5  

Generated dataset:

```
inputs  : [N, Cin, D, P, P]
targets : [N, Cout, D, P, P]
```

---

# Installation

Recommended environment:

```
Python >= 3.10
PyTorch >= 2.1
NumPy
SciPy
h5py
matplotlib
helita
```

Install dependencies:

```bash
pip install torch numpy scipy h5py matplotlib
```

For Bifrost / MULTI3D support:

```bash
pip install helita
```

---

# Configuration

All experiment parameters are defined in:

```
config.py
```

Example model configuration:

```python
MODEL_CONFIG = dict(
    width=64,
    modes_y=16,
    modes_x=16,
    n_layers=6,
    z_kernel=9,
    dropout=0.1,
    mlp_expansion=2
)
```

Dataset parameters:

```python
PATCH = 96
STRIDE = 48
NDEP = 400
SCALES = (1,2,3,4)
```

---

# Running the Pipeline

The full workflow is executed via

```bash
python pipeline.py
```

Steps automatically executed:

```
1. Load MULTI3D simulations
2. Build training dataset
3. Build validation dataset
4. Train FFNO model
5. Predict departure coefficients
```

---

# Training

Training uses:

- AdamW optimizer
- mixed precision (AMP)
- early stopping
- gradient clipping

Example parameters:

```python
NUM_EPOCHS = 50
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
```

---

# Prediction

Prediction supports two modes.

## Full cube inference

```
model(x, dx, dy)
```

## Tiled inference

Used for large grids.

```
tiled=True
patch=96
stride=48
```

Predictions are blended using a **Hann window** to avoid boundary artifacts.

---

# Output

Predictions are saved as

```
departure_coefficients
```

with dimensions

```
[1, Nlevels, Ndep, Nx, Ny]
```

in files

```
output_*.hdf5
```

---

# Physics Loss

The composite loss is

```
L = L_data + λ L_source
```

where

- `L_data` = weighted MSE on log departure coefficients
- `L_source` = consistency loss on source functions

The source function is computed from atomic level populations and transition energies.

---

# Applications

This framework enables:

- fast NLTE population synthesis
- surrogate modeling for radiative transfer
- ML acceleration of spectral synthesis
- large-scale solar atmosphere modeling

---

# License

MIT License

---

# Author

Harsh Mathur  
Computational Astrophysics / Radiative Transfer
