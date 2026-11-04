# FFNO3D uncertainty and targeted training: report on claude-next-steps-uq

Branch `uq-calibration`. Instructions: `docs/claude-next-steps-uq.txt`. Plan and earlier results:
`docs/ffno3d-uncertainty-active-learning-plan.md`. All numbers are from 24 labelled H snapshots and
23 labelled CA snapshots (ch024031_by200bz005/950 has no CA), predicted on eagle7 with the
production full-domain operator (`scripts/uq_predict.py --route full`; 256² tiles for the 384³
boxes). Errors are e = log10(n_pred) − log10(n_truth) in dex. Regimes are mutually exclusive:
photosphere z < 0.3 Mm; above it, chromosphere log T < 4.5, TR 4.5–5.5, corona > 5.5.
These cases have all been inspected and are development benchmarks; no pristine labelled
evaluation block remains, so a final evaluation needs a new MULTI3D snapshot.

## Short answers

**Which configurations fail?**

- **H:**
  - Hot corona, log T > 6.05. There is no training coverage there; the errors are snapshot-wide offsets.
  - Cool, low-n_e pockets (n_e < 10^16.3–16.8 m⁻³) below 1.5 Mm. These are local errors, MAE 0.43, present in 19 of 24 cases, including training snapshots.
  - Upper chromosphere of nw012023 1050/1120, a negative offset of 0.2–0.5 dex.
  - The sub-surface layers of the two 384³ boxes.
- **CA:**
  - Cool (log T < 4.33), rarefied (log ρ < −9.8) gas above 2.5 Mm. Held-out MAE 0.56 against 0.13 on training snapshots, with normal training coverage.
  - The worst instance is ch012012_hion_759: Ca II is 1.3 dex too low at 2–3 Mm, so the model over-ionizes Ca II to Ca III there.

**Missing coverage, incompatible physics, or inference context?** All three occur, in different places:

| Cause | Where | Evidence |
|---|---|---|
| Inference context | ch024031 and helium sub-surface/photospheric offsets (−0.2 to −0.8 dex) | Local inputs are inside the training distribution, but the column span (16.8 Mm) is not. Cropping the column flips the sign of the offsets. |
| Missing coverage | H corona above log T 6.05 | Training coverage ≈ 0 |
| Poor generalization, not coverage | Cool rarefied chromospheric gas (both species); temporal drift of nw012023 | Training coverage is normal, but held-out error is 3–4× training error. The nw1120 offset survives every crop. |
| Physics, unconfirmed | Helium | Its n_e at fixed (T, ρ) scatters by ±0.3 dex against the other simulations; cropping removes only part of its offset. |

**Does adding one targeted snapshot help?** See "Controlled fine-tuning" below. Full fine-tuning does not fit eagle7 or eagle9, so the local experiment fine-tunes only the pointwise output head. The full version is prepared for Olivia (`uq/olivia_finetune.sbatch`).

**Which existing snapshots should enter training next?**

- **First, nw012023/1050**, the targeted branch. Keep nw012023/1120 for evaluating transfer in time.
- **Then snapshots spread in time** in ch012012_hion (759) and ch012006 (795). Their CA errors are among the largest of the same-simulation snapshots (0.82 and 0.24 dex).
- **Hold ch024031 and helium** until the column-extent issue is handled. Their offsets come from the inference context, which more labels from the same grid would not fix.

## 1. Validation and corrections

- **Spearman:** now tie-aware, using average ranks; it returns NaN for constant input.
- **Regimes:** mutually exclusive and exhaustive, and shared by all scripts (`uq_error_atlas.REGIMES`).
- **Atlas field names:** the grids holding median |e| are now `medae_*`.
- **Support distance:** for training snapshots, references from the same snapshot within a 1 Mm box (periodic horizontally) are excluded.
  - This lowered the H scores: case level ρ 0.50 → 0.40, and same-simulation relative ρ 0.54 → 0.26.
  - The earlier statement that "the training snapshot is always nearest" was mostly self-matching.
- **Alignment** (`scripts/uq_verify_alignment.py`, `uq/alignment_{H,CA}.csv`). For every case:
  - truth T versus input T matches to 3e-7 dex, while flips and the transpose are off by about 2 dex
  - truth z equals mesh z, truth dx equals mesh dx, and the n_e units agree
  - Σn_truth equals ntot
  - the identity is the best of the 720 level permutations; the closest alternative swaps H levels 4 and 5, which are only 0.06 dex apart in the photosphere
  - for H, Σn/n_H equals the abundance; for CA in the `_hion` simulations it is off by 0.021 dex, because readnh = 1 means `atmos.nh` holds the simulation's non-equilibrium hydrogen

## 2. Regimes (`uq/missing_regimes_{H,CA}.csv`)

A regression tree of |e| (H levels 2–5, CA levels 1–5) on these descriptors:

- T, n_e, ρ, v_z and |v_h|
- z and column mass
- vertical gradients of log T, log ρ and v_z
- the horizontal log T gradient and divergence, per Mm
- the log T contrast against a 1 Mm horizontal neighbourhood

The tree pools 100k voxels per case, without case identity. Each row gives:

- the parameter ranges
- training coverage
- MAE on training and on held-out snapshots
- the number of independent cases and simulations
- the fraction explained by per-case regime offsets (`snapshot_wide_frac`)
- a representative voxel

Tree R² is 0.35 for H and 0.33 for CA. These are associations, not causes.

## 3. ch024031 and helium

| Check | ch024031 | helium | Training simulations |
|---|---|---|---|
| MULTI3D input (atom, solver, boundaries) | identical | identical | — |
| ρ/(n_H m_H) | 1.357 | 1.357 | 1.357 |
| z(τ500 = 1) | ±0.07 Mm | +0.04 Mm | −0.07 to +0.13 Mm |
| log b_i near τ = 1 | thermalized | thermalized | thermalized |
| n_e at fixed (T, ρ) vs nw012023 | ±0.04 dex | ±0.3 dex | ±0.05 (non-hion) |
| local input percentiles in the biased layers | same | same | — |
| column span | **16.8 Mm** | **16.8 Mm** | 3.1, 10.4, 15.3, 15.4 Mm |

The offset is −0.13 to −0.30 dex for ch024031 and −0.5 to −0.9 dex for helium. It sits below τ = 1 and grows with depth, in layers the truth holds in LTE. Production on Olivia (full domain) has the same bias, so the 256² tiles are not the cause.

**Column-crop test** (`scripts/uq_predict.py --z-range`): bias of levels 2–5 at the same heights.

| Case | Band | Full column | z ≥ −0.97 Mm | z ≤ 7.98 Mm |
|---|---|---|---|---|
| ch024031 450 | −1.0 to −0.5 Mm | −0.27 | +0.20 | −0.14 |
| ch024031 450 | 8–14 Mm | +0.27 | −0.14 | — |
| helium 109 | −1.0 to −0.5 Mm | −0.78 | −0.53 | −0.44 |
| nw012023 915 (train) | 0.3–1 Mm | +0.02 | +0.15 | — |
| nw012023 1120 | 0.3–1 Mm | −0.49 | −0.37 | — |

Readings:
- **The operator depends strongly on the vertical extent of the column.** Even the training snapshot shifts by 0.1–0.3 dex under a moderate crop, and a hard crop (z ≤ 4 Mm) gives errors of 0.5–0.9 dex.
- **ch024031's offsets are a context effect:** they flip sign with the crop.
- **nw1120's offset is real temporal drift:** it stays at −0.2 to −0.9 dex under every crop.
- **Helium carries both a context component and a residual** consistent with its different n_e–T–ρ relation.

The vertical branch takes z_span and (z − z_min)/z_span as input features (`models/ffno_model.py:194`). Predictions should therefore only be trusted for column extents seen in training, or the model should be trained with varied column crops.

## 4. Uncertainty indicators

| Indicator | H voxel ranking | H case ranking | CA voxel ranking | CA case ranking | Cost |
|---|---|---|---|---|---|
| MC dropout (20 samples) | ρ −0.2 to 0.25; ~0 on nw1050 | 3 cases only | ρ 0.2–0.5; 0.41–0.49 on nw1050 | 3 cases only | 21 forward passes |
| Support distance (k = 10, self-excluded) | ρ −0.35 to +0.26 in the chromosphere, mostly negative | ρ 0.40 | ρ 0.38 mean in the chromosphere | ρ 0.71 | kNN on CPU |
| Conservation residual \|log Σn_pred − log ntot\| | ρ 0.4–0.8 on the worst cases; negative for qs levels 4–6 | ρ 0.55; top-5 by residual = 5 of the 6 worst cases | ρ 0.0–0.9, mixed | ρ 0.65; blind to ch012012_759 | free, label-free |

- **MC dropout spread is far too narrow.** RMS(e)/RMS(σ) is 6–9 on in-distribution cases and 25 on nw1050, so the spread does not grow with shift.
- **The conservation residual needs no labels.** ntot follows from ρ, the abundance and the fixed mean mass of 1.357, so it is checkable at inference time. It cannot see errors that conserve the element total, such as ionization-balance errors within Ca.

For the 3 MC cases, the MC-dropout entries are whole-volume Spearman per level from the atlas summaries; the support-distance voxel entries are chromosphere-only.

## 5. Spectra (`uq/spectral_{H,CA}.csv`)

`scripts/uq_synth.jl` reproduces Forward.jl's Muspel synthesis: its truth spectra match `IO/intensity_bifrost_*` exactly. A few columns are non-finite for the truth populations too; these are excluded and counted.

Production model, line-core intensity error (MAE):

| Case | H-alpha | Ca II 8542 |
|---|---|---|
| nw012023 915 (train) | 6.8% | 6.6% |
| nw012023 940 (val) | 8.5% | 10.2% |
| nw012023 1050 | 9.4% | 16.6% |
| nw012023 1120 | 11.3% | 13.7% |
| en024048_hion 700 (val) | 18.5% (bias +17%) | 20.7% (bias +18%) |
| ch012012_hion 759 | 7.2% | 10.2% |

Spectral and population errors are not monotonically related:
- **ch012012_hion_759:** a Ca II population error of 1.3 dex at 2–3 Mm costs only 10% in the 8542 core.
- **en024048_hion_700:** the populations are moderate, but both line cores are 17–18% too bright.

So population-level failures must be tracked even when a chosen line is insensitive to them.

## 6. Tiling

- **Variance:** stitched MC variance is now Σ w² var / W², the variance of the blended realization when tiles draw independent masks. `tile_std_log` is documented as a tile-disagreement diagnostic, not a predictive variance.
- **Large tiles vs full domain:**

| Case | Tiles | Median \|Δ\| (dex) | p99 (dex) | Regime MAE change |
|---|---|---|---|---|
| en385 | 192² | 0.009 | 0.05 | chromosphere unchanged; corona/TR +0.01 dex |
| nw915 | 192² | 0.0034 | 0.033 | ≤ 0.001 dex |
| ch012012_910 | 192² | 0.0025 | 0.041 | ≤ 0.001 dex |
| ch024031 450 vs Olivia | 256² | 0.006 | 0.025 | — |
| helium vs Olivia | 256² | 0.018 | 0.11 | — |

- **40² tiles are a different operator,** with a median difference of 0.064 dex.

## 7. Controlled fine-tuning (step 4–5)

Pending; filled in when `uq/retrain/{H,CA}_*` finish.
