# FFNO3D uncertainty and targeted MULTI3D acquisition plan

Date: 2026-10-29

## Recommendation

Start on Hyades with the existing predictions and MULTI3D truth. Build an atmospheric error atlas before spending GPU time training ensembles or generating new MULTI3D outputs. Then validate an uncertainty estimator and use it to rank physically distinct candidate snapshots. Olivia is optional for the initial analysis; use it for workloads that exceed the measured Hyades memory or throughput budget.

This is an implementation plan, not completed UQ analysis. The current shell is on `eden.uio.no`, not an authenticated Hyades GPU session. Hyades GPU specifications and scheduler configuration have not been verified. Python in this shell lacks NumPy, h5py and PyTorch; file presence and representative HDF5 headers were inspected using filesystem tools and `h5dump`. No numerical error statistics were calculated in this review.

## Evidence from the project

- `FFNONet.py::_prepare_input_features` uses log10(T), vx, vy, vz, log10(ne), log10(rho).
- `_make_targets_ch_first` predicts **log10 NLTE populations**, not log10 departure coefficients. Prediction writers store linear `nlte_populations` in m^-3. Departure coefficients are reconstructed for diagnostics/physics using LTE populations.
- `config.py` currently selects H, but separate H and CA checkpoints, population predictions and intensity files exist. Analyze each species explicitly; changing ACTIVE_ATOMS alone is not an experiment registry.
- Training uses five snapshots: en024048_hion/465, nw012023/915, ch012012_hion/910, ch012006/836, qs006003_sap/689. Validation uses 700, 940, 984, 849, 900 respectively. Validation already participates in checkpoint selection and is not an untouched final test set.
- Patches are 40 x 40, stride 20, scales 1..8; depth remains native. Dataset files are about 18–19 GB each. Overlapping patches and scale variants are dependent observations.
- `errorplots.py` loads predictions and MULTI3D truth and computes error metrics. Its main path currently reports H levels 2/3 and H-alpha: generalize the computation explicitly for all levels and CA instead of reusing those selections unchanged.
- `make_paper_plots.py` reads paired synthesized intensity files for four original test atmospheres. Reuse its wavelength/layout validation, rather than copying hardcoded plot data as measurements.
- `Forward.jl` provides threaded column formal synthesis with supplied populations. This is distinct from a fresh full 3D MULTI3D population solve.
- Serial prediction uses `_predict_tiled`; distributed full-volume prediction uses `ffno_predict_populations_distributed_full` and `distributed_inference.py`. Their context differs. Keep prediction strategy fixed within a UQ experiment and compare the routes separately.
- Current MODEL_CONFIG has spectral and vertical dropout 0.08 (general `dropout=0.0`). **Resolved (2026-10-29): both production checkpoints were trained with this dropout.** Commit 0b16637 (2026-07-10) raised `spec_dropout` 0.03→0.08 and, in the same commit, `hidden_vertical` 32→128; the dropout lines have not changed since. Both `3D_sim_train_H.pt` (epoch 171, saved 2026-07-16) and `3D_sim_train_CA.pt` (epoch 252, saved 2026-07-30) have `blocks.0.vertical.in_proj.0.weight` of shape (128, 48, 1), i.e. the post-0b16637 architecture, read directly from the pickle without torch. Residual caveat: a cluster-side uncommitted edit to the rates cannot be excluded, but nothing points to one.
- Dropout sits only in `FFNOBlock3dBalanced.spec_drop` / `.vert_drop` (`models/ffno_model.py:435-437`), applied after each branch activation in all four blocks. The inner MLP/vertical `nn.Dropout`s are `Identity` because `dropout=0.0`. All normalization is GroupNorm (no running statistics), so MC dropout is just `model.eval()` followed by `.train()` on those eight modules.
- Prediction files record epoch/validation loss, but lack enough provenance to establish exact checkpoint identity and inference settings. Checkpoints store normalization and channel metadata, but the current saver does not record a complete architecture/seed/data manifest.
- Training HDF5 groups are numbered (`train_1`, etc.) and lack explicit snapshot identity and original patch coordinates. New dataset builds must add these; existing groups require a verified sidecar mapping.
- `train_gpu.sh` and `predict_gpu.sh` are Olivia-specific eight-node launchers. They are not Hyades launch scripts and do not establish that eight nodes are required.

### Existing configured prediction cases

File-presence audit of the configured source locations and current model directory:

| Cases | H and CA predictions | Standard MULTI3D population files beside atmosphere | Initial role |
|---|---|---|---|
| en024048_hion_385 | Present | Present | Error atlas; same-family generalization |
| nw012023_1050 | Present | Present | Error atlas; same-family generalization |
| ch024031_by200bz005_450 | Present | Present | Error atlas; family absent from TRAIN_SPLIT |
| en024031_by100_helium_109 | Present | Present | Error atlas; family absent from TRAIN_SPLIT |
| en024048_hion_504_385; nw012023_512_1050 | Present | Not found | Resolution/context diagnostics pending provenance |
| ch024031_by200bz005_768_450; en024031_by100_helium_768_109 | Present | Not found | Resolution/context diagnostics pending provenance |
| ar098192_270000 | Present | Not found | Unlabelled candidate; truth may exist elsewhere |

“Present” does not establish solver convergence, matched grids or valid contents. Audit these before computing errors. Do not label the higher-resolution variants independent simulations until their derivation is verified. Interpolating a coarse MULTI3D solution does not create high-resolution 3D NLTE truth.

## Action items, in execution order

### 1. Inventory and freeze the baseline — CPU on Hyades

- [ ] Add `scripts/uq_inventory.py` and an explicit `uq_cases.json` manifest with data roots, species, truth paths, checkpoint paths and case roles. Support path overrides rather than editing global config for every run.
- [ ] Record checkpoint SHA256, Git revision, training/validation membership, atom/level ordering, normalization, model configuration, epoch, inference route and numerical precision. Record unavailable historical information as unknown.
- [ ] Inspect HDF5 headers and representative blocks; validate dimensions, spatial ordering, depth direction, coordinates, units, finiteness and positive populations. Match atmosphere, prediction and truth before any flattening.
- [ ] Confirm MULTI3D convergence and atomic/RT assumptions from available solver logs/configuration; distinguish doubtful labels from model error.
- [ ] Explicitly preserve units: population/ne m^-3; rho kg/m^3; source velocity km/s; physical derivatives in documented units. FFNONet converts z to Mm internally, while dx/dy are in metres. Do not silently mix them.
- [ ] Reserve future independent simulation/time blocks for final evaluation. The four existing cases have already been inspected in plots, so describe them as development benchmarks rather than pristine final tests.

Deliverable: machine-readable inventory and a list of compatible labelled, unlabelled and excluded cases.

### 2. Build an actual-error atlas — CPU on Hyades, no new predictions

- [ ] Add `scripts/uq_error_atlas.py`, factoring reusable readers/metrics out of `errorplots.py` without requiring plotting or CUDA imports for CPU analysis.
- [ ] Stream HDF5 and memory-mapped MULTI3D populations in blocks; avoid existing full-array float64 conversions. Process one case/species at a time with a configurable memory budget.
- [ ] Calculate signed and absolute log error: e_i = log10(n_pred,i) - log10(n_truth,i). Report bias, median absolute error, p90/p95 and threshold exceedance by level. For context, 0.1 dex corresponds to a multiplicative factor of about 1.26, not a symmetric 26% interval.
- [ ] Count invalid/zero values separately instead of silently discarding difficult cells. Report both cell-weighted and physical-volume-weighted summaries, and per-case summaries so large boxes do not dominate.
- [ ] Associate errors with logT, logne, logrho, all velocities, z, vertical and horizontal gradients, compression/shear, column mass and neighborhood temperature contrasts at fixed physical distances. Derivatives must use actual coordinate spacing and respect depth direction.
- [ ] Preserve full vertical context in sampled column/patch descriptors. Add approximate opacity/optical-depth descriptors only with a documented source; do not require a new RT solve for these features.
- [ ] Fit descriptor scaling and training-support references from training inputs only. Use sampled columns/patches and multivariate nearest-neighbor distances; univariate min/max ranges miss missing combinations.
- [ ] Build conditional error maps with sample counts, independent-case counts and training coverage: e.g. logT versus logne at fixed density/height bands. Avoid exhaustive six-dimensional histograms.
- [ ] Use spatial/time blocks for uncertainty on summary statistics. Millions of correlated voxels are not millions of independent validation samples.
- [ ] Produce a ranked table of problematic regimes, with representative case/column/region coordinates and evidence across simulations. Associations are diagnostic, not causal parameter effects.

Deliverables: `population_error_summary.csv`, compact sampled descriptor/error tables, diagnostic figures and `missing_regimes.csv`. This stage alone can identify useful new training regimes.

### 3. Connect population errors to existing spectral errors — CPU on Hyades

- [ ] Read paired `IO/intensity_bifrost_<case>_<atom>.h5` and `training_.../intensity_ml_<case>_FFNO3D_<atom>.h5` for the four original cases. Verify wavelength grids, intensity units, atmosphere, electron-density treatment and synthesis settings agree.
- [ ] Report line-core and profile errors separately for H and CA, with continuum-normalized errors where appropriate. Compare profiles computed by the same formal solver when isolating surrogate population error.
- [ ] Link column population errors to spectral errors; retain population-level failures even when one selected line is insensitive to them. Do not assign a particular height's importance from a column intensity error without contribution/response information.
- [ ] Define the target tolerance and intended deployment distribution before choosing acquisition weights. Keep broad population accuracy and diagnostic-specific priorities separately visible.

Deliverable: `spectral_error_summary.csv` and a list of regimes that materially affect intended observables.

### 4. Pilot uncertainty without retraining — Hyades GPU if it fits

- [ ] Add a separate `scripts/uq_predict.py` driver accepting checkpoint, model configuration, input case, inference strategy, seed, sample count and output directory. Keep production prediction deterministic and unchanged by default.
- [x] Check historical training provenance for spectral/vertical dropout before attempting MC dropout. Done: p=0.08 on both branches in all four blocks for H and CA (see Evidence). MC dropout is supported; no ensemble training is needed for the pilot.
- [ ] If supported by provenance, keep the model in eval mode and activate only its intended Dropout modules. Run a pilot of 10 stochastic passes, expanding to 20–30 if estimates have not stabilized. Do not enable unrelated training behavior.
- [ ] Start with one labelled original-size atmosphere. Benchmark full-domain memory/time at one and then multiple GPUs available on Hyades. If using tiles, calibrate that strategy explicitly; tiled uncertainty is not automatically uncertainty for the full-domain operator.
- [ ] Accumulate mean/variance online in **denormalized log-population units**, preserving level order. Store compact statistics and selected samples rather than all full-volume realizations. Describe exp10(mean log n) as a geometric mean, not an arithmetic population mean.
- [ ] Test deterministic repeatability, nonzero stochastic spread, log-space aggregation against explicit samples, layout conversion, and restoration of eval state. For distributed runs validate random streams and statistical/topology behavior.
- [ ] Treat dropout spread as a candidate ranking signal, not an automatically valid Bayesian posterior for this FFNO architecture. Compare it against measured held-out errors and the CPU training-support baseline.

Deliverables: memory/runtime benchmark, uncertainty maps and an initial uncertainty-versus-error evaluation. Skip MC dropout if training provenance cannot support it.

### 5. Train an ensemble if the pilot is insufficient — Hyades or Olivia after benchmarking

- [ ] Add explicit experiment seed and run-root options to `pipeline.py`/training setup. TRAINSELECT_SEED controls patch subset resampling, not all model/training randomness; at TRAINSELECT=1 it does not provide data diversity.
- [ ] Record complete architecture, dropout rates, training cases, seeds, normalization and patch provenance in new checkpoints and prediction manifests. Restore them for inference rather than relying on current global MODEL_CONFIG.
- [ ] Start with three independently initialized members; increase toward five if useful. Treat multiple epochs of the same training run and H-versus-CA checkpoints as different objects, not independent ensemble members for the same target.
- [ ] Keep training/validation membership fixed. If bootstrapping, resample whole snapshots/blocks and record the scheme; only five training snapshots exist, so assess the loss of regime coverage explicitly. Independent initialization on all five is the simpler baseline.
- [ ] Add snapshot identity, patch origin, stride and scale to new HDF5 builds. Preserve existing scale balancing when introducing hard-region sampling, and include representative old data during retraining.
- [ ] Benchmark one member on Hyades before requesting Olivia. Train sequentially or use independent jobs across available devices; there is no requirement to keep the ensemble resident simultaneously.

Deliverables: separate checkpoints and ensemble mean/spread in a common physical log-population convention.

### 6. Calibrate and reject unhelpful uncertainty — CPU plus selected GPU inference

- [ ] Evaluate whether high uncertainty captures high actual errors: rank correlation, error among retained low-uncertainty predictions, and fraction of large errors captured by the top-ranked fraction of regions.
- [ ] Compare dropout/ensemble spread, training-support distance, and their combination. Explicitly inspect low-spread/high-error regions for shared model bias.
- [ ] Use case/family-blocked cross-validation for the error predictor/calibrator. Four original benchmark cases limit conclusions; report that limitation rather than claiming universal coverage.
- [ ] For any advertised intervals, measure empirical coverage and width by atom, level, regime and simulation. Voxelwise intervals do not imply simultaneous coverage of the entire 3D field.
- [ ] Separate surrogacy error from uncertain atmospheric inputs, observational noise and MULTI3D modelling assumptions. This plan initially targets surrogacy error for fixed inputs/physics.
- [ ] Investigate systematic errors against patch boundaries, resolution and context. Some failures require architecture/inference changes or better labels, and will not be solved by adding more atmospheres.

Deliverable: a validated error-ranking method, or an explicit decision that available uncertainty is unreliable and selection should initially use measured gaps plus diversity.

### 7. Rank the next MULTI3D calculations — CPU; inference where benchmark permits

- [ ] Scan additional existing Bifrost snapshots and their metadata before proposing new MHD simulations. Labelled but unused snapshots can support a retrospective acquisition experiment.
- [ ] Group resampled variants and adjacent snapshots by common parent simulation/event. Keep them together when measuring diversity or evaluating generalization.
- [ ] Score each candidate using predicted error/exceedance, difficult-region extent, spectral relevance, novelty and estimated label cost. Publish score components; uncertainty alone is not guaranteed expected retraining benefit.
- [ ] Select diverse snapshots with a small exploration allocation. Produce `candidate_ranking.csv` with case, regime descriptors/ranges, supporting regions, selection reason, required atoms, solver settings, expected cost and parent identity.
- [ ] Select full domains or boundary/context-converged subdomains for MULTI3D. A hotspot is not automatically a valid isolated NLTE domain.
- [ ] If desired regimes are absent from the snapshot pool, translate descriptors into a request for physically consistent simulation scenarios. Do not independently randomize T, rho and ne or infer that a magnetic-field parameter is a direct FFNO input.

Deliverable: a specific simulation/snapshot acquisition list. Use Olivia for the selected MULTI3D jobs only if their resource needs warrant it.

### 8. Measure whether targeted acquisition works

- [ ] Compare random acquisition, descriptor-diverse acquisition and uncertainty-plus-diversity acquisition at matched MULTI3D cost and matched retraining budgets.
- [ ] First run a retrospective experiment using labels already available but withheld from selection/model training. Audit unused snapshots and reserve a separate final evaluation block before choosing this pool.
- [ ] Keep normalization policy fixed/documented, regenerate affected datasets when needed, and never include evaluation labels in fitting, acquisition or early stopping.
- [ ] Report error improvement per added snapshot and per solver-hour, including tail errors, spectral errors and regressions on previously covered regimes. Repeat enough runs to distinguish improvements from training randomness.

Deliverable: evidence that targeted data acquisition beats its cheaper baselines before launching a large new MULTI3D campaign.

## Results so far — H, 24 labelled snapshots (2026-10-29, branch `uq-calibration`)

Predictions were made on eagle7 (5 × T4) with `scripts/uq_predict.py --route full`, the same full-domain distributed operator as production. The 384³ boxes (ch024031_by200bz005 450/700/950, en024031_by100_helium_109) do not fit at full domain and used 256² Hann-blended tiles of that operator (`--tile 256`). Errors come from `scripts/uq_error_atlas.py`, and the table from `scripts/uq_cross_snapshot.py` (`uq/cross_snapshot_H.csv`). Regimes: photosphere z < 0.3 Mm; chromosphere logT < 4.5 and z ≥ 0.3 Mm; TR 4.5–5.5; corona > 5.5.

**Inference route.** On en024048_hion_385 the full route reproduces production to a median of 0.002 dex. The tiling penalty, measured against the full route:

| Route | Median \|Δ\| vs full | p99 | Chromosphere MAE, levels 2–5 |
|---|---|---|---|
| Full domain | — | — | 0.099–0.113 |
| 192² tiles | 0.009 | 0.05 | 0.099–0.113, unchanged |
| 40² tiles | 0.064 | 0.33 | 0.115–0.134 |

The coronal and TR MAE of the 192² tiles is about 0.01 dex higher than the full route. Large tiles are therefore an acceptable fallback, but 40² tiles are a different operator and should not be used.

**Chromosphere, H levels 2–5, mean MAE (dex) and bias:**

| Group | Cases (MAE) |
|---|---|
| Train | qs 689 0.059, ch012006 836 0.078, nw 915 0.083, ch012012 910 0.091, en 465 0.098 |
| Val | ch012006 849 0.106, qs 900 0.118, nw 940 0.123, ch012012 984 0.137, en 700 0.143 |
| Held out, same simulation | qs 1297 0.055, qs 1100 0.058, ch012006 820 0.090, en 385/386 0.106/0.108, ch012012 834 0.106, ch012006 795 0.108, ch012012 759 0.143, nw 1050 0.244, nw 1120 0.341 |
| Held out, new simulation | ch024031 450 0.135, 700 0.221, 950 0.313; helium 109 0.408 |

Findings:

- **Within a simulation, error grows with time from the training snapshot.** The clearest case is nw012023:

  | nw012023 snapshot | Steps from training (915) | Chromosphere MAE (dex) |
  |---|---|---|
  | 940 | +25 | 0.12 |
  | 1050 | +135 | 0.24 |
  | 1120 | +205 | 0.34 |

  ch012012_hion and en024048_hion show the same trend more weakly. qs006003_sap is flat.
- **Large errors are coherent offsets, not noise.** Where chromospheric MAE exceeds 0.2 dex, |bias| is 0.14–0.32 dex. Examples: nw 1120 has bias −0.32 against MAE 0.34, and ch024031 950 has −0.26 against 0.31.
  - Voxelwise spread estimates can only rank such failures by their spatial pattern. A snapshot-level novelty score is the more natural detector.
- **Photospheric errors flag a simulation-level mismatch.** All three ch024031 snapshots carry the same photospheric bias in levels 2–5, −0.17 to −0.29 dex; every other simulation except helium is within ±0.06.
  - Because the bias is the same at every snapshot, the cause is probably a difference in simulation setup (EOS, abundances, hydrogen treatment) rather than undersampling. Check this before acquiring more ch024031 labels.
  - The helium run is off everywhere (photosphere −0.3 to −0.8 dex) and should be treated as out of scope until its physics is confirmed compatible.
- **Coronal and TR errors are snapshot-wide offsets that change sign between snapshots.** Examples: nw 1120 −0.56, helium +0.80, ch024031 450 +0.27 dex at level 2. They come from non-local labels (see the earlier coronal notes) and matter little optically.
- **MC dropout (p = 0.08, 20 samples; nw 915, en 385, nw 1050) is a weak signal.**
  - Whole-volume Spearman ρ is −0.19 to 0.24, and in the chromosphere 0.26–0.39.
  - RMS(error)/RMS(spread) is about 7, so the spread is badly underdispersed.
  - It does not see the snapshot-level offsets above. Keep it only as a secondary within-snapshot ranking term.

- **Training-support distance** (`scripts/uq_support_distance.py`, `uq/support_distance_H_all.csv`) is the mean distance from a voxel to its 10 nearest training voxels. The descriptors are the six inputs at the voxel and at −0.5/+0.5/+1.5 Mm, plus z, standardized on the training set.
  - **Per voxel it is useless for H.** Chromospheric Spearman ρ against |e| for levels 2–5 lies between −0.35 and +0.26, and is mostly negative.
  - **Per snapshot it is a moderate signal.** The rank correlation of median chromospheric distance with chromospheric MAE is ρ = 0.50 over all 24 cases (0.54 without helium). Measured as the change relative to the simulation's own training snapshot, it is ρ = 0.54 over 15 cases.
  - In every simulation the training snapshot has the smallest distance. The ordering within nw012023 is right (915 1.30 → 940 1.65 → 1050/1120 about 1.78).
  - **Absolute distances are not comparable across simulations.** qs006003_sap scores 0.25–0.51, while the training snapshot ch012006_836 scores 1.52. ch024031 950 (MAE 0.31) scores lower than ch012012_hion_834 (MAE 0.11). It cannot be used on its own to pick candidates.
  - **What would improve it:** per-simulation normalization of the distance, or a descriptor that sees snapshot-wide state, such as the distribution of column structure rather than single voxels. Both are untested.

**Implications for acquisition (step 7):**
1. The cheapest gain is temporal coverage within the existing simulations: snapshots far from 915/910/465 in time. nw012023 1050/1120 are already labelled and are the obvious first retraining additions.
2. Investigate ch024031's photospheric offset before labelling more of it.
3. Neither MC dropout nor support distance is reliable enough to rank snapshots alone. Select initially on time distance plus diversity, and keep the retrospective test of step 8.

CA has not yet been run through the 24-snapshot queue; only the 4 original cases have CA atlases (`uq/atlas_prod`).

## Resource placement

| Work | Hyades suitability | Olivia trigger |
|---|---|---|
| Inventory, error atlas, physical descriptors, existing spectra, calibration, ranking | CPU jobs; use block reads and bounded samples | Only if local I/O/RAM/time budgets are inadequate |
| MC dropout or ensemble inference | CUDA GPU jobs after a memory benchmark | Domain/activation memory or throughput exceeds Hyades allocation |
| Independent ensemble training on 40 x 40 native-depth patches | Benchmark one GPU, then one node | Required training turnaround cannot be met locally |
| New formal spectra with supplied populations | Threaded CPU jobs using compatible Julia/Muspel environment | Local CPU throughput insufficient |
| Fresh full 3D MULTI3D population solutions | Depends on domain and available CPU/MPI capacity | Large production runs exceed Hyades resources |
| Production Julia MPI/FSDP inversion | Not needed for this initial UQ project | Separate production-runtime requirements |

Observed sizes justify streaming: the standard en024048_hion_385 output is (252,252,467,6) float32, about 0.66 GiB uncompressed per species before auxiliary arrays; ar098192 H output occupies about 31 GB on disk. Input, truth, LTE and float64 temporary arrays multiply this footprint. Neither file size nor a small checkpoint establishes GPU activation memory requirements.

## First implementation batch

1. Implement inventory and streaming population/descriptor analysis for en024048_hion_385/H.
2. Validate alignment and metrics against a small slice using an independent manual calculation.
3. Extend to the four original cases, H and CA, and existing paired spectra.
4. Produce the first missing-regimes report and benchmark Hyades resources.
5. MC dropout is justified by checkpoint provenance (resolved 2026-10-29); run the step-4 pilot and only fall back to a three-member ensemble if its spread fails calibration against measured error.

No new MULTI3D calculation is required for this first batch.

## Method references

- Deep ensembles: https://arxiv.org/abs/1612.01474
- MC dropout (a methodological basis, not automatic validation for this architecture): https://proceedings.mlr.press/v48/gal16.html
- PyTorch Dropout evaluation behavior: https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
- Dependence/distribution shift and calibration caveats: https://www.stat.cmu.edu/~ryantibs/papers/nexcp.pdf
