# FFNO level fractions, cbh24 data and capacity: plan

Written 2026-11-13, updated 2026-11-16. Living copy: https://claude.ai/artifact/5qMZnySRnLHVWUqSubHykT

**Status (2026-11-16).** Phases 0–3 are done. The level-fraction head now predicts **departure coefficients**
relative to MULTI3D's own LTE populations (see *Level-fraction output*), and Phase 4 runs on that head: `cap0_base`
(Olivia job 2978340) and `cap1_zspec` (2978341). Ca waits until an H model has passed every phase, then gets one
training run.

## Summary

The work runs in **7 phases** and needs about **16,000 Olivia GPU-hours**, including both final trainings (H and Ca),
or about 12,000 without the Ca retrain. That is roughly the project's whole remaining GPU quota, so the final model's
capacity has to be capped at about 2× today's cost per epoch.

- **Goal:** retrain the FFNO on 6 simulations for training and 6 for validation, with cbh24 as the sixth. It will
  predict each species' total and its departure coefficients from MULTI3D's LTE split among the levels, and have more
  capacity.
- **cbh24 snapshots:** train on `s0910_half`, validate on `s0928_half`. They are 18 snapshots apart, and both are at
  least 29 from the `s0957` test snapshot.
- **Order:** data and code first, all on eagle7 and hyades. Then a level-fraction A/B test, then capacity changes one
  at a time, then the final trainings.
- **Olivia use:** only GPU training and `--fsdppredict`. Builds, scoring and synthesis stay on eagle7 and hyades.
- **Biggest uncertainty:** the cost of the larger model. Every multiplier here is an estimate until the Phase 3
  benchmark measures it.

## Starting point

Before this plan, the network mapped 6 inputs (ρ, T, v_x, v_y, v_z, nₑ) to 6 normalised log₁₀ level populations per
atom. Nothing tied the populations to ρ, so the forward solver rescales them afterwards. (The table describes that
starting point; the level-fraction head below replaces the output.)

| Item | Now | Where |
| --- | --- | --- |
| Output | log₁₀ n per level, standardised with `mean_Y`/`std_Y` | `models/ffno_model.py` head (`proj1`, `proj2`) |
| Levels | H: 5 bound + H II; Ca: 5 Ca II + Ca III | `config.py` `ATOM_CONFIG` |
| Model | width 48, 4 blocks, `hidden_vertical` 128, `hidden_spectral` 128 | `config.py` `MODEL_CONFIG` |
| Horizontal mixing | one complex gate per (k_x, k_y), shared by all channels | `SpectralConv2dFull.freq_mlp` |
| Side paths | `res_pw`, `res_mlp` start at 0.15; MLP output passes through tanh | `FFNOBlock3dBalanced` |
| Loss | composite loss on log₁₀ n, with source-function terms | `loss/nlte_composite_loss.py` |
| Training split | 5 simulations × 1 snapshot, validated on 5 | `config.py` `TRAIN_SPLIT`, `VAL_SPLIT` |
| Production training cost | 4,032 GPU-hours (figure from the paper draft) | Olivia, 8 nodes × 4 GPUs |
| Learning-curve `sims1` (1 snapshot) | 199 epochs in 14.7 h, about 2.2 min per epoch, 473 GPU-hours | Olivia job 2751141 |
| Learning-curve `sims4` (4 snapshots) | about 15 min per epoch, 189 epochs in 48 h, 1,536 GPU-hours | Olivia jobs 2751812, 2809685 |

The learning curve showed the network is short of diverse data: four simulations halve the held-out error from
log τ = −6 down, while three snapshots of one simulation do not. It also showed an error floor of 0.07–0.09 dex in
the upper chromosphere, even on training snapshots. That floor is what the output and capacity changes aim at.

## Phase plan

| Phase | Where | Olivia GPU-hours | Gate to the next phase | Status |
| --- | --- | --- | --- | --- |
| 0 cbh24 data and target checks | eagle7, hyades | 0 | both snapshots pass the checks | done; cbh24 waits for the Betzy rerun |
| 1 Level-fraction head and tests | eagle7 | 0 | all head tests pass | done; departure coefficients added 2026-11-16 |
| 2 Fraction head vs log populations, `sims1` split | Olivia, 1 job | 768 (measured) | unseen error not worse | passed (user, 2026-11-15) |
| 3 Capacity code, memory and time benchmark | eagle7, then 1 Olivia GPU | 0.6 (measured) | every configuration under 100 GB per GPU | passed: at most 88.1 GB |
| 4 Capacity screening, one change at a time | Olivia, 6 + 1 jobs of 100 epochs | about 2,190 | best set within 2× cost | running |
| 5 Final training, 6 + 6 simulations | Olivia | H about 9,600, Ca about 4,000 | early stopping ends it | — |
| 6 Scoring and synthesis against MULTI3D | eagle7, hyades | 0 | — | — |

Phases 0, 1 and 6 use no Olivia time. Phase 3's code can be written while the Phase 2 job runs.

| Phase | Delivers | Main files | Exit criterion |
| --- | --- | --- | --- |
| 0 Data | `cbh24` in `SIMULATIONS` and the splits; links in `bifrost_data/cbh24`; packed data file on Olivia | `config.py`, `packed_data.py` | Checks in *cbh24 data* pass for `s0910` and `s0928` |
| 1 Fraction head | `output_mode = "level_fraction"`: total plus composition, per-atom m, departure coefficients from MULTI3D's LTE | `models/ffno_model.py`, `ffno_model_factory.py`, `FFNONet.py`, `config.py`, `tests/` | The tests in *Level-fraction output* pass |
| 2 A/B | `sims1` retrained with the fraction head, H, scored on eagle7 | `scripts/learning_curve.ini` (new section) | Unseen error compared with the existing `sims1`, per log τ band |
| 3 Capacity code | 3 switches and 3 sizes in `MODEL_CONFIG`; measured memory and step time per configuration | `models/ffno_model.py`, `scripts/capacity_benchmark.py` | Measured costs replace the estimates in *Capacity changes* |
| 4 Screening | One 100-epoch run per change, cumulative | `scripts/learning_curve.ini` | A kept set within the 2× cost cap |
| 5 Final | H, then Ca, on 6 + 6 simulations with the chosen configuration | `config.py`, `train_gpu.sh` | Early stopping, with the best checkpoint kept |
| 6 Evaluation | Scores on `s0957` and the unseen simulations; H-alpha and Ca II 8542 synthesis | `scripts/population_error_diagnostic.py`, `Forward.jl` | Compared with today's production model |

## Level-fraction output

Decided 2026-11-13, after the Phase 0 checks: **total plus composition.** Per species the head predicts one value per
level and one total correction. Since 2026-11-16 the level values are departure coefficients from MULTI3D's LTE
populations:

    log10 n_i = log10(ρ / m) + s + log10 softmax(z)_i,    s = 6 tanh(u / 6),    z_i = ln f*_i(T, nₑ) + δ_i

- **Departure coefficients (2026-11-16).** f*ᵢ is MULTI3D's LTE share of level i, computed in the head from the input
  T and nₑ exactly as MULTI3D's `calc_lte_populations` does (Saha–Boltzmann with Debye lowering, MULTI3D's constants
  and its atom data in `ATOM_CONFIG[atom]["multi3d_lte"]`; Muspel's LTE is never used). Then
  softmax(z)ᵢ = f*ᵢ bᵢ / Σⱼ f*ⱼ bⱼ, so δᵢ is ln bᵢ, the departure coefficient, up to a constant common to the species'
  levels (the total is set by s), and δ = 0 gives MULTI3D's LTE populations. The network outputs the same 7 numbers
  per species as before, with no new parameters or inputs, and the model still returns log₁₀ n.
  - **Why:** at log τ −4..0 MULTI3D's n₂/n₁ is Boltzmann at the cell's T (median +0.001 dex in every simulation), but
    the `sims1` models learned its T slope at only 10–23% of the physical response, which biases unseen snapshots by up
    to −0.52 dex. With f* in the head the steep T dependence is exact, and the network only learns departures.
  - **Exactness:** the formula reproduces MULTI3D's stored `nstar` to 7×10⁻⁸ dex from `atm3d`'s float32 T and nₑ
    (H and Ca, nw012023/915 and en024048_hion/385). In the head, T and nₑ come from the float32 normalised inputs,
    which leaves at most 1.3×10⁻⁵ dex (accepted by the user). MULTI3D's Debye loop zeroes `ndxi(ion₁)` part-way, so
    Ca III is lowered by 3 dxi, not 2; the head does the same. Checks: `ffnoml_runs/multi3d_lte_check/`.
  - **Always on:** `output_mode = "level_fraction"` has no switch for it. Level-fraction checkpoints from before
    (commit `066d327` and earlier, the `sims1_lf` run) load only with that code.
- **Composition:** softmax(z) is each level's share of MULTI3D's own total. MULTI3D's 6 levels always sum to its
  `totn` (to 1.000000 in every labelled snapshot), so this is a true split of the total, with no remainder.
- **Total:** ρ / m is the reference, with m = 2.2697×10⁻²⁴ g per H atom and 1.0375×10⁻¹⁸ g per Ca atom. s is the
  log₁₀ ratio of MULTI3D's total to it. It starts at 0, and the network learns the rest: the `_hion` offset
  (−0.021 dex), the deficit below log T = 3.7 (down to −0.5 dex, probably H₂), and the readnh = 0 artefacts.
- **Complete:** 7 outputs carry 6 degrees of freedom, the same as the old head. Every positive set of levels whose
  total lies within 6 dex of ρ / m is reproduced exactly. The measured MULTI3D range over all labelled snapshots is
  −3.16 to +5.40 dex for H and exactly 0 for Ca, so a perfect network reproduces every target, edge cases included.
- **Finite:** the 6 dex bound keeps populations about 10⁹ below float32 overflow.
- **Not a hard conservation law:** the head can predict an excess wherever MULTI3D has one. A remainder-slot
  version that capped the levels at ρ / m was rejected, because it could not reproduce those cells.

**Design choices**

- **The output stays log₁₀ n, standardised as now.** The model builds the levels, then applies `mean_Y`/`std_Y`
  itself. The composite loss, the FSDP service, the Julia VJP/JVP paths and the scoring scripts stay unchanged.
- **The total uses ρ in physical units.** The model un-normalises its ρ channel with the stored `mean_X`/`std_X`,
  which become buffers in the checkpoint. Gradients with respect to ρ then include the total, which the inversion
  needs.
- **Old checkpoints still load.** `MODEL_CONFIG["output_mode"]` is `"log_population"` by default (today's head) and
  `"level_fraction"` with `FFNOML_OUTPUT_MODE=level_fraction`.
- **m is a fixed constant per species,** in `ATOM_CONFIG`, not fitted.

**Tests before any Olivia run**

- [x] every target with totals from −3.16 to +5.40 dex and level ratios over 30 dex is reproduced to 1e-9
  (`tests/test_level_fraction_head.py`)
- [x] the total starts at ρ / m; populations stay finite at the bound
- [x] gradcheck of the head in float64; through the model, d log total / dρ is exact
- [x] the batched backend equals `forward`; the default `log_population` model and its state dict are unchanged
- [x] FSDP predict, VJP and JVP through `DistributedFSDPBackend` on 2 eagle7 GPUs, on a 40 × 40 patch of
  nw012023/940: populations finite and positive; ⟨c, J t⟩ and ⟨Jᵀc, t⟩ agree to 4.7×10⁻⁷
- [x] `pipeline.py --train` on eagle7 with a small model (width 16, 2 blocks) on the `sims1` datasets: 2 epochs,
  162 steps, train loss 1.44 → 0.27, validation 0.45 → 0.144, no NaN, checkpoint saved and reloaded
- [x] departure coefficients (`tests/test_departure_coefficients.py`): δ = 0 gives a line-by-line transcription of
  `calc_lte_populations` for H and Ca over 2,000–60,000 K and nₑ 10¹⁴–10²³ m⁻³; Ca III's ×3 Debye lowering;
  gradcheck; d log₁₀(n₂/n₁)/d log₁₀ T is Boltzmann's; the batched backend equals `forward`; the head refuses to build
  without the atom data

**Finding from the smoke run** (`ffnoml_runs/level_fraction_smoke`): after 2 epochs the predicted total sits
0.49 dex below MULTI3D's (median; true s is 0 there). The composition is still poor (level 1 −1.18 dex, minor
levels +0.3 to +0.8 dex), and lowering s offsets the overshooting minor levels in the log-space loss: the level
error is 0.556 dex with s, against 0.647 dex for the composition alone. The loss on log populations does not pin the
total by itself, so s can absorb composition error.

Decided (user approved): a loss term on each species' total, λ |log₁₀ Σ n_pred − log₁₀ Σ n_true| / mean std_Y,
against MULTI3D's own total (`FFNOML_LAM_TOTAL`, off by default; logged as `Ltot`). Same smoke setup, 2 epochs,
compared with MULTI3D on a 40 × 40 patch of nw012023/940:

| λ | val L1 (levels) | predicted − true total, median (p5, p95) | median abs level error |
|---|---|---|---|
| 0 (first run) | 0.144 | −0.49 dex | 0.556 dex |
| 0 (repeat) | 0.201 | −0.58 (−2.89, +0.03) dex | 0.713 dex |
| 0.3 | 0.160 | +0.006 (−0.11, +0.11) dex | 0.620 dex |
| 1 | 0.396 | −0.013 (−0.26, +0.11) dex | 0.949 dex |

Without the term the total drifts about 0.5 dex low in both runs. λ = 0.3 fixes the total and keeps the level loss
within the run-to-run spread of λ = 0 (0.144–0.201); λ = 1 also fixes it but costs the levels. Used from Phase 2
on: λ = 0.3.

**Phase 2 result** (`sims1_lf`, Olivia job 2859141, plain softmax head, 325 epochs to the 24 h limit, best validation
0.01058 at epoch 320, 768 GPU-hours). Unseen median error, `sims1` → `sims1_lf`: lower level 0.186 → 0.155 dex, level
ratio 0.129 → 0.093 dex, but at log τ −2..0 the lower level 0.123 → 0.260 dex. That band's error is the n₂/n₁ slope
in T described above (scripts in `ffnoml_runs/learning_curve/sims1_lf/check_tau_m2_0/`); it led to the departure
coefficients. The user passed the gate.

## cbh24 data

cbh24 enters as `s0910` for training and `s0928` for validation, both from the half-resolution (252 × 252) runs, the
same grid as the `s0957` test copy. They are the two usable H runs farthest apart in time and farthest from `s0957`.

| Snapshot | Use | atm3d | mesh | H output | MULTI3D settings |
| --- | --- | --- | --- | --- | --- |
| `s0910_half` | train | yes | yes, same as `s0957`'s | `halpha/s0910_half/output` | start_acc 2500, maxiter 465, nng 0, readnh 0 |
| `s0928_half` | validation | yes | yes, same as `s0957`'s | `halpha/s0928_half/output` | start_acc 2500, maxiter 465, nng 0, readnh 0 |
| `s0957_half` | test, never trained on | yes | `mesh.dat` | already in `bifrost_data/cbh24/s0957` | start_acc 2500, maxiter 600, nng 0 |

Sources: atmospheres and meshes in `/mn/stornext/u3/tiago/data/cbh24`, H output in
`/mn/stornext/u3/tiago/data/rhout/cbh24/halpha`.

Not used:

- **`s0916_half`:** its atmosphere exists only as `atm3d.cbh24_s0916_half.wrong`, and its run used other settings
  (start_acc 250, nng 2).
- **`s0955`–`s0959`:** each is within 2 snapshots of the test snapshot, so validating on one would leak the test
  into model selection.
- **`s0912`, `s0914`:** maxiter 10 and 5, so likely restarts of a neighbour's run.

**Config entries.** The snapshot names leave out `_half`, as the existing `s0957` does. The prediction code splits
`NAME` at its last underscore, so an underscore in the snapshot name would break it.

```
"cbh24": {
    "base_path": DATA_DIR + "bifrost_data/cbh24",
    "snaps": ["s0910", "s0928", "s0957"],
},
TRAIN_SPLIT["cbh24"] = ["s0910"]
VAL_SPLIT["cbh24"] = ["s0928"]
```

In `bifrost_data/cbh24/s0910/` and `s0928/`: `atm3d` and `mesh` link to the files in
`/mn/stornext/u3/tiago/data/cbh24`, and `H/` links to the run's `output/`. That is the layout `s0957` already has.

**Checks before a snapshot goes in**

- [x] `out_atm` equals `atm3d` (velocities differ only by the unit factor 10⁵)
- [x] Σ H levels against ρ / m: within −0.48 and +0.024 dex, like the training simulations
- [x] Convergence proxy: populations at matched (T, nₑ, ρ) against the training simulations. **Failed:** level 1 is
  0.73–0.81 dex low. cbh24 stays out of training until the Betzy rerun of s0957 H shows whether this is
  non-convergence (see *Phase 0 results*)
- [ ] Same y orientation as the other snapshots, checked against the loader's flip
- [ ] A packed data file for Olivia, built on eagle7 and copied to `/cluster/work/projects/nn2834k/harshm/input`

**Ca.** The cbh24 directory holds MULTI3D output only for H-alpha and Mg II. Ca II runs for `s0910` and `s0928` are
being made on Betzy (from the multi3d session) and copied to `bifrost_data/cbh24/<snap>/CA`.

## Capacity changes

Each change is added on top of the ones kept before it, trained for a fixed 100 epochs on the `sims1` split, and kept
only if it lowers held-out error. The final model takes the best set whose cost per epoch is at most about 2×
today's.

| Step | Change | Code | Cost per epoch vs today (estimate) | Measured (Phase 3): time, memory per GPU | Run |
| --- | --- | --- | --- | --- | --- |
| 0 | Level-fraction head with departure coefficients (reference) | Phase 1 | 1.0× | 1.00×, 47.0 GB | `cap0_base` |
| 1 | z-spectral branch: padded FFT along depth, per-mode weights for the lowest 64 modes | `SpectralConvZ` in each vertical branch, gated (`z_spectral_modes`) | 1.15× | 1.20×, 53.4 GB | `cap1_zspec` |
| 2 | Per-frequency low-rank channel mixing: U(k): C×r and V(k): r×C, r = 8 | `SpectralConv2dFull` (`freq_mix_rank`) | 1.2× | 1.10×, 53.4 GB | — |
| 3 | Side paths: `res_pw`, `res_mlp` start at 0.5, no tanh on the MLP output | `FFNOBlock3dBalanced` (`open_side_paths`) | 1.2× | 1.15×, 53.4 GB | — |
| 4 | `width` 48 → 96 | `MODEL_CONFIG` | 1.9× | 1.16×, 62.2 GB | — |
| 5 | `n_layers` 4 → 6 (user, 2026-11-16; 8 blocks measured 2.25×, over the cap) | `MODEL_CONFIG` | 3.9× (8 blocks) | about 1.7× (estimate), 63.6 GB at 8 | — |
| 6 | `hidden_vertical` 128 → 192 | `MODEL_CONFIG` | 7× | 2.16× (with 8 blocks), 88.1 GB | — |

Costs are cumulative. The measured column is one training step on a 40 × 40 × 496 patch on one GH200
(`scripts/capacity_benchmark.py`, Olivia job 2945345; timing noise about ±10%). The GH200 has about 96 GB usable, not
120. The sections for steps 2–6 go into `scripts/learning_curve.ini` once the step before is scored, since what was
kept decides their contents.

**Rules**

- **New code is off by default.** Each capacity change is a `MODEL_CONFIG` switch (`FFNOML_MODEL_CONFIG` in a run),
  so old checkpoints load and give the same outputs. Tests check that with every switch off.
- **Measure before any long run.** Done in Phase 3 (table above).
- **A fair reference.** The cosine learning-rate schedule decays over the run's own epochs, so every step, the
  reference included, trains exactly 100 epochs; `cap0_base` is that reference.
- **Keep a change only if it helps held-out data** (user, 2026-11-15): the unseen median |Δ log₁₀ n| falls by
  ≥ 0.01 dex at log τ < −4, and no other log τ bin gets worse by more than 0.02 dex. A change that lowers only the
  training error points to a data limit, not a capacity one.
- **Read train against validation at each step.** If the gap grows while training error falls, the model is limited
  by data, and later capacity steps can be skipped.
- **If memory runs short,** checkpoint each residual sub-block inside the vertical stack. The current `chunk=4` loop
  doesn't reduce training memory.
- **Not tried:** width 128 with `hidden_vertical` 256 (about 100–110 GB per GPU, too close to 120 GB).

**Why `sims1` and not more data.** One training snapshot is the cheapest: about 237 GPU-hours per 100 epochs at
today's size (measured: 473 GPU-hours for 199 epochs, 768 for 325). The cost is that a single snapshot exposes data
limits sooner than capacity gains. A change that is borderline there is re-tested in the final training, not dropped.
The user chose to screen on `sims1` anyway, assuming more data fixes data limits (2026-11-15).

## Olivia compute budget

| Phase | GPU-hours | Running total |
| --- | --- | --- |
| 2 Fraction A/B | 768 (measured) | 768 |
| 3 Benchmark | 0.6 (measured) | 769 |
| 4 Screening, 6 steps + `cap0_base` | about 2,190 | 2,959 |
| 5 Final H | about 9,600 | 12,559 |
| 5 Final Ca | about 4,000 | 16,559 |

On 2026-11-15 `cost -p nn2834k` showed 16,353.86 GPU-hours left of 20,000 (after Phase 2). Phase 4 onward needs
about 15,790, leaving about 560. The final H run is cheaper if the screening keeps fewer changes.

**How each figure is estimated**

- **Phase 2:** measured, 768 GPU-hours (325 epochs before the 24 h limit).
- **Phase 3:** measured, 35 minutes on one GPU.
- **Phase 4:** 7 runs of 100 epochs on `sims1` (the reference and 6 steps), at about 237 GPU-hours per 100 epochs at
  today's size, times each step's measured multiplier (6 blocks estimated).
- **Phase 5, H:** production cost 4,032 GPU-hours for 5 + 5 snapshots; 6 + 6 gives about 4,800; the 2× capacity cap
  gives about 9,600.
- **Phase 5, Ca:** today's capacity with the fraction head: about 4,000.
- **Quota:** 19,899 GPU-hours were left before the learning-curve jobs, which used up to 3,545, so at most about
  16,300 remain. `cost -p nn2834k` gives the real figure.

Every run that hits the 24 h limit is resumed with `--resume` (the `sims4` route). That adds start-up time but no
repeated epochs, so no margin is added. A slower model than estimated is the main way this goes over.

## Risks and open decisions

**Risks**

- **The budget is tight.** The plan uses roughly all the GPU-hours left. The full capacity combination (about 7×)
  would cost about 34,000 GPU-hours for the H final alone, so the 2× cap is a hard limit, not a preference.
- **cbh24 targets may not be converged.** Every cbh24 run has Ng acceleration off, and the logs for `s0910` and
  `s0928` are missing. The log of `s0959`, run with the same settings, still shows maximum population changes of
  about 5% per iteration near iteration 100. If the Phase 0 checks flag these snapshots, the sixth simulation needs a
  replacement, or new MULTI3D runs.
- **The fraction head can conflict with a target.** Wherever MULTI3D's total doesn't follow ρ, as in en024031, the
  head cannot fit it. That is correct behaviour, but it raises training loss on such cells. The Phase 0 check catches
  them before training.
- **Single-snapshot screening may undersell capacity.** Mitigated by re-testing borderline changes in the final run.
- **Cost estimates may be wrong.** The plan's `sims1` cost was 2× too low (117 against a measured 237 GPU-hours per
  100 epochs); Phase 4 is sized with the measured figure.
- **Memory headroom is small.** The largest configuration measured 88.1 GB against about 96 GB usable.

**Decisions made**

- [x] Phase 2 gate: passed (2026-11-15).
- [x] Keep rule: ≥ 0.01 dex lower unseen median at log τ < −4, and no other log τ bin worse by > 0.02 dex (2026-11-15).
- [x] Ca: no Ca work until an H model has passed every phase; then one Ca training run. Ca is expected to do better
  than H, since its source function has more of B and H's is almost all J (2026-11-15).
- [x] Balance: 16,353.86 GPU-hours left (`cost -p nn2834k`, 2026-11-15).
- [x] Screening stays on `sims1`, assuming more data fixes the data limits (2026-11-15).
- [x] Step 5 is 6 blocks, not 8 (2026-11-16).
- [x] A 100-epoch `cap0_base` reference, and cumulative steps run one after another (2026-11-16).
- [x] Departure coefficients from MULTI3D's exact LTE in the level-fraction head, always on; Muspel's LTE never
  (2026-11-16); its 1.3×10⁻⁵ dex float32 residual is accepted.

## Phase 0 results (2026-11-13)

Scripts and logs: `/mn/stornext/d9/data/harshm/ffnoml_runs/cbh24_s0910_s0928_checks/`.

- **Links:** `bifrost_data/cbh24/s0910` and `s0928` hold `atm3d`, `mesh` and `H/` as links to the source runs.
- **out_atm against atm3d:** nₑ, T and ρ are identical; the velocities differ only by the unit factor 10⁵ (cm/s
  against km/s).
- **Ca total:** in all 5 training snapshots and cbh24 s0957, the 6 levels hold all of the Ca total (`totn`), and
  ρ / total Ca is 1.0375×10⁻¹⁸ g in every cell. Ca's total correction is exactly 0.
- **H total is not one constant of ρ:**
  - ρ / Σ levels is 2.2697×10⁻²⁴ g in nw012023, ch012006, qs006003 and cbh24, but 2.3805 and 2.3839×10⁻²⁴ g in
    en024048_hion and ch012012_hion.
  - In 10–21% of cells (nw012023, ch012006, cbh24) the levels hold less H than ρ / 2.2697×10⁻²⁴ g, by up to a factor
    of about 2.6.
  - qs006003/689 (training, readnh = 0) has the opposite: 3.6% of cells above it by more than 0.1 dex, up to 1.93 dex.
  - So the head predicts the total as well as the split (see *Level-fraction output*).
- **cbh24 levels at matched (T, nₑ, ρ):** median offset from the training snapshots, levels 1–6, in dex. The training
  snapshots agree with each other to within 0.03.

| Snapshot | 1 | 2 | 3 | 4 | 5 | 6 | Cells covered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cbh24 s0910 | −0.73 | +0.15 | +0.32 | +0.24 | +0.20 | +0.08 | 96.0% |
| cbh24 s0928 | −0.81 | +0.15 | +0.35 | +0.26 | +0.21 | +0.09 | 95.7% |
| cbh24 s0957 (test) | −1.02 | +0.17 | +0.47 | +0.37 | +0.32 | +0.11 | 95.1% |

  The two new cbh24 snapshots have the same signature as the test snapshot, only smaller. Either cbh24's MULTI3D
  runs are unconverged (Ng off, 465 iterations), or cbh24 is non-local in a way the training data never shows. The
  Betzy rerun of s0957 H (from the old output, to 900 iterations) can tell the two apart: if its populations moved
  towards the training values, the old cbh24 runs are unconverged.
