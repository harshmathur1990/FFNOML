# FFNO3D UQ: status brief (2026-10-30)

For Harsh and Codex. Branch `uq-calibration`; the full write-up is in `docs/ffno3d-uq-report.md`.
This covers Codex's `docs/claude-next-steps-uq.txt`, and Codex is asked to suggest next steps (bottom).

## Where we stand

| Step | H | CA |
|---|---|---|
| 1 Validate analysis (Spearman, self-exclusion, regimes, alignment) | done | done |
| 2 Missing regimes table | done | done |
| 3 ch024031 / helium offsets | done | done |
| 4–5 Controlled retraining A/B/C + spectra | running (eagle7, ETA ~13:00 today) | training on eagle9 (ETA tomorrow), then eval |
| 6 Tiling check | done | n/a |
| 7 H vs CA comparison | partial | partial |
| 8 Decide further UQ work | waits on 4–5 | waits on 4–5 |

## Five things to know

1. **Data and predictions are aligned.** All 47 snapshot/species pairs pass orientation, grid, unit, total-density and level-order checks.
2. **The model depends on column height.** ch024031 and helium are 16.8 Mm tall, taller than anything in training. Cropping the column flips the sign of their deep offsets, and even the training snapshot nw915 shifts 0.1–0.3 dex under a crop. The likely cause is that z_span and normalized z are input features (`models/ffno_model.py:194`). More labels on that grid won't fix this.
3. **nw012023 1050/1120 drift over time.** Their offset survives every crop. This is the case retraining should help.
4. **Failure regimes:**
   - H, hot corona above log T 6.05: no training coverage.
   - Both species, cool rarefied chromospheric gas: covered, but held-out error is 3–4× training error.
   - CA, ch012012_759: Ca II is 1.3 dex too low at 2–3 Mm, yet this costs only 10% in the 8542 core. Spectra alone would miss it.
5. **Uncertainty indicators:**
   - MC dropout: spread is 6–25× too small; useless for H.
   - Support distance with self-matches removed: weak for H (case ρ 0.40), good for CA (0.71).
   - Conservation residual |log Σn_pred − log ntot|: free and label-free, and the best H indicator (case ρ 0.55; its top 5 are 5 of the 6 worst). It can't see errors that conserve the total, such as the Ca ionization shift.

## Retraining experiment (running)

- **Branches** from the same production checkpoint: A = existing data, B = + nw012023/1050, C = + random qs006003_sap/1297 (seed 20261029). nw1120 stays held out.
- **Caveat:** only the output head (proj1/proj2, 5k parameters) is trained. A full fine-tune runs out of memory on T4 and MI100. This is a weak test of "does one snapshot help". The full version is `uq/olivia_finetune.sbatch`, ready but not submitted.
- **Evaluation:** 20 snapshots, per-level MAE/p95/bias (`uq/retrain/<atom>_<branch>/cross_snapshot.csv`), plus H-alpha / Ca 8542 spectra (`uq/spectra_run.sh`).
- **H so far:** A has finished and been evaluated; B is training.

## Open decisions (Harsh)

- Submit the Olivia full fine-tune? It is the only way to test retraining properly.
- A new MULTI3D snapshot is needed eventually: every labelled case has now been inspected, so there is no clean test set.

## Codex: please suggest next steps on

1. **Column-height dependence.** Options: train with random vertical crops, drop z_span/z_norm features, or restrict inference to trained extents. Which, and how to test it without new MULTI3D runs?
2. **If head-only B shows little gain over A**, is that enough to decide, or must we wait for the Olivia full fine-tune?
3. **The conservation residual as an acquisition signal.** Should it be combined with support distance (e.g. a rank sum), given that it is blind to ionization shifts within the total?
4. **Is an ensemble worth it** (step 8), given MC dropout's failure? Proposed cheap version: 3 head-only members with different seeds, compared on case-ranking ρ.
5. **Which snapshot to label next in MULTI3D**, for a clean held-out test: a new time in nw012023, or a new simulation?
