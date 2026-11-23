# FALC on its own grid: zaug vs l004p against RH

The test behind FNOML-Paper Appendix D (Fig. D.1). The H I and Ca II networks predict the populations of FALC_82 on
its own 82-point height grid, from −0.10 to 2.16 Mm, with `[population.mesh] enabled = false`. The column is repeated
on a 5×5 horizontal grid. H-alpha and Ca II 8542 are synthesised at μ = 1 with FFNOInversion and compared with RH run
on FALC_82 with the same nₑ.

Only scripts and configurations are kept here. The run directories hold the data:

- `ffnoml_runs/falc_zaug_native`: zaug, the files in this directory.
- `ffnoml_runs/falc_l004p_native_rerun`: l004p, the first training stage, with the files in `l004p/`.

The `.toml` files are the same for both models apart from their comments.

## Layout of a run directory

`run_necmp.sh` and `run_ca.sh` run from the run directory `W` and expect these files in `W/falc/`:

- the `forward_falc_*.toml` files;
- `model_factory.jl`, `model_factory_ca.jl` and `ffno_atom_factory.py`;
- the inputs from `FFNOML/response_run/falc`: `falc_logm.h5`, `falc_logm_z.h5`, `falc_input_extended.h5`,
  `kurucz_halpha_blend.list` and `kurucz_8542_blend.list`.

The checkpoints are read from `FFNOMLTrainingWeights/{zaug,l004p}`.

## Runs (eagle7)

- `bash run_necmp.sh`: H-alpha with the H network, for six nₑ cases. The nₑ comes from FALC itself, the LTE EOS or the
  NLTE EOS, each on the log column mass (`m`) and height (`z`) axes.
- `RUNS=falcz PREFIX=ca_ bash run_ca.sh`: Ca II 8542 with FALC's nₑ on the height axis.
  - The H and Ca II networks run together in one batched service. The synthesis needs the FFNO H populations even for
    a Ca II line: their neutral-hydrogen density sets the van der Waals damping.
  - `ffno_atom_factory.py` builds each checkpoint with its own atom's level-fraction tables. Without it, both would use
    the atom set by `FFNOML_ACTIVE_ATOMS`.

## Plots

- `plot_falc_zaug.py`: H-alpha for the six nₑ cases, and the H populations against RH, l004p vs zaug.
- `compare_ca.py`: the Ca II 8542 line core, and the Ca II populations against RH at selected heights, l004p vs zaug.
- `paper_fig_falc.py`: the paper figure, `FNOML-Paper/images/falc_halpha_ca8542_zaug.pdf`.

## Result (FALC's nₑ, height axis)

| Line | zaug line core vs RH | l004p line core vs RH |
|---|---|---|
| H-alpha | +12.7% | 4.5 times as bright |
| Ca II 8542 | +38% | 2.3 times as bright |

With l004p, the total Ca II population is about 50 times too low at 1.1–1.8 Mm. With zaug, the total Ca II population
is within about 30% below 1.8 Mm. Its 8542 excess comes from n(4p ²P₃/₂)/n(3d ²D₅/₂), which is 1.1–1.5 times RH's at
1.0–1.8 Mm.

Relative to RH's far-wing intensity (at +419 km/s for H-alpha, −284 km/s for Ca II 8542; `wing_numbers.py`), the
zaug line cores are 2.0% (H-alpha) and 7.5% (Ca II 8542) of it brighter than RH's, and l004p's 56% and 26%: the numbers
in FNOML-Paper Appendix D.

## With the population mesh (not in the paper)

The same two falcz runs with `[population.mesh] enabled = true`, using the default mesh. It runs from −1 to 14 Mm, with
20 km spacing up to 4.5 Mm that then stretches to 100 km. The network predicts on the mesh, and the populations are
mapped back to FALC's 82 heights for the synthesis.

- Configs: `forward_falc_mesh_necmp_falcz.toml` and `forward_falc_mesh_ca_falcz.toml`.
- Runs: `RUNS=falcz PREFIX=mesh_necmp_ bash run_necmp.sh` and `RUNS=falcz PREFIX=mesh_ca_ bash run_ca.sh`.
- Comparison: `compare_mesh.py` writes `falc_mesh_vs_native.png`.

| Line | Run | Line core vs RH | rms (max) vs RH |
|---|---|---|---|
| H-alpha (±50 km/s) | own grid | +12.7% | 5.8% (12.9%) |
| H-alpha (±50 km/s) | population mesh | −2.6% | 5.8% (13.3%, inner wings) |
| Ca II 8542 (±25 km/s) | own grid | +38.0% | 17.1% (37.8%) |
| Ca II 8542 (±25 km/s) | population mesh | +15.5% | 7.4% (15.5%) |
