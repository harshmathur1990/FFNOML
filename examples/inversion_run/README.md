# FFNOInversion.jl run directory

**Start here:** [Short forward/inversion starter guide](START_HERE.md), including
the folder layout, scientific inputs, Olivia setup, submission, and outputs.

This is the per-run layout for a production inversion. The checked-in files are
a template; the scientific inputs and platform library must be symlinked from
permanent storage.

Required links:

- `inputs/initial_atmosphere.h5`
- `inputs/observations.h5`
- `inputs/kurucz_6302.list` (for the LTE Fe 6302 region)
- `inputs/atoms/atom.h6_tiago2.yaml` and `atom.ca2.yaml`
- `inputs/pf_Kurucz.input`
- `inputs/wittmann/libwitt_ffno.so` on Olivia
- H and Ca checkpoints in `training_FFNO3D_zscale_expand_lognlte`

Prepare and submit:

```bash
repo=/path/to/permanent/FFNOML
run=/path/to/hot/ffnoml_inversion_run1

mkdir -p "$run"
cp -R "$repo/examples/inversion_run/." "$run/"

ln -s /permanent/inputs/initial_atmosphere.h5 "$run/inputs/initial_atmosphere.h5"
ln -s /permanent/inputs/observations.h5 "$run/inputs/observations.h5"
ln -s /permanent/multi3d/input/atoms/atom.h6_tiago2.yaml "$run/inputs/atoms/atom.h6_tiago2.yaml"
ln -s /permanent/multi3d/input/atoms/atom.ca2.yaml "$run/inputs/atoms/atom.ca2.yaml"
ln -s "$repo/scripts/pf_Kurucz.input" "$run/inputs/pf_Kurucz.input"
ln -s /permanent/lib/libwitt_ffno.so "$run/inputs/wittmann/libwitt_ffno.so"
ln -s /permanent/models/3D_sim_train_H.pt "$run/training_FFNO3D_zscale_expand_lognlte/3D_sim_train_H.pt"
ln -s /permanent/models/3D_sim_train_CA.pt "$run/training_FFNO3D_zscale_expand_lognlte/3D_sim_train_CA.pt"
ln -s /permanent/inputs/kurucz_6302.list "$run/inputs/kurucz_6302.list"

cd "$run"
bash "$repo/FFNOInversion.jl/scripts/submit_olivia_inversion.sh"
```

The Olivia batch script derives the Julia thread count and the runtime
`threads_per_rank` value from SLURM's `cpus-per-task`. Do not hard-code
`threads_per_rank` in a copied run configuration; it is only needed for
non-SLURM launches where it defaults to Julia's active thread count.

`model_factory.jl` uses these relative paths through `FFNOML_RUN_DIR`. Set
`FFNO_TOP_DENSITY_KG_M3` to the appropriate boundary density before submission;
the default in the template is only a starting value and must be checked for the
chosen atmosphere.

For atmosphere-to-spectrum synthesis without fitting, use `forward.toml`:

```bash
cd "$run"
export FFNO_INVERSION_CONFIG="$run/forward.toml"
bash "$repo/FFNOInversion.jl/scripts/submit_olivia_inversion.sh"
```

This needs the atmosphere and model/physics assets listed above, but no
`inputs/observations.h5`. It returns `outputs/synthesis.h5` and
`outputs/forward_atmosphere.h5`. Alternatively set `max_iterations = 0` in an
inversion configuration; nodes and observations will be ignored. This is still
force-balance/EOS reconstruction followed by FFNO populations and synthesis,
with no optimization or gradient work. The example factory currently loads both
H and Ca assets even when only one species contributes to the configured lines.
