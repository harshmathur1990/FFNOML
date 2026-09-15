# FFNOInversion.jl run directory

This is the per-run layout for a production inversion. The checked-in files are
a template; the scientific inputs and platform library must be symlinked from
permanent storage.

Required links:

- `inputs/initial_atmosphere.h5`
- `inputs/observations.h5`
- `inputs/kurucz_8542.list` and `inputs/kurucz_6302.list`
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
# Add the two Kurucz line-list links in inputs/ as well.

cd "$run"
bash "$repo/FFNOInversion.jl/scripts/submit_olivia_inversion.sh"
```

`model_factory.jl` uses these relative paths through `FFNOML_RUN_DIR`. Set
`FFNO_TOP_DENSITY_KG_M3` to the appropriate boundary density before submission;
the default in the template is only a starting value and must be checked for the
chosen atmosphere.
