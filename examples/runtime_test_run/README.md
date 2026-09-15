# Olivia runtime-test run directory

Run the complete workflow from an Olivia login node with one command:

```bash
repo=/path/to/permanent/FFNOML
bash "$repo/FFNOInversion.jl/scripts/bootstrap_olivia_regression.sh" \
  --run-dir /path/to/hot/ffnoml_runtime_tests \
  --model-assets /permanent/training_FFNO3D_zscale_expand_lognlte \
  --atmosphere-dir /permanent/bifrost_data/en024048_hion/385 \
  --atom-dir /permanent/multi3d/input/atoms \
  --muspel-dir /permanent/julia-sources/Muspel.jl \
  --julia-depot /permanent/julia-depot-1.12.2
```

The helper validates all required files, creates the run layout and symlinks,
submits Julia package installation/precompilation on a compute node, and then
submits the regression chain with an `afterok` dependency. Slurm logs, test
evidence, archives, and temporary files are created in the run directory.

`Muspel.jl` is declared in `Project.toml` and locked by `Manifest.toml` to
commit `01ec68da389be75c9ce31494a910095d3590499a`. If `--muspel-dir` is absent on
disk, the bootstrap helper clones that exact revision there from the login
node. The compute job then develops Muspel from this local checkout and does
not need GitHub access for Muspel.

The prepared directory has this shape:

```text
ffnoml_runtime_tests/
├── reference/
│   ├── atmosphere -> permanent snapshot directory (mesh and atm3d)
│   └── atoms      -> permanent Multi3D atom directory
├── training_FFNO3D_zscale_expand_lognlte/
│   ├── 3D_sim_train_H.pt -> permanent checkpoint
│   ├── output_..._H.hdf5 -> permanent H populations
│   ├── output_..._CA.hdf5 -> permanent Ca populations
│   ├── intensity_..._H.h5 -> permanent H reference intensity
│   └── intensity_..._CA.h5 -> permanent Ca reference intensity
├── julia-environment/       (run-local Project, Manifest and preferences)
├── tmp/
└── Slurm logs and evidence archives
```
