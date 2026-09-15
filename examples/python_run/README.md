# Python run directory

Copy this directory to hot storage, symlink the configured train and validation
HDF5 files into `IO`, then submit the repository's batch script from here.

```bash
repo=/path/to/permanent/FFNOML
run=/path/to/hot/ffnoml_run1

mkdir -p "$run"
cp -R "$repo/examples/python_run/." "$run/"
ln -s /permanent/data/3D_sim_train_NAME.hdf5 "$run/IO/3D_sim_train_NAME.hdf5"
ln -s /permanent/data/3D_sim_test_NAME.hdf5 "$run/IO/3D_sim_test_NAME.hdf5"
cd "$run"
sbatch "$repo/train_gpu.sh"
```

The link names must exactly match `TRAIN_FILE` and `TEST_FILE` printed from
`config.py`. For prediction, place or link the checkpoint and any prebuilt
`3D_sim_predict_<NAME>.hdf5` solving set in the training-output directory.
