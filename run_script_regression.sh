bash "$repo/FFNOInversion.jl/scripts/bootstrap_olivia_regression.sh" \
  --run-dir /cluster/work/projects/nn2834k/harshm/ffnoml_runtime_tests \
  --model-assets /cluster/projects/nn2834k/harshm/train_outputs/training_FFNO3D_zscale_expand_lognlte \
  --atmosphere-dir /cluster/projects/nn2834k/harshm/bifrost_data/en024048_hion/385 \
  --atom-dir /cluster/projects/nn2834k/harshm/multi3d/input/atoms \
  --muspel-dir /cluster/projects/nn2834k/harshm/julia-sources/Muspel.jl \
  --julia-depot /cluster/home/harshm/julia-depot-1.12.2