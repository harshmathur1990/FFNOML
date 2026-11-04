#!/bin/bash
# MC-dropout predictions (full-domain route, 5 T4s) followed by the error atlas, for each labelled snapshot.
# Usage: bash uq/run_queue.sh ATOM SAMPLES sim/snap [sim/snap ...]
source ~/loadnvidia.sh >/dev/null 2>&1
cd /mn/stornext/d9/data/harshm/FFNOML
PY=~/nvidiaenv/bin/python3.12
B=/mn/stornext/d9/data/harshm/bifrost_data
atom=$1; samples=$2; shift 2
export PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPUS:-1,2,3,4,5}
for cs in "$@"; do
  name=${cs/\//_}
  out=uq/mc/${name}_${atom}.h5
  src=(--atmos $B/$cs/atm3d --mesh $B/$cs/mesh)
  [ -f training_FFNO3D_zscale_expand_lognlte/3D_sim_predict_${name}.hdf5 ] && src=()
  if [ ! -f $out ]; then
    echo "$(date +%T) predict $name $atom"
    ~/nvidiaenv/bin/torchrun --standalone --nproc_per_node=5 scripts/uq_predict.py --atom $atom --name $name \
      --samples $samples --tile ${TILE:-0} "${src[@]}" --out $out > uq/mc/${name}_${atom}.log 2>&1 || { echo "FAILED predict $name $atom"; continue; }
  fi
  echo "$(date +%T) atlas $name $atom"
  $PY scripts/uq_error_atlas.py --case $name --atom $atom --truth $B/$cs/$atom --uq $out \
    --atmos $B/$cs/atm3d --mesh $B/$cs/mesh --out-dir uq/atlas_mc > uq/mc/${name}_${atom}_atlas.log 2>&1 \
    || echo "FAILED atlas $name $atom"
done
echo "$(date +%T) queue done"
