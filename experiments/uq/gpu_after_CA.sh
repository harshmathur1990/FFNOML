#!/bin/bash
# GPU work queued behind the CA queue: column-crop context experiments, extra large-tile validation,
# and a one-epoch training benchmark. Log: uq/mc/gpu_after_CA.log
source ~/loadnvidia.sh >/dev/null 2>&1
cd /mn/stornext/d9/data/harshm/FFNOML
while ! grep -q "CA all done" uq/mc/queue_CA.log; do sleep 60; done
export PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1,2,3,4,5
PY=~/nvidiaenv/bin/python3.12; TR=~/nvidiaenv/bin/torchrun; B=/mn/stornext/d9/data/harshm/bifrost_data
mkdir -p uq/context uq/tilecheck
pred() {  # name cs out extra...
  local name=$1 cs=$2 out=$3; shift 3
  [ -f $out ] && return
  echo "$(date +%T) predict $out"
  $TR --standalone --nproc_per_node=5 scripts/uq_predict.py --atom H --name $name --samples 0 \
    --atmos $B/$cs/atm3d --mesh $B/$cs/mesh --out $out "$@" > ${out%.h5}.log 2>&1 || echo "FAILED $out"
}
for cs in ch024031_by200bz005/450 en024031_by100_helium/109; do
  name=${cs/\//_}
  pred $name $cs uq/context/${name}_H_zge-0.97.h5 --tile 256 --z-range -0.97 20
  pred $name $cs uq/context/${name}_H_zle7.98.h5 --tile 256 --z-range -5 7.98
done
pred nw012023_915 nw012023/915 uq/tilecheck/nw012023_915_H_tile192.h5 --tile 192
pred ch012012_hion_910 ch012012_hion/910 uq/tilecheck/ch012012_hion_910_H_tile192.h5 --tile 192
echo "$(date +%T) benchmark training epoch"
FFNOML_ACTIVE_ATOMS=H CUDA_VISIBLE_DEVICES= $PY scripts/uq_retrain_setup.py --branch bench --notes "1-epoch benchmark" > /dev/null
FFNOML_ACTIVE_ATOMS=H FFNOML_NUM_EPOCHS=1 FFNOML_RUN_DIR=$PWD/uq/retrain/H_bench \
  $TR --standalone --nproc_per_node=5 pipeline.py --train --resume --bestpath > uq/retrain/H_bench/train.log 2>&1 \
  || echo "FAILED benchmark"
echo "$(date +%T) gpu_after_CA done"
