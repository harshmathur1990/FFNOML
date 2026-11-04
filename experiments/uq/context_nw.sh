#!/bin/bash
# Column-crop context test on nw012023 (training 915, held-out 1120). Log: uq/mc/context_nw.log
source ~/loadnvidia.sh >/dev/null 2>&1
cd /mn/stornext/d9/data/harshm/FFNOML
export PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1,2,3,4,5
B=/mn/stornext/d9/data/harshm/bifrost_data
for snap in 915 1120; do
  for zr in "-0.97 7.98" "0.0 20" "-0.97 4.0"; do
    set -- $zr; out=uq/context/nw012023_${snap}_H_z${1}_${2}.h5
    [ -f $out ] || ~/nvidiaenv/bin/torchrun --standalone --nproc_per_node=5 scripts/uq_predict.py --atom H \
      --name nw012023_$snap --samples 0 --atmos $B/nw012023/$snap/atm3d --mesh $B/nw012023/$snap/mesh \
      --z-range $1 $2 --out $out > ${out%.h5}.log 2>&1 || echo "FAILED $out"
    echo "$(date +%T) $out"
  done
done
echo "context_nw done"
