#!/bin/bash
# After the H queue: validate 192^2 tiling on en024048_hion_385, then the 384^3 boxes with 256^2 tiles.
cd /mn/stornext/d9/data/harshm/FFNOML
while pgrep -f "run_queue.sh H 0 en024048_hion/465" > /dev/null; do sleep 30; done
source ~/loadnvidia.sh >/dev/null 2>&1
export PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1,2,3,4,5
echo "$(date +%T) tile validation" >> uq/mc/queue_H.log
~/nvidiaenv/bin/torchrun --standalone --nproc_per_node=5 scripts/uq_predict.py --atom H --name en024048_hion_385 \
  --samples 0 --tile 192 --out uq/test_tile192_en024048_hion_385_H.h5 > uq/mc/test_tile192.log 2>&1 \
  || echo "FAILED tile validation" >> uq/mc/queue_H.log
rm -f uq/mc/ch024031_by200bz005_450_H.h5 uq/mc/en024031_by100_helium_109_H.h5
TILE=256 bash uq/run_queue.sh H 0 ch024031_by200bz005/450 en024031_by100_helium/109 ch024031_by200bz005/700 ch024031_by200bz005/950 >> uq/mc/queue_H.log 2>&1
echo "$(date +%T) followup done" >> uq/mc/queue_H.log
