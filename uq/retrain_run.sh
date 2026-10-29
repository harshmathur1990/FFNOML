#!/bin/bash
# Controlled fine-tuning experiment (claude-next-steps-uq step 4/5). Usage: bash uq/retrain_run.sh ATOM E
# Branch dirs come from scripts/uq_retrain_setup.py. A trains round(1.2 E) epochs (805 patches/epoch),
# B and C train E epochs (966 patches/epoch), so optimizer steps match. Checkpoint selection is the
# pipeline's best validation loss on the fixed VAL_SPLIT; evaluation snapshots never enter training,
# selection or early stopping. Log: uq/retrain/run_<ATOM>.log
source ~/loadnvidia.sh >/dev/null 2>&1
cd /mn/stornext/d9/data/harshm/FFNOML
atom=$1; E=$2
export PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPUS:-1,2,3,4,5} FFNOML_ACTIVE_ATOMS=$atom
PY=~/nvidiaenv/bin/python3.12; TR=~/nvidiaenv/bin/torchrun; B=/mn/stornext/d9/data/harshm/bifrost_data
EVAL="nw012023/1120 nw012023/1050 nw012023/940 nw012023/915 en024048_hion/385 en024048_hion/386 en024048_hion/465
  en024048_hion/700 ch012012_hion/759 ch012012_hion/834 ch012012_hion/910 ch012012_hion/984 ch012006/795 ch012006/820
  ch012006/836 ch012006/849 qs006003_sap/689 qs006003_sap/900 qs006003_sap/1100 qs006003_sap/1297"
for br in A_base B_nw1050 C_random; do
  run=uq/retrain/${atom}_$br
  ep=$E; [ $br = A_base ] && ep=$(( (12 * E + 5) / 10 ))
  if [ ! -f $run/train.done ]; then
    echo "$(date +%T) train $br $ep epochs"
    FFNOML_NUM_EPOCHS=$ep FFNOML_RUN_DIR=$PWD/$run $TR --standalone --nproc_per_node=5 pipeline.py --train --resume --bestpath \
      > $run/train.log 2>&1 && touch $run/train.done || { echo "FAILED train $br"; continue; }
  fi
  ckpt=$run/training_FFNO3D_zscale_expand_lognlte/3D_sim_train_$atom.pt
  mkdir -p $run/pred $run/atlas
  for cs in $EVAL; do
    name=${cs/\//_}; out=$run/pred/${name}_$atom.h5
    if [ ! -f $out ]; then
      $TR --standalone --nproc_per_node=5 scripts/uq_predict.py --atom $atom --name $name --checkpoint $ckpt --samples 0 \
        --atmos $B/$cs/atm3d --mesh $B/$cs/mesh --out $out > ${out%.h5}.log 2>&1 || { echo "FAILED predict $br $name"; continue; }
    fi
    [ -f $run/atlas/${name}_${atom}_summary.csv ] || CUDA_VISIBLE_DEVICES= $PY scripts/uq_error_atlas.py --case $name --atom $atom \
      --truth $B/$cs/$atom --uq $out --atmos $B/$cs/atm3d --mesh $B/$cs/mesh --out-dir $run/atlas --sample 1000000 \
      > $run/atlas/${name}.log 2>&1 &
  done
  wait
  CUDA_VISIBLE_DEVICES= $PY scripts/uq_cross_snapshot.py --atlas $run/atlas/*_${atom}_atlas.h5 --out $run/cross_snapshot.csv
  echo "$(date +%T) evaluated $br"
done
echo "$(date +%T) retrain $atom done"
