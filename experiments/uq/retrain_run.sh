#!/bin/bash
# Controlled fine-tuning experiment (claude-next-steps-uq step 4/5) on eagle7. Usage: bash uq/retrain_run.sh ATOM E
# Branch dirs come from scripts/uq_retrain_setup.py. A trains round(1.2 E) epochs (805 patches/epoch),
# B and C train E epochs (966 patches/epoch), so optimizer steps match. Checkpoint selection is the
# pipeline's best validation loss on the fixed VAL_SPLIT; evaluation snapshots never enter training,
# selection or early stopping. Log: uq/retrain/run_<ATOM>.log
# PHASES=train,eval (default both). On eagle9 (2 x MI100, ROCm) only training fits; evaluate on eagle7.
cd /mn/stornext/d9/data/harshm/FFNOML
atom=$1; E=$2; PHASES=${PHASES:-train,eval}
if [ "$(hostname -s)" = eagle9 ]; then
  source ~/loadamd.sh >/dev/null 2>&1
  PY=~/amdenv/bin/python3.12; TR=~/amdenv/bin/torchrun; NPROC=2
  export HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_P2P_DISABLE=1
else
  source ~/loadnvidia.sh >/dev/null 2>&1
  PY=~/nvidiaenv/bin/python3.12; TR=~/nvidiaenv/bin/torchrun; NPROC=5
  export CUDA_VISIBLE_DEVICES=${GPUS:-1,2,3,4,5}
fi
export PYTORCH_ALLOC_CONF=expandable_segments:True FFNOML_ACTIVE_ATOMS=$atom
# Full fine-tuning does not fit eagle7 (T4) or eagle9 (MI100): one block's backward at 40x40x467 exceeds
# 25 GB. Only the pointwise output head is trained here; FFNOML_TRAIN_ONLY="" would train everything.
export FFNOML_TRAIN_ONLY=${FFNOML_TRAIN_ONLY-proj1,proj2}
B=/mn/stornext/d9/data/harshm/bifrost_data
EVAL="nw012023/1120 nw012023/1050 nw012023/940 nw012023/915 en024048_hion/385 en024048_hion/386 en024048_hion/465
  en024048_hion/700 ch012012_hion/759 ch012012_hion/834 ch012012_hion/910 ch012012_hion/984 ch012006/795 ch012006/820
  ch012006/836 ch012006/849 qs006003_sap/689 qs006003_sap/900 qs006003_sap/1100 qs006003_sap/1297"
for br in A_base B_nw1050 C_random; do
  run=uq/retrain/${atom}_$br
  ep=$E; [ $br = A_base ] && ep=$(( (12 * E + 5) / 10 ))
  if [[ $PHASES == *train* ]] && [ ! -f $run/train.done ]; then
    echo "$(date +%T) train $br $ep epochs on $(hostname -s) x$NPROC"
    FFNOML_NUM_EPOCHS=$ep FFNOML_RUN_DIR=$PWD/$run $TR --standalone --nproc_per_node=$NPROC pipeline.py --train --resume --bestpath \
      > $run/train.log 2>&1 && touch $run/train.done || { echo "FAILED train $br"; exit 1; }
  fi
  [[ $PHASES == *eval* ]] || continue
  [ -f $run/train.done ] || { echo "skip eval $br: not trained"; continue; }
  ckpt=$run/training_FFNO3D_zscale_expand_lognlte/3D_sim_train_$atom.pt
  mkdir -p $run/pred $run/atlas
  for cs in $EVAL; do
    name=${cs/\//_}; out=$run/pred/${name}_$atom.h5
    if [ ! -f $out ]; then
      $TR --standalone --nproc_per_node=$NPROC scripts/uq_predict.py --atom $atom --name $name --checkpoint $ckpt --samples 0 \
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
