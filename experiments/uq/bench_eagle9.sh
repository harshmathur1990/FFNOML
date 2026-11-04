#!/bin/bash
# One-epoch fine-tuning benchmark on eagle9 (2 x MI100). Log: uq/retrain/H_bench/train_eagle9.log
source ~/loadamd.sh >/dev/null 2>&1
cd /mn/stornext/d9/data/harshm/FFNOML
export HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_P2P_DISABLE=1  # RCCL on eagle9 MI100s needs P2P off
export FFNOML_ACTIVE_ATOMS=H FFNOML_NUM_EPOCHS=1 FFNOML_RUN_DIR=$PWD/uq/retrain/H_bench PYTORCH_ALLOC_CONF=expandable_segments:True
cp uq/retrain/H_A_base/training_FFNO3D_zscale_expand_lognlte/3D_sim_train_H.pt uq/retrain/H_bench/training_FFNO3D_zscale_expand_lognlte/
rm -f uq/retrain/H_bench/training_FFNO3D_zscale_expand_lognlte/*last*
date +%T
~/amdenv/bin/torchrun --standalone --nproc_per_node=2 pipeline.py --train --resume --bestpath
date +%T
