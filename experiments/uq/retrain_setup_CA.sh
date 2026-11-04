#!/bin/bash
# Branches of the CA fine-tuning experiment (CPU). Log: uq/retrain/setup_CA.log
source ~/loadnvidia.sh >/dev/null 2>&1
cd /mn/stornext/d9/data/harshm/FFNOML
export CUDA_VISIBLE_DEVICES= FFNOML_ACTIVE_ATOMS=CA
PY=~/nvidiaenv/bin/python3.12
$PY scripts/uq_retrain_setup.py --branch A_base --notes "existing training data only; 1.2x epochs to match optimizer steps"
$PY scripts/uq_retrain_setup.py --branch B_nw1050 --add nw012023/1050 --notes "targeted: nw012023 temporal transfer"
$PY scripts/uq_retrain_setup.py --branch C_random --add qs006003_sap/1297 \
  --notes "random control: random.Random(20261029).choice over en024048_hion 385/386, ch012012_hion 759/834, ch012006 795/820, qs006003_sap 1100/1297"
echo "setup done"
