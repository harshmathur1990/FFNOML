#!/bin/bash
# FALC given (nₑ, H populations) on the three axes, on eagle7.
R=/mn/stornext/d9/data/harshm/FFNOML/response_run; cd $R/falc
source ~/loadnvidia.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=4,5 FFNO_FSDP_NPROC=2
~/nvidiaenv/bin/python build_falc_logm.py || exit 1
cd $R/falc
for v in m tau z; do
  FALC_CONFIG=falc/forward_falc_given$v.toml bash ./run_falc_eagle7.sh > falc_given${v}_run.log 2>&1; echo "$v exit $?"
done
