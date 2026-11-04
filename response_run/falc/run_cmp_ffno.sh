#!/bin/bash
# FFNOInversion forward on the two FALC atmospheres on eagle7, one after the other.
cd /mn/stornext/d9/data/harshm/FFNOML/response_run/falc
export CUDA_VISIBLE_DEVICES=3,4 FFNO_FSDP_NPROC=2
for v in atmos 5x5; do
  FALC_CONFIG=falc/forward_falc_cmp_$v.toml bash ./run_falc_eagle7.sh > falc_cmp_${v}_run.log 2>&1; echo "$v exit $?"
done; echo FFNO_DONE
