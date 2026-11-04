#!/bin/bash
# The two FALC column-mass runs on eagle7.
R=/mn/stornext/d9/data/harshm/FFNOML/response_run; cd $R/falc
source ~/loadnvidia.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=2,3,4,5
~/nvidiaenv/bin/python build_falc_logm.py || exit 1
for v in logm logmne; do
  FALC_CONFIG=falc/forward_falc_$v.toml bash ./run_falc_eagle7.sh > falc_${v}_run.log 2>&1; echo "$v exit $?"
  tail -1 falc_${v}_run.log | cut -c1-200
done
~/nvidiaenv/bin/python compare_logm.py
