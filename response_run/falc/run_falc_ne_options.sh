#!/bin/bash
# falc_ne_options.jl on eagle7.
source ~/loadnvidia.sh >/dev/null 2>&1
export PATH=/tmp/julia-1.12.3/bin:$PATH
P=/mn/stornext/d9/data/harshm/FFNOML/FFNOInversion.jl
R=/mn/stornext/d9/data/harshm/FFNOML/response_run
export FFNOML_RUN_DIR=$R
export FFNO_FSDP_LAUNCHER=$P/scripts/ffno_fsdp_service_local.sh
export FFNO_FSDP_PYTHON=${FFNO_FSDP_PYTHON:-$HOME/nvidiaenv/bin/python}
export FFNO_GPU_CONTROL_HOST=127.0.0.1
export FFNO_GPU_DIAGNOSTICS_DIR=$R/diagnostics
NT=$(nproc --all)
export OMP_NUM_THREADS=1
cd $R
julia --startup-file=no --project=$P --threads=$NT falc/falc_ne_options.jl
