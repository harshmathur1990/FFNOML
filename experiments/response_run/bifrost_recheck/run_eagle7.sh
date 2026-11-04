#!/bin/bash
# Bifrost recheck on eagle7: falc/model_factory.jl (H and Ca II) with the config given as $1.
source ~/loadnvidia.sh >/dev/null 2>&1
export PATH=/tmp/julia-1.12.3/bin:$PATH
P=/mn/stornext/d9/data/harshm/FFNOML/FFNOInversion.jl; R=/mn/stornext/d9/data/harshm/FFNOML/response_run
export FFNOML_RUN_DIR=$R FFNO_FSDP_LAUNCHER=$P/scripts/ffno_fsdp_service_local.sh FFNO_FSDP_PYTHON=${FFNO_FSDP_PYTHON:-$HOME/nvidiaenv/bin/python}
export FFNO_GPU_CONTROL_HOST=127.0.0.1 FFNO_GPU_DIAGNOSTICS_DIR=$R/diagnostics OMP_NUM_THREADS=1
cd $R
julia --startup-file=no --project=$P --threads=$(nproc --all) -e "using FFNOInversion; factory=include(\"falc/model_factory.jl\"); run_inversion_files!(\"$1\",factory); println(\"DONE\")"
