#!/bin/bash
# usage (on eagle7, bash -l): run_tangents_eagle7.sh OUTPUT.h5 [ffno_population_tangents.jl options, e.g. --stride 4]
# FFNO population temperature tangents on this host's GPUs: the FSDP service runs under the local torchrun launcher.
source ~/loadnvidia.sh >/dev/null 2>&1
export PATH=/tmp/julia-1.12.3/bin:$PATH
P=/mn/stornext/d9/data/harshm/FFNOML/FFNOInversion.jl
R=/mn/stornext/d9/data/harshm/FFNOML/response_run
export FFNOML_RUN_DIR=$R
export FFNO_FSDP_LAUNCHER=$P/scripts/ffno_fsdp_service_local.sh
export FFNO_FSDP_PYTHON=${FFNO_FSDP_PYTHON:-$HOME/nvidiaenv/bin/python}
export FFNO_GPU_CONTROL_HOST=127.0.0.1          # the service is on this host; avoid resolving its hostname
export FFNO_GPU_DIAGNOSTICS_DIR=$R/diagnostics
export OMP_NUM_THREADS=1
cd $R
out=$1; shift
julia --startup-file=no --project=$P --threads=$(nproc --all) $P/scripts/ffno_population_tangents.jl \
  $R/forward.toml $R/ffno_models.jl /mn/stornext/d9/data/harshm/FFNOML/outputs/forward_atmosphere.h5 "$out" "$@"
