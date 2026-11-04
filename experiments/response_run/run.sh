#!/bin/bash
# usage: run.sh OUTPUT.h5 [column_response.jl options]; runs on the current hyades node with all its cores.
# CONFIG and ATMOSPHERE override the forward config and atmosphere (e.g. forward_multi3d.toml, inputs/multi3d_atmosphere.h5).
source ~/loadgcc.sh >/dev/null 2>&1
export PATH=/tmp/julia-1.12.3/bin:$PATH
P=/mn/stornext/d9/data/harshm/FFNOML/FFNOInversion.jl
R=/mn/stornext/d9/data/harshm/FFNOML/response_run
export FFNOML_RUN_DIR=$R
cd $R
out=$1; shift
julia --startup-file=no --project=$P --threads=$(nproc) $P/scripts/column_response.jl \
  ${CONFIG:-$R/forward.toml} $R/paths.jl ${ATMOSPHERE:-/mn/stornext/d9/data/harshm/FFNOML/outputs/forward_atmosphere.h5} "$out" "$@"
