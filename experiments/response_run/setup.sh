#!/bin/bash
# Build the Wittmann library for this node type and instantiate the FFNOInversion environment (hyades, Julia 1.12.3).
source ~/loadgcc.sh >/dev/null 2>&1
export PATH=/tmp/julia-1.12.3/bin:$PATH CXX=g++ JULIA_PKG_USE_CLI_GIT=true
P=/mn/stornext/d9/data/harshm/FFNOML/FFNOInversion.jl
R=/mn/stornext/d9/data/harshm/FFNOML/response_run
julia --startup-file=no $P/scripts/build_wittmann_backend.jl $R/inputs/wittmann/libwitt_ffno.so
julia --startup-file=no --project=$P -e 'using Pkg; Pkg.instantiate(); using FFNOInversion, Muspel, HDF5; println("environment OK")'
