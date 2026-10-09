#!/bin/bash
# Accelerator-node runtime settings. This file is sourced by the setup and
# production batch jobs only, never by the login-node submission helper.

: "${FFNOML_RUN_DIR:?FFNOML_RUN_DIR must be set before loading the runtime environment}"

export OLIVIA_REPO_DIR=/cluster/projects/nn2834k/harshm/FFNOML/FFNOInversion.jl
export OLIVIA_ENV_SCRIPT=/cluster/home/harshm/loadnvidiampi.sh
export OLIVIA_JULIA=/cluster/software/NRIS/neoverse_v2/software/Julia/1.12.2/bin/julia
export OLIVIA_PYTHON=/cluster/home/harshm/nvidiaenv/bin/python3
export OLIVIA_CXX=c++
export OLIVIA_JULIA_DEPOT=/cluster/home/harshm/julia-depot-1.12.2
export OLIVIA_JULIA_PROJECT="${FFNOML_RUN_DIR}/julia-environment"
export OLIVIA_LOCAL_MUSPEL_DIR=/cluster/projects/nn2834k/harshm/julia-sources/Muspel.jl
export JULIA_DEPOT_PATH="${OLIVIA_JULIA_DEPOT}"

# Initialization guess at the top boundary; subsequent density comes from EOS.
export FFNO_TOP_DENSITY_KG_M3=1e-10
