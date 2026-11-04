#!/bin/bash
# usage: run_mpi.sh HOSTS OUTPUT.h5 [column_response.jl options]
#   HOSTS: MPICH host list, first host gets the writer rank too, e.g. hyades5:2,hyades11,hyades14
# hyadesN resolves to an IPv6 link-local address, so MPICH is pinned to MPI_IFACE with the libfabric TCP provider:
# eno8303 (default) on the Zen 3 nodes hyades5 and hyades7-16, eno1 on hyades3/4. Use nodes of one CPU type, or
# each type compiles its own package images in the shared depot, one at a time.
# Every host needs Julia 1.12.3 at /tmp/julia-1.12.3 (node-local; stornext Julia fails to map segments).
# CONFIG and ATMOSPHERE override the forward config and atmosphere (e.g. forward_multi3d.toml, inputs/multi3d_atmosphere.h5).
# Run with bash -l: ~/loadgcc.sh needs the module command; mpiexec passes this environment to every rank.
source ~/loadgcc.sh >/dev/null 2>&1
export PATH=/tmp/julia-1.12.3/bin:$PATH
P=/mn/stornext/d9/data/harshm/FFNOML/FFNOInversion.jl
R=/mn/stornext/d9/data/harshm/FFNOML/response_run
export FFNOML_RUN_DIR=$R
cd $R
hosts=$1; out=$2; shift 2
iface=${MPI_IFACE:-eno8303}
nranks=$(echo "$hosts" | tr ',' '\n' | awk -F: '{n+=($2==""?1:$2)} END{print n}')
MPIEXEC=$(julia --startup-file=no --project=$P -e 'using MPI; print(MPI.mpiexec().exec[1])')
"$MPIEXEC" -iface "$iface" -genv FI_PROVIDER tcp -genv FI_TCP_IFACE "$iface" -hosts "$hosts" -n "$nranks" \
  /tmp/julia-1.12.3/bin/julia --startup-file=no --project=$P --threads=auto \
  $P/scripts/column_response.jl ${CONFIG:-$R/forward.toml} $R/paths.jl ${ATMOSPHERE:-/mn/stornext/d9/data/harshm/FFNOML/outputs/forward_atmosphere.h5} \
  "$out" "$@"
