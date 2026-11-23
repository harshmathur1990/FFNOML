#!/bin/bash
# The six FALC necmp runs (FFNOML/response_run/falc/run_necmp_eagle7.sh) with the l004p H model on FALC's own grid
# ([population.mesh] enabled = false), H-alpha only, on eagle7: FFNOML_lf's FFNOInversion and batched FFNO service, with the
# model's configuration. Run assets (atoms, Wittmann library, partition functions) from FFNOML/response_run/inputs.
source ~/loadnvidia.sh >/dev/null 2>&1
W=/mn/stornext/d9/data/harshm/ffnoml_data/models/l004p/falc/native_rerun; P=/mn/stornext/d9/data/harshm/FFNOML_lf/FFNOInversion.jl
export PATH=/tmp/julia-1.12.3/bin:$PATH FFNOML_RUN_DIR=$W CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
set -a; source <(grep -E '^FFNOML_(ACTIVE_ATOMS|OUTPUT_MODE|LAM_TOTAL|MODEL_CONFIG)=' /mn/stornext/d9/data/harshm/ffnoml_data/models/l004p/H/experiment.env); set +a
export FFNO_FSDP_LAUNCHER=$P/scripts/ffno_batched_service_local.sh FFNO_FSDP_PYTHON=$HOME/nvidiaenv/bin/python FFNO_BATCHED_VERTICAL_NORM_SLABS=1
export FFNO_GPU_CONTROL_HOST=127.0.0.1 FFNO_GPU_DIAGNOSTICS_DIR=$W/diagnostics OMP_NUM_THREADS=1
cd $W
for v in ${RUNS:-falcm ltem nltem falcz ltez nltez}; do
  FALC_CONFIG=falc/forward_falc_${PREFIX-necmp_}$v.toml julia --startup-file=no --project=$P --threads=$(nproc --all) -e '
using FFNOInversion
factory=include("falc/model_factory.jl")
run_inversion_files!(ENV["FALC_CONFIG"],factory)
println("FORWARD_FALC_DONE")' > falc/falc_${PREFIX-necmp_}${v}_run.log 2>&1; echo "$v exit $? $(tail -1 falc/falc_${PREFIX-necmp_}${v}_run.log | cut -c1-150)"
done
echo "all done $(date)"
