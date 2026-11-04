#!/bin/bash
# Runs on eagle7 after run_tangents_eagle7.sh: column responses of the stride-4 linearization atmosphere with the
# FFNO population term (column_response_s4_ffno.h5).
R=/mn/stornext/d9/data/harshm/FFNOML/response_run
cd $R
while pgrep -f ffno_population_tangents.jl >/dev/null; do sleep 60; done
grep -q "wrote $R/outputs/population_tangents_s4.h5" tangents_s4.log || { echo "tangent run did not finish"; exit 1; }
export CONFIG=$R/forward.toml ATMOSPHERE=$R/outputs/population_tangents_s4_atmosphere.h5
./run.sh outputs/column_response_s4_ffno.h5 --variables temperature,vz --stride 1 \
  --population-tangents outputs/population_tangents_s4.h5
echo "after_tangents_s4 done"
