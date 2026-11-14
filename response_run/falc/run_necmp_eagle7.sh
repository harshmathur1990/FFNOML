#!/bin/bash
# FALC_82 with three nₑ: its own (given, no HSEMHS), the LTE EOS and the NLTE EOS (HSEMHS column HSE), on the log column mass
# and height axes, on eagle7. Rebuilds the Wittmann library first (witt_kurucz_populations_given_ne, for the NLTE Kurucz lines).
R=/mn/stornext/d9/data/harshm/FFNOML/response_run; P=/mn/stornext/d9/data/harshm/FFNOML/FFNOInversion.jl; cd $R/falc
source ~/loadnvidia.sh >/dev/null 2>&1
export PATH=/tmp/julia-1.12.3/bin:$PATH CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5} FFNO_FSDP_NPROC=2
L=$R/inputs/wittmann/libwitt_ffno.so
if ! nm -D $L | grep -q witt_kurucz_populations_given_ne; then
  cp -n $L $L.before_nlte_eos
  CXX=g++ julia --startup-file=no --project=$P $P/scripts/build_wittmann_backend.jl $L || exit 1
fi
~/nvidiaenv/bin/python build_falc_logm.py || exit 1
for v in falcm ltem nltem falcz ltez nltez; do
  FALC_CONFIG=falc/forward_falc_necmp_$v.toml bash ./run_falc_eagle7.sh > falc_necmp_${v}_run.log 2>&1; echo "$v exit $?"
  tail -1 falc_necmp_${v}_run.log | cut -c1-200
done
