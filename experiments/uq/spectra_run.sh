#!/bin/bash
# CPU: line profiles (scripts/uq_synth.jl) for truth, production-equivalent and each fine-tuned branch, and
# their errors (scripts/uq_spectral_errors.py) into uq/spectral_<ATOM>.csv. Usage: bash uq/spectra_run.sh ATOM
cd /mn/stornext/d9/data/harshm/FFNOML
atom=$1; B=/mn/stornext/d9/data/harshm/bifrost_data
[ -x /tmp/julia-1.12.3/bin/julia ] || cp -a /mn/stornext/astro/local/julia/julia-1.12.3 /tmp/
J() { bash -lc "source ~/loadnvidia_mpi.sh >/dev/null 2>&1; cd $PWD; /tmp/julia-1.12.3/bin/julia --threads=32 $*"; }
PY() { bash -c "source ~/loadnvidia.sh >/dev/null 2>&1; cd $PWD; ~/nvidiaenv/bin/python3.12 $*"; }
CASES="nw012023/1120 nw012023/1050 nw012023/940 nw012023/915 en024048_hion/700 ch012012_hion/984 ch012006/849
  qs006003_sap/900 en024048_hion/385 ch012012_hion/759"
mkdir -p uq/spectra
for cs in $CASES; do
  name=${cs/\//_}; t=uq/spectra/${name}_${atom}_truth.h5
  [ -f $t ] || J scripts/uq_synth.jl --atom $atom --mesh $B/$cs/mesh --atmos $B/$cs/atm3d --out-pop $B/$cs/$atom/out_pop --out $t > /dev/null 2>&1
  for lab in prod A_base B_nw1050 C_random; do
    src=uq/mc/${name}_$atom.h5; [ $lab = prod ] || src=uq/retrain/${atom}_$lab/pred/${name}_$atom.h5
    [ -f $src ] || continue
    o=uq/spectra/${name}_${atom}_$lab.h5
    [ -f $o ] || J scripts/uq_synth.jl --atom $atom --mesh $B/$cs/mesh --atmos $B/$cs/atm3d --uq $src --out $o > /dev/null 2>&1
    grep -q "^$name,$atom,$lab," uq/spectral_$atom.csv 2>/dev/null || \
      PY scripts/uq_spectral_errors.py --truth $t --pred $o --atom $atom --case $name --label $lab --out uq/spectral_$atom.csv > /dev/null
  done
  echo "$(date +%T) spectra $name $atom"
done
echo "$(date +%T) spectra $atom done"
