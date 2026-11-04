#!/bin/bash
# CPU: atlases with the physical context descriptors (uq/atlas_v3) for every prediction of ATOM in uq/mc,
# then the missing-regimes table. Usage: bash uq/atlas_v3.sh ATOM LEVELS
source ~/loadnvidia.sh >/dev/null 2>&1
cd /mn/stornext/d9/data/harshm/FFNOML
export CUDA_VISIBLE_DEVICES= PY=~/nvidiaenv/bin/python3.12 B=/mn/stornext/d9/data/harshm/bifrost_data
atom=$1; levels=$2
one() {
  f=$1; atom=$2; name=$(basename $f _$atom.h5); cs=${name%_*}/${name##*_}
  [ -f uq/atlas_v3/${name}_${atom}_atlas.h5 ] && return
  $PY scripts/uq_error_atlas.py --case $name --atom $atom --truth $B/$cs/$atom --uq $f \
    --atmos $B/$cs/atm3d --mesh $B/$cs/mesh --out-dir uq/atlas_v3 > uq/mc/${name}_${atom}_atlas_v3.log 2>&1 \
    || echo "FAILED atlas $name"
  echo "$(date +%T) atlas_v3 $name $atom"
}
export -f one
ls uq/mc/*_$atom.h5 | xargs -P 4 -I{} bash -c "one {} $atom"
$PY scripts/uq_missing_regimes.py --atlas uq/atlas_v3/*_${atom}_atlas.h5 --levels $levels \
  --out uq/missing_regimes_${atom}.csv > uq/missing_regimes_${atom}.log 2>&1 || echo "FAILED missing regimes"
echo "$(date +%T) atlas_v3 $atom done"
