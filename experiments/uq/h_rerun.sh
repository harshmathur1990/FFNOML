#!/bin/bash
# CPU-only: alignment checks and corrected atlases for every H prediction in uq/mc, then
# support distance and the cross-snapshot table. Log: uq/mc/h_rerun.log
source ~/loadnvidia.sh >/dev/null 2>&1
cd /mn/stornext/d9/data/harshm/FFNOML
export CUDA_VISIBLE_DEVICES= PY=~/nvidiaenv/bin/python3.12 B=/mn/stornext/d9/data/harshm/bifrost_data
atom=${1:-H}
one() {
  f=$1; atom=$2; name=$(basename $f _$atom.h5)
  sim=${name%_*}; snap=${name##*_}; cs=$sim/$snap
  $PY scripts/uq_verify_alignment.py --case $name --atom $atom --truth $B/$cs/$atom --uq $f \
    --atmos $B/$cs/atm3d --mesh $B/$cs/mesh --out uq/alignment_$atom.csv > uq/mc/${name}_${atom}_align.log 2>&1 \
    || echo "FAILED align $name"
  $PY scripts/uq_error_atlas.py --case $name --atom $atom --truth $B/$cs/$atom --uq $f \
    --atmos $B/$cs/atm3d --mesh $B/$cs/mesh --out-dir uq/atlas_v2 > uq/mc/${name}_${atom}_atlas_v2.log 2>&1 \
    || echo "FAILED atlas $name"
  echo "$(date +%T) done $name"
}
export -f one
rm -f uq/alignment_$atom.csv
ls uq/mc/*_$atom.h5 | xargs -P 4 -I{} bash -c "one {} $atom"
$PY scripts/uq_cross_snapshot.py --atlas uq/atlas_v2/*_${atom}_atlas.h5 --out uq/cross_snapshot_${atom}_v2.csv
$PY scripts/uq_support_distance.py --atlas uq/atlas_v2/*_${atom}_atlas.h5 --out uq/support_distance_${atom}_v2.csv \
  > uq/support_distance_${atom}_v2.log 2>&1 || echo "FAILED support distance"
echo "$(date +%T) rerun $atom done"
