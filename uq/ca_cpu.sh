#!/bin/bash
# CPU analysis for CA: alignment checks, descriptor atlases + missing regimes (atlas_v3.sh),
# cross-snapshot table and support distance on the v3 atlases. Log: uq/mc/ca_cpu.log
source ~/loadnvidia.sh >/dev/null 2>&1
cd /mn/stornext/d9/data/harshm/FFNOML
export CUDA_VISIBLE_DEVICES=; PY=~/nvidiaenv/bin/python3.12; B=/mn/stornext/d9/data/harshm/bifrost_data
rm -f uq/alignment_CA.csv
for f in uq/mc/*_CA.h5; do
  name=$(basename $f _CA.h5); cs=${name%_*}/${name##*_}
  $PY scripts/uq_verify_alignment.py --case $name --atom CA --truth $B/$cs/CA --uq $f --atmos $B/$cs/atm3d \
    --mesh $B/$cs/mesh --out uq/alignment_CA.csv > uq/mc/${name}_CA_align.log 2>&1 || echo "FAILED align $name"
done &
bash uq/atlas_v3.sh CA 1-5
wait
$PY scripts/uq_cross_snapshot.py --atlas uq/atlas_v3/*_CA_atlas.h5 --out uq/cross_snapshot_CA.csv
$PY scripts/uq_support_distance.py --atlas uq/atlas_v3/*_CA_atlas.h5 --out uq/support_distance_CA.csv \
  > uq/support_distance_CA.log 2>&1 || echo "FAILED support distance"
echo "$(date +%T) ca_cpu done"
