#!/bin/bash
# rhf1d and solveray on the two FALC atmospheres, unedited atoms.
source ~/loadgcc.sh >/dev/null 2>&1
H=/mn/stornext/d9/data/harshm/rh_falc/rhf1d; F=/mn/stornext/d9/data/harshm/FFNOML/response_run/falc; PY=~/env/bin/python3.12
for v in atmos 5x5; do
  ( cd $H/run_cmp_$v; ../rhf1d > rhf1d.log 2>&1; echo "$v rhf1d exit $?"; ../solveray > solveray.log 2>&1; echo "$v solveray exit $?"
    cd $F; $PY read_rh_output.py $H/run_cmp_$v falc_cmp_${v}_rh.h5; echo "$v read exit $?" ) &
done
wait; echo RH_DONE
