#!/bin/bash
source ~/loadgcc.sh >/dev/null 2>&1; export PATH=/tmp/julia-1.12.3/bin:$PATH
H=/mn/stornext/d9/data/harshm/rh_falc/rhf1d; F=/mn/stornext/d9/data/harshm/FFNOML/response_run/falc; PY=~/env/bin/python3.12
cd $F; $PY rh_cont_setup.py || exit 1
cd $H/run_falc_cont; ../solveray > solveray.log 2>&1; echo "solveray exit $?"; tail -3 solveray.log | cut -c1-100
air=$(cd $F; $PY -c "
import struct,numpy as np
H='$H/run_falc_cont/'; b=open(H+'spectrum.out','rb').read(); n=struct.unpack('>i',b[:4])[0]; lam=np.frombuffer(b,'>f8',n,4).astype(float)
r=open(H+'ray.input').read().split(); ids=[int(x) for x in r[2:]]; print(','.join('%.6f'%lam[i] for i in ids))")
echo "air $air"; cd $F/..; julia --project=../FFNOInversion.jl -t 4 falc/falc_cont_ours.jl "$air" 2>&1 | grep -vE '^\[ Info|Zeeman|FORWARD' | tail -12 | cut -c1-170
