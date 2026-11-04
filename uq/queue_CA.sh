#!/bin/bash
# CA pass over the labelled snapshots: MC dropout on the three pilot cases, deterministic elsewhere,
# 256^2 tiles of the full-domain operator for the 384^3 boxes. Log: uq/mc/queue_CA.log
cd /mn/stornext/d9/data/harshm/FFNOML
bash uq/run_queue.sh CA 20 nw012023/915 en024048_hion/385 nw012023/1050
bash uq/run_queue.sh CA 0 en024048_hion/465 nw012023/1120 en024048_hion/386 ch012012_hion/759 ch012012_hion/834 \
  ch012006/795 ch012006/820 qs006003_sap/1100 qs006003_sap/1297 ch012012_hion/910 ch012006/836 qs006003_sap/689 \
  en024048_hion/700 nw012023/940 ch012012_hion/984 ch012006/849 qs006003_sap/900
TILE=256 bash uq/run_queue.sh CA 0 ch024031_by200bz005/450 ch024031_by200bz005/700 en024031_by100_helium/109
echo "$(date +%T) CA all done"
