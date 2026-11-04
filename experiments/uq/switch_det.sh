#!/bin/bash
# After the MC runs for 385 and 1050, replace the remaining H queue with deterministic predictions.
cd /mn/stornext/d9/data/harshm/FFNOML
until grep -q "predict ch024031_by200bz005_450 H" uq/mc/queue_H.log; do sleep 15; done
pkill -f "run_queue.sh H 20"; pkill -f "uq_predict.py --atom H --name ch024031_by200bz005_450"; sleep 20
rm -f uq/mc/ch024031_by200bz005_450_H.h5.tmp
echo "$(date +%T) switched to deterministic" >> uq/mc/queue_H.log
bash uq/run_queue.sh H 0 en024048_hion/465 ch024031_by200bz005/450 en024031_by100_helium/109 nw012023/1120 en024048_hion/386 ch012012_hion/759 ch012012_hion/834 ch012006/795 ch012006/820 qs006003_sap/1100 qs006003_sap/1297 ch024031_by200bz005/700 ch024031_by200bz005/950 ch012012_hion/910 ch012006/836 qs006003_sap/689 en024048_hion/700 nw012023/940 ch012012_hion/984 ch012006/849 qs006003_sap/900 >> uq/mc/queue_H.log 2>&1
