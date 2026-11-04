#!/bin/bash
pkill -f "uq/atlas_v3[.]sh"
pkill -f "xargs -P 4"
pkill -f "out-dir uq/atlas_v3"
sleep 3
pgrep -f "out-dir uq/atlas_v3" | wc -l
