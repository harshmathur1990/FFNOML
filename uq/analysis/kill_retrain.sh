#!/bin/bash
pkill -f "uq/retrain_run[.]sh"
pkill -f "pipeline[.]py --train"
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "
