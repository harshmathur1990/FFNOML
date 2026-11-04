"""Full-domain distributed prediction with the current code, written to a chosen path, as a reference for
compare_batched_with_reference.py:  torchrun --standalone --nproc_per_node=N scripts/fsdp_reference_predict.py SOLVE.hdf5 OUT.hdf5 CHECKPOINT.pt"""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ffno"))
from config import MODEL, MODEL_CONFIG, ATOM_CONFIG
from FFNONet import ffno_predict_populations_distributed_full

solve, out, checkpoint = sys.argv[1:4]
species = os.path.basename(checkpoint).split("_")[-1].split(".")[0]
atom = ATOM_CONFIG[species]
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")
ffno_predict_populations_distributed_full(model=MODEL, checkpoint_path=checkpoint, solve_h5=solve, save_path=out,
    model_config=MODEL_CONFIG, lines=[atom["lines"]], wave=[atom["wave"]], chi=[atom["chi"]], levels=[atom["levels"]],
    atom_names=[species], cuda=True)
dist.destroy_process_group()
