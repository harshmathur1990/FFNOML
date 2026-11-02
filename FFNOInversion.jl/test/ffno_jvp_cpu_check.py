"""CPU check of the FFNO forward-mode JVP through the distributed H-slab adapters (no GPU, no FSDP).

A small random-weight FFNO3D in float64 runs distributed over H slabs (gloo) with dual tensors; the gathered
population-feature tangent is compared with the JVP of the same model on one process and with central finite
differences. Run from the FFNOML checkout:
    python -m torch.distributed.run --nproc_per_node=2 FFNOInversion.jl/test/ffno_jvp_cpu_check.py
"""
import copy
import os
import sys

import torch
import torch.autograd.forward_ad as fwAD
import torch.distributed as dist

sys.path.insert(0, os.getcwd())
from distributed_inference import enable_distributed_inference, partition_range  # noqa: E402
from models.ffno_model import FFNO3D  # noqa: E402


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.set_default_dtype(torch.float64)      # the spectral layers build their k features in the default dtype
    torch.manual_seed(0)
    model = FFNO3D(in_channels=6, out_channels=6, width=8, n_layers=2, dropout=0.0,
                   checkpoint_blocks=True, hidden_spectral=16, hidden_vertical=16).double().eval()
    serial = copy.deepcopy(model)
    distributed = enable_distributed_inference(model)

    D, H, W = 24, 8, 10
    x = torch.randn(1, 6, D, H, W, dtype=torch.float64)
    z = torch.linspace(1.0, -0.5, D, dtype=torch.float64)[None, :, None, None].expand(1, D, H, W).contiguous()
    tangent = torch.zeros_like(x); tangent[:, 0, D // 2] = 1.0          # channel 0 at one depth, every column
    dx = dy = torch.tensor(0.048, dtype=torch.float64)
    h0, h1 = partition_range(H, rank, world)

    with torch.no_grad(), fwAD.dual_level():
        local = distributed(fwAD.make_dual(x[:, :, :, h0:h1].contiguous(), tangent[:, :, :, h0:h1].contiguous()),
                            z[:, :, h0:h1].contiguous(), dx, dy)
        local_tangent = fwAD.unpack_dual(local).tangent.contiguous()
        h = 1e-6
        xs, ts, zs = x[:, :, :, h0:h1].contiguous(), tangent[:, :, :, h0:h1].contiguous(), z[:, :, h0:h1].contiguous()
        local_primal = distributed(xs, zs, dx, dy)
        local_fd = (distributed(xs + h * ts, zs, dx, dy) - distributed(xs - h * ts, zs, dx, dy)) / (2 * h)

    def gather(local):
        parts = [torch.empty_like(local[:, :, :, :1].expand(-1, -1, -1, partition_range(H, r, world)[1]
                 - partition_range(H, r, world)[0], -1).contiguous()) for r in range(world)]
        dist.all_gather(parts, local.contiguous())
        return torch.cat(parts, dim=3)
    jvp_distributed = gather(local_tangent); primal_distributed = gather(local_primal)
    fd_distributed = gather(local_fd)

    if rank == 0:
        with torch.no_grad(), fwAD.dual_level():
            jvp_serial = fwAD.unpack_dual(serial(fwAD.make_dual(x, tangent), z, dx, dy)).tangent
            h = 1e-6
            fd = (serial(x + h * tangent, z, dx, dy) - serial(x - h * tangent, z, dx, dy)) / (2 * h)
            primal_serial = serial(x, z, dx, dy)
        scale = jvp_serial.abs().max().item()
        print(f"distributed vs serial forward (rel) {(primal_distributed - primal_serial).abs().max().item() / primal_serial.abs().max().item():.2e}")
        print(f"distributed JVP vs distributed FD   {(jvp_distributed - fd_distributed).abs().max().item() / scale:.2e}")
        print(f"max |JVP|                           {scale:.3e}")
        print(f"distributed JVP vs serial JVP (rel) {(jvp_distributed - jvp_serial).abs().max().item() / scale:.2e}")
        print(f"serial JVP vs finite difference     {(jvp_serial - fd).abs().max().item() / scale:.2e}")
        far = jvp_serial[:, :, : D // 2 - 6].abs().max().item() / scale
        print(f"tangent spread: max |JVP| more than 6 depth points above the seed / max = {far:.2e}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
