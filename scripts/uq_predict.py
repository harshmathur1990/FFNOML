"""Deterministic and MC-dropout FFNO3D population prediction for uncertainty studies.

Writes, in denormalized log10 population units (log10 m^-3), arrays of shape [nx, ny, nz, nlev]
matching the MULTI3D out_pop layout:

    det_log      eval-mode prediction (dropout off)
    mc_mean_log  mean of the MC-dropout samples
    mc_std_log   standard deviation of the MC-dropout samples

Two routes:
  --route full  (default) the production operator: whole domain, H-slab sharded over torchrun
                ranks with DistributedGroupNorm, as FFNONet.ffno_predict_populations_distributed_full.
                Each MC sample is a full-domain forward, so mc_std_log is the plain sample std.
                With --tile N, boxes larger than N x N are covered by N x N sub-boxes at stride N/2,
                each run through the same distributed operator and Hann-blended. Tiles draw independent
                dropout masks, so the std of the stitched realization sum_t w_t y_t / W is
                mc_std_log = sqrt(sum_t w_t^2 var_t) / W. Use this when the whole box does not fit the GPUs.
  --route tiled Hann-stitched patch x patch tiles (FFNONet._predict_tiled). GroupNorm statistics
                are then per tile, so this is a different operator (~40% larger error for
                en024048_hion_385/H with 40x40 tiles); kept for comparison. mc_std_log is the std of
                the stitched realization, as above. tile_std_log is a tile-disagreement diagnostic (the
                Hann-weighted spread of overlapping deterministic tiles), not a predictive variance.

MC dropout keeps the model in eval mode and switches only the FFNOBlock3dBalanced spec_drop /
vert_drop modules to train mode. The production checkpoints were trained with p=0.08 on both
(see docs/ffno3d-uncertainty-active-learning-plan.md). All normalization is GroupNorm, so no other
layer changes behaviour. Samples are drawn in batches; dropout masks are elementwise, so batch
members are independent samples.

Usage:
    torchrun --nproc_per_node=5 scripts/uq_predict.py --atom H --name en024048_hion_385 --samples 20 --out OUT.h5
    python scripts/uq_predict.py --route tiled --atom CA --atmos .../atm3d --mesh .../mesh --name X --out OUT.h5
"""
import argparse
import hashlib
import itertools
import os
import subprocess
import sys
import time

import h5py
import numpy as np
import torch
import torch.distributed as dist

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from config import MODEL_CONFIG, MODEL_DIR  # noqa: E402
from distributed_inference import enable_distributed_inference, partition_range  # noqa: E402
from FFNONet import (  # noqa: E402
    _distributed_full_forward,
    _hann_2d,
    _load_inference_metadata_from_checkpoint,
    _tile_positions,
)
from models.ffno_model import FFNO3D, FFNOBlock3dBalanced  # noqa: E402
from train_utils import load_checkpoint  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atom", required=True, choices=["H", "CA"])
    p.add_argument("--name", required=True, help="case name, e.g. en024048_hion_385")
    p.add_argument("--checkpoint", help="default: MODEL_DIR/3D_sim_train_<atom>.pt")
    p.add_argument("--input", help="prebuilt solving set; default: MODEL_DIR/3D_sim_predict_<name>.hdf5")
    p.add_argument("--atmos", help="build inputs from this atm3d instead of --input")
    p.add_argument("--mesh", help="mesh for --atmos")
    p.add_argument("--out", required=True)
    p.add_argument("--route", choices=["full", "tiled"], default="full")
    p.add_argument("--tile", type=int, default=0, help="full route: sub-box size for large domains; 0 = whole box")
    p.add_argument("--samples", type=int, default=20, help="MC-dropout samples; 0 = deterministic only")
    p.add_argument("--batch", type=int, default=4, help="tiled route: MC samples per forward pass")
    p.add_argument("--patch", type=int, default=40, help="tiled route")
    p.add_argument("--stride", type=int, default=20, help="tiled route")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--z-range", type=float, nargs=2, metavar=("ZMIN", "ZMAX"),
                   help="crop the columns to ZMIN <= z <= ZMAX (Mm) before prediction (context experiments)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_revision():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_inputs(args, h_slab=slice(None)):
    """Return raw (unnormalized) X [1,Cin,D,nx,ny], z_scale [1,D,nx,ny] in Mm, dx, dy in m,
    restricted to rows h_slab of nx and, with --z-range, to the selected heights."""
    X, z, dx, dy, src = _load_inputs(args, h_slab)
    if args.z_range:
        keep = (z[0, :, 0, 0] >= args.z_range[0]) & (z[0, :, 0, 0] <= args.z_range[1])
        X, z = X[:, :, keep], z[:, keep]
    return X, z, dx, dy, src


def _load_inputs(args, h_slab):
    if args.atmos:
        from pipeline import load_pred_data
        from FFNONet import _make_inputs_ch_first, _normalize_z_scale, _expand_z_to_match_rho
        rho, z_scale, temp, vx, vy, vz, ne, dx, dy = load_pred_data(mesh_file=args.mesh, atmos_file=args.atmos)
        X = _make_inputs_ch_first(rho, temp, vx, vy, vz, ne)[None]
        z = _normalize_z_scale(_expand_z_to_match_rho(z_scale, rho))
        z = np.transpose(z, (2, 0, 1))[None].astype(np.float32)
        return (X[:, :, :, h_slab], z[:, :, h_slab], np.array([dx], np.float32), np.array([dy], np.float32),
                args.atmos)
    path = args.input or os.path.join(MODEL_DIR, f"3D_sim_predict_{args.name}.hdf5")
    with h5py.File(path, "r") as f:
        return (f["inputs"][:, :, :, h_slab], f["z_scale"][:, :, h_slab].astype(np.float32), f["dx"][...],
                f["dy"][...], path)


def input_nx(args):
    if args.atmos:
        from pipeline import read_mesh
        return read_mesh(args.mesh)[0]
    with h5py.File(args.input or os.path.join(MODEL_DIR, f"3D_sim_predict_{args.name}.hdf5"), "r") as f:
        return f["inputs"].shape[3]


def build_model(ckpt_path, device):
    Cin, Cout, mean_X, std_X, mean_Y, std_Y = _load_inference_metadata_from_checkpoint(ckpt_path)
    cfg = dict(MODEL_CONFIG, in_channels=Cin, out_channels=Cout, checkpoint_blocks=False)
    model = FFNO3D(**cfg).to(device)
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt, cfg, (mean_X, std_X, mean_Y, std_Y)


def dropout_modules(model):
    mods = []
    for block in model.modules():
        if isinstance(block, FFNOBlock3dBalanced):
            mods += [m for m in (block.spec_drop, block.vert_drop) if isinstance(m, torch.nn.Dropout)]
    return mods


def set_mc(mods, on):
    for m in mods:
        m.train(on)


@torch.no_grad()
def predict_tiled(model, mods, X, z, dx, dy, patch, stride, n_samples, batch, mean_Y, std_Y):
    """Tiled prediction; returns denormalized log10 arrays [C, D, nx, ny] on CPU (float32)."""
    _, _, D, nx, ny = X.shape
    dev = X.device
    w2 = _hann_2d(patch, device=dev, dtype=torch.float32)[None, None, :, :]  # [1,1,p,p]
    mY = torch.as_tensor(mean_Y, device=dev, dtype=torch.float32)[:, None, None, None]
    sY = torch.as_tensor(std_Y, device=dev, dtype=torch.float32)[:, None, None, None]
    C = len(mean_Y)

    acc = {k: torch.zeros((C, D, nx, ny), dtype=torch.float32) for k in ("det", "det2", "mc", "var")}
    W = torch.zeros((1, 1, nx, ny), dtype=torch.float32)
    w2c = w2.cpu()

    xs, ys = _tile_positions(nx, patch, stride), _tile_positions(ny, patch, stride)
    t0 = time.time()
    for k, (i0, j0) in enumerate(itertools.product(xs, ys)):
        sl = (slice(None), slice(None), slice(i0, i0 + patch), slice(j0, j0 + patch))
        xt = X[:, :, :, i0:i0 + patch, j0:j0 + patch]
        zt = z[:, :, i0:i0 + patch, j0:j0 + patch]

        set_mc(mods, False)
        y = model(xt, zt, dx, dy)[0] * sY + mY  # [C,D,p,p]
        acc["det"][sl] += (y * w2).cpu()
        acc["det2"][sl] += (y * y * w2).cpu()

        if n_samples > 0:
            set_mc(mods, True)
            s1 = torch.zeros_like(y)
            s2 = torch.zeros_like(y)
            done = 0
            while done < n_samples:
                b = min(batch, n_samples - done)
                ys_ = model(xt.expand(b, -1, -1, -1, -1), zt.expand(b, -1, -1, -1), dx, dy) * sY + mY
                s1 += ys_.sum(0)
                s2 += (ys_ * ys_).sum(0)
                done += b
            mean = s1 / n_samples
            var = (s2 - n_samples * mean * mean).clamp_min(0) / max(n_samples - 1, 1)
            acc["mc"][sl] += (mean * w2).cpu()
            acc["var"][sl] += (var * w2 * w2).cpu()
            set_mc(mods, False)

        W[:, :, i0:i0 + patch, j0:j0 + patch] += w2c
        if k % 20 == 0:
            print(f"tile {k + 1}/{len(xs) * len(ys)}  {time.time() - t0:.0f}s", flush=True)

    if (W == 0).any():
        raise RuntimeError("tiles did not cover the domain")
    det = acc["det"] / W
    out = {
        "det_log": det,
        "tile_std_log": (acc["det2"] / W - det * det).clamp_min(0).sqrt(),
    }
    if n_samples > 0:
        out["mc_mean_log"] = acc["mc"] / W
        out["mc_std_log"] = acc["var"].sqrt() / W
    return out


@torch.no_grad()
def predict_full(model, mods, X, z, dx, dy, n_samples, mean_Y, std_Y):
    """Full-domain forwards on this rank's H slab; returns [C, D, h, ny] arrays on CPU."""
    dev = X.device
    mY = torch.as_tensor(mean_Y, device=dev, dtype=torch.float32)[:, None, None, None]
    sY = torch.as_tensor(std_Y, device=dev, dtype=torch.float32)[:, None, None, None]

    def forward():
        return (_distributed_full_forward(model, X, z, dx, dy)[0][0] * sY + mY).cpu()

    t0 = time.time()
    set_mc(mods, False)
    out = {"det_log": forward()}
    log0(f"deterministic pass {time.time() - t0:.0f}s")
    if n_samples > 0:
        set_mc(mods, True)
        s1 = torch.zeros_like(out["det_log"], dtype=torch.float64)
        s2 = torch.zeros_like(s1)
        for k in range(n_samples):
            y = forward().double()
            s1 += y
            s2 += y * y
            log0(f"sample {k + 1}/{n_samples} {time.time() - t0:.0f}s")
        set_mc(mods, False)
        mean = s1 / n_samples
        out["mc_mean_log"] = mean.float()
        out["mc_std_log"] = ((s2 - n_samples * mean * mean).clamp_min(0) / max(n_samples - 1, 1)).sqrt().float()
    return out


def log0(msg):
    if not (dist.is_initialized() and dist.get_rank() != 0):
        print(msg, flush=True)


def main():
    args = parse_args()
    if os.path.exists(args.out):
        raise IOError(f"Output exists: {args.out}")
    device = args.device if torch.cuda.is_available() else "cpu"
    ckpt_path = args.checkpoint or os.path.join(MODEL_DIR, f"3D_sim_train_{args.atom}.pt")

    full = args.route == "full"
    rank, world = 0, 1
    if full:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
        rank, world = dist.get_rank(), dist.get_world_size()
        device = f"cuda:{local_rank}"
    # independent dropout masks per rank (slab) and per run
    torch.manual_seed(args.seed * 1000 + rank)

    model, ckpt, cfg, (mean_X, std_X, mean_Y, std_Y) = build_model(ckpt_path, device)
    if full:
        model = enable_distributed_inference(model).eval()
    mods = dropout_modules(model)
    rates = sorted({m.p for m in mods})
    log0(f"{len(mods)} MC-dropout modules, p={rates}")
    if args.samples > 0 and not mods:
        raise RuntimeError("checkpoint/config has no spec/vertical dropout; MC dropout impossible")

    if full:
        X, z, dx, dy, src = load_inputs(args)
        nx, ny = X.shape[3], X.shape[4]
        tx, ty = (min(args.tile, nx), min(args.tile, ny)) if args.tile else (nx, ny)
        xs, ys = _tile_positions(nx, tx, max(tx // 2, 1)), _tile_positions(ny, ty, max(ty // 2, 1))
    else:
        nx = input_nx(args)
        X, z, dx, dy, src = load_inputs(args)
    mX = torch.as_tensor(mean_X, dtype=torch.float32)[None, :, None, None, None]
    sX = torch.as_tensor(std_X, dtype=torch.float32)[None, :, None, None, None]
    dxt, dyt = torch.from_numpy(np.asarray(dx, np.float32)).to(device), torch.from_numpy(np.asarray(dy, np.float32)).to(device)
    log0(f"{args.name}/{args.atom}: route={args.route}, ranks={world}, input {tuple(X.shape)}, "
         f"dx={float(dx[0]):.4e} m, samples={args.samples}"
         + (f", tiles {len(xs)}x{len(ys)} of {tx}x{ty}" if full else ""))

    t0 = time.time()
    if full:
        multi = len(xs) * len(ys) > 1
        if multi:
            hx = torch.hann_window(tx, dtype=torch.float32).clamp_min(1e-3)
            hy = torch.hann_window(ty, dtype=torch.float32).clamp_min(1e-3)
            w2 = torch.outer(hx, hy).numpy()[None, None]  # [1,1,tx,ty]
        else:
            w2 = np.ones((1, 1, tx, ty), np.float32)
        acc, W = {}, np.zeros((1, 1, nx, ny), np.float32)
        h0, h1 = partition_range(tx, rank, world)
        for k, (i0, j0) in enumerate(itertools.product(xs, ys)):
            Xs = X[:, :, :, i0 + h0:i0 + h1, j0:j0 + ty]
            Xt = ((torch.from_numpy(np.ascontiguousarray(Xs)).float() - mX) / sX).to(device)
            zt = torch.from_numpy(np.ascontiguousarray(z[:, :, i0 + h0:i0 + h1, j0:j0 + ty])).to(device)
            res = predict_full(model, mods, Xt, zt, dxt, dyt, args.samples, mean_Y, std_Y)
            del Xt, zt
            gathered = [None] * world if rank == 0 else None
            dist.gather_object({k_: v.numpy() for k_, v in res.items()}, gathered, dst=0)
            if rank == 0:
                # slabs are along nx, axis 2 of [C, D, nx, ny]
                tile = {k_: np.concatenate([g[k_] for g in gathered], axis=2) for k_ in res}
                if "mc_std_log" in tile:
                    tile["mc_var"] = tile.pop("mc_std_log") ** 2
                sl = (slice(None), slice(None), slice(i0, i0 + tx), slice(j0, j0 + ty))
                for k_, v in tile.items():
                    if k_ not in acc:
                        acc[k_] = np.zeros(v.shape[:2] + (nx, ny), np.float32)
                    acc[k_][sl] += v * (w2 * w2 if k_ == "mc_var" else w2)
                W[..., i0:i0 + tx, j0:j0 + ty] += w2[0, 0]
            if multi:
                log0(f"tile {k + 1}/{len(xs) * len(ys)} at ({i0},{j0}) {time.time() - t0:.0f}s")
        if rank == 0:
            res = {k_: torch.from_numpy(v / (W * W if k_ == "mc_var" else W)) for k_, v in acc.items()}
            if "mc_var" in res:
                res["mc_std_log"] = res.pop("mc_var").sqrt()
    else:
        Xt = ((torch.from_numpy(X).float() - mX) / sX).to(device)
        zt = torch.from_numpy(z).to(device)
        res = predict_tiled(model, mods, Xt, zt, dxt, dyt, args.patch, args.stride, args.samples, args.batch,
                            mean_Y, std_Y)
    elapsed = time.time() - t0
    if rank != 0:
        dist.barrier()
        dist.destroy_process_group()
        return

    tmp = args.out + ".tmp"
    with h5py.File(tmp, "w") as f:
        for k, v in res.items():
            # [C,D,nx,ny] -> [nx,ny,D,C], the out_pop / output_3D_sim layout
            f.create_dataset(k, data=v.permute(2, 3, 1, 0).numpy(), compression="gzip", compression_opts=4,
                             shuffle=True, chunks=True)
        f.create_dataset("z_scale_Mm", data=z[0, :, 0, 0])
        a = f.attrs
        a.update(dict(
            name=args.name, atom=args.atom, units="log10 m^-3", checkpoint=os.path.abspath(ckpt_path),
            checkpoint_sha256=sha256(ckpt_path), epoch=int(ckpt.get("epoch", -1)),
            val_loss=float(ckpt.get("val_loss", np.nan)), git_revision=git_revision(), input=os.path.abspath(src),
            route=args.route, ranks=world, patch=args.patch if not full else (args.tile or 0),
            stride=args.stride if not full else (args.tile // 2 if args.tile else 0), n_samples=args.samples,
            seed=args.seed, dropout_rates=str(rates), n_dropout_modules=len(mods),
            z_range=str(args.z_range) if args.z_range else "full",
            model_config=repr(cfg), device=torch.cuda.get_device_name() if device.startswith("cuda") else "cpu",
            seconds=elapsed,
        ))
    os.replace(tmp, args.out)
    print(f"wrote {args.out} in {elapsed:.0f}s")
    if full:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
