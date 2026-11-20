"""Phase 3 of docs/level-fractions-capacity-plan.md: GPU memory and time of one training step for each capacity
configuration, the level-fraction head on H, fp32 and AdamW as in training, one patch per GPU.

    python scripts/capacity_benchmark.py [--patch 40] [--depth 496] [--steps 5] [--config NAME ...] [--out FILE]

The configurations are cumulative, as the screening adds them (plan, "Capacity changes"): 0 the Phase 2 model, then
+ z-spectral branch, + low-rank frequency mixing, + open side paths, width 96, 8 blocks, hidden_vertical 192. Each runs
in a fresh process on the first visible GPU, so a configuration that runs out of memory is reported and the rest go on.
Parameters are counted unsharded; under FSDP each GPU holds 1/N of them and their optimiser state, which is small next
to the activations.
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = [
    ("0_base", {}),
    ("1_z_spectral", {"z_spectral_modes": 64}),
    ("2_freq_rank", {"freq_mix_rank": 8}),
    ("3_side_paths", {"open_side_paths": True}),
    ("4_width96", {"width": 96}),
    ("5_layers8", {"n_layers": 8, "spec_dropout_layers": list(range(8)), "vertical_dropout_layers": list(range(8))}),
    ("6_hidden_vertical192", {"hidden_vertical": 192}),
]


def cumulative():
    out, overrides = [], {}
    for name, change in CONFIGS:
        overrides = {**overrides, **change}
        out.append((name, dict(overrides)))
    return out


def measure(overrides, patch, depth, steps):
    """One configuration in this process: {params, peak_gb, seconds_per_step}."""
    os.environ["FFNOML_OUTPUT_MODE"] = "level_fraction"
    os.environ["FFNOML_ACTIVE_ATOMS"] = "H"
    os.environ["FFNOML_MODEL_CONFIG"] = json.dumps(overrides)
    sys.path.insert(0, REPO)
    import torch
    import config
    from ffno_model_factory import create_ffno3d

    device = torch.device("cuda")
    torch.manual_seed(0)
    model = create_ffno3d(6, 6).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(1, 6, depth, patch, patch, device=device)
    z = torch.linspace(14.0, -1.0, depth, device=device)[None, :, None, None].expand(1, depth, patch, patch).contiguous()
    dx = dy = 4.8e4

    def step():
        optimizer.zero_grad(set_to_none=True)
        model(x, z, dx, dy).square().mean().backward()
        optimizer.step()

    step()                                              # warm-up: cuFFT plans, cuBLAS handles, optimiser state
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    return {"params": sum(p.numel() for p in model.parameters()), "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
            "seconds_per_step": (time.time() - started) / steps, "model_config": dict(config.MODEL_CONFIG, species=None)}


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--patch", type=int, default=40, help="patch width and height (training: 40)")
    cli.add_argument("--depth", type=int, default=496)
    cli.add_argument("--steps", type=int, default=5)
    cli.add_argument("--config", action="append", help="names to run (default: all)")
    cli.add_argument("--out", help="also write the results here, one JSON line per configuration")
    cli.add_argument("--one", help=argparse.SUPPRESS)  # internal: JSON overrides to measure in this process
    args = cli.parse_args()

    if args.one is not None:
        print(json.dumps(measure(json.loads(args.one), args.patch, args.depth, args.steps)))
        return

    results, base = [], None
    for name, overrides in cumulative():
        if args.config and name not in args.config:
            continue
        run = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", json.dumps(overrides), "--patch",
                              str(args.patch), "--depth", str(args.depth), "--steps", str(args.steps)],
                             capture_output=True, text=True)
        lines = [line for line in run.stdout.splitlines() if line.startswith("{")]
        if run.returncode or not lines:
            oom = "out of memory" in run.stderr.lower()
            result = {"name": name, "overrides": overrides, "error": "out of memory" if oom else run.stderr[-2000:]}
            print(f"{name:22s} {'out of memory' if oom else 'failed'}", flush=True)
        else:
            result = {"name": name, "overrides": overrides, **json.loads(lines[-1])}
            base = base or (result if name == "0_base" else None)
            ratio = (f"  x{result['seconds_per_step'] / base['seconds_per_step']:.2f} time, "
                     f"x{result['peak_gb'] / base['peak_gb']:.2f} memory" if base else "")
            print(f"{name:22s} {result['params'] / 1e6:7.2f} M params  {result['peak_gb']:6.2f} GB  "
                  f"{result['seconds_per_step']:6.3f} s/step{ratio}", flush=True)
        results.append(result)
    if args.out:
        with open(args.out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in results)


if __name__ == "__main__":
    main()
