"""Build the native log10 column-mass input atmosphere for the FFNO forward model and inversion from a Bifrost snapshot.

Column mass per column, from the top: m(z) = P_top/g + integral rho dz (trapezoid; as HSEMHS make_input_mhs_logm.py).
T (linear in ln T), velocities and B are interpolated in log10 m onto a common uniform axis, which must lie inside every
column's range. Layout of build_input_ltau.py with log10m [m in kg m-2] for log10tau500: h5py (x, y, m), top first.

Usage:
    python build_input_logm.py ATMOSPHERE.nc OUTPUT.nc [--lm-min -4.9] [--lm-max 2.6] [--dlm 0.1] [--g 274.0]
"""
import argparse

import h5py
import numpy as np

FIELDS = ("velocity_x", "velocity_y", "velocity_z", "B_x", "B_y", "B_z")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("atmosphere")
    parser.add_argument("output")
    parser.add_argument("--lm-min", type=float, default=-4.9)
    parser.add_argument("--lm-max", type=float, default=2.6)
    parser.add_argument("--dlm", type=float, default=0.1)
    parser.add_argument("--g", type=float, default=274.0)
    args = parser.parse_args()

    grid = args.lm_min + args.dlm * np.arange(int(round((args.lm_max - args.lm_min) / args.dlm)) + 1)
    with h5py.File(args.atmosphere, "r") as f:
        z = f["z"][0, :].astype(np.float64)
        if not np.all(np.diff(z) < 0):
            raise ValueError("z must decrease (top first)")
        rho = f["density"][0].astype(np.float64)                 # (x, y, height)
        pressure = f["gas_pressure"][0].astype(np.float64)
        logm = np.empty_like(rho)
        logm[:, :, 0] = pressure[:, :, 0] / args.g
        logm[:, :, 1:] = logm[:, :, :1] + np.cumsum(0.5 * (rho[:, :, :-1] + rho[:, :, 1:]) * -np.diff(z), axis=2)
        logm = np.log10(logm)
        if grid[0] < logm[:, :, 0].max() or grid[-1] > logm[:, :, -1].min():
            raise ValueError(f"the axis {grid[0]} .. {grid[-1]} must lie inside every column: top up to "
                             f"{logm[:, :, 0].max():.3f}, bottom from {logm[:, :, -1].min():.3f}")
        nx, ny, _ = logm.shape

        def on_axis(field):
            out = np.empty((nx, ny, len(grid)))
            for x in range(nx):
                for y in range(ny):
                    out[x, y] = np.interp(grid, logm[x, y], field[x, y])
            return out

        out = {"temperature": np.exp(on_axis(np.log(f["temperature"][0].astype(np.float64))))}
        print("temperature done", flush=True)
        for name in FIELDS:
            out[name] = on_axis(f[name][0].astype(np.float64))
            print(f"{name} done", flush=True)

    with h5py.File(args.output, "w") as f:
        f["log10m"] = grid
        f["log10m"].attrs["units"] = "kg m-2"
        for name, value in out.items():
            f[name] = value
    print(f"written {args.output}")


if __name__ == "__main__":
    main()
