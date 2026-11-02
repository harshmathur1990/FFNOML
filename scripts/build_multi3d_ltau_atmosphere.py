"""Build a native log tau500 forward atmosphere (the format of FFNOInversion's forward_atmosphere.h5) from the Bifrost
snapshot and its MULTI3D hydrogen populations, for scripts/column_response.jl.

Everything is taken from the simulation instead of HSEMHS/FFNO: T, velocities, pgas, rho, ne and z from the Bifrost
netCDF, H populations (6 levels) from MULTI3D output_aux.hdf5, log tau500 from the supplementary file. chi500 follows
from log tau500(z): chi500 = ln 10 * tau * d log tau / d(-z). MULTI3D used heights 0..466 of the 496 in the netCDF.

Orientations, relative to the netCDF (x, y), verified by correlating maps:
    supplementary ltau500[0, a, b, :]      = column (x=b, y=a)                      (T(z=0) vs ltau500: r = 0.97)
    output_aux populations[:, a, b, :]     = column (x=a, y=503-b)                  (log n1 vs T: r = -0.83)
    output_ray intensity[a, b, :]          = column (x=a, y=503-b)                  (H-alpha wing vs T(z=0): r = 0.76)

Every column is interpolated linearly in log tau500 onto log tau500 = -8.0 ... 1.4, step 0.1 (as build_input_ltau.py),
populations, pgas, rho, ne and chi500 in their logarithm.

Usage:
    python build_multi3d_ltau_atmosphere.py ATMOSPHERE.nc SUPPLEMENTARY.nc OUTPUT_AUX.hdf5 OUTPUT.h5
"""
import argparse

import h5py
import numpy as np

HEIGHTS = np.arange(0, 467)
GRID = np.arange(-8.0, 1.45, 0.1)
LINEAR = {"temperature": "temperature", "vx": "velocity_x", "vy": "velocity_y", "vz": "velocity_z"}
LOGARITHMIC = {"pgas": "gas_pressure", "rho": "density", "ne": "electron_density"}


def weights(ltau):
    """Index and weight of the left neighbour of every GRID point in every column (np.interp, clamped)."""
    n = ltau.shape[-1]
    index = np.empty(ltau.shape[:-1] + GRID.shape, dtype=np.int64)
    for c in np.ndindex(ltau.shape[:-1]):
        index[c] = np.clip(np.searchsorted(ltau[c], GRID) - 1, 0, n - 2)
    left = np.take_along_axis(ltau, index, -1); right = np.take_along_axis(ltau, index + 1, -1)
    w = np.clip((GRID - left) / (right - left), 0, 1)
    return index, w


def interpolate(field, index, w):
    """field (..., height) -> (..., GRID) with the weights of weights()."""
    return (1 - w) * np.take_along_axis(field, index, -1) + w * np.take_along_axis(field, index + 1, -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("atmosphere"); parser.add_argument("supplementary"); parser.add_argument("aux")
    parser.add_argument("output")
    args = parser.parse_args()

    with h5py.File(args.supplementary, "r") as f:
        ltau = np.transpose(f["ltau500"][0][..., HEIGHTS], (1, 0, 2))           # (x, y, height)
    if not np.all(np.diff(ltau, axis=2) > 0):
        raise ValueError("ltau500 must increase with height index (top first) in every column")
    nx, ny, _ = ltau.shape
    index, w = weights(ltau); print("weights done", flush=True)

    out = {}
    with h5py.File(args.atmosphere, "r") as f:
        z = f["z"][0][HEIGHTS]
        for name, source in LINEAR.items():
            out[name] = interpolate(f[source][0][..., HEIGHTS].astype(np.float64), index, w)
            print(name, "done", flush=True)
        for name, source in LOGARITHMIC.items():
            out[name] = np.exp(interpolate(np.log(f[source][0][..., HEIGHTS].astype(np.float64)), index, w))
            print(name, "done", flush=True)
        x_m = f["x"][:] * 1.0; y_m = f["y"][:] * 1.0
    out["z"] = interpolate(np.broadcast_to(z, ltau.shape), index, w)
    chi500 = np.log(10) * 10.0**ltau * np.gradient(ltau, -z, axis=2)
    if not np.all(chi500 > 0):
        raise ValueError("chi500 is not positive everywhere")
    out["chi500"] = np.exp(interpolate(np.log(chi500), index, w)); del chi500
    out["logtau500"] = np.broadcast_to(GRID, ltau.shape[:2] + GRID.shape).copy()
    out["vturb"] = np.zeros_like(out["temperature"])

    with h5py.File(args.aux, "r") as f:
        populations = f["atom_H/populations"][...]                                   # (level, a, b, height)
    populations = np.transpose(populations[:, :, ::-1, :][..., HEIGHTS], (1, 2, 3, 0))    # (x, y, height, level)
    H = np.empty((nx, ny, len(GRID), populations.shape[-1]))
    for level in range(populations.shape[-1]):
        H[..., level] = np.exp(interpolate(np.log(populations[..., level].astype(np.float64)), index, w))
    del populations; print("populations done", flush=True)

    with h5py.File(args.output, "w") as f:
        f.attrs["axis_order"] = "time,z,y,x; populations=level,z,y,x"
        f.attrs["vertical_coordinate"] = "log10tau500"; f.attrs["vertical_order"] = "top_first"
        f.attrs["description"] = ("Bifrost en024048_hion snapshot with MULTI3D H populations on native log tau500; "
                                  "built by scripts/build_multi3d_ltau_atmosphere.py")
        f["logtau_500"] = GRID; f["x_m"] = x_m; f["y_m"] = y_m
        for name, value in out.items():
            f.create_dataset(name, data=value[..., None], chunks=(32, 32, len(GRID), 1))
        f.create_dataset("populations/H", data=H, chunks=(32, 32, len(GRID), H.shape[-1]))
    print("written", args.output)


if __name__ == "__main__":
    main()
