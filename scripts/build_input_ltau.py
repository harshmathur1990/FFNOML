"""Build the native log tau500 input atmosphere for the FFNO inversion from a Bifrost snapshot.

Each column's T, velocities and B (heights 0..466 of the Bifrost netCDF) are interpolated linearly in log tau500
(np.interp) onto the 95-point grid log tau500 = -8.0 ... 1.4, step 0.1.

The supplementary MULTI3D file stores ltau500 with x and y swapped relative to the Bifrost netCDF: its element
[0, a, b, :] is the column at netCDF position (x=b, y=a). Pixel (x, y) of the atmosphere therefore takes
ltau500[0, y, x, :]. (The previous input-mhs-ltau.nc used ltau500[0, x, y, :], i.e. each column's temperature on
the optical-depth scale of the mirrored column; `--as-stored` reproduces that file exactly.)

Usage:
    python build_input_ltau.py ATMOSPHERE.nc SUPPLEMENTARY.nc OUTPUT.nc [--as-stored]
"""
import argparse

import h5py
import numpy as np

FIELDS = ("temperature", "velocity_x", "velocity_y", "velocity_z", "B_x", "B_y", "B_z")
HEIGHTS = np.arange(0, 467, 1)
GRID = np.arange(-8.0, 1.45, 0.1)          # as in the original input-mhs-ltau.nc (95 points)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("atmosphere")
    parser.add_argument("supplementary")
    parser.add_argument("output")
    parser.add_argument("--as-stored", action="store_true",
                        help="use ltau500 without swapping x and y (reproduces the old, misregistered input)")
    args = parser.parse_args()

    with h5py.File(args.supplementary, "r") as f:
        ltau = f["ltau500"][0][..., HEIGHTS]                     # (a, b, height) in the file's layout
    if not args.as_stored:
        ltau = np.transpose(ltau, (1, 0, 2))                     # -> (x, y, height) of the atmosphere
    nx, ny, _ = ltau.shape
    if not np.all(np.diff(ltau, axis=2) > 0):
        raise ValueError("ltau500 must increase with height index (top first) in every column")

    out = {name: np.empty((nx, ny, len(GRID))) for name in FIELDS}
    with h5py.File(args.atmosphere, "r") as f:
        for name in FIELDS:
            field = f[name][0][..., HEIGHTS].astype(np.float64)   # (x, y, height)
            if field.shape[:2] != (nx, ny):
                raise ValueError(f"{name} has horizontal shape {field.shape[:2]}, ltau500 {(nx, ny)}")
            for x in range(nx):
                for y in range(ny):
                    out[name][x, y] = np.interp(GRID, ltau[x, y], field[x, y])
            print(f"{name} done", flush=True)

    with h5py.File(args.output, "w") as f:
        f["log10tau500"] = GRID
        for name in FIELDS:
            f[name] = out[name]
    print(f"written {args.output}")


if __name__ == "__main__":
    main()
