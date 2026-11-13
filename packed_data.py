"""One HDF5 file holding every MULTI3D snapshot input a run reads, so a cluster needs one copy instead of bifrost_data/.

    python packed_data.py OUT.h5 --ini scripts/learning_curve.ini [--snapshot sim/snap ...] [--atoms H,CA] [--data-dir D]

packs every snapshot the input file's experiments use (train, val and held_out of every section) and any --snapshot,
read from D/bifrost_data (default config.DATA_DIR), and writes OUT.h5.json, a manifest of what it holds. Arrays keep
MULTI3D's Fortran order, so their axes are reversed: the atmosphere is (z, y, x) and populations (level, z, y, x).

    /<sim>/<snap>/atmos/{ne, temp, vx, vy, vz, rho}    atm3d, cgs
    /<sim>/<snap>/mesh/{x, y, z}                        mesh [cm], read as pipeline.read_mesh does; attrs dx, dy [m]
    /<sim>/<snap>/<atom>/{n, nstar}                     out_pop NLTE and LTE populations [cm^-3]
    /<sim>/<snap>/<atom>/z                              out_par geometry z [cm]
    /<sim>/<snap>/<atom>/tau500                         the atom's tau500, or H's when it has none

With FFNOML_PACKED_DATA=OUT.h5, pipeline.py --build and --buildforpredict and scripts/population_error_diagnostic.py
read the snapshots from it instead of bifrost_data/, getting the same arrays as from the files.
"""
import argparse
import configparser
import json
import os
import sys

import h5py
import numpy as np

ATMOS = ("ne", "temp", "vx", "vy", "vz", "rho")


def _group(f, sim, snap, atom=None):
    path = f"{sim}/{snap}" + (f"/{atom}" if atom else "")
    if path not in f:
        raise KeyError(f"{f.filename} holds no {path}")
    return f[path]


def _fortran(dataset):
    """A stored array back in MULTI3D's axis order (x, y, z[, level])."""
    return dataset[()].T


def atmosphere(path, sim, snap):
    """atm3d as Multi3dAtmos reads it, (x, y, z) per quantity, and the mesh: (fields, x, y, z, dx, dy)."""
    with h5py.File(path, "r") as f:
        g = _group(f, sim, snap)
        fields = {q: _fortran(g["atmos"][q]) for q in ATMOS}
        mesh = g["mesh"]
        return fields, mesh["x"][()], mesh["y"][()], mesh["z"][()], float(mesh.attrs["dx"]), float(mesh.attrs["dy"])


def populations(path, sim, snap, atom, *, lte=True):
    """out_pop as Multi3dOut reads it, (x, y, z, level) [cm^-3], and out_par's z [cm]: (n, nstar or None, z)."""
    with h5py.File(path, "r") as f:
        g = _group(f, sim, snap, atom)
        return _fortran(g["n"]), _fortran(g["nstar"]) if lte else None, g["z"][()]


def tau500(path, sim, snap, atom):
    """The atom's tau500, (x, y, z)."""
    with h5py.File(path, "r") as f:
        return _fortran(_group(f, sim, snap, atom)["tau500"])


def holds(path, sim, snap, atom=None):
    with h5py.File(path, "r") as f:
        return (f"{sim}/{snap}" + (f"/{atom}" if atom else "")) in f


def ini_snapshots(ini):
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#",))
    if not parser.read(ini):
        raise FileNotFoundError(ini)
    found = []
    for name in parser.sections():
        for key in ("train", "val", "held_out"):
            found += parser[name].get(key, "").split()
    return found


def pack(out, snapshots, atoms, data_dir):
    from helita.sim.multi3d import Multi3dAtmos, Multi3dOut
    from pipeline import compute_dx_dy, read_mesh

    manifest = {"atoms": atoms, "snapshots": {}}
    with h5py.File(out + ".tmp", "w") as f:
        f.attrs["data_dir"] = data_dir
        for item in snapshots:
            sim, snap = item.split("/")
            src = os.path.join(data_dir, "bifrost_data", sim, snap)
            g = f.create_group(f"{sim}/{snap}")
            g.attrs["source"] = src
            nx, ny, nz, x, y, z = read_mesh(os.path.join(src, "mesh"))
            dx, dy = compute_dx_dy(os.path.join(src, "mesh"))
            mesh = g.create_group("mesh")
            mesh["x"], mesh["y"], mesh["z"] = x, y, z
            mesh.attrs["dx"], mesh.attrs["dy"] = dx, dy
            atm = Multi3dAtmos(os.path.join(src, "atm3d"), nx, ny, nz)
            for q in ATMOS:
                g[f"atmos/{q}"] = np.asarray(getattr(atm, q)).T
            entry = {"shape": [nx, ny, nz], "atoms": {}}
            for atom in atoms:
                m3d = Multi3dOut(directory=os.path.join(src, atom), printinfo=False)
                m3d.readall()
                if m3d.atom.n.shape[:3] != (nx, ny, nz):
                    raise ValueError(f"{src}/{atom}: populations {m3d.atom.n.shape} on a {nx}x{ny}x{nz} mesh")
                tau = m3d.atmos.tau500
                if tau is None:
                    h = Multi3dOut(directory=os.path.join(src, "H"), printinfo=False)
                    h.readall()
                    tau = h.atmos.tau500
                if tau is None:
                    raise FileNotFoundError(f"{src}: no tau500 for {atom} or H")
                a = g.create_group(atom)
                for name, array in (("n", m3d.atom.n), ("nstar", m3d.atom.nstar), ("tau500", tau)):
                    array = np.asarray(array)
                    if not np.isfinite(array).all():   # e.g. a MULTI3D run that diverged
                        raise ValueError(f"{src}/{atom}: {name} is not finite everywhere")
                    a[name] = array.T
                a["z"] = np.asarray(m3d.geometry.z)
                entry["atoms"][atom] = {"levels": int(m3d.atom.n.shape[3])}
            manifest["snapshots"][item] = entry
            print(f"packed {item}: {nx}x{ny}x{nz}, {', '.join(atoms)}", flush=True)
    os.replace(out + ".tmp", out)
    with open(out + ".json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {out} ({os.path.getsize(out) / 2**30:.1f} GiB) and {out}.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--ini", action="append", default=[], help="pack the snapshots of this experiments file")
    parser.add_argument("--snapshot", action="append", default=[], help="sim/snap, repeatable")
    parser.add_argument("--atoms", default="H,CA")
    parser.add_argument("--data-dir", default=None, help="directory holding bifrost_data/ (default: config.DATA_DIR)")
    args = parser.parse_args()
    if args.data_dir is None:
        from config import DATA_DIR
        args.data_dir = DATA_DIR
    snapshots = list(dict.fromkeys(sum((ini_snapshots(i) for i in args.ini), []) + args.snapshot))
    if not snapshots:
        sys.exit("nothing to pack: give --ini or --snapshot")
    pack(args.out, snapshots, args.atoms.split(","), args.data_dir)


if __name__ == "__main__":
    main()
