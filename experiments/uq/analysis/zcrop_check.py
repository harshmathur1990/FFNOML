"""Column-crop context test: error by height band for full-column vs cropped predictions."""
import sys
import h5py
import numpy as np
sys.path.insert(0, "/mn/stornext/d9/data/harshm/FFNOML/scripts")
sys.path.insert(0, "/mn/stornext/d9/data/harshm/FFNOML")
from uq_error_atlas import load_truth
B = "/mn/stornext/d9/data/harshm/bifrost_data"
bands = [(-2.5, -1.0), (-1.0, -0.5), (-0.5, 0.0), (0.0, 0.3), (0.3, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 14.3)]
for name, cs in (("ch024031_by200bz005_450", "ch024031_by200bz005/450"), ("en024031_by100_helium_109", "en024031_by100_helium/109")):
    lt, z, _ = load_truth(f"{B}/{cs}/H")
    runs = {"full": (f"uq/mc/{name}_H.h5", np.ones_like(z, bool))}
    runs["z>=-0.97"] = (f"uq/context/{name}_H_zge-0.97.h5", z >= -0.97)
    runs["z<=7.98"] = (f"uq/context/{name}_H_zle7.98.h5", z <= 7.98)
    print(f"== {name}: bias of levels 2-5 (mean over levels) / MAE, by height band")
    print("   band(Mm)       " + "   ".join(f"{k:>16s}" for k in runs))
    res = {}
    for k, (p, sel) in runs.items():
        e = h5py.File(p)["det_log"][...] - lt[:, :, sel]
        res[k] = (e[..., 1:5].mean(-1), z[sel])
    for lo, hi in bands:
        cells = []
        for k, (e, zz) in res.items():
            m = (zz >= lo) & (zz < hi)
            cells.append(f"{e[:, :, m].mean():+.3f}/{np.abs(e[:, :, m]).mean():.3f}" if m.any() else "               -")
        print(f"   {lo:5.1f}..{hi:5.1f}   " + "   ".join(f"{c:>16s}" for c in cells))
