"""Column-crop context test on nw012023: bias/MAE of levels 2-5 by height band, full vs cropped."""
import os, sys
import h5py
import numpy as np
sys.path.insert(0, "/mn/stornext/d9/data/harshm/FFNOML/scripts"); sys.path.insert(0, "/mn/stornext/d9/data/harshm/FFNOML")
from uq_error_atlas import load_truth
B = "/mn/stornext/d9/data/harshm/bifrost_data"
bands = [(-1.0, 0.0), (0.0, 0.3), (0.3, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 14.3)]
crops = {"full": None, "-0.97..7.98": ("-0.97", "7.98"), "0..top": ("0.0", "20"), "-0.97..4": ("-0.97", "4.0")}
for snap in sys.argv[1:]:
    lt, z, _ = load_truth(f"{B}/nw012023/{snap}/H")
    res = {}
    for k, zr in crops.items():
        p = f"uq/mc/nw012023_{snap}_H.h5" if zr is None else f"uq/context/nw012023_{snap}_H_z{zr[0]}_{zr[1]}.h5"
        if not os.path.exists(p):
            continue
        sel = np.ones_like(z, bool) if zr is None else (z >= float(zr[0])) & (z <= float(zr[1]))
        e = h5py.File(p)["det_log"][...] - lt[:, :, sel]
        res[k] = (e[..., 1:5].mean(-1), z[sel])
    print(f"== nw012023_{snap}: bias/MAE levels 2-5 by band   " + "  ".join(f"{k:>14s}" for k in res))
    for lo, hi in bands:
        cells = []
        for e, zz in res.values():
            m = (zz >= lo) & (zz < hi)
            cells.append(f"{e[:, :, m].mean():+.3f}/{np.abs(e[:, :, m]).mean():.3f}" if m.any() else "-")
        print(f"   {lo:5.1f}..{hi:5.1f}                         " + "  ".join(f"{c:>14s}" for c in cells))
