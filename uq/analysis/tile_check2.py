import sys, h5py, numpy as np
sys.path.insert(0, "/mn/stornext/d9/data/harshm/FFNOML/scripts"); sys.path.insert(0, "/mn/stornext/d9/data/harshm/FFNOML")
from uq_error_atlas import load_truth, REGIMES
from pipeline import load_pred_data
B = "/mn/stornext/d9/data/harshm/bifrost_data"
for name, cs in (("nw012023_915", "nw012023/915"), ("ch012012_hion_910", "ch012012_hion/910")):
    lt, z, _ = load_truth(f"{B}/{cs}/H")
    T = np.log10(load_pred_data(mesh_file=f"{B}/{cs}/mesh", atmos_file=f"{B}/{cs}/atm3d")[2])
    zz = np.broadcast_to(z[None, None], T.shape)
    full = h5py.File(f"uq/mc/{name}_H.h5")["det_log"][...]
    t192 = h5py.File(f"uq/tilecheck/{name}_H_tile192.h5")["det_log"][...]
    d = np.abs(t192 - full)
    print(f"{name}: |tile192 - full| median {np.median(d):.4f} p99 {np.percentile(d, 99):.4f} max {d.max():.3f}")
    for r, fn in REGIMES.items():
        m = fn(T, zz)
        if m.any():
            print(f"   {r:12s} MAE L2-5 full {np.abs(full - lt)[m][:, 1:5].mean():.3f}  tile192 {np.abs(t192 - lt)[m][:, 1:5].mean():.3f}")
