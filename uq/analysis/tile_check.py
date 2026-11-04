import sys, h5py, numpy as np
sys.path.insert(0, "/mn/stornext/d9/data/harshm/FFNOML")
sys.path.insert(0, "/mn/stornext/d9/data/harshm/FFNOML/scripts")
from uq_error_atlas import load_truth
import os
from config import MODEL_DIR
lt, z, _ = load_truth("/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385/H")
with h5py.File("uq/test_full_det.h5") as f: full = f["det_log"][...]
with h5py.File("uq/test_tile192_en024048_hion_385_H.h5") as f: t192 = f["det_log"][...]
with h5py.File("uq/test_det_en024048_hion_385_H.h5") as f: t40 = f["det_log"][...]
with h5py.File(os.path.join(MODEL_DIR, "3D_sim_predict_en024048_hion_385.hdf5")) as f:
    logT = np.transpose(f["inputs"][0, 0], (1, 2, 0))
zz = np.broadcast_to(z[None, None], logT.shape)
reg = {"phot": zz < 0.3, "chrom": (logT < 4.5) & (zz >= 0.3), "TR": (logT >= 4.5) & (logT <= 5.5), "corona": logT > 5.5}
for name, p in (("full", full), ("tile192", t192), ("tile40", t40)):
    d = np.abs(p - full); e = np.abs(p - lt)
    print(name, "|p-full| median %.4f p99 %.4f" % (np.median(d), np.percentile(d, 99)))
    for r, m in reg.items():
        print("   %-6s MAE vs truth per level:" % r, " ".join("%.3f" % e[..., l][m].mean() for l in range(e.shape[-1])))
