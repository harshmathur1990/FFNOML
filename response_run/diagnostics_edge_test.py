# Do identical input columns give identical FFNO outputs, and where do they differ? One real column repeated over the grid.
import torch, h5py, numpy as np, sys
sys.path.insert(0, ".")
from models.ffno_model import FFNO3D
from config import MODEL_CONFIG
torch.set_num_threads(16)
ck = torch.load("training_FFNO3D_zscale_expand_lognlte/3D_sim_train_H.pt", map_location="cpu", weights_only=False)
ns = ck["normalization_stats"]; cfg = dict(MODEL_CONFIG); cfg["checkpoint_blocks"] = False
m = FFNO3D(**cfg); m.load_state_dict(ck["model_state"]); m.eval()
f = h5py.File([x for x in __import__("glob").glob("IO/3D_sim_test*atomsH*")][0])
g = f["test_2"]; x0 = g["inputs"][0, :, :, 20, 20]; z0 = g["z_scale"][0, :, 20, 20]; dx = float(g["dx"][0]); dy = float(g["dy"][0])
std = np.asarray(ns["std_Y"])
for n in (5, 9, 13, 21):
    x = torch.tensor(np.repeat(np.repeat(x0[:, :, None, None], n, 2), n, 3))[None]
    z = torch.tensor(np.repeat(np.repeat(z0[:, None, None], n, 1), n, 2))[None]
    with torch.no_grad(): y = m(x, z, torch.tensor(dx), torch.tensor(dy))[0].numpy()   # normalized log10 n
    dex = y * std[:, None, None, None]
    c = n // 2; dist = np.minimum.reduce(np.indices((n, n)) .tolist() + [(n - 1 - np.indices((n, n))).min(0).tolist()]) if False else None
    ix, iy = np.indices((n, n)); edge = np.minimum.reduce([ix, iy, n - 1 - ix, n - 1 - iy])
    out = []
    for d in range(edge.max() + 1):
        mask = edge == d
        out.append(float(np.abs(dex[:, :, mask] - dex[:, :, c, c][:, :, None]).max()))
    print(f"n={n}: max |log10 n - centre| [dex] by distance from the edge:", [round(v, 6) for v in out], flush=True)
