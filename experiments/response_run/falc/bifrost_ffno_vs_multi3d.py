# The FFNO on Bifrost en024048_hion 385 against MULTI3D, by height: the same quantities that are off for FALC (Ca II n5/n3, n3; H n2,
# n3/n2, total H). Ratios FFNO / MULTI3D per pixel on every 4th column: median and 5-95 percentiles.
import h5py, numpy as np
B = "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385/"; T = "/mn/stornext/d9/data/harshm/FFNOML/training_FFNO3D_zscale_expand_lognlte/"
v = np.array(open(B + "mesh").read().split(), dtype=float)
nx = int(v[0]); ny = int(v[1 + nx]); nz = int(v[2 + nx + ny]); z = v[3 + nx + ny:3 + nx + ny + nz] / 1e5
def truth(atom):
    return np.memmap(B + atom + "/out_pop", dtype=np.float32, mode="r", order="F", shape=(nx, ny, nz, 6))
def ffno(atom):
    return h5py.File(T + f"output_3D_sim_s5_en024048_hion_385_FFNO3D_{atom}.hdf5")["nlte_populations"]
tc, th, fc, fh = truth("CA"), truth("H"), ffno("CA"), ffno("H")
a = np.memmap(B + "atm3d", dtype="f4", mode="r", shape=(12, nz, ny, nx))
def stats(x):
    return "%6.2f (%5.2f,%6.2f)" % (np.median(x), np.quantile(x, .05), np.quantile(x, .95))
print("z_km  medT | Ca n5/n3 FFNO/M3D | Ca n3 FFNO/M3D | H n2 FFNO/M3D | H n3/n2 FFNO/M3D | total H FFNO/M3D | total Ca FFNO/M3D")
for h in (3000, 2000, 1600, 1349, 1000, 700, 500, 250, 100, 0, -100):
    k = int(np.argmin(abs(z - h))); s = (slice(None, None, 4), slice(None, None, 4), k)
    C = np.array(tc[s + (slice(None),)]) * 1e6; H = np.array(th[s + (slice(None),)]) * 1e6
    FC = np.array(fc[s + (slice(None),)]).astype(float); FH = np.array(fh[s + (slice(None),)]).astype(float)
    print(int(z[k]), int(np.median(a[1, k, ::4, ::4])), "|", stats((FC[..., 4] / FC[..., 2]) / (C[..., 4] / C[..., 2])), "|", stats(FC[..., 2] / C[..., 2]), "|", stats(FH[..., 1] / H[..., 1]), "|",
          stats((FH[..., 2] / FH[..., 1]) / (H[..., 2] / H[..., 1])), "|", stats(FH.sum(-1) / H.sum(-1)), "|", stats(FC.sum(-1) / C.sum(-1)))
