# V5b of docs/viewing-angle/README.md: FFNO H-alpha from the Bifrost cube (response_run/view_check/forward_bifrost_view.jl,
# every 4th column; views mu = 1, mu = 0.8 inclined and plane-parallel at azimuth 180 deg) against MULTI3D: the vertical
# rays (Multi-3D-H_6_level_populations/output_ray.hdf5) and the mu = 0.8 rays (output_ray_mu08.hdf5). Both MULTI3D maps
# are footprint (x = a, y = 503 - b). Profiles are compared on dlambda from each model's own mean-profile minimum (the
# FFNO atom's line centre differs from MULTI3D's) after converting MULTI3D to per nm. Prints, for every FFNO view
# against each MULTI3D view, the map correlation and relative rms at the line core and in the wings.
import sys, h5py, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ffno_path = sys.argv[1] if len(sys.argv) > 1 else "response_run/view_check/forward_view_s4.h5"
out = sys.argv[2] if len(sys.argv) > 2 else "figures/viewing_angle/ffno_vs_multi3d_mu08.png"
M3D = {"MULTI3D mu=1": "/mn/stornext/d9/data/harshm/halpha_output/Multi-3D-H_6_level_populations/output_ray.hdf5",
       "MULTI3D mu=0.8": "/mn/stornext/d9/data/harshm/halpha_output/output_ray_mu08.hdf5"}
C = 299792458.0
with h5py.File(ffno_path) as f:
    xi = f["x_index"][:]; yi = f["y_index"][:]; wf = f["wavelength_air_m"][:] * 1e9
    ffno = {name: f[f"{name}/intensity"][:].transpose(2, 1, 0) for name in ("mu1", "inclined", "plane_parallel")}  # Julia (nl, nx, ny)
ref = {}
for name, path in M3D.items():
    with h5py.File(path) as f:
        w = f["wavelength"][:]; sel = np.where((w > 655.6) & (w < 657.2))[0]
        I = f["intensity"][:, :, sel[0]:sel[-1] + 1]                       # (a, b, nl)
        I = I[np.ix_(xi, 503 - yi)]                                       # footprints (x, y) of the FFNO columns
        lam = w[sel]
        ref[name] = (lam, np.moveaxis(I * C / (lam * 1e-9) ** 2 * 1e-9 * 1e-3, 2, 0))   # kW m^-2 sr^-1 nm^-1, (nl, nx, ny)

def centred(lam, I):
    return lam - lam[np.argmin(I.mean(axis=(1, 2)))]
grid = np.linspace(-0.45, 0.45, 91)
def on_grid(lam, I):
    d = centred(lam, I)
    return np.stack([np.interp(grid, d, I[:, i, j]) for i in range(I.shape[1]) for j in range(I.shape[2])], 1) \
        .reshape(len(grid), I.shape[1], I.shape[2])
F = {k: on_grid(wf, v) for k, v in ffno.items()}
R = {k: on_grid(*v) for k, v in ref.items()}
core = np.abs(grid) <= 0.03; wing = (np.abs(grid) >= 0.08) & (np.abs(grid) <= 0.15)
def stats(a, b, mask):
    A = a[mask].mean(0).ravel(); B = b[mask].mean(0).ravel()
    return np.corrcoef(A, B)[0, 1], np.sqrt(np.mean((A / B - 1) ** 2))
print("map correlation / relative rms of FFNO against MULTI3D (core |dl| <= 0.03 nm, wings 0.08-0.15 nm):")
for fn in F:
    for rn in R:
        rc, ec = stats(F[fn], R[rn], core); rw, ew = stats(F[fn], R[rn], wing)
        print(f"  FFNO {fn:15s} vs {rn:15s}: core r={rc:.3f} rms={ec:.3f}   wings r={rw:.3f} rms={ew:.3f}")

fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), constrained_layout=True)
ax = axes[0, 0]
for (k, v), c in zip(F.items(), ["#2a78d6", "#eb6834", "#1baf7a"]):
    ax.plot(grid, v.mean((1, 2)), color=c, lw=2, label=f"FFNO {k}")
for (k, v), ls in zip(R.items(), ["--", ":"]):
    ax.plot(grid, v.mean((1, 2)), "k", ls=ls, lw=1.6, label=k)
ax.set_xlabel("dlambda from the mean-profile minimum [nm]"); ax.set_ylabel("mean I [kW m^-2 sr^-1 nm^-1]")
ax.legend(fontsize=8, frameon=False); ax.set_title("mean profiles")
panels = [(axes[0, 1], "MULTI3D mu=1", R["MULTI3D mu=1"]), (axes[0, 2], "FFNO mu=1", F["mu1"]),
          (axes[1, 0], "MULTI3D mu=0.8", R["MULTI3D mu=0.8"]), (axes[1, 1], "FFNO mu=0.8 inclined", F["inclined"]),
          (axes[1, 2], "FFNO mu=0.8 plane-parallel", F["plane_parallel"])]
lo, hi = np.percentile(R["MULTI3D mu=1"][core].mean(0), [1, 99])
for a, n, v in panels:
    im = a.imshow(v[core].mean(0).T, origin="lower", cmap="inferno", vmin=lo, vmax=hi)
    a.set_title(f"{n}: line core (|dl| <= 0.03 nm)", fontsize=10); a.set_xlabel("x [px of 4]"); a.set_ylabel("y [px of 4]")
fig.colorbar(im, ax=axes[:, 1:], shrink=0.6, label="I [kW m^-2 sr^-1 nm^-1]")
fig.suptitle("H-alpha: FFNO (Bifrost cube, every 4th column) vs MULTI3D, vertical and mu = 0.8 (azimuth 180 deg, tilt toward -x)")
fig.savefig(out, dpi=90); print("wrote", out)
