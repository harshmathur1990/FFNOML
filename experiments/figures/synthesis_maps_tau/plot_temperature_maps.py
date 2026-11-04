"""Temperature maps from forward_atmosphere.h5 at fixed log tau500 and at fixed geometric height z."""
import sys, h5py, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
path, outdir = sys.argv[1], sys.argv[2]
with h5py.File(path, "r") as f:
    lt = f["logtau_500"][:]
    T = f["temperature"][:, :, :, 0]            # (x, y, depth), top first
    z = f["z"][:, :, :, 0] / 1e3                # km

def panel(ax, img, title):
    good = np.isfinite(img)
    lo, hi = np.percentile(img[good], [1, 99])
    im = ax.imshow(img.T, origin="lower", cmap="inferno", vmin=lo, vmax=hi)
    hot = (img[good] > 2e4).mean() * 100
    extra = f", T>2e4 K: {hot:.2f}%" if hot > 0 else ""
    missing = (~good).mean() * 100
    extra += f", outside column: {missing:.1f}%" if missing > 0 else ""
    ax.set_title(f"{title}\nmedian {np.median(img[good]):.0f} K, max {img[good].max():.3g} K{extra}", fontsize=9)
    ax.set_xlabel("x [pixel]"); ax.set_ylabel("y [pixel]")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="T [K]")

def at_height(target):
    """Linear interpolation of T at height `target` (km) per column; NaN where the column does not reach it."""
    below = (z > target).sum(axis=2)            # z decreases with index: points above the target
    k = np.clip(below, 1, z.shape[2] - 1)       # interval [k-1, k] brackets the target
    z0 = np.take_along_axis(z, (k - 1)[..., None], 2)[..., 0]; z1 = np.take_along_axis(z, k[..., None], 2)[..., 0]
    T0 = np.take_along_axis(T, (k - 1)[..., None], 2)[..., 0]; T1 = np.take_along_axis(T, k[..., None], 2)[..., 0]
    w = (z0 - target) / (z0 - z1)
    out = T0 + w * (T1 - T0)
    out[(below == 0) | (below == z.shape[2])] = np.nan
    return out

taus = [-1, -3, -5.1, -5.7]
fig, axes = plt.subplots(1, 4, figsize=(22, 5.6))
for ax, t in zip(axes, taus):
    k = int(np.argmin(np.abs(lt - t)))
    panel(ax, T[:, :, k], f"log tau500 = {lt[k]:.1f}")
fig.suptitle("Temperature at fixed log tau500 (forward_atmosphere.h5)")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{outdir}/temperature_logtau.png", dpi=100)

heights = [0, 300, 500, 1000, 1200, 1800]
fig, axes = plt.subplots(2, 3, figsize=(17, 11))
for ax, h in zip(axes.ravel(), heights):
    panel(ax, at_height(h), f"z = {h} km")
fig.suptitle("Temperature at fixed height z, column-HSE heights (forward_atmosphere.h5)")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{outdir}/temperature_height.png", dpi=100)
