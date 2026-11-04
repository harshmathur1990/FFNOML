import sys, h5py, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
path, outdir = sys.argv[1], sys.argv[2]
with h5py.File(path, "r") as f:
    w = f["wavelength_m"][:] * 1e10
    I = f["intensity"][:, :, :, 0, 0]          # (x, y, wavelength)
regions = [("H-alpha 6563", slice(0, 88), 0.7, 6562.80), ("Ca II 8542", slice(88, 176), 1.0, 8542.09), ("Fe I 6301.5", slice(176, 222), 0.12, 6301.50)]
fig, axes = plt.subplots(3, 4, figsize=(19, 14), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.1]))
for row, (name, s, offset, nominal) in zip(axes, regions):
    wr, Ir = w[s], I[:, :, s]
    mean = Ir.mean(axis=(0, 1))
    k0 = int(np.argmin(np.abs(wr - nominal))); core = wr[k0]
    picks = [("blue wing", int(np.argmin(np.abs(wr - (core - offset))))), ("core", k0),
             ("red wing", int(np.argmin(np.abs(wr - (core + offset)))))]
    for ax, (label, k) in zip(row[:3], picks):
        img = Ir[:, :, k].T
        # H-alpha core: a few columns reach 1e4, so clip to the bulk (median ~6, p90 ~15).
        clip = [0.5, 99.5]
        lo, hi = np.percentile(img, clip)
        im = ax.imshow(img, origin="lower", cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(f"{name} {label}\n{wr[k]:.3f} Å ({wr[k]-core:+.3f} Å)")
        ax.set_xlabel("x [pixel]"); ax.set_ylabel("y [pixel]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax = row[3]
    lo, hi = np.percentile(Ir, [5, 95], axis=(0, 1))
    ax.fill_between(wr, lo, hi, color="0.8", label="5–95 %")
    ax.plot(wr, mean, "k", label="mean")
    for label, k in picks:
        ax.axvline(wr[k], ls="--", color={"core": "C3"}.get(label, "C0"))
    ax.set_xlabel("air wavelength [Å]"); ax.set_ylabel("intensity (as stored)")
    ax.set_title(f"{name} profile"); ax.legend()
    ax.ticklabel_format(useOffset=False); ax.tick_params(axis="x", labelrotation=30)
fig.suptitle("forward synthesis, column-HSE atmosphere (504×504)")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(f"{outdir}/synthesis_maps.png", dpi=110)
