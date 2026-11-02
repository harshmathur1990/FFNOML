# Temperature response of the H-alpha and Ca II 8542 cores with the FFNO populations responding (forward-mode
# tangents through the FFNO, scripts/ffno_population_tangents.jl, stride-4 grid) for
# 4 random columns of the stride-4 grid: ∫ R_T dλ over each core (H-alpha 6562.96 ± 0.15 Å, the model's line centre;
# Ca II 8542.09 ± 0.25 Å), intensity × Å per K per unit log τ500. Also prints the map-wide formation depths.
# The bottom depth is left out: T there sets the slope of the extrapolated mesh below the atmosphere, so its tangent
# changes the populations of the whole column (an artifact of the deep extension).
import sys, h5py, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
full = sys.argv[1] if len(sys.argv) > 1 else "response_run/outputs/column_response_s4_ffno.h5"
out = sys.argv[2] if len(sys.argv) > 2 else "figures/column_response/ffno_population_response.png"
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
lines = [("H-alpha", 6562.96, 0.15), ("Ca II 8542", 8542.09, 0.25)]
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # categorical slots 1-4

with h5py.File(full) as f:
    w = f["wavelength_air_m"][:] * 1e10; lt = f["logtau500"][:]
    xi = f["x_index"][:]; yi = f["y_index"][:]
    Rf = f["R/temperature"][:, :, :-1]                              # (x, y, z, λ), bottom depth dropped
lt = lt[:-1]
windows = [np.where(abs(w - c) <= h + 1e-6)[0] for _, c, h in lines]
core = lambda R, s: np.trapezoid(R[..., s], w[s], axis=-1)          # (..., z)

rng = np.random.default_rng(seed)
pix = [(int(rng.integers(len(xi))), int(rng.integers(len(yi)))) for _ in range(4)]
fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
for r, ((name, c, h), s) in enumerate(zip(lines, windows)):
    for p, (i, j) in enumerate(pix):
        ax = axes[r, p]
        ax.plot(lt, core(Rf[i, j], s), "-", color=colors[p], lw=2)
        ax.axhline(0, color="0.6", lw=0.8, zorder=0); ax.grid(alpha=0.25, lw=0.6)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.set_title(f"{name} core, x={xi[i]}, y={yi[j]}", fontsize=10)
        if r == 1: ax.set_xlabel("log$_{10}$ τ$_{500}$")
    axes[r, 0].set_ylabel("∫R$_T$ dλ  [I·Å / K per unit log τ$_{500}$]", fontsize=9)
fig.suptitle("Line-core temperature response with FFNO populations responding (dual-number JVP, neighbours "
             "included); ne, pgas, z fixed; stride-4 grid; bottom depth omitted", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out, dpi=110)
print("pixels:", [(int(xi[i]), int(yi[j])) for i, j in pix]); print("wrote", out)

def depth_stats(R, s):
    A = abs(core(R, s)); A = A / A.sum(-1, keepdims=True)
    return (A * lt).sum(-1).ravel(), A[..., lt > -4.5].sum(-1).ravel()
for (name, _, _), s in zip(lines, windows):
    centroid, deep = depth_stats(Rf, s)
    print(f"{name:10s} centroid log τ: median {np.median(centroid):.2f} "
          f"(25-75% {np.percentile(centroid, 25):.2f} to {np.percentile(centroid, 75):.2f}); "
          f"share deeper than log τ -4.5: median {np.median(deep):.2f}")
cH, _ = depth_stats(Rf, windows[0]); cC, _ = depth_stats(Rf, windows[1])
print(f"H-alpha minus Ca II centroid: median {np.median(cH - cC):+.2f}, "
      f"H-alpha higher in {np.mean(cH < cC):.0%} of columns")
