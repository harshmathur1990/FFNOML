# Absolute, normalised wavelength-integrated column response functions at the H-alpha and Ca II 8542 line cores:
# |∫ R_v(λ, log τ500) dλ| / max over log τ500, per pixel and panel, for the same 4 random pixels as plot_core_response.py.
import sys, h5py, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
path = sys.argv[1] if len(sys.argv) > 1 else "response_run/outputs/column_response.h5"
out = sys.argv[2] if len(sys.argv) > 2 else "figures/column_response/core_response_abs_norm.png"
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
windows = [("H-alpha", 6562.80, 0.15), ("Ca II 8542", 8542.09, 0.25)]
variables = [("temperature", "T"), ("vz", "v_z")]
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # categorical slots 1-4

with h5py.File(path, "r") as f:
    w = f["wavelength_air_m"][:] * 1e10; lt = f["logtau500"][:]
    xi = f["x_index"][:]; yi = f["y_index"][:]
    rng = np.random.default_rng(seed)
    pix = [(int(rng.integers(len(xi))), int(rng.integers(len(yi)))) for _ in range(4)]
    R = {v: np.stack([f["R/" + v][i, j] for i, j in pix]) for v, _ in variables}   # (pixel, z, λ)

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True)
for c, (name, centre, half) in enumerate(windows):
    sel = np.where(abs(w - centre) <= half + 1e-6)[0]
    for r, (v, label) in enumerate(variables):
        ax = axes[r, c]
        integrated = abs(np.trapezoid(R[v][:, :, sel], w[sel], axis=2))      # (pixel, z)
        integrated /= integrated.max(axis=1, keepdims=True)
        for p, (i, j) in enumerate(pix):
            ax.plot(lt, integrated[p], color=colors[p], lw=2, label=f"x={xi[i]}, y={yi[j]}")
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_title(f"{label} response, {name}  {centre:.2f} ± {half:.2f} Å", fontsize=10)
        if c == 0: ax.set_ylabel(f"|∫R$_{{{label}}}$ dλ| / max", fontsize=9)
        if r == 1: ax.set_xlabel("log$_{10}$ τ$_{500}$")
axes[0, 0].set_ylim(0, 1.05)
axes[0, 0].legend(title="pixel (0-based)", fontsize=9, title_fontsize=9, frameon=False)
fig.suptitle("Normalised |line-core column response|, 4 random pixels (populations and HSE structure held fixed)",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out, dpi=120)
print("pixels:", [(int(xi[i]), int(yi[j])) for i, j in pix]); print("wrote", out)
