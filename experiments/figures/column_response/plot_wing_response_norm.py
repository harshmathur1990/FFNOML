# Absolute, normalised column response functions at one wavelength far from each line core (the bluest point of each
# line's grid): |R_v(λ, log τ500)| / max over log τ500, per pixel and panel, for the same 4 random pixels as
# plot_core_response.py.
import sys, h5py, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
path = sys.argv[1] if len(sys.argv) > 1 else "response_run/outputs/column_response.h5"
out = sys.argv[2] if len(sys.argv) > 2 else "figures/column_response/wing_response_abs_norm.png"
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
lines = [("H-alpha", 6562.80), ("Ca II 8542", 8542.09)]
variables = [("temperature", "T"), ("vz", "v_z")]
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # categorical slots 1-4

with h5py.File(path, "r") as f:
    w = f["wavelength_air_m"][:] * 1e10; lt = f["logtau500"][:]
    xi = f["x_index"][:]; yi = f["y_index"][:]
    rng = np.random.default_rng(seed)
    pix = [(int(rng.integers(len(xi))), int(rng.integers(len(yi)))) for _ in range(4)]
    R = {v: np.stack([f["R/" + v][i, j] for i, j in pix]) for v, _ in variables}   # (pixel, z, λ)

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True)
for c, (name, centre) in enumerate(lines):
    near = np.where(abs(w - centre) < 50)[0]
    l = near[np.argmin(w[near])]                       # bluest grid point of this line
    for r, (v, label) in enumerate(variables):
        ax = axes[r, c]
        response = abs(R[v][:, :, l])                  # (pixel, z)
        response /= response.max(axis=1, keepdims=True)
        for p, (i, j) in enumerate(pix):
            ax.plot(lt, response[p], color=colors[p], lw=2, label=f"x={xi[i]}, y={yi[j]}")
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_title(f"{label} response, {name}  {w[l]:.2f} Å  (Δλ = {w[l] - centre:+.2f} Å)", fontsize=10)
        if c == 0: ax.set_ylabel(f"|R$_{{{label}}}$| / max", fontsize=9)
        if r == 1: ax.set_xlabel("log$_{10}$ τ$_{500}$")
axes[0, 0].set_ylim(0, 1.05)
axes[0, 0].legend(title="pixel (0-based)", fontsize=9, title_fontsize=9, frameon=False)
fig.suptitle("Normalised |column response| far from the line cores, 4 random pixels "
             "(populations and HSE structure held fixed)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out, dpi=120)
print("pixels:", [(int(xi[i]), int(yi[j])) for i, j in pix]); print("wrote", out)
