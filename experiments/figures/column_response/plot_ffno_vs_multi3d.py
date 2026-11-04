# H-alpha column response with FFNO populations (HSEMHS atmosphere) against MULTI3D populations (Bifrost atmosphere),
# same 4 random pixels as plot_core_response.py: absolute, normalised, wavelength-integrated over the core
# (centred on the model's rest wavelength 6562.96 Å air, from the level energies of atom.h6_tiago2.yaml) and at the
# bluest grid point (wing). Solid: FFNO, dashed: MULTI3D.
import sys, h5py, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ffno = sys.argv[1] if len(sys.argv) > 1 else "response_run/outputs/column_response.h5"
m3d = sys.argv[2] if len(sys.argv) > 2 else "response_run/outputs/multi3d_column_response.h5"
out = sys.argv[3] if len(sys.argv) > 3 else "figures/column_response/ffno_vs_multi3d_halpha.png"
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
CENTRE, HALF = 6562.96, 0.15
variables = [("temperature", "T"), ("vz", "v_z")]
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # categorical slots 1-4

def load(path, pix):
    with h5py.File(path, "r") as f:
        w = f["wavelength_air_m"][:] * 1e10
        h = np.where(abs(w - 6562.8) < 5)[0]
        R = {v: np.stack([f["R/" + v][i, j][:, h] for i, j in pix]) for v, _ in variables}   # (pixel, z, λ)
        return w[h], f["logtau500"][:], R

with h5py.File(ffno, "r") as f:
    xi = f["x_index"][:]; yi = f["y_index"][:]
rng = np.random.default_rng(seed)
pix = [(int(rng.integers(len(xi))), int(rng.integers(len(yi)))) for _ in range(4)]
runs = [(load(ffno, pix), "-", "FFNO"), (load(m3d, pix), "--", "MULTI3D")]

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True)
for (w, lt, R), style, label in runs:
    core = np.where(abs(w - CENTRE) <= HALF + 1e-6)[0]; wing = np.argmin(w)
    for r, (v, name) in enumerate(variables):
        panels = [abs(np.trapezoid(R[v][:, :, core], w[core], axis=2)), abs(R[v][:, :, wing])]
        for c, response in enumerate(panels):
            response = response / response.max(axis=1, keepdims=True)
            for p, (i, j) in enumerate(pix):
                axes[r, c].plot(lt, response[p], style, color=colors[p], lw=1.8,
                                label=f"x={xi[i]}, y={yi[j]}" if style == "-" else None)
w = runs[0][0][0]
titles = [f"core {CENTRE:.2f} ± {HALF:.2f} Å (integrated)", f"wing {w.min():.2f} Å"]
for r, (v, name) in enumerate(variables):
    for c in range(2):
        ax = axes[r, c]; ax.grid(alpha=0.25, lw=0.6)
        ax.set_title(f"{name} response, H-alpha {titles[c]}", fontsize=10)
        if c == 0: ax.set_ylabel(f"|R$_{{{name}}}$| / max", fontsize=9)
        if r == 1: ax.set_xlabel("log$_{10}$ τ$_{500}$")
axes[0, 0].set_ylim(0, 1.05)
first = axes[0, 0].legend(title="pixel (0-based)", fontsize=9, title_fontsize=9, frameon=False, loc="upper left")
axes[0, 0].add_artist(first)
axes[0, 0].legend([plt.Line2D([], [], color="0.3", ls=s) for s in ("-", "--")], ["FFNO populations", "MULTI3D populations"],
                  fontsize=9, frameon=False, loc="upper right")
fig.suptitle("H-alpha column response: FFNO vs MULTI3D populations (populations held fixed)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out, dpi=120)
print("pixels:", [(int(xi[i]), int(yi[j])) for i, j in pix]); print("wrote", out)
