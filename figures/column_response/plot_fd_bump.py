# Finite-difference H-alpha temperature response with a Gaussian bump in log tau500 (response_run/fd_bump_response.jl,
# MULTI3D populations on the Bifrost atmosphere, populations fixed): (I+ - I-)/2A against the bump centre, integrated
# over the core (6562.96 ± 0.15 Å air, the model's line centre) and at the bluest grid point (wing), signed and
# normalised by max |.|. Dashed grey: the adjoint column response (multi3d_column_response.h5) convolved with the
# same bump, Σ_k R_k Δlogτ_k g_k, as a cross-check.
import sys, h5py, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
fd = sys.argv[1] if len(sys.argv) > 1 else "response_run/outputs/fd_bump_4pixels.h5"
adj = sys.argv[2] if len(sys.argv) > 2 else "response_run/outputs/multi3d_column_response.h5"
out = sys.argv[3] if len(sys.argv) > 3 else "figures/column_response/fd_bump_halpha.png"
CENTRE, HALF = 6562.96, 0.15
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # categorical slots 1-4

with h5py.File(fd) as f:
    R = f["R_temperature"][:]                     # (pixel, z, λ)
    w = f["wavelength_air_m"][:] * 1e10; lt = f["logtau500"][:]; pixels = f["pixels"][:]
    sigma = f.attrs["sigma_dex"]; fraction = f.attrs["fraction"]
g = np.exp(-(lt[:, None] - lt[None, :])**2 / (2 * sigma**2))     # (k0, k)
with h5py.File(adj) as f:
    d = f["dlogtau500"][:]
    Ra = np.stack([f["R/temperature"][x, y] for x, y in pixels])  # (pixel, z, λ)
Ra = np.einsum("jk,pkl->pjl", g * d[None, :], Ra)

core = np.where(abs(w - CENTRE) <= HALF + 1e-6)[0]; wing = np.argmin(w)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for c, (ax, title) in enumerate(zip(axes, [f"core {CENTRE:.2f} ± {HALF:.2f} Å (integrated)", f"wing {w[wing]:.2f} Å"])):
    for p, (x, y) in enumerate(pixels):
        for data, style, lw, col in ((R, "-", 2, colors[p]), (Ra, "--", 1, "0.25")):
            r = np.trapezoid(data[p][:, core], w[core], axis=1) if c == 0 else data[p][:, wing]
            ax.plot(lt, r / abs(r).max(), style, color=col, lw=lw, label=f"x={x}, y={y}" if style == "-" else None)
    ax.axhline(0, color="0.6", lw=0.8, zorder=0); ax.grid(alpha=0.25, lw=0.6)
    ax.set_title(f"H-alpha {title}", fontsize=10); ax.set_xlabel("log$_{10}$ τ$_{500}$ of bump centre")
axes[0].set_ylabel("(I$_+$ − I$_-$) / 2A, normalised by max |·|", fontsize=9); axes[0].set_ylim(-1.05, 1.05)
first = axes[0].legend(title="pixel (0-based)", fontsize=9, title_fontsize=9, frameon=False, loc="lower left")
axes[0].add_artist(first)
axes[0].legend([plt.Line2D([], [], color="0.3"), plt.Line2D([], [], color="0.25", ls="--", lw=1)],
               ["finite difference", "adjoint ⊗ bump"], fontsize=9, frameon=False, loc="upper left")
fig.suptitle(f"H-alpha temperature response, Gaussian bump σ = {sigma} dex, A = {fraction:.0%} T "
             "(MULTI3D populations, Bifrost atmosphere, populations fixed)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out, dpi=120); print("wrote", out)
for p, (x, y) in enumerate(pixels):
    r = np.trapezoid(R[p][:, core], w[core], axis=1); ra = np.trapezoid(Ra[p][:, core], w[core], axis=1)
    print((x, y), "core: max |FD - adjoint| / max|FD| = %.3f" % (abs(r - ra).max() / abs(r).max()))
