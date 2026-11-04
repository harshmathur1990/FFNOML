# FFNO input maps for an inclined native-log-tau atmosphere (docs/viewing-angle/README.md V6), from
# response_run/view_check/ffno_input_maps.jl: rows z = 0, 300, 1200 km (HSEMHS gauge); columns the Bifrost cube at that
# height, the Cartesian features the FFNO receives (rays shifted by -t z), and the unshifted ray features (what the
# FFNO would see without the shift). One colour scale per row. Prints the correlation of each input map with Bifrost.
import sys, h5py, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
path = sys.argv[1] if len(sys.argv) > 1 else "response_run/view_check/ffno_input_maps_mu08.h5"
var = sys.argv[2] if len(sys.argv) > 2 else "T"
out = sys.argv[3] if len(sys.argv) > 3 else f"figures/viewing_angle/ffno_input_maps_{var}.png"
label = {"T": "T [kK]", "vz": "v_z [km/s]", "log10_ne": "log10 n_e [m^-3]", "log10_rho": "log10 rho [kg m^-3]"}[var]
scale = {"T": 1e-3, "vz": 1e-3}.get(var, 1.0)
cmap = {"vz": "RdBu_r"}.get(var, "inferno")
with h5py.File(path) as f:
    mu = f.attrs["view_mu"]; phi = f.attrs["view_azimuth_deg"]; L = f.attrs["dx_m"] * 504 / 1e6
    groups = sorted(f.keys())
    fig, axes = plt.subplots(len(groups), 3, figsize=(13.5, 4.3 * len(groups)), constrained_layout=True)
    for r, g in enumerate(groups):
        h = f[g]["mesh_height_m"][()] / 1e3
        maps = [f[g][f"bifrost_{var}"][:] * scale, f[g][f"ffno_{var}"][:] * scale, f[g][f"ray_{var}"][:] * scale]
        lo, hi = np.percentile(maps[0], [1, 99])
        if var == "vz": hi = max(abs(lo), abs(hi)); lo = -hi
        titles = ["Bifrost cube", "FFNO input (rays shifted to x, y)", "ray features, not shifted"]
        for c, (m, title) in enumerate(zip(maps, titles)):
            ax = axes[r, c]
            im = ax.imshow(m.T, origin="lower", extent=(0, L, 0, L), cmap=cmap, vmin=lo, vmax=hi)
            rcorr = np.corrcoef(m.ravel(), maps[0].ravel())[0, 1]
            ax.set_title(f"z = {h:.0f} km: {title}" + ("" if c == 0 else f"  (r = {rcorr:.2f})"), fontsize=10)
            ax.set_xlabel("x [Mm]"); ax.set_ylabel("y [Mm]") if c == 0 else None
            if c > 0: print(f"{var} z={h:.0f} km {title}: correlation with Bifrost {rcorr:.3f}")
        fig.colorbar(im, ax=axes[r, :], shrink=0.85, label=label)
    fig.suptitle(f"What the FFNO sees: inclined view mu = {mu:g} (theta = {np.degrees(np.arccos(mu)):.1f} deg), "
                 f"azimuth {phi:g} deg (tilt along +x); HSEMHS column balance on tau_LOS", fontsize=12)
    fig.savefig(out, dpi=90); print("wrote", out)
