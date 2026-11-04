# RH (rhf1d) and FFNOInversion forward on FALC_82.atmos and on the centre pixel of FALC_82_5x5.hdf5, with each atmosphere's nₑ and hydrogen populations and each code's own atoms
# (line_center = "atom"): air wavelengths, SI. One figure per line. response_run/falc/build_falc_cmp.py, run_cmp_rh.sh, run_cmp_ffno.sh.
import h5py, numpy as np, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__)); D = here + "/../../response_run/falc/"
def rh(v):
    with h5py.File(D + f"falc_cmp_{v}_rh.h5") as f:
        l, I = f["wavelength_air_nm"][:], f["intensity"][:]
    q = np.argsort(l); return l[q], I[q]
def ffno(v):
    with h5py.File(D + f"falc_cmp_{v}_synthesis.h5") as f:
        assert f.attrs["wavelength_frame"] in ("air", b"air")
        return f["wavelength_m"][:] * 1e9, f["intensity"][2, 2, :, 0, 0]
# each 5x5 curve dashed in its own colour over its solid .atmos curve, which it nearly overlaps
series = (("RH, FALC_82.atmos", rh("atmos"), "#2a78d6", "-", 2.8), ("RH, FALC_82_5x5 center pixel", rh("5x5"), "#1baf7a", (0, (3, 3)), 1.4),
          ("FFNO, FALC_82.atmos", ffno("atmos"), "#eb6834", "-", 2.8), ("FFNO, FALC_82_5x5 center pixel", ffno("5x5"), "#eda100", (0, (3, 3)), 1.4))
for name, title, lo, hi in (("halpha", "Hα", 655.8, 657.2), ("ca8542", "Ca II 8542", 853.4, 854.6)):
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    for label, (l, I), color, ls, lw in series:
        m = (l >= lo) & (l <= hi); ax.plot(l[m], I[m], color=color, ls=ls, lw=lw, label=label)
    ax.set_xlim(lo, hi); ax.set_xlabel("Air wavelength [nm]"); ax.set_ylabel("Intensity [J s⁻¹ m⁻² Hz⁻¹ sr⁻¹]"); ax.set_title(title, loc="left", color="#0b0b0b")
    ax.ticklabel_format(axis="x", useOffset=False); ax.grid(color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(here + f"/falc_cmp_{name}.png", dpi=200)
