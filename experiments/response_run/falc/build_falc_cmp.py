# The two FALC atmospheres for the RH / FFNO comparison, unedited atoms in both codes: FALC_82_5x5.hdf5 as an FFNO native-z input (all columns; x, y are 0-4 in
# the file, so 48 km as in falc_logm.h5) and its centre column as an rhf1d height-scale .atmos, plus the two rhf1d run directories. FALC_82.atmos itself is
# falc_logm.h5 for FFNO (checked here against the file) and is read directly by rhf1d.
import h5py, numpy as np, os, re, shutil
here = os.path.dirname(os.path.abspath(__file__)); rhf = "/mn/stornext/d9/data/harshm/rh_falc/rhf1d/"
f = h5py.File(here + "/FALC_82_5x5.hdf5")
get = lambda n: np.asarray(f[n][0], float)
T, ne, vz, vt, h, z = get("temperature"), get("electron_density"), get("velocity_z"), get("velocity_turbulent"), get("hydrogen_populations"), np.asarray(f["z"][0], float)
assert np.all(np.diff(z) < 0)
A = [l.split() for l in open("/mn/stornext/d9/data/harshm/rh_falc/Atmos/FALC_82.atmos") if l.strip() and not l.startswith("*")]
nd = int(A[3][0]); a = np.array(A[4:4 + nd], float); hh = np.array(A[4 + nd:4 + 2 * nd], float)
print("5x5 vs .atmos, max relative difference: T %.2e ne %.2e vturb %.2e" % tuple(np.abs(x[2, 2] / y - 1).max() for x, y in ((T, a[:, 1]), (ne, a[:, 2] * 1e6), (vt, a[:, 4] * 1e3))))
print("  H levels", ["%.2e" % np.abs(h[i, 2, 2] / (hh[:, i] * 1e6) - 1).max() for i in range(6)])
with h5py.File(here + "/falc_logm.h5") as o:
    print("falc_logm.h5 vs .atmos: T %.2e ne %.2e H %.2e" % (np.abs(o["temperature"][:, 2, 2] / a[:, 1] - 1).max(), np.abs(o["ne"][:, 2, 2] / (a[:, 2] * 1e6) - 1).max(),
          np.abs(o["hydrogen_populations"][:, :, 2, 2].T / (hh * 1e6) - 1).max()))
zyx = lambda v: np.moveaxis(v, -1, -3).swapaxes(-1, -2)    # (..., x, y, z) -> (..., z, y, x)
with h5py.File(here + "/falc_5x5_z.h5", "w") as o:
    o["z_m"] = z; o["x_m"] = 48e3 * np.arange(5.0); o["y_m"] = 48e3 * np.arange(5.0)
    o["temperature"] = zyx(T); o["ne"] = zyx(ne); o["vz"] = zyx(vz); o["vturb"] = zyx(vt); o["vx"] = 0 * zyx(vz); o["vy"] = 0 * zyx(vz)
    o["hydrogen_populations"] = zyx(h)
    assert o["hydrogen_populations"].shape == (6, nd, 5, 5) and np.array_equal(o["temperature"][:, 2, 2], T[2, 2])
# the centre column for rhf1d: height [km], T, ne [cm-3], v, vturb [km/s], then the populations [cm-3]
at = rhf + "FALC_5x5_center.atmos"
with open(at, "w") as o:
    o.write("* FALC_82_5x5.hdf5 (ITA-Solar/rh), column (2, 2), on its own height scale\n*\n  FALC_5x5_center\n  Height scale\n*\n* log g [cm s^-2]\n  4.44\n*\n* Ndep\n  %d\n*\n" % nd)
    o.write("* height [km]  Temperature  Ne  V  Vturb\n")
    for k in range(nd): o.write("  %.8E  %.6E  %.6E  %.6E  %.6E\n" % (z[k] / 1e3, T[2, 2, k], ne[2, 2, k] / 1e6, vz[2, 2, k] / 1e3, vt[2, 2, k] / 1e3))
    o.write("*\n* Hydrogen populations\n*    nh(1)  nh(2)  nh(3)  nh(4)  nh(5)  np\n")
    for k in range(nd): o.write("  " + "  ".join("%.6E" % (h[i, 2, 2, k] / 1e6) for i in range(6)) + "\n")
for name, atmos in (("run_cmp_atmos", "../../Atmos/FALC_82.atmos"), ("run_cmp_5x5", "../FALC_5x5_center.atmos")):
    d = rhf + name + "/"
    if os.path.exists(d): shutil.rmtree(d)
    shutil.copytree(rhf + "run_falc", d)
    for g in os.listdir(d):
        if g.endswith((".out", ".log", ".dat")) or g.startswith("spectrum_"): os.remove(d + g)
    s = open(d + "keyword.input").read(); s, n = re.subn(r"(?m)^(\s*ATMOS_FILE\s*=\s*).*$", r"\g<1>" + atmos, s); assert n == 1
    open(d + "keyword.input", "w").write(s)
