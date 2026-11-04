# The FALC column on a wider horizontal grid (identical columns), for the FFNO's horizontal Fourier layers: falc_input_ltau.h5 and
# falc_input_extended.h5 replicated to n x n columns at 48 km.
import h5py, numpy as np, os, sys
here = os.path.dirname(os.path.abspath(__file__)); n = int(sys.argv[1]) if len(sys.argv) > 1 else 96
for src, dst in (("falc_input_ltau.h5", f"falc_input_ltau_{n}.h5"), ("falc_input_extended.h5", f"falc_input_extended_{n}.h5")):
    with h5py.File(os.path.join(here, src)) as g, h5py.File(os.path.join(here, dst), "w") as o:
        for name in g:
            v = g[name][...]
            if name in ("x_m", "y_m"): o[name] = 48e3 * np.arange(float(n))
            elif v.ndim == 3: o[name] = np.broadcast_to(v[:, 2:3, 2:3], (v.shape[0], n, n)).copy()
            else: o[name] = v
        print(dst, g["temperature"].shape[0], "levels")
