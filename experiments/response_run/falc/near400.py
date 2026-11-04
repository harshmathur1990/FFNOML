import h5py, numpy as np
r = h5py.File("falc_rh_air.h5"); l = r["wavelength_air_nm"][:]; I = r["intensity"][:]; q = np.argsort(l); l, I = l[q], I[q]
m = (l > 380) & (l < 425)
for a, b in zip(l[m], I[m]): print("%.4f  %.4e" % (a, b))
