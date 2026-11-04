import struct, h5py, numpy as np
R = "/mn/stornext/d9/data/harshm/rh_falc/rhf1d/run_falc_cont/"
sp = open(R + "spectrum.out", "rb").read(); nspect = struct.unpack(">i", sp[:4])[0]; lam = np.frombuffer(sp, ">f8", nspect, 4).astype(float)
b = open(R + "spectrum_1.00", "rb").read(); p = 8 + 8 * nspect; I = np.frombuffer(b, ">f8", nspect, 8).astype(float)
n = struct.unpack(">i", b[p:p + 4])[0]; p += 4; nz = 82; ids = []; C = []; S = []
for _ in range(n):
    i = struct.unpack(">i", b[p:p + 4])[0]; p += 4; ids.append(i)
    C.append(np.frombuffer(b, ">f8", nz, p).astype(float)); p += 8 * nz; S.append(np.frombuffer(b, ">f8", nz, p).astype(float)); p += 8 * nz
with h5py.File("falc_rh_chi.h5", "w") as f:
    f["air_nm"] = lam[ids]; f["I_rh"] = I[ids]; f["chi"] = np.array(C).T; f["S"] = np.array(S).T   # h5py (z, lambda)
