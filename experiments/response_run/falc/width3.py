import h5py, numpy as np
r = h5py.File("falc_rh.h5"); H = r["H_n"][...]; Ca = r["CA_n"][...]; print("RH shapes", H.shape, Ca.shape)
a = h5py.File("falc_givenm_atmosphere.h5"); z = a["z"][2, 2, :, 0] / 1e3; C = a["populations/CA"][2, 2, :, :]; A = a["populations/H"][2, 2, :, :]; print("FFNO shapes", A.shape, C.shape)
print("   z km | Ca nl(3) FFNO/RH  Ca nu(5) FFNO/RH | H nl(2) FFNO/RH  H nu(3) FFNO/RH")
for zk in (0, 100, 200, 300, 500, 700, 1000, 1300, 1600, 2000):
    k = int(np.argmin(abs(z - zk)))
    print("%7d | %14.2f %16.2f | %14.2f %15.2f" % (zk, C[k, 2] / Ca[k, 2], C[k, 4] / Ca[k, 4], A[k, 1] / H[k, 1], A[k, 2] / H[k, 2]))
