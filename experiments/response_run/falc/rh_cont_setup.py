# RH's chi and S at line-free wavelengths: run_falc_air/ray.input in a copy of the run (run_falc_cont), the same atmosphere and J.
import shutil, os, struct, numpy as np
H = "/mn/stornext/d9/data/harshm/rh_falc/rhf1d/"; a = H + "run_falc_cont/"
if os.path.exists(a): shutil.rmtree(a)
shutil.copytree(H + "run_falc_air", a)
b = open(a + "spectrum.out", "rb").read(); n = struct.unpack(">i", b[:4])[0]; lam = np.frombuffer(b, ">f8", n, 4 + 0).astype(float)
print("n", n, "first", lam[:2], "monotonic", bool(np.all(np.diff(lam) > 0)))
want = (360, 399.88, 422.17, 450, 500, 550, 600, 650, 700, 800, 853.4, 1000)
idx = sorted({int(np.argmin(abs(lam - w))) for w in want})
print([(i, round(float(lam[i]), 3)) for i in idx])
open(a + "ray.input", "w").write("1.00\n%d\n%s\n" % (len(idx), " ".join(map(str, idx))))
