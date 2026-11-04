# Inputs for two tests of the corona above FALC: (1) the 1 MK column as native log tau (log tau500 from the z run falc_zext, so the
# corona is on the tau grid before HSE); (2) the z column with a 0.5 MK corona (the temperature's effect).
import h5py, numpy as np, os
here = os.path.dirname(os.path.abspath(__file__))
with h5py.File(os.path.join(here, "falc_input_extended.h5")) as g, h5py.File(os.path.join(here, "falc_zext_atmosphere.h5")) as a:
    lt = a["logtau500"][2, 2, :, 0]; print("log tau top, FALC top, bottom", lt[0], lt[-82], lt[-1], "monotonic", np.all(np.diff(lt) > 0))
    with h5py.File(os.path.join(here, "falc_input_ltau_extended.h5"), "w") as o:
        o["logtau_500"] = lt
        for n in ("x_m", "y_m", "temperature", "vx", "vy", "vz", "vturb"): o[n] = g[n][...]
    with h5py.File(os.path.join(here, "falc_input_extended_05MK.h5"), "w") as o:
        for n in ("z_m", "x_m", "y_m", "vx", "vy", "vz", "vturb"): o[n] = g[n][...]
        T = g["temperature"][...].copy(); added = T.shape[0] - 82
        T0 = T[added, 2, 2]; prof = T[:added, 2, 2]
        T[:added] = (10 ** (np.log10(T0) + (np.log10(5e5) - np.log10(T0)) * (np.log10(prof) - np.log10(T0)) / (6 - np.log10(T0))))[:, None, None]
        o["temperature"] = T; print("0.5 MK top", T[0, 2, 2])
