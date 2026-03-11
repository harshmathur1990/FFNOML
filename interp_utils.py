import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid as cumtrapz


# ============================================================
# ---------------- INTERPOLATION ------------------------------
# ============================================================

def interpolate_everything(rho_xyz, z_scale, values_xyzq, new_cmass):
    """
    Vectorized interpolation onto common column-mass grid.
    """

    def interpolate_column(rho, z, values, new_scale):
        cmass = cumtrapz(rho, -z, initial=0)
        f = interp1d(
            cmass,
            values,
            axis=0,
            kind="linear",
            fill_value="extrapolate",
        )
        return f(new_scale)

    vec = np.vectorize(
        interpolate_column,
        signature="(n),(n),(n,o),(m)->(m,o)",
    )

    return vec(rho_xyz, z_scale, values_xyzq, new_cmass).astype(np.float32, copy=False)
