# New inversion-path synthesis (outputs/synthesis.h5, PSF applied) vs FFNO height-grid synthesis of the same snapshot
# (intensity_ml_en024048_hion_504_385, PSF applied here), at the reference (vacuum) wavelengths inside our windows.
import h5py, numpy as np, sys
from scipy.ndimage import gaussian_filter
old_path = sys.argv[1] if len(sys.argv) > 1 else None
T="training_FFNO3D_zscale_expand_lognlte/"
with h5py.File("outputs/synthesis.h5") as f: wo=f["wavelength_m"][:]*1e10; O=f["intensity"][:,:,:,0,0].astype(float)
def air2vac(a):
    s2=(1e4/a)**2; return a*(1+0.0000834254+0.02406147/(130-s2)+0.00015998/(38.9-s2))
wov=air2vac(wo); z=lambda a:(a-a.mean())/a.std()
for g,sl,offs in (("H",slice(0,88),(-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6)),("CA",slice(88,176),(-1.6,-1.0,-0.5,-0.2,0.0,0.2,0.5,1.0,1.6))):
    with h5py.File(T+f"intensity_ml_en024048_hion_504_385_FFNO3D_{g}.h5") as f: M=f[g+"/intensity"][:].astype(float); wm=f[g+"/wave"][:]*10
    M=gaussian_filter(M,sigma=(2/2.3548,2/2.3548,0),mode="wrap")
    Og=O[:,:,sl][:,::-1]; wg=wov[sl]                     # y flipped onto the reference grid
    kmin=np.argmin(M.mean((0,1))); cont=M[:,:,0]
    print(f"\n== {g}: mean intensity, pixel correlation, and granulation imprint corr(map, continuum)")
    print("  Δλ [Å] | height-grid  ours | corr(ours, height) | granulation: height  ours")
    for d in offs:
        k=np.argmin(abs(wm-(wm[kmin]+d)))
        if not (wg[0]<=wm[k]<=wg[-1]): continue
        j=np.searchsorted(wg,wm[k]); a=(wm[k]-wg[j-1])/(wg[j]-wg[j-1]); om=(1-a)*Og[:,:,j-1]+a*Og[:,:,j]
        print("  %+5.2f  |   %6.2f   %6.2f |      %5.2f         |       %5.2f   %5.2f"%(wm[k]-wm[kmin],M[:,:,k].mean(),om.mean(),
            (z(om)*z(M[:,:,k])).mean(),(z(M[:,:,k])*z(cont)).mean(),(z(om)*z(cont)).mean()))
