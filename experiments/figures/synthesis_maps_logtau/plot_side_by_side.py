# Inversion path (log tau, HSEMHS, FFNO on the population mesh) vs FFNO on the native height grid, same snapshot
# (en024048_hion 385, 504^2), same PSF, at the reference vacuum wavelengths. Maps shown as [:, ::-1].T of ours.
import h5py, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
T="training_FFNO3D_zscale_expand_lognlte/"
with h5py.File("outputs/synthesis.h5") as f: wo=f["wavelength_m"][:]*1e10; O=f["intensity"][:,:,:,0,0].astype(float)
s2=(1e4/wo)**2; wov=wo*(1+0.0000834254+0.02406147/(130-s2)+0.00015998/(38.9-s2))
rows=[("H-alpha",  "H", slice(0,88),  (-0.6,-0.4,0.0,0.4), {0.0:(0,12)}),
      ("Ca II 8542","CA",slice(88,176),(-1.6,-0.5,0.0,0.5), {0.0:(0,12)})]
fig,axes=plt.subplots(4,4,figsize=(17,17))
for r,(name,g,sl,offs,fixed) in enumerate(rows):
    with h5py.File(T+f"intensity_ml_en024048_hion_504_385_FFNO3D_{g}.h5") as f: M=f[g+"/intensity"][:].astype(float); wm=f[g+"/wave"][:]*10
    M=gaussian_filter(M,sigma=(2/2.3548,2/2.3548,0),mode="wrap"); kmin=np.argmin(M.mean((0,1)))
    Og=O[:,:,sl][:,::-1]; wg=wov[sl]
    for c,d in enumerate(offs):
        k=np.argmin(abs(wm-(wm[kmin]+d))); j=np.searchsorted(wg,wm[k]); a=(wm[k]-wg[j-1])/(wg[j]-wg[j-1])
        om=(1-a)*Og[:,:,j-1]+a*Og[:,:,j]; ref=M[:,:,k]
        lo,hi=fixed.get(d,np.percentile(np.concatenate([om.ravel(),ref.ravel()]),[1,99]))
        for i,(img,lab) in enumerate(((ref,"height grid (FFNO, MULTI3D-like)"),(om,"inversion path (log tau + mesh)"))):
            ax=axes[2*r+i,c]; im=ax.imshow(img.T,origin="lower",cmap="gray",vmin=lo,vmax=hi)
            ax.set_title(f"{name} {wm[k]-wm[kmin]:+.2f} Å\n{lab}",fontsize=9); ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im,ax=ax,fraction=0.046,pad=0.02)
fig.suptitle("en024048_hion 385: inversion path vs native height grid (same PSF, same colour scale per column)")
fig.tight_layout(rect=(0,0,1,0.97)); fig.savefig("figures/synthesis_maps_logtau/side_by_side.png",dpi=90)
