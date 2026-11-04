# Is the photospheric T mismatch a shift of the log tau scale? Per pixel, the shift d (dex) that best maps the input
# T(log tau) onto Bifrost's T at MULTI3D log tau + d over log tau500 in [-2, 0.5].
import numpy as np, h5py
B="/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385/"; nx=ny=252; D=467; n=nx*ny*D
TB=np.asarray(np.memmap(B+"atm3d",dtype="<f4",mode="r",shape=(nx,ny,D),order="F",offset=4*n))
lt=np.log10(np.maximum(np.asarray(np.memmap(B+"H/tau500",dtype="<f4",mode="r",shape=(nx,ny,D),order="F")),1e-30))
f=h5py.File("outputs/forward_atmosphere.h5"); L=f["logtau_500"][:]
ks=[k for k in range(len(L)) if -2.0<=L[k]<=0.5]
To=np.stack([f["temperature"][0::2,1::2,k,0][:,::-1] for k in ks],-1)
def bifT(l):
    k=np.clip(np.argmax(lt>=l,axis=2),1,D-1)
    a0=np.take_along_axis(lt,(k-1)[...,None],2)[...,0]; a1=np.take_along_axis(lt,k[...,None],2)[...,0]
    w=np.clip((l-a0)/np.where(a1>a0,a1-a0,1),0,1)
    return np.exp((1-w)*np.log(np.take_along_axis(TB,(k-1)[...,None],2)[...,0])+w*np.log(np.take_along_axis(TB,k[...,None],2)[...,0]))
shifts=np.arange(-0.6,0.61,0.05); err=[]
for d in shifts:
    Tb=np.stack([bifT(L[k]+d) for k in ks],-1); err.append(np.sqrt(np.mean(np.log(To/Tb)**2,axis=2)))
err=np.array(err); best=shifts[np.argmin(err,axis=0)]
print("best log tau shift per pixel (dex): median %+.2f, p10 %+.2f, p90 %+.2f, |shift|>=0.2: %.0f%%"%(np.median(best),np.percentile(best,10),np.percentile(best,90),100*np.mean(abs(best)>=0.2)))
print("rms ln T mismatch: no shift %.3f, best shift %.3f"%(np.median(err[np.argmin(abs(shifts))]),np.median(err.min(0))))
