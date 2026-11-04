# HSEMHS forward atmosphere (504^2 run, taken at [0::2, 1::2][:, ::-1]) vs Bifrost en024048_hion 385 (252^2 =
# 504^2[::2, ::2]) at fixed log tau500, Bifrost's log tau from MULTI3D tau500. Pixel-wise ratios ours/Bifrost.
import numpy as np, h5py
B="/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385/"; nx=ny=252; D=467; n=nx*ny*D
mm=lambda i: np.asarray(np.memmap(B+"atm3d",dtype="<f4",mode="r",shape=(nx,ny,D),order="F",offset=4*n*i))
neB=mm(0)*1e6; TB=mm(1); rhoB=mm(5)*1e3
t=np.fromfile(B+"mesh",sep=" ",dtype=np.float64); zB=t[nx+ny+3:nx+ny+3+D]/100
lt=np.log10(np.maximum(np.asarray(np.memmap(B+"H/tau500",dtype="<f4",mode="r",shape=(nx,ny,D),order="F")),1e-30))
f=h5py.File("outputs/forward_atmosphere.h5"); L=f["logtau_500"][:]
sel=lambda name,k: f[name][0::2,1::2,k,0][:,::-1]
def bif(q,l,log=True):
    k=np.clip(np.argmax(lt>=l,axis=2),1,D-1)
    a0=np.take_along_axis(lt,(k-1)[...,None],2)[...,0]; a1=np.take_along_axis(lt,k[...,None],2)[...,0]
    w=np.clip((l-a0)/np.where(a1>a0,a1-a0,1),0,1)
    q=np.broadcast_to(q,(nx,ny,D)) if q.ndim==1 else q
    q0=np.take_along_axis(q,(k-1)[...,None],2)[...,0]; q1=np.take_along_axis(q,k[...,None],2)[...,0]
    return np.exp((1-w)*np.log(q0)+w*np.log(q1)) if log else (1-w)*q0+w*q1
print(" log tau | T ours/Bif [p10,p90]  corr | rho ours/Bif [p10,p90] | ne ours/Bif [p10,p90] | P~rhoT ratio | z ours-Bif km: median  std")
for l in (1.0,0.5,0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-4.5,-5.0,-5.5):
    k=int(np.argmin(abs(L-l)))
    To,ro,no,zo=sel("temperature",k),sel("rho",k),sel("ne",k),sel("z",k)
    Tb,rb,nb=bif(TB,L[k]),bif(rhoB,L[k]),bif(neB,L[k]); zb=bif(zB[None,None,:]*np.ones((1,1,1)),L[k],log=False) if False else bif(np.broadcast_to(zB,(nx,ny,D)),L[k],log=False)
    q=lambda r:"%5.2f [%4.2f,%4.2f]"%(np.median(r),np.percentile(r,10),np.percentile(r,90))
    c=np.corrcoef(np.log(To).ravel(),np.log(Tb).ravel())[0,1]; dz=(zo-zb)/1e3
    print(" %5.1f   | %s %5.2f | %s     | %s    |   %5.2f      | %7.0f %6.0f"%(L[k],q(To/Tb),c,q(ro/rb),q(no/nb),np.median(ro*To/(rb*Tb)),np.median(dz),np.std(dz)))
