"""Simulation-physics diagnostics for the ch024031/helium offsets (step 3 of claude-next-steps-uq)."""
import sys
import numpy as np
from helita.sim.multi3d import Multi3dOut
B = "/mn/stornext/d9/data/harshm/bifrost_data"
cases = sys.argv[1:]
ref_ne = None
for cs in cases:
    m = Multi3dOut(directory=f"{B}/{cs}/H"); m.readall()
    z = np.asarray(m.geometry.z) * 1e-8                       # Mm, top-down? report as is
    tau = np.asarray(m.atmos.tau500, np.float64)
    T = np.log10(np.asarray(m.atmos.tg, np.float64))
    rho = np.log10(np.asarray(m.atmos.rho, np.float64))
    ne = np.log10(np.asarray(m.atmos.ne, np.float64))
    nh = np.asarray(m.atmos.nh, np.float64).sum(-1)
    n = np.asarray(m.atom.n, np.float64); ns = np.asarray(m.atom.nstar, np.float64)
    # tau500 = 1 height per column
    lt = np.log10(np.maximum(tau, 1e-30))
    k1 = np.abs(lt).argmin(axis=2)
    z1 = z[k1]
    # rho -> nH conversion (composition / mean mass)
    mu = np.median(10**rho / nh) / 1.6726e-24
    # photosphere slab: 0-0.3 Mm above the tau=1 surface, i.e. in coordinates relative to z1
    dz = z[None, None, :] - z1[:, :, None]
    above = dz  # z is height (stored top-down), so height above the tau=1 surface is z - z1
    sl = (above > 0.0) & (above < 0.3)
    deep = (above > -0.2) & (above < 0.0)
    b = np.log10(np.maximum(n, 1e-30) / np.maximum(ns, 1e-30))
    print(f"== {cs}: z[0]={z[0]:.2f} z[-1]={z[-1]:.2f} Mm, dz_min={np.abs(np.diff(z)).min()*1e3:.1f} km, "
          f"z(tau500=1) median {np.median(z1):.3f} p5 {np.percentile(z1,5):.3f} p95 {np.percentile(z1,95):.3f}")
    print(f"   rho/(nH mH) = {mu:.4f};  ne/nH at tau=1 median log {np.median((ne - np.log10(nh))[deep]):.3f}")
    print("   log b_i, 0-0.3 Mm above tau=1:", " ".join(f"{np.median(b[..., l][sl]):+.3f}" for l in range(n.shape[-1])))
    print("   log b_i, 0-0.2 Mm below tau=1:", " ".join(f"{np.median(b[..., l][deep]):+.3f}" for l in range(n.shape[-1])))
    # ne at fixed (logT, logrho) in the photosphere/low chromosphere, against the first case
    ti = np.clip(((T - 3.4) / 0.05).astype(int), 0, 39); ri = np.clip(((rho + 10) / 0.1).astype(int), 0, 59)
    msk = (T < 4.2)
    key = ti[msk] * 60 + ri[msk]
    med = np.full(2400, np.nan)
    order = np.argsort(key); ks, vs = key[order], ne[msk][order]
    u, st = np.unique(ks, return_index=True); en = np.append(st[1:], len(ks))
    for kk, a, e in zip(u, st, en):
        if e - a > 50:
            med[kk] = np.median(vs[a:e])
    if ref_ne is None:
        ref_ne = med
    else:
        d = med - ref_ne
        print(f"   ne at fixed (T<10^4.2, rho) vs {cases[0]}: median {np.nanmedian(d):+.3f} dex, "
              f"p10 {np.nanpercentile(d,10):+.3f} p90 {np.nanpercentile(d,90):+.3f}, bins {np.isfinite(d).sum()}")
