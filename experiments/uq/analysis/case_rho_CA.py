import numpy as np
from scipy.stats import spearmanr
rows=[l.split() for l in open('uq/analysis/cd_CA.txt')]
c=[r[0] for r in rows]; d=np.array([float(r[1]) for r in rows]); m=np.array([float(r[2]) for r in rows])
print('all 24', spearmanr(d,m)[0])
ex=[i for i,x in enumerate(c) if 'helium' not in x]
print('no helium', spearmanr(d[ex],m[ex])[0])
tr={'qs006003_sap':'689','en024048_hion':'465','ch012012_hion':'910','nw012023':'915','ch012006':'836'}
rel=[];mm=[]
for i,x in enumerate(c):
  s=x.rsplit('_',1)[0]
  if s in tr and not x.endswith(tr[s]):
    j=c.index(s+'_'+tr[s]); rel.append(d[i]-d[j]); mm.append(m[i]-m[j])
print('same-sim delta vs train snapshot n=%d'%len(rel), spearmanr(rel,mm)[0])
