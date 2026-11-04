import csv, glob, numpy as np
from scipy.stats import spearmanr
cases, u, m = [], [], []
for p in sorted(glob.glob("uq/atlas_v3/*_H_summary.csv")):
    rows = list(csv.DictReader(open(p)))
    cases.append(rows[0]["case"]); u.append(float(rows[0]["conservation_log:median_u"]))
    m.append(np.mean([float(r["mae"]) for r in rows[1:5]]))
cases, u, m = np.array(cases), np.array(u), np.array(m)
for c, a, b in sorted(zip(cases, u, m), key=lambda t: t[1]): print("%-28s median residual %.3f  MAE(L2-5) %.3f" % (c, a, b))
print("case-level Spearman all:", round(spearmanr(u, m)[0], 2), " without helium:", round(spearmanr(u[cases != "en024031_by100_helium_109"], m[cases != "en024031_by100_helium_109"])[0], 2))
