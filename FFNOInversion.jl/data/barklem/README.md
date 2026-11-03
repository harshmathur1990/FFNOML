# ABO van der Waals tables

These are copies of RH's `Atoms/Barklem_*data.dat`. Each file holds σ (in a₀², at 10 km/s) and α, gridded on the effective principal quantum numbers of the two levels in steps of 0.1:

| File | Transition | Source | n_eff ranges (lower l, higher l) |
| --- | --- | --- | --- |
| `Barklem_spdata.dat` | s-p | Anstee & O'Mara 1995, MNRAS 276, 859 | 1.0–3.0, 1.3–3.0 |
| `Barklem_pddata.dat` | p-d | Barklem & O'Mara 1997, MNRAS 290, 102 | 1.3–3.0, 2.3–4.0 |
| `Barklem_dfdata.dat` | d-f | Barklem, O'Mara & Ross 1998, MNRAS 296, 1057 | 2.3–4.0, 3.3–5.0 |

`src/Barklem.jl` reads them as RH does in `barklem.c` and interpolates them with RH's cubic convolution.
