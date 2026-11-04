# Experiments

Records of validation and uncertainty runs, kept so that their results can be traced. Nothing in the packages depends on them.

| Path | What it holds |
| --- | --- |
| `response_run/` | response functions, FALC against RH, Bifrost against MULTI3D, viewing-angle and mixed-region checks |
| `uq/` | driver scripts of the uncertainty and active-learning study, with `scripts/uq_*.py` |
| `figures/` | plotting scripts of those runs |
| `diagnostics/` | investigation notes |

Run the plotting scripts from this directory: they use paths such as `response_run/falc/` and `figures/falc/`.

These scripts were written before the repository was reorganised and renamed. Many of them name the old checkout, `/mn/stornext/d9/data/harshm/FFNOML`, as an absolute path, because their results, which git ignores, are there. They need that checkout, with its old layout, to run as written.
