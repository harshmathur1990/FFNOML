# Run-directory examples

These directories are templates for keeping code in a permanent checkout while
all large inputs, generated files, temporary files, and scheduler logs live in a
separate run location.

- `python_run`: Python dataset/training/prediction jobs.
- `inversion_run`: production `FFNOInversion.jl` job. Its [`START_HERE.md`](inversion_run/START_HERE.md) describes how the run treats the atmosphere: the population mesh, the corona above a column's top, the extension below its bottom, a given electron density or a whole given atmosphere (forward only), line centres and the wavelength frame.
- `runtime_test_run`: Olivia Julia/MPI/GPU regression jobs.

Copy the appropriate template to the target filesystem. Replace the placeholder
files with symlinks to permanent assets; do not copy large scientific files into
the Git checkout.
