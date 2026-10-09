# Run-directory examples

These directories are templates for keeping code in a permanent checkout while
all large inputs, generated files, temporary files, and scheduler logs live in a
separate run location.

- `python_run`: Python dataset/training/prediction jobs.
- `inversion_run`: production `FFNOInversion.jl` job.
- `runtime_test_run`: Olivia Julia/MPI/GPU regression jobs.

Copy the appropriate template to the target filesystem. Replace the placeholder
files with symlinks to permanent assets; do not copy large scientific files into
the Git checkout.
