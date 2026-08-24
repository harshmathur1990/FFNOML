# Current FFNOML seams inventory

## Python population path

- `FFNONet.py::_make_inputs_ch_first` constructs the six atmospheric input channels.
- `ffno_predict_populations` loads solving-set HDF5, runs tiled/full inference, converts log populations to linear space, and writes HDF5.
- The reusable seam for Phase 2 is the model call plus normalization and population conversion. File loading, checkpoint loading, and output writing must move outside the hot loop.
- Relevant hazards: channel/axis permutations, model normalization metadata, z-scale normalization, checkpoint reload, CUDA ownership, and tiling halos.

## Julia synthesis path

- `Forward.jl::synthesize_intensity_3d` is the closest callable physics kernel.
- `split_atoms`, atom loading, Voigt/background setup, and `save_synthesis_results` are currently mixed into a file-driven workflow.
- Phase 3 should cache atomic data and interpolation tables, make redistribution explicit, return `SpectralCube`, and keep HDF5 plus CLI logic outside the synthesis call.
- Relevant hazards: hard-coded intensity array order, scalar-only return type, global configuration, atom-level channel mapping, threading buffers, and filesystem checks.
