# ADR-002: Initial FFNO bridge

Status: Phase 2 accepted for the available CPU environment; GPU execution remains a performance follow-up.

Use a persistent PythonCall.jl-backed population model first. PythonCall is a weak dependency and activates only when explicitly loaded, so HE3D/MHS and record/replay runs do not initialize or install Python. The Python backend loads the PyTorch checkpoint, normalization statistics and model weights once, then exchanges atmospheric and population arrays in memory with no per-evaluation HDF5 or process launch.

The canonical FFNO input channel order is `temperature, vx, vy, vz, log10_ne, log10_rho`, stored as `(channel,nz,nx,ny)`. Corrugated geometrical height is passed separately as `(nz,nx,ny)` together with physical `dx` and `dy`. Python returns linear populations in m^-3 as `(nz,nx,ny,nlevels)`.

Checkpoint I/O metadata, normalization arrays, level count, level names, SHA-256 checkpoint hash, finite values and positive populations are validated at the boundary. The runtime refuses metadata mismatches rather than guessing channel or level order.

`RecordedPopulationModel` stores an exact request/response case and can be serialized for CPU tests. It checks both features and z before replaying. Mock, recorded and Python implementations share `predict_populations!`; an RPC or native backend can therefore replace PythonCall without changing `ForwardModel`.

The production Python runtime accepts a model-factory module/function because the existing FFNO construction requires project-specific line/model configuration. It owns the persistent `torch.nn.Module`, device tensors and normalization state. CPU and GPU integration tests remain separate.
