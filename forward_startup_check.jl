#!/usr/bin/env julia

#=
Load Forward.jl exactly as a real job does and verify its enabled configuration
without reading the atmosphere, initializing MPI, launching Python/PyTorch, or
performing synthesis.

Run from the repository directory, for example:

    julia --project="$JULIA_PROJECT" --threads=1 forward_startup_check.jl
=#

using Dates

const CHECK_STARTED_AT = time()

function check_message(message)
    elapsed = round(time() - CHECK_STARTED_AT; digits=1)
    println("[", elapsed, " s] ", message)
    flush(stdout)
end

cd(@__DIR__)
check_message("Starting Forward.jl login-shell check")
check_message("Repository: $(pwd())")
check_message("Julia executable: $(Base.julia_cmd().exec[1])")
check_message("Julia project: $(Base.active_project())")
check_message("Julia depot: $(DEPOT_PATH[1])")

include(joinpath(@__DIR__, "Forward.jl"))
check_message("Forward.jl and all Julia dependencies loaded")

predictions = load_multi3d_pred_data()
isempty(predictions) && error(
    "config.py MULTI3D_PRED_DATA has no entries enabled for Forward.jl"
)
check_message("config.py loaded $(length(predictions)) enabled prediction(s)")

launcher = joinpath(@__DIR__, "forward_fsdppredict.sh")
context = serial_parallel_context()

for prediction in predictions
    configured_iterations = prediction.charge_conservation_max_iterations
    max_iterations = if prediction.nonlte_ne === false
        0
    elseif configured_iterations !== nothing
        configured_iterations
    elseif prediction.nonlte_ne === true
        error(
            "$(prediction.name) has NONLTE_NE=True but no per-dataset " *
            "CHARGE_CONSERVATION_MAX_ITERATIONS value"
        )
    else
        0
    end

    cfg = build_config(prediction)
    check_message("Checking $(cfg.name) ($(cfg.mode) mode)")
    check_message("  mesh: $(cfg.mesh_file)")
    check_message("  atmosphere: $(cfg.atmos_file)")
    check_message("  atoms: $(join([atom.atom_file for atom in cfg.atoms], ", "))")
    preflight_forward_config(cfg, max_iterations, launcher, context)
    check_message("Preflight passed for $(cfg.name)")
end

check_message(
    "SUCCESS: startup and preflight passed; no atmosphere, MPI, ML, SE, or " *
    "synthesis calculation was run",
)
