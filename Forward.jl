#!/usr/bin/env julia
# -----------------------------------------------------------------------------
# ForwardSynthesis.jl
#
# A scriptified version of Forward.ipynb:
#   - Reads Bifrost/Multi3D atmosphere
#   - Builds LTE populations via Saha-Boltzmann
#   - Remaps atmosphere + populations to a new column-mass (cmass) scale
#   - Two synthesis modes:
#       (A) "ml"     : NLTE populations read directly from the ML prediction
#       (B) "bifrost": NLTE populations read from Multi3D out_pop (and remapped)
#   - Synthesizes 1D line profiles for all columns (nx, ny) using Muspel
#   - Writes intensity + wavelength to an HDF5 file
#   - Writes two diagnostic plots (PNG)
# -----------------------------------------------------------------------------

# Julia fixes the size of its thread pool at startup.  Make the common
# `julia Forward.jl` invocation use all CPUs available to this process; an
# explicit `julia --threads=N Forward.jl` invocation is left untouched.
const _THREAD_RESTART_ENV = "FNOML_FORWARD_THREADS_RESTARTED"
const _FORWARD_SCRIPT = abspath(@__FILE__)
if abspath(PROGRAM_FILE) == _FORWARD_SCRIPT &&
   Threads.nthreads() == 1 &&
   Sys.CPU_THREADS > 1 &&
   Base.JLOptions().nthreads == 0 &&
   !haskey(ENV, "JULIA_NUM_THREADS") &&
   get(ENV, _THREAD_RESTART_ENV, "0") != "1"
    restart_env = copy(ENV)
    restart_env[_THREAD_RESTART_ENV] = "1"
    project_dir = dirname(Base.active_project())
    restart_cmd = `$(Base.julia_cmd()) --project=$project_dir --threads=auto $_FORWARD_SCRIPT $(ARGS)`
    println("Restarting Forward.jl with --threads=auto ($(Sys.CPU_THREADS) CPUs available)...")
    run(setenv(restart_cmd, restart_env))
    exit()
end

ENV["GKSwstype"] = "100"     # file / offscreen
ENV["GKS_WSTYPE"] = "100"    # some setups use this spelling

using Muspel
using StaticArrays
using AtomicData
using HDF5
using ProgressMeter
using Base.Threads
using Interpolations
using Plots
gr()                         # ensure GR backend
default(show=false)


# ============================================================
# CONFIGURATION
# ============================================================

# -----------------------------
# Config loaded from config.py
# -----------------------------

const RUN_MODE = :ml
# const FORWARD_ATOMS = ["H"]
const FORWARD_ATOMS = ["CA"]
# const FORWARD_ATOMS = ["H", "CA"]

function atom_tag(atom_names)
    return join(atom_names, "_")
end

function load_multi3d_pred_data(predname=nothing)
    script = """
import os
import sys
import types

sys.modules.setdefault(
    "numpy",
    types.SimpleNamespace(
        array=lambda values, dtype=None: values,
        float32=float,
    ),
)

import config

model_dir = config.MODEL_DIR.rstrip("/")
prediction_name = sys.argv[1] if len(sys.argv) > 1 else None
pred_items = config.MULTI3D_PRED_DATA

if prediction_name is not None:
    pred_items = [item for item in pred_items if item["NAME"] == prediction_name]

    if not pred_items:
        try:
            sim_name, snap = prediction_name.rsplit("_", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --predname {prediction_name!r}; expected <atmosphere>_<snap>."
            ) from exc

        if not sim_name or not snap:
            raise ValueError(
                f"Invalid --predname {prediction_name!r}; expected <atmosphere>_<snap>."
            )

        snapshot_dir = os.path.join(
            config.PRED_DIR,
            "bifrost_data",
            sim_name,
            snap,
        )
        pred_items = [{
            "NAME": prediction_name,
            "SIM_NAME": sim_name,
            "SNAP": snap,
            "MESH": os.path.join(snapshot_dir, "mesh"),
            "MULTI3D_ATMOS": os.path.join(snapshot_dir, "atm3d"),
            "TRAIN_DIR": model_dir,
        }]

for item in pred_items:
    print("\\t".join([
        item["NAME"],
        item.get("SIM_NAME", "_".join(item["NAME"].split("_")[:-1])),
        item.get("SNAP", item["NAME"].split("_")[-1]),
        item["MESH"],
        item["MULTI3D_ATMOS"],
        item.get("TRAIN_DIR", model_dir).rstrip("/"),
        config.MODEL,
    ]))
"""

    output = cd(@__DIR__) do
        command = isnothing(predname) ? `python3 -c $script` : `python3 -c $script $predname`
        read(command, String)
    end

    pred_data = []
    for line in split(chomp(output), "\n")
        isempty(line) && continue
        fields = split(line, "\t")
        length(fields) == 7 || error("Unexpected config.py output: $(line)")
        name, sim_name, snap, mesh_file, atmos_file, train_dir, model = fields
        push!(
            pred_data,
            (
                name = name,
                sim_name = sim_name,
                snap = snap,
                mesh_file = mesh_file,
                atmos_file = atmos_file,
                train_dir = train_dir,
                model = model,
            ),
        )
    end

    return pred_data
end

function print_usage(io::IO=stdout)
    println(io, "Usage: julia Forward.jl [--predname <atmosphere>_<snap>]")
end

function parse_cli_args(args)
    predname = nothing
    i = 1

    while i <= length(args)
        arg = args[i]

        if arg == "--predname"
            isnothing(predname) || error("--predname may only be specified once")
            i == length(args) && error("--predname requires a value")
            predname = args[i + 1]
            isempty(predname) && error("--predname requires a non-empty value")
            i += 2
        elseif arg == "--help" || arg == "-h"
            print_usage()
            exit(0)
        else
            error("Unknown argument: $(arg). Run with --help for usage.")
        end
    end

    return (predname = predname,)
end

function all_atom_configs(pred)
    snapshot_dir = dirname(pred.mesh_file)

    return [
        (
            name = "H",
            atom_file = "/mn/stornext/u3/harshm/Documents/WorkRepo/multi3d/input/atoms/atom.h6_tiago2.yaml",
            pops_file = joinpath(snapshot_dir, "H", "out_pop"),
            nlevels = 6,
            line_index = 5,
            lower_level = 2,
            upper_level = 3
        ),
        (
            name = "CA",
            atom_file = "/mn/stornext/u3/harshm/Documents/WorkRepo/multi3d/input/atoms/atom.ca2.yaml",
            pops_file = joinpath(snapshot_dir, "CA", "out_pop"),
            nlevels = 6,
            line_index = 5,
            lower_level = 3,
            upper_level = 5
        )
    ]
end

function atom_configs(pred, atom_names)
    available = Dict(a.name => a for a in all_atom_configs(pred))
    unknown = [name for name in atom_names if !haskey(available, name)]

    if !isempty(unknown)
        error("Unknown Forward atom(s): $(join(unknown, ", ")). Available: $(join(sort(collect(keys(available))), ", "))")
    end

    return [available[name] for name in atom_names]
end

function atom_configs(pred)
    return atom_configs(pred, FORWARD_ATOMS)
end

function base_voigt_config()
    return (
        a_min = 1f-4,
        a_max = 1f1,
        a_n   = 20000,
        v_min = 0f0,
        v_max = 5f2,
        v_n   = 2500
    )
end

function config_ml(pred)
    atoms = atom_configs(pred)
    atoms_tag = atom_tag([a.name for a in atoms])

    return (
        mode = :ml,
        name = pred.name,
        sim_name = pred.sim_name,
        snap = pred.snap,
        atoms = atoms,
        atoms_tag = atoms_tag,
        mesh_file = pred.mesh_file,
        atmos_file = pred.atmos_file,
        model = pred.model,
        pred_h5 = joinpath(
            pred.train_dir,
            "output_3D_sim_s5_$(pred.name)_$(pred.model)_$(atoms_tag).hdf5"
        ),
        pred_key = "nlte_populations",
        plot_diagnostics = false,
        out_h5 = joinpath(
            pred.train_dir,
            "intensity_ml_$(pred.name)_$(pred.model)_$(atoms_tag).h5"
        ),
        out_prefix = joinpath(pred.train_dir, "diag_ml"),
        x_pick = 33,
        y_pick = 21,
        voigt = base_voigt_config()
    )
end

# -----------------------------
# MODE 2 — Original Bifrost NLTE pops
# -----------------------------
function config_bifrost(pred)
    atoms = atom_configs(pred)
    atoms_tag = atom_tag([a.name for a in atoms])

    return (
        mode = :bifrost,
        name = pred.name,
        sim_name = pred.sim_name,
        snap = pred.snap,
        atoms = atoms,
        atoms_tag = atoms_tag,
        mesh_file = pred.mesh_file,
        atmos_file = pred.atmos_file,
        out_h5 = "IO/intensity_bifrost_$(pred.name)_$(atoms_tag).h5",
        out_prefix = "diag_bifrost",
        x_pick = 33,
        y_pick = 21,
        voigt = base_voigt_config()
    )
end

function config_tiago(pred)
    atoms = atom_configs(pred)
    atoms_tag = atom_tag([a.name for a in atoms])

    return (
        mode = :tiago,
        name = pred.name,
        sim_name = pred.sim_name,
        snap = pred.snap,
        atoms = atoms,
        atoms_tag = atoms_tag,
        mesh_file = pred.mesh_file,
        atmos_file = pred.atmos_file,
        out_h5 = "IO/intensity_bifrost_TIAGO_MODE_$(pred.name)_$(atoms_tag).h5",
        out_prefix = "diag_bifrost",
        x_pick = 33,
        y_pick = 21,
        voigt = base_voigt_config()
    )
end

# ============================================================
# USER CHOOSES WHICH ONE TO RUN
# ============================================================

function build_config(pred; mode=RUN_MODE)
    if mode == :ml
        return config_ml(pred)
    elseif mode == :bifrost
        return config_bifrost(pred)
    elseif mode == :tiago
        return config_tiago(pred)
    else
        error("Unknown mode: $(mode)")
    end
end


function split_atoms(dep_coeff, atoms)

    # dep_coeff shape: (nx, ny, nz, total_levels)

    offsets = cumsum([0; [a.nlevels for a in atoms]])

    out = Dict{String,Any}()

    for (i,a) in enumerate(atoms)
        s = offsets[i] + 1
        e = offsets[i+1]

        println("Atom ", a.name, ": levels ", s, ":", e)

        out[a.name] = view(dep_coeff, :, :, :, s:e)
    end

    return out
end

# -----------------------------
# Populations + diagnostics helpers
# -----------------------------
function lte_pops_saha(atom, atmos::Atmosphere3D)

    # --------------------------------------------------
    # Total hydrogen density (all hydrogen particles)
    # --------------------------------------------------
    nH = atmos.hydrogen1_density .+ atmos.proton_density

    # --------------------------------------------------
    # Convert abundance to ratio N_species / N_H
    # abundance stored in log scale: log10(N/H)+12
    # --------------------------------------------------
    ratio = 10.0^(atom.abundance - 12.0)

    # --------------------------------------------------
    # Total density of this species
    # --------------------------------------------------
    n_species = ratio .* nH

    # --------------------------------------------------
    # LTE populations from Muspel
    # --------------------------------------------------
    pops = Muspel.saha_boltzmann.(
        Ref(atom),
        atmos.temperature,
        atmos.electron_density,
        n_species
    )

    # --------------------------------------------------
    # Convert Vector{SVector} → Float32 array
    # --------------------------------------------------
    pops_s = SVector{atom.nlevels,Float32}.(pops)
    reint  = reshape(reinterpret(Float32, pops_s), atom.nlevels, size(pops_s)...)

    # → (nz, nx, ny, nlevels)
    pops4d = permutedims(reint, (2,3,4,1))

    return pops4d
end

function load_pred_depcoeff(pred_h5::String, pred_key::String)

    h5open(pred_h5, "r") do f
        raw = read(f[pred_key])

        dep_coeff = PermutedDimsArray(raw, (2, 3, 4, 1))
        return dep_coeff

    end
end

function load_multi3d_pops(pops_file::String, atmos::Atmosphere3D, nlevels::Int)
    pops_out_nlte, pops_out_lte = read_pops_multi3d(pops_file, atmos.nx, atmos.ny, atmos.nz, nlevels)
    return pops_out_nlte, pops_out_lte
end


# -----------------------------
# Synthesis
# -----------------------------
function default_background_atom_files()
    bckgr_atoms = [
        "Al.yaml","C.yaml","Ca.yaml","Fe.yaml","H_6.yaml","He.yaml","KI.yaml","Mg.yaml",
        "N.yaml","Na.yaml","NiI.yaml","O.yaml","S.yaml","Si.yaml",
    ]
    return [joinpath(AtomicData.get_atom_dir(), a) for a in bckgr_atoms]
end

function synthesize_intensity_3d(
    atms::Atmosphere3D, h_atom,
    line_index::Int,
    nltepops_nz_nx_ny_nlev,
    lower_level::Int,
    upper_level::Int;
    voigt_cfg=(a_min=1f-4,a_max=1f1,a_n=20000,v_min=0f0,v_max=5f2,v_n=2500)
)
    if Threads.nthreads() == 1 && Sys.CPU_THREADS > 1
        @warn "synthesize_intensity_3d has only one Julia thread; start Julia with --threads=auto (or --threads=N) to use multiple CPU cores"
    end

    my_line = h_atom.lines[line_index]

    a = LinRange(Float32(voigt_cfg.a_min), Float32(voigt_cfg.a_max), voigt_cfg.a_n)
    v = LinRange(Float32(voigt_cfg.v_min), Float32(voigt_cfg.v_max), voigt_cfg.v_n)
    voigt_itp = create_voigt_itp(a, v)

    atom_files = default_background_atom_files()
    σ_itp = get_σ_itp(atms, my_line.λ0, atom_files)

    intensity = Array{Float32,3}(undef, my_line.nλ, atms.ny, atms.nx)
    ncolumns = atms.nx * atms.ny
    nworkers = min(Threads.nthreads(), ncolumns)
    p = Progress(ncolumns; desc="Synthesis columns")

    n_u = nltepops_nz_nx_ny_nlev[:, :, :, upper_level]
    n_l = nltepops_nz_nx_ny_nlev[:, :, :, lower_level]

    # Expose every (x, y) column to the thread pool.  The old outer-x loop
    # could only keep min(nx, nthreads()) threads busy.  One buffer per worker
    # avoids both repeated allocations and shared mutable state.
    Threads.@threads :static for worker in 1:nworkers
        buf = RTBuffer(atms.nz, my_line.nλ, Float32)
        for column in worker:nworkers:ncolumns
            j = mod1(column, atms.ny)
            i = fld(column - 1, atms.ny) + 1
            calc_line_prep!(my_line, buf, atms[:, j, i], σ_itp)
            calc_line_1D!(my_line, buf, my_line.λ, atms[:, j, i],
                          n_u[:, j, i], n_l[:, j, i], voigt_itp)
            intensity[:, j, i] = buf.intensity
            next!(p)
        end
    end

    return (intensity=intensity, wave=my_line.λ, line=my_line)
end

# -----------------------------
# Diagnostics + output
# -----------------------------
function save_intensity_h5(out_h5::String, intensity, wave)
    f = h5open(out_h5, "w")
    f["intensity"] = intensity
    f["wave_HA"]   = wave
    close(f)
end


function output_atom_done(out_h5::String, atom_name::String)
    isfile(out_h5) || return false

    h5open(out_h5, "r") do f
        atom_name in keys(f) || return false
        grp = f[atom_name]
        return "intensity" in keys(grp) && "wave" in keys(grp)
    end
end


function pending_atoms(cfg)
    return [a for a in cfg.atoms if !output_atom_done(cfg.out_h5, a.name)]
end


function missing_population_inputs(atoms)
    missing = []

    for a in atoms
        pops_dir = dirname(a.pops_file)

        if !isdir(pops_dir)
            push!(missing, (atom = a.name, path = pops_dir, kind = "directory"))
        elseif !isfile(a.pops_file)
            push!(missing, (atom = a.name, path = a.pops_file, kind = "file"))
        end
    end

    return missing
end


function ensure_forward_inputs(cfg)
    required = [
        (label = "mesh", path = cfg.mesh_file),
        (label = "atmosphere", path = cfg.atmos_file),
    ]

    if cfg.mode == :ml
        push!(required, (label = "ML prediction", path = cfg.pred_h5))
    end

    missing = [item for item in required if !isfile(item.path)]
    isempty(missing) && return

    lines = join(["  - $(item.label): $(item.path)" for item in missing], "\n")
    error("Missing inputs for --predname $(repr(cfg.name)):\n$(lines)")
end


function save_synthesis_results(out_h5::String, results)
    mode = isfile(out_h5) ? "r+" : "w"

    h5open(out_h5, mode) do f
        for (name, syn) in results
            if name in keys(f)
                delete_object(f, name)
            end

            grp = create_group(f, name)
            grp["intensity"] = syn.intensity
            grp["wave"]      = syn.wave
        end
    end
end


function calc_multi3d_hα(mesh_file, atmos_file, pops_file, atom_file)
    h_atom = read_atom(atom_file)
    my_line = h_atom.lines[5]  #  index 5 for Halpha

    atmos, h_pops = read_atmos_hpops_multi3d(mesh_file, atmos_file, pops_file)
    n_u = h_pops[:, :, :, 3]
    n_l = h_pops[:, :, :, 2]

    # Continuum opacity structures
    bckgr_atoms = [
        "Al.yaml",
        "C.yaml",
        "Ca.yaml",
        "Fe.yaml",
        "H_6.yaml",
        "He.yaml",
        "KI.yaml",
        "Mg.yaml",
        "N.yaml",
        "Na.yaml",
        "NiI.yaml",
        "O.yaml",
        "S.yaml",
        "Si.yaml",
    ]
    atom_files = [joinpath(AtomicData.get_atom_dir(), a) for a in bckgr_atoms]
    σ_itp = get_σ_itp(atmos, my_line.λ0, atom_files)

    a = LinRange(1f-4, 1.5f1, 20000)
    v = LinRange(0f2, 5f2, 2500)
    voigt_itp = create_voigt_itp(a, v)

    intensity = Array{Float32, 3}(undef, my_line.nλ, atmos.ny, atmos.nx)
    p = ProgressMeter.Progress(atmos.nx)

    Threads.@threads for i in 1:atmos.nx
        buf = RTBuffer(atmos.nz, my_line.nλ, Float32)  # allocate inside for local scope
        for j in 1:atmos.ny
            calc_line_prep!(my_line, buf, atmos[:, j, i], σ_itp)
            calc_line_1D!(my_line, buf, my_line.λ, atmos[:, j, i], n_u[:, j, i], n_l[:, j, i], voigt_itp)
            intensity[:, j, i] = buf.intensity
        end
        ProgressMeter.next!(p)
    end

    return (intensity=intensity, wave=my_line.λ, line=my_line)
end


# -----------------------------
# Main pipeline
# -----------------------------
function main(cfg)

    println("Mode        : ", cfg.mode)
    println("Dataset     : ", cfg.name)
    println("Simulation  : ", cfg.sim_name)
    println("Snapshot    : ", cfg.snap)
    println("Threads     : ", Threads.nthreads())
    println("Output      : ", cfg.out_h5)

    atoms_to_run = pending_atoms(cfg)

    if isempty(atoms_to_run)
        println("Skipping $(cfg.name) snap $(cfg.snap): all requested atom outputs already exist.")
        return
    end

    println("Pending atoms: ", join([a.name for a in atoms_to_run], ", "))

    if cfg.mode == :bifrost
        missing = missing_population_inputs(atoms_to_run)

        if !isempty(missing)
            println(
                "Skipping $(cfg.name) snap $(cfg.snap): level populations are not available."
            )
            for item in missing
                println("  - $(item.atom) missing $(item.kind): $(item.path)")
            end
            return
        end
    end

    ensure_forward_inputs(cfg)

    println("Reading atmosphere...")
    atmos = read_atmos_multi3d(cfg.mesh_file, cfg.atmos_file)

    # ============================================================
    # ML MODE
    # ============================================================
    if cfg.mode == :ml

        nlte_pop_full = load_pred_depcoeff(cfg.pred_h5, cfg.pred_key)

        # ---------------------------------------------------
        # Split ML populations per atom
        # ---------------------------------------------------
        nlte_per_atom = split_atoms(nlte_pop_full, cfg.atoms)

        # Dictionary to store final NLTE pops per atom
        nlte_atoms = Dict{String,Any}()

        for a in atoms_to_run

            h_atom = Muspel.read_atom(a.atom_file)

            nlte_pop = nlte_per_atom[a.name]

            @assert size(nlte_pop,4) == h_atom.nlevels "Level mismatch for atom $(a.name)"

            nlte_atoms[a.name] = nlte_pop

        end

    # ============================================================
    # BIFROST MODE
    # ============================================================
    elseif cfg.mode == :bifrost

        println("Reading Multi3D pops...")

        nlte_atoms = Dict{String,Any}()

        for a in atoms_to_run

            atom = Muspel.read_atom(a.atom_file)

            pops_out_nlte, pops_out_lte =
                read_pops_multi3d(a.pops_file, atmos.nx, atmos.ny, atmos.nz, atom.nlevels)

            nlte_atoms[a.name] = pops_out_nlte
        end

    elseif cfg.mode == :tiago
        println("tiago mode")
    else
        error("Unknown mode")
    end

    # for (k,v) in nlte_atoms
    #     println(k, " NLTE shape = ", size(v))
    # end

    println("Synthesizing line profiles...")
    results = Dict{String,Any}()

    for a in atoms_to_run

        println("Synthesizing atom: ", a.name)

        if cfg.mode == :tiago
            results[a.name] = calc_multi3d_hα(cfg.mesh_file, cfg.atmos_file, a.pops_file, a.atom_file)
        else
            h_atom = Muspel.read_atom(a.atom_file)

            syn = synthesize_intensity_3d(
                atmos,
                h_atom,
                a.line_index,
                nlte_atoms[a.name],
                a.lower_level,
                a.upper_level;
                voigt_cfg = cfg.voigt
            )

            results[a.name] = syn
        end
    end

    println("Saving output...")
    save_synthesis_results(cfg.out_h5, results)

    println("Done.")
end

# Run
function run_all(; predname=nothing)
    pred_data = load_multi3d_pred_data(predname)
    isempty(pred_data) && error("No prediction atmospheres were selected")

    for (idx, pred) in enumerate(pred_data)
        println("")
        println("============================================================")
        println("Forward synthesis $(idx)/$(length(pred_data)): $(pred.name)")
        println("============================================================")

        cfg = build_config(pred)
        main(cfg)
    end
end

cli_args = parse_cli_args(ARGS)
run_all(predname=cli_args.predname)
