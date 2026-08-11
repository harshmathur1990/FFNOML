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
#   - Optional ML consistency iteration, either charge-only or 3D
#     hydrogen statistical equilibrium followed by charge conservation
#   - Optional MPI x-slab decomposition for synthesis and both consistency modes
#     with a single rank-0-owned distributed FFNoML launcher
#   - Synthesizes 1D line profiles for all columns (nx, ny) using Muspel
#   - Writes intensity + wavelength to an HDF5 file
#   - Writes two diagnostic plots (PNG)
# -----------------------------------------------------------------------------

ENV["GKSwstype"] = "100"     # file / offscreen
ENV["GKS_WSTYPE"] = "100"    # some setups use this spelling

function forward_startup_log(stage)
    slurm_rank = get(ENV, "SLURM_PROCID", "?")
    local_rank = get(ENV, "SLURM_LOCALID", "?")
    println(
        stderr,
        "[Forward startup] host=$(gethostname()) " *
        "slurm_rank=$(slurm_rank) " *
        "local_rank=$(local_rank) stage=$(stage)",
    )
    flush(stderr)
end

forward_startup_log("loading Julia packages")
using Muspel
using StaticArrays
using AtomicData
using HDF5
using ProgressMeter
using Base.Threads
using Interpolations
using Serialization
using Sockets
forward_startup_log("Julia packages loaded")

include("ForwardMPI.jl")
include("ForwardDiagnostics.jl")
include("HydrogenSE.jl")
forward_startup_log("source files loaded")


# ============================================================
# CONFIGURATION
# ============================================================

# -----------------------------
# Config loaded from config.py
# -----------------------------

const RUN_MODE = :ml
const FORWARD_ATOMS = ["H"]
# const FORWARD_ATOMS = ["CA"]
# const FORWARD_ATOMS = ["H", "CA"]

const DEFAULT_CHARGE_CONSERVATION_TOLERANCE = 1f-4
const DEFAULT_FSDP_NPROC_PER_NODE = 4
const ELECTRON_DENSITY_INPUT_CHANNEL = 5
const DEFAULT_POPULATION_CONSISTENCY_MODE = :charge_only
const DEFAULT_HYDROGEN_SE_RELAXATION = 1.0
const DEFAULT_HYDROGEN_SE_WAVELENGTH_STRIDE = 1
const CONSISTENCY_CACHE_FORMAT_VERSION = 5
const DEFAULT_ATOM_DIR = "/cluster/work/projects/nn2834k/harshm/multi3d/input/atoms"

function atom_tag(atom_names)
    return join(atom_names, "_")
end

function load_multi3d_pred_data()
    script = """
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
for item in config.MULTI3D_PRED_DATA:
    forward = item.get("FORWARD", True)
    if not isinstance(forward, bool):
        raise TypeError(
            f'MULTI3D_PRED_DATA[{item.get("NAME", "<unnamed>")!r}]["FORWARD"] '
            "must be True or False"
        )
    if not forward:
        continue

    nonlte_ne = item.get("NONLTE_NE")
    if nonlte_ne is not None and not isinstance(nonlte_ne, bool):
        raise TypeError(
            f'MULTI3D_PRED_DATA[{item.get("NAME", "<unnamed>")!r}]["NONLTE_NE"] '
            "must be True, False, or None"
        )

    max_iterations = item.get("CHARGE_CONSERVATION_MAX_ITERATIONS")
    if max_iterations is not None and (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations < 0
    ):
        raise TypeError(
            f'MULTI3D_PRED_DATA[{item.get("NAME", "<unnamed>")!r}]'
            '["CHARGE_CONSERVATION_MAX_ITERATIONS"] must be a non-negative '
            "integer or None"
        )

    consistency_mode = item.get("POPULATION_CONSISTENCY_MODE")
    if consistency_mode is not None and not isinstance(consistency_mode, str):
        raise TypeError(
            f'MULTI3D_PRED_DATA[{item.get("NAME", "<unnamed>")!r}]'
            '["POPULATION_CONSISTENCY_MODE"] must be a string or None'
        )

    wavelength_stride = item.get("HYDROGEN_SE_WAVELENGTH_STRIDE")
    if wavelength_stride is not None and (
        not isinstance(wavelength_stride, int)
        or isinstance(wavelength_stride, bool)
        or wavelength_stride <= 0
    ):
        raise TypeError(
            f'MULTI3D_PRED_DATA[{item.get("NAME", "<unnamed>")!r}]'
            '["HYDROGEN_SE_WAVELENGTH_STRIDE"] must be a positive integer or None'
        )

    print("\\t".join([
        item["NAME"],
        item.get("SIM_NAME", "_".join(item["NAME"].split("_")[:-1])),
        item.get("SNAP", item["NAME"].split("_")[-1]),
        item["MESH"],
        item["MULTI3D_ATMOS"],
        item.get("TRAIN_DIR", model_dir).rstrip("/"),
        config.MODEL,
        config.MODEL_FILE,
        "auto" if nonlte_ne is None else str(nonlte_ne).lower(),
        "auto" if max_iterations is None else str(max_iterations),
        "auto" if consistency_mode is None else consistency_mode,
        "auto" if wavelength_stride is None else str(wavelength_stride),
    ]))
"""

    output = cd(@__DIR__) do
        read(`python3 -c $script`, String)
    end

    pred_data = []
    for line in split(chomp(output), "\n")
        isempty(line) && continue
        fields = split(line, "\t")
        length(fields) == 12 || error("Unexpected config.py output: $(line)")
        name, sim_name, snap, mesh_file, atmos_file, train_dir, model, model_file,
            nonlte_ne_value, max_iterations_value, consistency_mode_value,
            wavelength_stride_value = fields
        nonlte_ne = nonlte_ne_value == "auto" ? nothing : nonlte_ne_value == "true"
        max_iterations = max_iterations_value == "auto" ?
                         nothing : parse(Int, max_iterations_value)
        consistency_mode = consistency_mode_value == "auto" ?
                           nothing : consistency_mode_value
        wavelength_stride = wavelength_stride_value == "auto" ?
                            nothing : parse(Int, wavelength_stride_value)
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
                model_file = model_file,
                nonlte_ne = nonlte_ne,
                charge_conservation_max_iterations = max_iterations,
                population_consistency_mode = consistency_mode,
                hydrogen_se_wavelength_stride = wavelength_stride,
            ),
        )
    end

    return pred_data
end

function all_atom_configs(pred)
    snapshot_dir = dirname(pred.mesh_file)
    atom_dir = get(ENV, "FNOML_ATOM_DIR", DEFAULT_ATOM_DIR)

    return [
        (
            name = "H",
            atom_file = joinpath(atom_dir, "atom.h6_tiago2.yaml"),
            pops_file = joinpath(snapshot_dir, "H", "out_pop"),
            nlevels = 6,
            line_index = 5,
            lower_level = 2,
            upper_level = 3
        ),
        (
            name = "CA",
            atom_file = joinpath(atom_dir, "atom.ca2.yaml"),
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
        model_file = pred.model_file,
        pred_h5 = joinpath(
            pred.train_dir,
            "output_3D_sim_s5_$(pred.name)_$(pred.model)_$(atoms_tag).hdf5"
        ),
        pred_key = "nlte_populations",
        solve_h5 = joinpath(pred.train_dir, "3D_sim_predict_$(pred.name).hdf5"),
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


function split_atoms(dep_coeff, atoms; verbose::Bool=true)

    # dep_coeff shape after loading: (nz, ny, nx, total_levels)

    expected_levels = sum(a.nlevels for a in atoms)
    size(dep_coeff, 4) == expected_levels || error(
        "Prediction has $(size(dep_coeff, 4)) levels, but FORWARD_ATOMS requires $(expected_levels)"
    )

    offsets = cumsum([0; [a.nlevels for a in atoms]])

    out = Dict{String,Any}()

    for (i,a) in enumerate(atoms)
        s = offsets[i] + 1
        e = offsets[i+1]

        verbose && println("Atom ", a.name, ": levels ", s, ":", e)

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

function load_pred_depcoeff(
    pred_h5::String,
    pred_key::String;
    x_range=nothing,
)

    h5open(pred_h5, "r") do f
        dataset = f[pred_key]
        raw = x_range === nothing ? read(dataset) : dataset[:, :, :, x_range]

        dep_coeff = PermutedDimsArray(raw, (2, 3, 4, 1))
        return dep_coeff

    end
end

function load_multi3d_pops(pops_file::String, atmos::Atmosphere3D, nlevels::Int)
    pops_out_nlte, pops_out_lte = read_pops_multi3d(pops_file, atmos.nx, atmos.ny, atmos.nz, nlevels)
    return pops_out_nlte, pops_out_lte
end


# -----------------------------
# ML charge conservation
# -----------------------------
function charge_conservation_background_atoms()
    abundances = AtomicData.get_solar_abundances()
    atoms = []

    for atom_file in default_background_atom_files()
        atom = Muspel.read_atom(atom_file)
        atom.element == :H && continue

        abundance = get(abundances, atom.element, nothing)
        abundance === nothing && error(
            "No solar abundance is available for background atom $(atom.element)"
        )
        push!(atoms, (atom = atom, abundance = abundance))
    end

    return atoms
end


function hydrogen_positive_charge(nlte_h, hydrogen_atom)
    size(nlte_h, 4) == hydrogen_atom.nlevels || error(
        "Hydrogen population level count does not match the hydrogen atom"
    )

    charge = zeros(eltype(nlte_h), size(nlte_h)[1:3])
    for level in 1:hydrogen_atom.nlevels
        ion_charge = hydrogen_atom.stage[level] - 1
        ion_charge == 0 && continue
        @views charge .+= ion_charge .* nlte_h[:, :, :, level]
    end
    return charge
end


function update_atmosphere_hydrogen_densities!(atmos::Atmosphere3D, nlte_h, hydrogen_atom)
    neutral_levels = hydrogen_atom.stage .== 1
    ionized_levels = hydrogen_atom.stage .> 1
    atmos.hydrogen1_density .= dropdims(sum(nlte_h[:, :, :, neutral_levels]; dims=4); dims=4)
    atmos.proton_density .= dropdims(sum(nlte_h[:, :, :, ionized_levels]; dims=4); dims=4)
    return nothing
end


"""
Evaluate Eq. (A.5) of Bjørgen et al. (2019) for a fixed set of predicted
hydrogen populations. Hydrogen contributes its non-LTE positive charge; all
other elements contribute LTE charge from Saha-Boltzmann populations evaluated
at the electron density used for the prediction.
"""
function charge_conservation_electron_density(
    atmos::Atmosphere3D,
    input_ne,
    nlte_h,
    hydrogen_atom,
    background_atoms,
    ;
    total_hydrogen_density=nothing,
)
    size(input_ne) == size(atmos.temperature) || error(
        "Electron-density shape $(size(input_ne)) does not match atmosphere shape $(size(atmos.temperature))"
    )
    size(nlte_h)[1:3] == size(input_ne) || error(
        "Hydrogen-population shape $(size(nlte_h)[1:3]) does not match electron-density shape $(size(input_ne))"
    )

    hydrogen_density = total_hydrogen_density === nothing ?
                       atmos.hydrogen1_density .+ atmos.proton_density :
                       total_hydrogen_density
    hydrogen_charge = hydrogen_positive_charge(nlte_h, hydrogen_atom)
    output_ne = similar(input_ne)

    Threads.@threads for index in eachindex(output_ne)
        temperature = Float64(atmos.temperature[index])
        ne = max(Float64(input_ne[index]), eps(Float64))
        n_h = Float64(hydrogen_density[index])
        total_charge = Float64(hydrogen_charge[index])

        for item in background_atoms
            populations = Muspel.saha_boltzmann(
                item.atom,
                temperature,
                ne,
                item.abundance * n_h,
            )
            for level in 1:item.atom.nlevels
                total_charge += (item.atom.stage[level] - 1) * populations[level]
            end
        end

        output_ne[index] = total_charge
    end

    return output_ne
end


function maximum_relative_change(new_values, old_values)
    size(new_values) == size(old_values) || error("Relative-change array shapes differ")
    floor_value = eps(Float64)
    maximum_change = 0.0

    @inbounds for index in eachindex(new_values, old_values)
        new_value = Float64(new_values[index])
        old_value = Float64(old_values[index])
        denominator = max(abs(new_value), abs(old_value), floor_value)
        maximum_change = max(maximum_change, abs(new_value - old_value) / denominator)
    end

    return maximum_change
end


function read_solving_electron_density(
    solve_h5::String,
    atmos::Atmosphere3D;
    global_nx::Int=atmos.nx,
    x_range=1:atmos.nx,
)
    h5open(solve_h5, "r") do file
        inputs = file["inputs"]
        dims = size(inputs)
        reverse_dims = (atmos.ny, global_nx, atmos.nz, 6, 1)
        canonical_dims = (1, 6, atmos.nz, global_nx, atmos.ny)

        if dims == reverse_dims
            log_ne = inputs[:, x_range, :, ELECTRON_DENSITY_INPUT_CHANNEL, 1]
            return exp10.(permutedims(log_ne, (3, 1, 2)))
        elseif dims == canonical_dims
            log_ne = inputs[1, ELECTRON_DENSITY_INPUT_CHANNEL, :, x_range, :]
            return exp10.(permutedims(log_ne, (1, 3, 2)))
        else
            error(
                "Unexpected solving-set inputs shape $(dims); expected $(reverse_dims) " *
                "(HDF5.jl order) or $(canonical_dims)"
            )
        end
    end
end


function write_solving_electron_density!(solve_h5::String, electron_density)
    all(isfinite, electron_density) || error("Charge conservation produced non-finite electron densities")
    all(>(0), electron_density) || error("Charge conservation produced non-positive electron densities")

    log_ne = Float32.(log10.(electron_density))
    nz, ny, nx = size(electron_density)
    h5open(solve_h5, "r+") do file
        inputs = file["inputs"]
        dims = size(inputs)
        reverse_dims = (ny, nx, nz, 6, 1)
        canonical_dims = (1, 6, nz, nx, ny)

        if dims == reverse_dims
            inputs[:, :, :, ELECTRON_DENSITY_INPUT_CHANNEL, 1] = permutedims(log_ne, (2, 3, 1))
        elseif dims == canonical_dims
            inputs[1, ELECTRON_DENSITY_INPUT_CHANNEL, :, :, :] = permutedims(log_ne, (1, 3, 2))
        else
            error(
                "Unexpected solving-set inputs shape $(dims); expected $(reverse_dims) " *
                "(HDF5.jl order) or $(canonical_dims)"
            )
        end
    end
end


function consistency_method_tag(mode::Symbol)
    return replace(String(mode), "_" => "-")
end


function consistency_result_path(cfg, mode::Symbol)
    model = hasproperty(cfg, :model) ? cfg.model : "model"
    atoms_tag = hasproperty(cfg, :atoms_tag) ? cfg.atoms_tag : atom_tag([a.name for a in cfg.atoms])
    filename = "nonlte_electron_density_$(cfg.name)_$(model)_$(atoms_tag)_" *
               "$(consistency_method_tag(mode)).hdf5"
    return joinpath(dirname(abspath(cfg.solve_h5)), filename)
end


function consistency_prediction_work_path(result_h5::String)
    stem, _ = splitext(result_h5)
    return stem * ".prediction-work.hdf5"
end


function hdf_attribute(attributes_object, name, default)
    name in keys(attributes_object) || return default
    return read(attributes_object[name])
end


function set_hdf_attribute!(object, name::String, value)
    name in keys(attributes(object)) && delete_attribute(object, name)
    attributes(object)[name] = value
    return nothing
end


function replace_hdf_dataset!(parent, name::String, values)
    name in keys(parent) && delete_object(parent, name)
    parent[name] = values
    return parent[name]
end


function source_file_signature(path::AbstractString)
    info = stat(path)
    return (path=abspath(path), size=Int64(info.size), mtime=Float64(info.mtime))
end


const CONSISTENCY_SCALAR_HISTORY_FIELDS = (
    "iteration",
    "converged",
    "electron_density_delta",
    "ffno_population_delta",
    "se_population_correction",
    "se_residual",
    "background_scattering_iterations",
    "background_scattering_residual",
    "electron_density_min",
    "electron_density_max",
    "ffno_population_min",
    "ffno_population_max",
    "prediction_seconds",
    "prediction_read_seconds",
    "se_seconds",
    "charge_seconds",
    "io_seconds",
    "iteration_seconds",
)

const CONSISTENCY_LEVEL_HISTORY_FIELDS = (
    "ffno_population_level_delta",
    "ffno_population_level_min",
    "ffno_population_level_max",
)


function consistency_history_complete(file, iterations::Int)
    iterations == 0 && return true
    "iteration_history" in keys(file) || return false
    history = file["iteration_history"]
    all(name -> name in keys(history) && length(history[name]) >= iterations,
        CONSISTENCY_SCALAR_HISTORY_FIELDS) || return false
    return all(
        name -> name in keys(history) && ndims(history[name]) == 2 &&
                size(history[name], 2) >= iterations,
        CONSISTENCY_LEVEL_HISTORY_FIELDS,
    )
end


function trim_consistency_history!(result_h5::String, iterations::Int)
    h5open(result_h5, "r+") do file
        "iteration_history" in keys(file) || return
        history = file["iteration_history"]
        for name in CONSISTENCY_SCALAR_HISTORY_FIELDS
            name in keys(history) || continue
            values = vec(read(history[name]))
            length(values) > iterations || continue
            replace_hdf_dataset!(history, name, values[1:iterations])
        end
        for name in CONSISTENCY_LEVEL_HISTORY_FIELDS
            name in keys(history) || continue
            values = read(history[name])
            size(values, 2) > iterations || continue
            replace_hdf_dataset!(history, name, values[:, 1:iterations])
        end
    end
    return nothing
end


function inspect_consistency_result(
    result_h5::String,
    cfg,
    mode::Symbol,
    tolerance::Real,
    hydrogen_se_relaxation::Real,
    hydrogen_se_wavelength_stride::Int,
)
    isfile(result_h5) || return (
        compatible=false,
        converged=false,
        iterations=0,
        reason="not found",
    )
    source = source_file_signature(cfg.solve_h5)
    hydrogen_config = only(filter(atom -> atom.name == "H", cfg.atoms))
    hydrogen_source = source_file_signature(hydrogen_config.atom_file)
    model_source = hasproperty(cfg, :model_file) ? source_file_signature(cfg.model_file) : nothing
    return h5open(result_h5, "r") do file
        attrs = attributes(file)
        compatible = hdf_attribute(attrs, "forward_consistency_format_version", 0) ==
                     CONSISTENCY_CACHE_FORMAT_VERSION &&
                     hdf_attribute(attrs, "method", "") == String(mode) &&
                     hdf_attribute(attrs, "source_solve_h5", "") == source.path &&
                     hdf_attribute(attrs, "source_size_bytes", Int64(-1)) == source.size &&
                     hdf_attribute(attrs, "hydrogen_atom_file", "") == hydrogen_source.path &&
                     hdf_attribute(attrs, "hydrogen_atom_size_bytes", Int64(-1)) == hydrogen_source.size &&
                     isapprox(
                         Float64(hdf_attribute(attrs, "source_mtime", -1.0)),
                         source.mtime;
                         atol=1e-6,
                     ) &&
                     isapprox(
                         Float64(hdf_attribute(attrs, "hydrogen_atom_mtime", -1.0)),
                         hydrogen_source.mtime;
                         atol=1e-6,
                     )
        if mode == :hydrogen_se_3d
            compatible = compatible && isapprox(
                Float64(hdf_attribute(attrs, "hydrogen_se_relaxation", -1.0)),
                Float64(hydrogen_se_relaxation);
                atol=eps(Float64),
            ) && hdf_attribute(attrs, "hydrogen_se_wavelength_stride", -1) ==
                 hydrogen_se_wavelength_stride &&
                 hdf_attribute(attrs, "background_scattering_max_iterations", -1) ==
                 HSE_SCATTERING_MAX_ITERATIONS &&
                 isapprox(
                     Float64(hdf_attribute(attrs, "background_scattering_tolerance", -1.0)),
                     HSE_SCATTERING_TOLERANCE;
                     atol=eps(Float64),
                 )
        end
        if model_source !== nothing
            compatible = compatible &&
                hdf_attribute(attrs, "model_file", "") == model_source.path &&
                hdf_attribute(attrs, "model_size_bytes", Int64(-1)) == model_source.size &&
                isapprox(
                    Float64(hdf_attribute(attrs, "model_mtime", -1.0)),
                    model_source.mtime;
                    atol=1e-6,
                )
        end
        compatible || return (
            compatible=false,
            converged=false,
            iterations=0,
            reason="metadata or source solving-set changed",
        )

        iterations = Int(hdf_attribute(attrs, "iterations_completed", 0))
        has_state = "electron_density" in keys(file) &&
                    cfg.pred_key in keys(file) &&
                    "last_ffno_hydrogen_populations" in keys(file)
        if iterations > 0 && !has_state
            return (
                compatible=false,
                converged=false,
                iterations=0,
                reason="incomplete consistency checkpoint",
            )
        end
        consistency_history_complete(file, iterations) || return (
            compatible=false,
            converged=false,
            iterations=0,
            reason="incomplete consistency iteration history",
        )
        electron_delta = Float64(hdf_attribute(attrs, "final_electron_density_delta", Inf))
        se_residual = Float64(hdf_attribute(attrs, "final_se_residual", Inf))
        population_correction = Float64(
            hdf_attribute(attrs, "final_se_population_correction", Inf)
        )
        tolerance_satisfied = electron_delta <= tolerance &&
            (mode == :charge_only ||
             (se_residual <= tolerance && population_correction <= tolerance))
        converged = Bool(hdf_attribute(attrs, "converged", false)) &&
                    has_state && tolerance_satisfied
        return (
            compatible=true,
            converged=converged,
            iterations=iterations,
            reason=converged ? "converged cache" : "unconverged checkpoint",
        )
    end
end


function prepare_consistency_result!(
    result_h5::String,
    cfg,
    mode::Symbol,
    tolerance::Real,
    hydrogen_se_relaxation::Real,
    hydrogen_se_wavelength_stride::Int,
)
    state = inspect_consistency_result(
        result_h5,
        cfg,
        mode,
        tolerance,
        hydrogen_se_relaxation,
        hydrogen_se_wavelength_stride,
    )
    if isfile(result_h5) && !state.compatible
        stale_path = result_h5 * ".stale-$(time_ns())"
        mv(result_h5, stale_path)
        @warn "Archived incompatible electron-density consistency file" result_h5 stale_path reason=state.reason
    end

    if !isfile(result_h5)
        cp(cfg.solve_h5, result_h5; force=false)
        source = source_file_signature(cfg.solve_h5)
        hydrogen_config = only(filter(atom -> atom.name == "H", cfg.atoms))
        hydrogen_source = source_file_signature(hydrogen_config.atom_file)
        model_source = hasproperty(cfg, :model_file) ? source_file_signature(cfg.model_file) : nothing
        h5open(result_h5, "r+") do file
            attrs = attributes(file)
            attrs["forward_consistency_format_version"] = CONSISTENCY_CACHE_FORMAT_VERSION
            attrs["method"] = String(mode)
            attrs["source_solve_h5"] = source.path
            attrs["source_size_bytes"] = source.size
            attrs["source_mtime"] = source.mtime
            attrs["hydrogen_atom_file"] = hydrogen_source.path
            attrs["hydrogen_atom_size_bytes"] = hydrogen_source.size
            attrs["hydrogen_atom_mtime"] = hydrogen_source.mtime
            attrs["hydrogen_se_relaxation"] = Float64(hydrogen_se_relaxation)
            attrs["hydrogen_se_wavelength_stride"] = hydrogen_se_wavelength_stride
            attrs["background_scattering_max_iterations"] = HSE_SCATTERING_MAX_ITERATIONS
            attrs["background_scattering_tolerance"] = HSE_SCATTERING_TOLERANCE
            if model_source !== nothing
                attrs["model_file"] = model_source.path
                attrs["model_size_bytes"] = model_source.size
                attrs["model_mtime"] = model_source.mtime
            end
            attrs["converged"] = false
            attrs["iterations_completed"] = 0
            attrs["requested_tolerance"] = Float64(tolerance)
            attrs["created_utc"] = diagnostic_timestamp()
            attrs["updated_utc"] = diagnostic_timestamp()
            "iteration_history" in keys(file) || create_group(file, "iteration_history")
        end
    end
    current_state = inspect_consistency_result(
        result_h5,
        cfg,
        mode,
        tolerance,
        hydrogen_se_relaxation,
        hydrogen_se_wavelength_stride,
    )
    current_state.compatible && trim_consistency_history!(
        result_h5,
        current_state.iterations,
    )
    return inspect_consistency_result(
        result_h5,
        cfg,
        mode,
        tolerance,
        hydrogen_se_relaxation,
        hydrogen_se_wavelength_stride,
    )
end


function read_consistency_electron_density(
    result_h5::String;
    x_range=nothing,
)
    h5open(result_h5, "r") do file
        dataset = file["electron_density"]
        return x_range === nothing ? read(dataset) : dataset[:, :, x_range]
    end
end


function append_history_scalar!(history, name::String, value)
    previous = name in keys(history) ? vec(read(history[name])) : Float64[]
    replace_hdf_dataset!(history, name, vcat(previous, Float64(value)))
    return nothing
end


function append_history_levels!(history, name::String, values)
    column = reshape(Float64.(values), :, 1)
    combined = if name in keys(history)
        previous = read(history[name])
        size(previous, 1) == size(column, 1) || error(
            "Consistency-history hydrogen level count changed for $(name)"
        )
        hcat(previous, column)
    else
        column
    end
    replace_hdf_dataset!(history, name, combined)
    return nothing
end


function write_initial_consistency_state!(result_h5::String, electron_density)
    h5open(result_h5, "r+") do file
        if !("initial_electron_density" in keys(file))
            dataset = replace_hdf_dataset!(file, "initial_electron_density", electron_density)
            attributes(dataset)["units"] = "m^-3"
            attributes(dataset)["dimensions"] = "z,y,x"
        end
    end
    return nothing
end


function finalize_consistency_checkpoint_timings!(
    result_h5::String,
    io_seconds::Real,
    iteration_seconds::Real,
)
    h5open(result_h5, "r+") do file
        history = file["iteration_history"]
        for (name, value) in (
            ("io_seconds", io_seconds),
            ("iteration_seconds", iteration_seconds),
        )
            values = vec(read(history[name]))
            isempty(values) && error("Consistency history $(name) is unexpectedly empty")
            values[end] = Float64(value)
            replace_hdf_dataset!(history, name, values)
        end
        set_hdf_attribute!(file, "updated_utc", diagnostic_timestamp())
    end
    return nothing
end


function write_consistency_checkpoint!(
    result_h5::String,
    work_prediction_h5::String,
    pred_key::String,
    electron_density,
    ffno_hydrogen_populations,
    corrected_hydrogen_populations,
    hydrogen_level_range,
    metrics;
    method::Symbol,
    tolerance::Real,
    max_iterations::Int,
    hydrogen_se_relaxation::Real,
    hydrogen_se_wavelength_stride::Int,
    converged::Bool,
)
    all(isfinite, electron_density) || error("Cannot checkpoint non-finite electron density")
    all(>(0), electron_density) || error("Cannot checkpoint non-positive electron density")
    all(isfinite, ffno_hydrogen_populations) || error(
        "Cannot checkpoint non-finite raw FFNoML hydrogen populations"
    )
    if corrected_hydrogen_populations !== nothing
        all(isfinite, corrected_hydrogen_populations) || error(
            "Cannot checkpoint non-finite corrected hydrogen populations"
        )
        h5open(work_prediction_h5, "r+") do prediction_file
            dataset = prediction_file[pred_key]
            corrected_raw = permutedims(corrected_hydrogen_populations, (4, 1, 2, 3))
            dataset[hydrogen_level_range, :, :, :] = corrected_raw
        end
    end

    raw_ffno_hydrogen = permutedims(ffno_hydrogen_populations, (4, 1, 2, 3))
    h5open(work_prediction_h5, "r") do prediction_file
        h5open(result_h5, "r+") do file
            ne_dataset = replace_hdf_dataset!(file, "electron_density", electron_density)
            attributes(ne_dataset)["units"] = "m^-3"
            attributes(ne_dataset)["dimensions"] = "z,y,x"
            pred_key in keys(file) && delete_object(file, pred_key)
            copy_object(prediction_file, pred_key, file, pred_key)
            attributes(file[pred_key])["dimensions"] = "level,z,y,x"
            ffno_dataset = replace_hdf_dataset!(
                file,
                "last_ffno_hydrogen_populations",
                raw_ffno_hydrogen,
            )
            attributes(ffno_dataset)["dimensions"] = "hydrogen_level,z,y,x"

            history = "iteration_history" in keys(file) ?
                      file["iteration_history"] : create_group(file, "iteration_history")
            append_history_scalar!(history, "iteration", metrics.iteration)
            append_history_scalar!(history, "converged", converged ? 1.0 : 0.0)
            append_history_scalar!(history, "electron_density_delta", metrics.electron_density_delta)
            append_history_scalar!(history, "ffno_population_delta", metrics.ffno_population_delta)
            append_history_scalar!(history, "se_population_correction", metrics.se_population_correction)
            append_history_scalar!(history, "se_residual", metrics.se_residual)
            append_history_scalar!(history, "background_scattering_iterations", metrics.background_scattering_iterations)
            append_history_scalar!(history, "background_scattering_residual", metrics.background_scattering_residual)
            append_history_scalar!(history, "electron_density_min", metrics.electron_density_min)
            append_history_scalar!(history, "electron_density_max", metrics.electron_density_max)
            append_history_scalar!(history, "ffno_population_min", metrics.ffno_population_min)
            append_history_scalar!(history, "ffno_population_max", metrics.ffno_population_max)
            append_history_scalar!(history, "prediction_seconds", metrics.prediction_seconds)
            append_history_scalar!(history, "prediction_read_seconds", metrics.prediction_read_seconds)
            append_history_scalar!(history, "se_seconds", metrics.se_seconds)
            append_history_scalar!(history, "charge_seconds", metrics.charge_seconds)
            append_history_scalar!(history, "io_seconds", metrics.io_seconds)
            append_history_scalar!(history, "iteration_seconds", metrics.iteration_seconds)
            append_history_levels!(history, "ffno_population_level_delta", metrics.ffno_level_changes)
            append_history_levels!(history, "ffno_population_level_min", metrics.population_level_min)
            append_history_levels!(history, "ffno_population_level_max", metrics.population_level_max)

            set_hdf_attribute!(file, "method", String(method))
            set_hdf_attribute!(file, "converged", converged)
            set_hdf_attribute!(file, "iterations_completed", metrics.iteration)
            set_hdf_attribute!(file, "convergence_iteration", converged ? metrics.iteration : -1)
            set_hdf_attribute!(file, "last_run_max_iterations", max_iterations)
            set_hdf_attribute!(file, "requested_tolerance", Float64(tolerance))
            set_hdf_attribute!(file, "final_electron_density_delta", Float64(metrics.electron_density_delta))
            set_hdf_attribute!(file, "final_ffno_population_delta", Float64(metrics.ffno_population_delta))
            set_hdf_attribute!(file, "final_se_population_correction", Float64(metrics.se_population_correction))
            set_hdf_attribute!(file, "final_se_residual", Float64(metrics.se_residual))
            set_hdf_attribute!(file, "final_background_scattering_iterations", Int(metrics.background_scattering_iterations))
            set_hdf_attribute!(file, "final_background_scattering_residual", Float64(metrics.background_scattering_residual))
            set_hdf_attribute!(file, "hydrogen_se_relaxation", Float64(hydrogen_se_relaxation))
            set_hdf_attribute!(file, "hydrogen_se_wavelength_stride", hydrogen_se_wavelength_stride)
            set_hdf_attribute!(file, "background_scattering_max_iterations", HSE_SCATTERING_MAX_ITERATIONS)
            set_hdf_attribute!(file, "background_scattering_tolerance", HSE_SCATTERING_TOLERANCE)
            set_hdf_attribute!(file, "updated_utc", diagnostic_timestamp())
        end
    end
    # The complete population state is now inside result_h5. Keeping another
    # full-volume working prediction would unnecessarily double disk use.
    isfile(work_prediction_h5) && rm(work_prediction_h5)
    # Update the copied solving-set input only after the durable result and
    # iteration history have been written. The original cfg.solve_h5 is never
    # opened for writing.
    write_solving_electron_density!(result_h5, electron_density)
    return nothing
end


function call_fsdppredict!(
    cfg;
    nproc_per_node::Int,
    launcher::Union{Nothing,String}=nothing,
    solve_h5::String=cfg.solve_h5,
    pred_h5::String=cfg.pred_h5,
)
    nproc_per_node > 0 || error("fsdppredict processes per node must be positive")
    isfile(solve_h5) || error("Prediction solving set not found: $(solve_h5)")

    command = if launcher === nothing
        `torchrun --nproc_per_node=$(nproc_per_node) pipeline.py --fsdppredict --predname $(cfg.name) --solve-h5 $(abspath(solve_h5)) --prediction-output $(abspath(pred_h5))`
    else
        isfile(launcher) || error("FFNoML launcher not found: $(launcher)")
        isexecutable(launcher) || error("FFNoML launcher is not executable: $(launcher)")
        `$(abspath(launcher)) $(cfg.name) $(abspath(solve_h5)) $(abspath(pred_h5))`
    end
    backup = pred_h5 * ".forward-backup"
    isfile(backup) && error(
        "Refusing to overwrite stale prediction backup: $(backup). " *
        "Restore or remove it before retrying."
    )

    # pipeline.py intentionally refuses to overwrite predictions. Move the old
    # result aside so a failed launch can restore it instead of losing it.
    had_prediction = isfile(pred_h5)
    had_prediction && mv(pred_h5, backup)
    try
        cd(@__DIR__) do
            run(command)
        end
        isfile(pred_h5) || error("fsdppredict did not create $(pred_h5)")
    catch
        if had_prediction && isfile(backup)
            isfile(pred_h5) && rm(pred_h5)
            mv(backup, pred_h5)
        end
        rethrow()
    end
    had_prediction && rm(backup)
end


function call_fsdppredict_collective!(
    cfg,
    context::ForwardParallelContext;
    nproc_per_node::Int,
    launcher::Union{Nothing,String},
    solve_h5::String=cfg.solve_h5,
    pred_h5::String=cfg.pred_h5,
    diagnostics=nothing,
)
    if !context.enabled
        call_fsdppredict!(
            cfg;
            nproc_per_node=nproc_per_node,
            launcher=launcher,
            solve_h5=solve_h5,
            pred_h5=pred_h5,
        )
        return nothing
    end

    # Do not leave the non-root Julia ranks inside an MPI collective while
    # rank 0 runs the overlapping Slurm/NCCL step. On Slingshot this can leave
    # the outer MPI collective permanently wedged even after Slurm reports the
    # nested step as COMPLETED. Establish a tiny TCP control channel before
    # the launch; non-root ranks then block on a socket read without polling or
    # keeping an MPI request active. MPI resumes only after rank 0 sends the
    # launcher result to every peer.
    timeout_seconds = tryparse(
        Float64,
        get(ENV, "FORWARD_FSDP_STATUS_TIMEOUT", "0"),
    )
    timeout_seconds === nothing && error(
        "FORWARD_FSDP_STATUS_TIMEOUT must be a number of seconds"
    )
    timeout_seconds >= 0 || error(
        "FORWARD_FSDP_STATUS_TIMEOUT must be zero (disabled) or positive"
    )

    diagnostic_checkpoint!(diagnostics, "fsdppredict_control_setup_start")
    server = parallel_isroot(context) ? listen(ip"0.0.0.0", 0) : nothing
    endpoint = parallel_bcast(
        if parallel_isroot(context)
            _, port = getsockname(server)
            (gethostname(), Int(port))
        else
            nothing
        end,
        context,
    )
    peer_sockets = TCPSocket[]
    control_socket = nothing
    if parallel_isroot(context)
        peer_ranks = Set{Int}()
        for _ in 1:(context.size - 1)
            socket = accept(server)
            peer_rank = Int(read(socket, Int32))
            peer_rank in peer_ranks && error(
                "Duplicate FFNoML control connection from MPI rank $(peer_rank)"
            )
            push!(peer_ranks, peer_rank)
            push!(peer_sockets, socket)
        end
        close(server)
    else
        control_socket = connect(endpoint[1], endpoint[2])
        write(control_socket, Int32(context.rank))
        flush(control_socket)
    end

    diagnostic_checkpoint!(diagnostics, "fsdppredict_prelaunch_barrier_start")
    parallel_barrier(context)
    diagnostic_checkpoint!(diagnostics, "fsdppredict_prelaunch_barrier_complete")
    diagnostic_checkpoint!(
        diagnostics,
        "fsdppredict_control_setup_complete";
        host=endpoint[1],
        port=endpoint[2],
    )

    success = true
    message = ""
    if parallel_isroot(context)
        launch_start = time()
        diagnostic_checkpoint!(diagnostics, "fsdppredict_launcher_start")
        try
            call_fsdppredict!(
                cfg;
                nproc_per_node=nproc_per_node,
                launcher=launcher,
                solve_h5=solve_h5,
                pred_h5=pred_h5,
            )
        catch exception
            success = false
            message = sprint(showerror, exception, catch_backtrace())
        end
        launch_seconds = time() - launch_start
        diagnostic_checkpoint!(
            diagnostics,
            "fsdppredict_launcher_returned";
            seconds=launch_seconds,
            success=success,
            prediction=abspath(pred_h5),
        )
        parallel_println(
            context,
            "FFNoML launcher returned after " *
            "$(round(launch_seconds; digits=2)) s; notifying MPI ranks...",
        )

        for socket in peer_sockets
            serialize(socket, (success=success, message=message))
            flush(socket)
            close(socket)
        end
    else
        diagnostic_checkpoint!(diagnostics, "fsdppredict_status_wait_start")
        wait_start = time()
        timed_out = Ref(false)
        timeout_timer = if timeout_seconds > 0
            Timer(timeout_seconds) do _
                timed_out[] = true
                close(control_socket)
            end
        else
            nothing
        end
        try
            status = deserialize(control_socket)
            success = status.success
            message = status.message
        catch exception
            if timed_out[]
                error(
                    "Timed out after $(timeout_seconds) s waiting for rank-0 " *
                    "FFNoML launcher status"
                )
            end
            error(
                "Lost the rank-0 FFNoML control connection: " *
                sprint(showerror, exception)
            )
        finally
            timeout_timer === nothing || close(timeout_timer)
            isopen(control_socket) && close(control_socket)
        end
        diagnostic_checkpoint!(
            diagnostics,
            "fsdppredict_status_received";
            seconds=time() - wait_start,
            success=success,
        )
    end

    # At this point the nested Slurm/NCCL step has returned and no outer rank
    # has had an MPI request outstanding during it. Every non-root rank has
    # been woken by rank 0 rather than by a polling loop. This barrier verifies
    # that the communicator is usable before prediction-file reads begin.
    diagnostic_checkpoint!(diagnostics, "fsdppredict_postlaunch_barrier_start")
    parallel_barrier(context)
    diagnostic_checkpoint!(diagnostics, "fsdppredict_postlaunch_barrier_complete")

    success || error("MPI rank-0 FFNoML launcher failed:\n$(message)")
    parallel_println(context, "FFNoML prediction complete; all MPI ranks resumed.")
    return nothing
end


function hydrogen_se_update_3d_parallel(
    atmos::Atmosphere3D,
    nlte_h,
    atom,
    atom_file::String,
    voigt_itp;
    parallel::ForwardParallelContext,
    global_nx::Int,
    kwargs...,
)
    parallel.enabled || return hydrogen_se_update_3d(
        atmos, nlte_h, atom, atom_file, voigt_itp; kwargs...,
    )

    # A characteristic needs complete horizontal planes.  Keep x/y intact on
    # one rank per node, divide wavelengths across those node leaders, and use
    # Julia threads to divide cell-local work into height slabs.  This avoids
    # treating an x/y rank boundary as a physical boundary while allowing all
    # allocated nodes to participate in the expensive SE rates.
    leaders = initialize_node_leader_context(parallel)
    try
        global_x = parallel_gather_x_to_node_leaders(
            atmos.x, parallel, leaders; dimension=1, label="hydrogen_se_x",
        )
        atmosphere_fields = (
            :temperature,
            :velocity_x,
            :velocity_y,
            :velocity_z,
            :electron_density,
            :hydrogen1_density,
            :proton_density,
        )
        global_fields = map(atmosphere_fields) do field
            parallel_gather_x_to_node_leaders(
                getfield(atmos, field),
                parallel,
                leaders;
                dimension=3,
                label="hydrogen_se_$(field)",
            )
        end
        global_populations = parallel_gather_x_to_node_leaders(
            nlte_h,
            parallel,
            leaders;
            dimension=3,
            label="hydrogen_se_populations",
        )

        leader_result = if leaders.isleader
            leaders.rank == 0 && println(
                "Hydrogen SE parallel layout: $(leaders.size) wavelength ranks, " *
                "$(Threads.nthreads()) height threads/rank",
            )
            wavelength_parallel = HSEWavelengthParallelContext(
                leaders.size > 1,
                leaders.rank,
                leaders.size,
                leaders.comm,
            )
            global_atmosphere = Atmosphere3D(
                global_nx,
                atmos.ny,
                atmos.nz,
                global_x,
                copy(atmos.y),
                copy(atmos.z),
                global_fields...,
            )
            hydrogen_se_update_3d(
                global_atmosphere,
                global_populations,
                atom,
                atom_file,
                voigt_itp;
                kwargs...,
                wavelength_parallel=wavelength_parallel,
            )
        else
            nothing
        end

        global_corrected = leaders.isleader && leaders.rank == 0 ?
                           leader_result[1] : nothing
        local_corrected = parallel_scatter_x_from_node_leader(
            global_corrected,
            nlte_h,
            parallel;
            dimension=3,
        )
        diagnostics = parallel_bcast(
            parallel_isroot(parallel) ? (leader_result[2], leader_result[3]) : nothing,
            parallel,
        )
        return local_corrected, diagnostics[1], diagnostics[2]
    finally
        finalize_node_leader_context(leaders)
    end
end


function predict_with_charge_conservation(
    cfg,
    atmos::Atmosphere3D;
    max_iterations::Int,
    tolerance::Real,
    nproc_per_node::Int,
    consistency_mode::Symbol=DEFAULT_POPULATION_CONSISTENCY_MODE,
    hydrogen_se_relaxation::Real=DEFAULT_HYDROGEN_SE_RELAXATION,
    hydrogen_se_wavelength_stride::Int=DEFAULT_HYDROGEN_SE_WAVELENGTH_STRIDE,
    parallel::ForwardParallelContext=serial_parallel_context(),
    global_nx::Int=atmos.nx,
    x_range=1:atmos.nx,
    fsdp_launcher::Union{Nothing,String}=nothing,
    diagnostics::Union{Nothing,ForwardDiagnostics}=nothing,
)
    set_diagnostic_context!(diagnostics; phase="consistency_setup")
    max_iterations > 0 || error("Charge-conservation max iterations must be positive")
    tolerance > 0 || error("Charge-conservation tolerance must be positive")
    consistency_mode in (:charge_only, :hydrogen_se_3d) || error(
        "Unknown population consistency mode: $(consistency_mode)"
    )

    hydrogen_cfg_index = findfirst(a -> a.name == "H", cfg.atoms)
    hydrogen_cfg_index === nothing && error(
        "Charge-conservation prediction requires H in FORWARD_ATOMS"
    )
    hydrogen_level_first = sum(
        (a.nlevels for a in cfg.atoms[1:hydrogen_cfg_index-1]);
        init=0,
    ) + 1
    hydrogen_level_range = hydrogen_level_first:(
        hydrogen_level_first + cfg.atoms[hydrogen_cfg_index].nlevels - 1
    )
    parallel_println(parallel, "Consistency setup: loading the hydrogen atom model...")
    hydrogen_atom_start = time()
    diagnostic_checkpoint!(diagnostics, "consistency_hydrogen_atom_read_start")
    hydrogen_atom = Muspel.read_atom(cfg.atoms[hydrogen_cfg_index].atom_file)
    diagnostic_checkpoint!(
        diagnostics,
        "consistency_hydrogen_atom_read_complete";
        seconds=time() - hydrogen_atom_start,
        levels=hydrogen_atom.nlevels,
    )

    parallel_println(parallel, "Consistency setup: loading background atom models...")
    background_atoms_start = time()
    diagnostic_checkpoint!(diagnostics, "consistency_background_atoms_read_start")
    background_atoms = charge_conservation_background_atoms()
    diagnostic_checkpoint!(
        diagnostics,
        "consistency_background_atoms_read_complete";
        seconds=time() - background_atoms_start,
        atoms=length(background_atoms),
    )
    result_h5 = consistency_result_path(cfg, consistency_mode)
    work_prediction_h5 = consistency_prediction_work_path(result_h5)
    parallel_println(parallel, "Consistency setup: checking checkpoint compatibility...")
    diagnostic_checkpoint!(
        diagnostics,
        "consistency_cache_check_start";
        path=result_h5,
    )
    cache_check_start = time()
    cache_state = parallel_root_call(parallel) do
        prepare_consistency_result!(
            result_h5,
            cfg,
            consistency_mode,
            tolerance,
            hydrogen_se_relaxation,
            hydrogen_se_wavelength_stride,
        )
    end
    cache_state = parallel_bcast(
        parallel_isroot(parallel) ? cache_state : nothing,
        parallel,
    )
    diagnostic_checkpoint!(
        diagnostics,
        "consistency_cache_check_complete";
        seconds=time() - cache_check_start,
        compatible=cache_state.compatible,
        converged=cache_state.converged,
        iterations=cache_state.iterations,
    )
    parallel_println(parallel, "Electron-density result: $(result_h5)")
    diagnostic_event!(
        diagnostics,
        "consistency_cache_checked";
        path=result_h5,
        compatible=cache_state.compatible,
        converged=cache_state.converged,
        iterations=cache_state.iterations,
        reason=cache_state.reason,
    )

    parallel_println(parallel, "Consistency setup: constructing fixed hydrogen-density array...")
    fixed_hydrogen_start = time()
    diagnostic_checkpoint!(
        diagnostics,
        "consistency_fixed_hydrogen_density_start";
        local_size=size(atmos.hydrogen1_density),
    )
    fixed_hydrogen_density = atmos.hydrogen1_density .+ atmos.proton_density
    diagnostic_checkpoint!(
        diagnostics,
        "consistency_fixed_hydrogen_density_complete";
        seconds=time() - fixed_hydrogen_start,
        local_mib=sizeof(eltype(fixed_hydrogen_density)) * length(fixed_hydrogen_density) / 2.0^20,
    )

    if cache_state.converged
        set_diagnostic_context!(diagnostics; phase="consistency_cache_hit")
        cached_ne = read_consistency_electron_density(result_h5; x_range=x_range)
        cached_populations = load_pred_depcoeff(
            result_h5,
            cfg.pred_key;
            x_range=x_range,
        )
        cached_hydrogen = split_atoms(
            cached_populations,
            cfg.atoms;
            verbose=false,
        )["H"]
        atmos.electron_density .= cached_ne
        update_atmosphere_hydrogen_densities!(atmos, cached_hydrogen, hydrogen_atom)
        parallel_println(
            parallel,
            "Using converged non-LTE electron density from $(result_h5) " *
            "($(cache_state.iterations) completed iteration(s)); skipping iteration.",
        )
        diagnostic_event!(
            diagnostics,
            "consistency_cache_hit";
            path=result_h5,
            iterations=cache_state.iterations,
            tolerance=tolerance,
        )
        return cached_populations
    end

    parallel_println(parallel, "Consistency setup: reading the starting electron density...")
    input_read_start = time()
    diagnostic_checkpoint!(
        diagnostics,
        "consistency_input_density_read_start";
        source=cache_state.iterations > 0 ? "checkpoint" : "solving_set",
        path=result_h5,
        x_first=first(x_range),
        x_last=last(x_range),
    )
    input_ne = if cache_state.iterations > 0
        read_consistency_electron_density(result_h5; x_range=x_range)
    else
        read_solving_electron_density(
            result_h5,
            atmos;
            global_nx=global_nx,
            x_range=x_range,
        )
    end
    diagnostic_checkpoint!(
        diagnostics,
        "consistency_input_density_read_complete";
        seconds=time() - input_read_start,
        local_size=size(input_ne),
        local_mib=sizeof(eltype(input_ne)) * length(input_ne) / 2.0^20,
    )
    parallel_println(
        parallel,
        "Consistency setup: starting electron density read in " *
        "$(round(time() - input_read_start; digits=2)) s; updating local atmosphere...",
    )
    atmosphere_update_start = time()
    atmos.electron_density .= input_ne
    diagnostic_checkpoint!(
        diagnostics,
        "consistency_atmosphere_density_update_complete";
        seconds=time() - atmosphere_update_start,
    )

    # Persist the untouched starting density once. On resume, also force the
    # copied solving-set input to agree with the last durable checkpoint.
    parallel_println(
        parallel,
        "Consistency setup: gathering electron density from $(parallel.size) MPI ranks...",
    )
    gather_start = time()
    global_input_ne = parallel_gather_x(
        input_ne,
        parallel;
        diagnostics=diagnostics,
        label="initial_electron_density",
    )
    diagnostic_checkpoint!(
        diagnostics,
        "consistency_input_density_gather_complete";
        seconds=time() - gather_start,
        global_size=parallel_isroot(parallel) ? size(global_input_ne) : nothing,
    )
    parallel_println(
        parallel,
        "Consistency setup: electron-density gather completed in " *
        "$(round(time() - gather_start; digits=2)) s; writing checkpoint...",
    )
    parallel_root_call(parallel) do
        initial_write_start = time()
        diagnostic_checkpoint!(
            diagnostics,
            "consistency_initial_density_write_start";
            path=result_h5,
            global_size=size(global_input_ne),
        )
        write_initial_consistency_state!(result_h5, global_input_ne)
        diagnostic_checkpoint!(
            diagnostics,
            "consistency_initial_density_write_complete";
            seconds=time() - initial_write_start,
            path=result_h5,
        )

        solving_write_start = time()
        diagnostic_checkpoint!(
            diagnostics,
            "consistency_solving_density_write_start";
            path=result_h5,
        )
        write_solving_electron_density!(result_h5, global_input_ne)
        diagnostic_checkpoint!(
            diagnostics,
            "consistency_solving_density_write_complete";
            seconds=time() - solving_write_start,
            path=result_h5,
        )
    end
    diagnostic_checkpoint!(diagnostics, "consistency_initial_checkpoint_write_complete")
    parallel_println(parallel, "Consistency setup: checkpoint written; entering MPI barrier...")
    barrier_start = time()
    diagnostic_checkpoint!(diagnostics, "consistency_initial_barrier_start")
    parallel_barrier(parallel)
    diagnostic_checkpoint!(
        diagnostics,
        "consistency_initial_barrier_complete";
        seconds=time() - barrier_start,
    )
    parallel_println(
        parallel,
        "Consistency setup: all MPI ranks passed the barrier in " *
        "$(round(time() - barrier_start; digits=2)) s.",
    )

    hse_setup_start = time()
    if consistency_mode == :hydrogen_se_3d
        set_diagnostic_context!(diagnostics; phase="hydrogen_se_setup")
        parallel_println(
            parallel,
            "Preparing hydrogen SE Voigt and RH background-continuum data...",
        )
    end
    voigt_setup_start = time()
    consistency_mode == :hydrogen_se_3d &&
        diagnostic_checkpoint!(diagnostics, "hydrogen_se_voigt_setup_start")
    voigt_itp = if consistency_mode == :hydrogen_se_3d
        a = LinRange(Float32(cfg.voigt.a_min), Float32(cfg.voigt.a_max), cfg.voigt.a_n)
        v = LinRange(Float32(cfg.voigt.v_min), Float32(cfg.voigt.v_max), cfg.voigt.v_n)
        create_voigt_itp(a, v)
    else
        nothing
    end
    if consistency_mode == :hydrogen_se_3d
        diagnostic_checkpoint!(
            diagnostics,
            "hydrogen_se_voigt_setup_complete";
            seconds=time() - voigt_setup_start,
        )
        parallel_println(
            parallel,
            "Hydrogen SE setup: Voigt interpolation prepared; loading background continua...",
        )
    end
    background_setup_start = time()
    consistency_mode == :hydrogen_se_3d &&
        diagnostic_checkpoint!(diagnostics, "hydrogen_se_background_setup_start")
    hse_background_data = consistency_mode == :hydrogen_se_3d ?
                          hse_background_continuum_data() : nothing
    if consistency_mode == :hydrogen_se_3d
        diagnostic_checkpoint!(
            diagnostics,
            "hydrogen_se_background_setup_complete";
            seconds=time() - background_setup_start,
            background_atoms=length(hse_background_data.atoms),
        )
    end
    if consistency_mode == :hydrogen_se_3d
        hse_setup_seconds = time() - hse_setup_start
        parallel_println(
            parallel,
            "Hydrogen SE data prepared in $(round(hse_setup_seconds; digits=2)) s.",
        )
        diagnostic_event!(
            diagnostics,
            "hydrogen_se_setup_complete";
            seconds=hse_setup_seconds,
            background_atoms=length(hse_background_data.atoms),
        )
    end

    final_populations = nothing
    previous_ffno_hydrogen = if cache_state.iterations > 0
        copy(load_pred_depcoeff(
            result_h5,
            "last_ffno_hydrogen_populations";
            x_range=x_range,
        ))
    else
        nothing
    end
    converged = false
    iterations_run = 0

    for run_iteration in 1:max_iterations
        iterations_run = run_iteration
        iteration = cache_state.iterations + run_iteration
        iteration_start = time()
        set_diagnostic_context!(
            diagnostics;
            iteration=iteration,
            phase="ffnoml_prediction",
        )
        parallel_println(
            parallel,
            "Charge-conservation iteration $(run_iteration)/$(max_iterations) " *
            "(total $(iteration)): running fsdppredict",
        )
        prediction_start = time()
        call_fsdppredict_collective!(
            cfg,
            parallel;
            nproc_per_node=nproc_per_node,
            launcher=fsdp_launcher,
            solve_h5=result_h5,
            pred_h5=work_prediction_h5,
            diagnostics=diagnostics,
        )
        prediction_seconds = time() - prediction_start

        set_diagnostic_context!(diagnostics; phase="prediction_read")
        load_start = time()
        predicted = load_pred_depcoeff(work_prediction_h5, cfg.pred_key; x_range=x_range)
        invalid_prediction = any(value -> !isfinite(value) || value < 0, predicted)
        parallel_allreduce_max(invalid_prediction ? 1.0 : 0.0, parallel) == 0 || error(
            "FFNoML produced non-finite or negative populations during iteration $(iteration)"
        )
        predicted_per_atom = split_atoms(
            predicted,
            cfg.atoms;
            verbose=parallel_isroot(parallel) && iteration == 1,
        )
        nlte_h = predicted_per_atom["H"]
        hydrogen_levels = size(nlte_h, 4)
        ffno_level_changes = if previous_ffno_hydrogen === nothing
            fill(NaN, hydrogen_levels)
        else
            [
                parallel_allreduce_max(
                    maximum_relative_change(
                        @view(nlte_h[:, :, :, level]),
                        @view(previous_ffno_hydrogen[:, :, :, level]),
                    ),
                    parallel,
                )
                for level in 1:hydrogen_levels
            ]
        end
        ffno_population_change = previous_ffno_hydrogen === nothing ? NaN :
                                 maximum(ffno_level_changes)
        previous_ffno_hydrogen = copy(nlte_h)
        population_min = parallel_allreduce_min(minimum(nlte_h), parallel)
        population_max = parallel_allreduce_max(maximum(nlte_h), parallel)
        population_level_min = [
            parallel_allreduce_min(minimum(@view(nlte_h[:, :, :, level])), parallel)
            for level in 1:hydrogen_levels
        ]
        population_level_max = [
            parallel_allreduce_max(maximum(@view(nlte_h[:, :, :, level])), parallel)
            for level in 1:hydrogen_levels
        ]
        isfinite(population_min) && isfinite(population_max) || error(
            "FFNoML produced non-finite hydrogen populations during iteration $(iteration)"
        )
        population_min >= 0 || error(
            "FFNoML produced a negative hydrogen population ($(population_min)) during iteration $(iteration)"
        )
        load_seconds = time() - load_start
        se_residual = 0.0
        population_correction = 0.0
        se_seconds = 0.0
        scattering_iterations = 0
        scattering_residual = 0.0
        if consistency_mode == :hydrogen_se_3d
            set_diagnostic_context!(diagnostics; phase="hydrogen_statistical_equilibrium")
            parallel_println(
                parallel,
                "  solving 3D hydrogen SE on $(global_nx)×$(atmos.ny)×$(atmos.nz) cells...",
            )
            se_start = time()
            corrected_h, se_residual, scattering_diagnostics =
                hydrogen_se_update_3d_parallel(
                atmos,
                nlte_h,
                hydrogen_atom,
                cfg.atoms[hydrogen_cfg_index].atom_file,
                voigt_itp;
                relaxation=hydrogen_se_relaxation,
                wavelength_stride=hydrogen_se_wavelength_stride,
                background_data=hse_background_data,
                parallel=parallel,
                global_nx=global_nx,
            )
            population_correction = maximum_relative_change(corrected_h, nlte_h)
            scattering_iterations = scattering_diagnostics.max_iterations
            scattering_residual = scattering_diagnostics.max_residual
            nlte_h .= corrected_h
            se_seconds = time() - se_start
        end
        set_diagnostic_context!(diagnostics; phase="charge_conservation")
        charge_start = time()
        update_atmosphere_hydrogen_densities!(atmos, nlte_h, hydrogen_atom)
        new_ne = charge_conservation_electron_density(
            atmos,
            input_ne,
            nlte_h,
            hydrogen_atom,
            background_atoms,
            total_hydrogen_density=fixed_hydrogen_density,
        )
        relative_change = maximum_relative_change(new_ne, input_ne)
        charge_seconds = time() - charge_start
        se_residual = parallel_allreduce_max(se_residual, parallel)
        population_correction = parallel_allreduce_max(population_correction, parallel)
        scattering_iterations = Int(parallel_allreduce_max(scattering_iterations, parallel))
        scattering_residual = parallel_allreduce_max(scattering_residual, parallel)
        relative_change = parallel_allreduce_max(relative_change, parallel)
        if consistency_mode == :hydrogen_se_3d
            parallel_println(parallel, "  max hydrogen SE residual           = $(se_residual)")
            parallel_println(
                parallel,
                "  max relative hydrogen correction   = $(population_correction)",
            )
            parallel_println(
                parallel,
                "  background scattering: max iterations=$(scattering_iterations), " *
                "residual=$(scattering_residual)",
            )
        end
        if isfinite(ffno_population_change)
            parallel_println(
                parallel,
                "  max relative FFNoML population change = $(ffno_population_change)",
            )
        else
            parallel_println(parallel, "  max relative FFNoML population change = n/a (first iteration)")
        end
        parallel_println(
            parallel,
            "  max relative electron-density residual = $(relative_change)",
        )

        final_populations = predicted
        atmos.electron_density .= new_ne
        ne_min = parallel_allreduce_min(minimum(new_ne), parallel)
        ne_max = parallel_allreduce_max(maximum(new_ne), parallel)
        isfinite(ne_min) && isfinite(ne_max) && ne_min > 0 || error(
            "Charge conservation produced invalid electron density during iteration $(iteration)"
        )
        populations_converged = consistency_mode == :charge_only ||
                                (se_residual <= tolerance && population_correction <= tolerance)
        iteration_converged = relative_change <= tolerance && populations_converged

        set_diagnostic_context!(diagnostics; phase="consistency_hdf5_write")
        io_start = time()
        global_ne = parallel_gather_x(new_ne, parallel)
        global_ffno_hydrogen = parallel_gather_x(previous_ffno_hydrogen, parallel)
        global_corrected_hydrogen = consistency_mode == :hydrogen_se_3d ?
                                    parallel_gather_x(nlte_h, parallel) : nothing
        parallel_root_call(parallel) do
            write_consistency_checkpoint!(
                result_h5,
                work_prediction_h5,
                cfg.pred_key,
                global_ne,
                global_ffno_hydrogen,
                global_corrected_hydrogen,
                hydrogen_level_range,
                (
                    iteration=iteration,
                    electron_density_delta=relative_change,
                    ffno_population_delta=ffno_population_change,
                    se_population_correction=population_correction,
                    se_residual=se_residual,
                    background_scattering_iterations=scattering_iterations,
                    background_scattering_residual=scattering_residual,
                    electron_density_min=ne_min,
                    electron_density_max=ne_max,
                    ffno_population_min=population_min,
                    ffno_population_max=population_max,
                    prediction_seconds=prediction_seconds,
                    prediction_read_seconds=load_seconds,
                    se_seconds=se_seconds,
                    charge_seconds=charge_seconds,
                    io_seconds=NaN,
                    iteration_seconds=NaN,
                    ffno_level_changes=ffno_level_changes,
                    population_level_min=population_level_min,
                    population_level_max=population_level_max,
                );
                method=consistency_mode,
                tolerance=tolerance,
                max_iterations=max_iterations,
                hydrogen_se_relaxation=hydrogen_se_relaxation,
                hydrogen_se_wavelength_stride=hydrogen_se_wavelength_stride,
                converged=iteration_converged,
            )
        end
        parallel_barrier(parallel)
        io_seconds = time() - io_start
        iteration_seconds = time() - iteration_start
        parallel_root_call(parallel) do
            finalize_consistency_checkpoint_timings!(
                result_h5,
                io_seconds,
                iteration_seconds,
            )
        end
        parallel_barrier(parallel)
        parallel_println(
            parallel,
            "  electron-density range = [$(ne_min), $(ne_max)] m^-3",
        )
        parallel_println(
            parallel,
            "  timings [s]: FFNoML=$(round(prediction_seconds; digits=2)) " *
            "read=$(round(load_seconds; digits=2)) SE=$(round(se_seconds; digits=2)) " *
            "charge=$(round(charge_seconds; digits=2)) I/O=$(round(io_seconds; digits=2)) " *
            "total=$(round(iteration_seconds; digits=2))",
        )
        diagnostic_event!(
            diagnostics,
            "consistency_iteration";
            local_x_first=first(x_range),
            local_x_last=last(x_range),
            ffno_population_change=ffno_population_change,
            ffno_population_level_changes=ffno_level_changes,
            ffno_population_min=population_min,
            ffno_population_max=population_max,
            ffno_population_level_min=population_level_min,
            ffno_population_level_max=population_level_max,
            se_population_correction=population_correction,
            se_residual=se_residual,
            background_scattering_iterations=scattering_iterations,
            background_scattering_residual=scattering_residual,
            electron_density_change=relative_change,
            electron_density_min=ne_min,
            electron_density_max=ne_max,
            prediction_seconds=prediction_seconds,
            prediction_read_seconds=load_seconds,
            se_seconds=se_seconds,
            charge_seconds=charge_seconds,
            io_seconds=io_seconds,
            iteration_seconds=iteration_seconds,
            result_h5=result_h5,
            converged=iteration_converged,
        )
        write_resource_snapshot!(diagnostics)

        if iteration_converged
            parallel_println(
                parallel,
                "  populations and electron density are consistent to tolerance $(tolerance)",
            )
            converged = true
            break
        end

        input_ne = new_ne
    end

    if !converged
        parallel_isroot(parallel) && @warn(
            "Population/electron consistency did not converge",
            consistency_mode,
            max_iterations,
            tolerance,
        )
    end
    set_diagnostic_context!(diagnostics; iteration=0, phase="consistency_complete")
    diagnostic_event!(
        diagnostics,
        "consistency_complete";
        converged=converged,
        result_h5=result_h5,
        iterations_completed=cache_state.iterations + iterations_run,
    )

    return final_populations
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
    voigt_cfg=(a_min=1f-4,a_max=1f1,a_n=20000,v_min=0f0,v_max=5f2,v_n=2500),
    show_progress::Bool=true,
    continuum_bounds=nothing,
)
    my_line = h_atom.lines[line_index]

    a = LinRange(Float32(voigt_cfg.a_min), Float32(voigt_cfg.a_max), voigt_cfg.a_n)
    v = LinRange(Float32(voigt_cfg.v_min), Float32(voigt_cfg.v_max), voigt_cfg.v_n)
    voigt_itp = create_voigt_itp(a, v)

    atom_files = default_background_atom_files()
    σ_itp = if continuum_bounds === nothing
        get_σ_itp(atms, my_line.λ0, atom_files)
    else
        value_type = eltype(atms.temperature)
        temperature_bounds = value_type[
            continuum_bounds.temperature_min,
            continuum_bounds.temperature_max,
        ]
        electron_bounds = value_type[
            continuum_bounds.electron_density_min,
            continuum_bounds.electron_density_max,
        ]
        dummy = zeros(value_type, 2)
        bounds_atmosphere = Atmosphere1D(
            1,
            1,
            2,
            value_type[0, 1],
            temperature_bounds,
            dummy,
            electron_bounds,
            dummy,
            dummy,
        )
        get_σ_itp(bounds_atmosphere, my_line.λ0, atom_files)
    end

    intensity = Array{Float32,3}(undef, my_line.nλ, atms.ny, atms.nx)
    column_count = atms.nx * atms.ny
    progress = show_progress ? Progress(column_count; desc="Synthesis columns (x,y)") : nothing
    buffers = [
        RTBuffer(atms.nz, my_line.nλ, Float32)
        for _ in 1:Threads.maxthreadid()
    ]

    n_u = nltepops_nz_nx_ny_nlev[:, :, :, upper_level]
    n_l = nltepops_nz_nx_ny_nlev[:, :, :, lower_level]

    # Static scheduling keeps threadid() stable while its private RTBuffer is
    # in use. Columns remain independent.
    Threads.@threads :static for column in 1:column_count
        i = div(column - 1, atms.ny) + 1
        j = mod(column - 1, atms.ny) + 1
        buf = buffers[Threads.threadid()]
        calc_line_prep!(my_line, buf, atms[:, j, i], σ_itp)
        calc_line_1D!(my_line, buf, my_line.λ, atms[:, j, i],
                      n_u[:, j, i], n_l[:, j, i], voigt_itp)
        intensity[:, j, i] = buf.intensity
        progress === nothing || next!(progress)
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


function read_atmosphere_parallel(
    mesh_file,
    atmos_file,
    context::ForwardParallelContext,
)
    full_atmosphere = parallel_root_call(context) do
        read_atmos_multi3d(mesh_file, atmos_file)
    end
    global_shape = parallel_bcast(
        parallel_isroot(context) ?
        (full_atmosphere.nx, full_atmosphere.ny, full_atmosphere.nz) : nothing,
        context,
    )
    global_nx, _, _ = global_shape
    x_range = parallel_local_xrange(global_nx, context)

    if !context.enabled
        return full_atmosphere, global_shape, x_range
    end

    metadata = parallel_bcast(
        parallel_isroot(context) ? (
            value_type=eltype(full_atmosphere.temperature),
            y=copy(full_atmosphere.y),
            z=copy(full_atmosphere.z),
        ) : nothing,
        context,
    )
    value_type = metadata.value_type
    x_counts = [
        length(parallel_partition(global_nx, rank, context.size))
        for rank in 0:context.size-1
    ]
    cell_counts = global_shape[2] * global_shape[3] .* x_counts
    local_shape = (global_shape[3], global_shape[2], length(x_range))

    local_x = parallel_scatterv_vector(
        parallel_isroot(context) ? full_atmosphere.x : nothing,
        x_counts,
        value_type,
        context,
    )
    local_fields = map(
        field -> reshape(
            parallel_scatterv_vector(
                parallel_isroot(context) ? getfield(full_atmosphere, field) : nothing,
                cell_counts,
                value_type,
                context,
            ),
            local_shape,
        ),
        (
            :temperature,
            :velocity_x,
            :velocity_y,
            :velocity_z,
            :electron_density,
            :hydrogen1_density,
            :proton_density,
        ),
    )
    local_atmosphere = Atmosphere3D(
        length(x_range),
        global_shape[2],
        global_shape[3],
        local_x,
        metadata.y,
        metadata.z,
        local_fields...,
    )
    return local_atmosphere, global_shape, x_range
end


function read_bifrost_populations_parallel(
    atom_config,
    global_shape,
    context::ForwardParallelContext,
)
    global_nx, global_ny, global_nz = global_shape
    full_populations = parallel_root_call(context) do
        nlte, _ = read_pops_multi3d(
            atom_config.pops_file,
            global_nx,
            global_ny,
            global_nz,
            atom_config.nlevels,
        )
        nlte
    end

    context.enabled || return full_populations
    population_type = parallel_bcast(
        parallel_isroot(context) ? eltype(full_populations) : nothing,
        context,
    )
    x_counts = [
        length(parallel_partition(global_nx, rank, context.size))
        for rank in 0:context.size-1
    ]
    cell_counts = global_ny * global_nz .* x_counts
    local_nx = x_counts[context.rank + 1]
    local_populations = Array{population_type,4}(
        undef,
        global_nz,
        global_ny,
        local_nx,
        atom_config.nlevels,
    )
    for level in 1:atom_config.nlevels
        global_level = parallel_isroot(context) ?
                       copy(full_populations[:, :, :, level]) : nothing
        local_populations[:, :, :, level] .= reshape(
            parallel_scatterv_vector(
                global_level,
                cell_counts,
                population_type,
                context,
            ),
            global_nz,
            global_ny,
            local_nx,
        )
    end
    return local_populations
end


function global_continuum_bounds(
    atmosphere::Atmosphere3D,
    context::ForwardParallelContext,
)
    temperature_min = parallel_allreduce_min(minimum(atmosphere.temperature), context)
    temperature_max = parallel_allreduce_max(maximum(atmosphere.temperature), context)
    electron_min = parallel_allreduce_min(minimum(atmosphere.electron_density), context)
    electron_max = parallel_allreduce_max(maximum(atmosphere.electron_density), context)

    # Muspel constructs spline ranges, which must have non-zero width.
    temperature_max = temperature_max == temperature_min ?
                      temperature_min + max(abs(temperature_min) * 1e-6, 1e-6) :
                      temperature_max
    electron_max = electron_max == electron_min ?
                   electron_min + max(abs(electron_min) * 1e-6, 1e-6) : electron_max
    return (
        temperature_min=temperature_min,
        temperature_max=temperature_max,
        electron_density_min=electron_min,
        electron_density_max=electron_max,
    )
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
    column_count = atmos.nx * atmos.ny
    p = ProgressMeter.Progress(column_count)
    buffers = [
        RTBuffer(atmos.nz, my_line.nλ, Float32)
        for _ in 1:Threads.maxthreadid()
    ]

    Threads.@threads :static for column in 1:column_count
        i = div(column - 1, atmos.ny) + 1
        j = mod(column - 1, atmos.ny) + 1
        buf = buffers[Threads.threadid()]
        calc_line_prep!(my_line, buf, atmos[:, j, i], σ_itp)
        calc_line_1D!(my_line, buf, my_line.λ, atmos[:, j, i], n_u[:, j, i], n_l[:, j, i], voigt_itp)
        intensity[:, j, i] = buf.intensity
        ProgressMeter.next!(p)
    end

    return (intensity=intensity, wave=my_line.λ, line=my_line)
end


function preflight_forward_config(
    cfg,
    charge_conservation_max_iterations::Int,
    fsdp_launcher,
    context::ForwardParallelContext,
)
    parallel_root_call(context) do
        required_files = [cfg.mesh_file, cfg.atmos_file]
        append!(required_files, [atom.atom_file for atom in cfg.atoms])
        if cfg.mode == :ml
            if charge_conservation_max_iterations > 0
                push!(required_files, cfg.solve_h5)
                hasproperty(cfg, :model_file) && push!(required_files, cfg.model_file)
                if fsdp_launcher === nothing
                    push!(required_files, joinpath(@__DIR__, "pipeline.py"))
                else
                    push!(required_files, fsdp_launcher)
                end
            else
                push!(required_files, cfg.pred_h5)
            end
        end
        missing = unique(filter(path -> !isfile(path), required_files))
        isempty(missing) || error(
            "Forward preflight found missing required files:\n  " *
            join(missing, "\n  ") *
            "\nPrediction data roots are controlled by FNOML_PRED_DIR; " *
            "atom YAML files by FNOML_ATOM_DIR."
        )
        fsdp_launcher === nothing || isexecutable(fsdp_launcher) || error(
            "FFNoML launcher is not executable: $(fsdp_launcher)"
        )
        output_directory = dirname(abspath(cfg.out_h5))
        isdir(output_directory) || error(
            "Forward output directory does not exist: $(output_directory)"
        )
        mktemp(output_directory) do _, io
            write(io, "Forward.jl output write test\n")
            flush(io)
        end
        if cfg.mode == :ml && charge_conservation_max_iterations > 0
            any(atom -> atom.name == "H", cfg.atoms) || error(
                "Charge-conservation iteration requires H in FORWARD_ATOMS"
            )
            h5open(cfg.solve_h5, "r") do file
                "inputs" in keys(file) || error(
                    "Solving-set HDF5 has no inputs dataset: $(cfg.solve_h5)"
                )
                ndims(file["inputs"]) == 5 || error(
                    "Solving-set inputs must be five-dimensional: $(cfg.solve_h5)"
                )
            end
        elseif cfg.mode == :ml
            h5open(cfg.pred_h5, "r") do file
                cfg.pred_key in keys(file) || error(
                    "Prediction HDF5 has no $(cfg.pred_key) dataset: $(cfg.pred_h5)"
                )
            end
        end
        nothing
    end
    parallel_barrier(context)
    return nothing
end


# -----------------------------
# Main pipeline
# -----------------------------
function main(
    cfg;
    charge_conservation_max_iterations::Int=0,
    charge_conservation_tolerance::Real=DEFAULT_CHARGE_CONSERVATION_TOLERANCE,
    fsdp_nproc_per_node::Int=DEFAULT_FSDP_NPROC_PER_NODE,
    population_consistency_mode::Symbol=DEFAULT_POPULATION_CONSISTENCY_MODE,
    hydrogen_se_relaxation::Real=DEFAULT_HYDROGEN_SE_RELAXATION,
    hydrogen_se_wavelength_stride::Int=DEFAULT_HYDROGEN_SE_WAVELENGTH_STRIDE,
    parallel::ForwardParallelContext=serial_parallel_context(),
    fsdp_launcher::Union{Nothing,String}=nothing,
    diagnostics::Union{Nothing,ForwardDiagnostics}=nothing,
)

    set_diagnostic_context!(diagnostics; dataset=cfg.name, iteration=0, phase="preflight")
    diagnostic_event!(diagnostics, "dataset_start"; mode=cfg.mode, output=cfg.out_h5)
    preflight_forward_config(
        cfg,
        charge_conservation_max_iterations,
        fsdp_launcher,
        parallel,
    )

    parallel_println(parallel, "Mode        : ", cfg.mode)
    parallel_println(parallel, "Dataset     : ", cfg.name)
    parallel_println(parallel, "Simulation  : ", cfg.sim_name)
    parallel_println(parallel, "Snapshot    : ", cfg.snap)
    parallel_println(parallel, "MPI ranks   : ", parallel.size)
    parallel_println(parallel, "Threads/rank: ", Threads.nthreads())
    parallel_println(parallel, "Output      : ", cfg.out_h5)
    if cfg.mode == :ml
        parallel_println(parallel, "Charge max  : ", charge_conservation_max_iterations)
        parallel_println(parallel, "Consistency : ", population_consistency_mode)
    end

    cfg.mode == :tiago && parallel.enabled && error("Tiago mode does not support --mpi")

    atoms_to_run = parallel_bcast(
        parallel_isroot(parallel) ? pending_atoms(cfg) : nothing,
        parallel,
    )

    if isempty(atoms_to_run)
        parallel_println(
            parallel,
            "Skipping $(cfg.name) snap $(cfg.snap): all requested atom outputs already exist.",
        )
        return
    end

    parallel_println(parallel, "Pending atoms: ", join([a.name for a in atoms_to_run], ", "))

    if cfg.mode == :bifrost
        missing = parallel_bcast(
            parallel_isroot(parallel) ? missing_population_inputs(atoms_to_run) : nothing,
            parallel,
        )

        if !isempty(missing)
            parallel_println(
                parallel,
                "Skipping $(cfg.name) snap $(cfg.snap): level populations are not available."
            )
            for item in missing
                parallel_println(
                    parallel,
                    "  - $(item.atom) missing $(item.kind): $(item.path)",
                )
            end
            return
        end
    end

    set_diagnostic_context!(diagnostics; phase="atmosphere_read_scatter")
    parallel_println(parallel, "Reading and distributing atmosphere...")
    atmosphere_start = time()
    atmos, global_shape, x_range = read_atmosphere_parallel(
        cfg.mesh_file,
        cfg.atmos_file,
        parallel,
    )
    global_nx, _, _ = global_shape
    x_base, x_remainder = divrem(global_nx, parallel.size)
    parallel_println(
        parallel,
        "Atmosphere shape: $(global_shape); x columns/rank=$(x_base)" *
        (x_remainder == 0 ? "" : " or $(x_base + 1)"),
    )
    diagnostic_event!(
        diagnostics,
        "atmosphere_distributed";
        global_shape=global_shape,
        local_shape=size(atmos.temperature),
        x_first=first(x_range),
        x_last=last(x_range),
        seconds=time() - atmosphere_start,
    )
    write_resource_snapshot!(diagnostics)

    # ============================================================
    # ML MODE
    # ============================================================
    if cfg.mode == :ml

        nlte_pop_full = if charge_conservation_max_iterations > 0
            predict_with_charge_conservation(
                cfg,
                atmos;
                max_iterations=charge_conservation_max_iterations,
                tolerance=charge_conservation_tolerance,
                nproc_per_node=fsdp_nproc_per_node,
                consistency_mode=population_consistency_mode,
                hydrogen_se_relaxation=hydrogen_se_relaxation,
                hydrogen_se_wavelength_stride=hydrogen_se_wavelength_stride,
                parallel=parallel,
                global_nx=global_nx,
                x_range=x_range,
                fsdp_launcher=fsdp_launcher,
                diagnostics=diagnostics,
            )
        else
            set_diagnostic_context!(diagnostics; phase="prediction_read")
            load_pred_depcoeff(cfg.pred_h5, cfg.pred_key; x_range=x_range)
        end

        # ---------------------------------------------------
        # Split ML populations per atom
        # ---------------------------------------------------
        nlte_per_atom = split_atoms(
            nlte_pop_full,
            cfg.atoms;
            verbose=parallel_isroot(parallel),
        )

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

        parallel_println(parallel, "Reading Multi3D pops...")

        nlte_atoms = Dict{String,Any}()

        for a in atoms_to_run

            atom = Muspel.read_atom(a.atom_file)

            nlte_atoms[a.name] = read_bifrost_populations_parallel(
                a,
                global_shape,
                parallel,
            )
        end

    elseif cfg.mode == :tiago
        println("tiago mode")
    else
        error("Unknown mode")
    end

    # for (k,v) in nlte_atoms
    #     println(k, " NLTE shape = ", size(v))
    # end

    parallel_println(parallel, "Synthesizing line profiles...")
    continuum_bounds = global_continuum_bounds(atmos, parallel)
    results = Dict{String,Any}()

    for a in atoms_to_run

        parallel_println(parallel, "Synthesizing atom: ", a.name)
        set_diagnostic_context!(diagnostics; phase="synthesis_$(a.name)")
        synthesis_start = time()

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
                voigt_cfg=cfg.voigt,
                show_progress=parallel_isroot(parallel),
                continuum_bounds=continuum_bounds,
            )

            results[a.name] = syn
        end
        synthesis_seconds = time() - synthesis_start
        parallel_println(
            parallel,
            "  $(a.name) synthesis completed in $(round(synthesis_seconds; digits=2)) s",
        )
        diagnostic_event!(
            diagnostics,
            "atom_synthesis_complete";
            atom=a.name,
            local_columns=atmos.nx * atmos.ny,
            seconds=synthesis_seconds,
        )
        write_resource_snapshot!(diagnostics)
    end

    set_diagnostic_context!(diagnostics; phase="synthesis_gather_write")
    parallel_println(parallel, "Gathering and saving output...")
    output_start = time()
    gathered_results = Dict{String,Any}()
    for atom in atoms_to_run
        syn = results[atom.name]
        global_intensity = parallel_gather_x(syn.intensity, parallel)
        if parallel_isroot(parallel)
            gathered_results[atom.name] = (
                intensity=global_intensity,
                wave=syn.wave,
                line=syn.line,
            )
        end
    end
    parallel_root_call(parallel) do
        save_synthesis_results(cfg.out_h5, gathered_results)
    end
    parallel_barrier(parallel)

    diagnostic_event!(
        diagnostics,
        "dataset_complete";
        output=cfg.out_h5,
        gather_write_seconds=time() - output_start,
    )
    set_diagnostic_context!(diagnostics; phase="dataset_complete")
    write_resource_snapshot!(diagnostics)
    parallel_println(parallel, "Done.")
end

# Run
function run_all(
    ;
    charge_conservation_max_iterations::Int=0,
    charge_conservation_tolerance::Real=DEFAULT_CHARGE_CONSERVATION_TOLERANCE,
    fsdp_nproc_per_node::Int=DEFAULT_FSDP_NPROC_PER_NODE,
    population_consistency_mode::Symbol=DEFAULT_POPULATION_CONSISTENCY_MODE,
    hydrogen_se_relaxation::Real=DEFAULT_HYDROGEN_SE_RELAXATION,
    hydrogen_se_wavelength_stride::Int=DEFAULT_HYDROGEN_SE_WAVELENGTH_STRIDE,
    parallel::ForwardParallelContext=serial_parallel_context(),
    fsdp_launcher::Union{Nothing,String}=nothing,
    diagnostics::Union{Nothing,ForwardDiagnostics}=nothing,
)
    pred_data = parallel_bcast(
        parallel_isroot(parallel) ? load_multi3d_pred_data() : nothing,
        parallel,
    )
    isempty(pred_data) && error(
        "config.py MULTI3D_PRED_DATA has no entries enabled for Forward.jl"
    )

    for (idx, pred) in enumerate(pred_data)
        configured_max_iterations = pred.charge_conservation_max_iterations
        dataset_max_iterations = if pred.nonlte_ne === false
            0
        elseif configured_max_iterations !== nothing
            configured_max_iterations
        elseif pred.nonlte_ne === nothing
            charge_conservation_max_iterations
        elseif pred.nonlte_ne
            charge_conservation_max_iterations > 0 || error(
                "$(pred.name) has NONLTE_NE=True in config.py, but no non-LTE " *
                "electron-density iterations were requested. Pass " *
                "--charge-conservation-max-iterations N with N > 0."
            )
            charge_conservation_max_iterations
        end
        pred.nonlte_ne === true && dataset_max_iterations == 0 && error(
            "$(pred.name) has NONLTE_NE=True in config.py, but its effective " *
            "CHARGE_CONSERVATION_MAX_ITERATIONS is zero."
        )

        dataset_consistency_mode = pred.population_consistency_mode === nothing ?
            population_consistency_mode :
            parse_population_consistency_mode(pred.population_consistency_mode)
        dataset_wavelength_stride = pred.hydrogen_se_wavelength_stride === nothing ?
            hydrogen_se_wavelength_stride : pred.hydrogen_se_wavelength_stride

        parallel_println(parallel, "")
        parallel_println(parallel, "============================================================")
        parallel_println(
            parallel,
            "Forward synthesis $(idx)/$(length(pred_data)): $(pred.name)",
        )
        parallel_println(parallel, "============================================================")

        cfg = build_config(pred)
        main(
            cfg;
            charge_conservation_max_iterations=dataset_max_iterations,
            charge_conservation_tolerance=charge_conservation_tolerance,
            fsdp_nproc_per_node=fsdp_nproc_per_node,
            population_consistency_mode=dataset_consistency_mode,
            hydrogen_se_relaxation=hydrogen_se_relaxation,
            hydrogen_se_wavelength_stride=dataset_wavelength_stride,
            parallel=parallel,
            fsdp_launcher=fsdp_launcher,
            diagnostics=diagnostics,
        )
    end
end

function print_usage()
    println("""
Usage: julia Forward.jl [options]

Options:
  --charge-conservation-max-iterations N
  --max-iterations N
      Repeatedly run fsdppredict and impose charge conservation, at most N
      additional times. A converged method-specific result HDF5 is reused;
      an unconverged checkpoint resumes. Default 0 keeps one-shot ML behavior.
  --charge-conservation-tolerance VALUE
      Maximum relative electron-density residual (default: 1e-4).
  --population-consistency-mode MODE
      Population update mode: charge-only (default) or hydrogen-se-3d.
      The latter performs a full-volume 3D formal solution and hydrogen
      statistical-equilibrium correction before charge conservation.
  --hydrogen-se-relaxation VALUE
      Fraction of the SE population correction to apply, in (0, 1] (default: 1).
  --hydrogen-se-wavelength-stride N
      Use every Nth transition wavelength in hydrogen-se-3d (default: 1).
  --fsdp-nproc-per-node N
      Processes passed to torchrun for fsdppredict (default: 4).
  --fsdp-launcher PATH
      Executable invoked once by MPI rank 0 as:
      PATH PREDICTION_NAME SOLVE_H5 PREDICTION_OUTPUT.
      Use this for the coordinated multi-node torchrun Slurm step.
  --mpi
      Partition x columns over MPI COMM_WORLD. Combine MPI ranks with Julia
      threads by launching Julia with --threads N.
  --diagnostics-dir PATH
      Directory for per-rank detailed event, failure, and resource files.
      Default: forward-diagnostics-slurm-JOBID (or a timestamped local path).
  --resource-monitor-interval SECONDS
      Resource CSV sampling interval (default: 30; use 0 to disable periodic
      sampling while retaining phase-boundary snapshots).
  --no-resource-monitor
      Alias for --resource-monitor-interval 0.
  -h, --help
      Show this help.
""")
end


function parse_population_consistency_mode(value::AbstractString)
    normalized = lowercase(replace(value, "_" => "-"))
    normalized == "charge-only" && return :charge_only
    normalized in ("preferred", "hydrogen-se", "hydrogen-se-3d") &&
        return :hydrogen_se_3d
    error(
        "Unknown population consistency mode $(value). " *
        "Use charge-only or hydrogen-se-3d."
    )
end


function parse_forward_args(args)
    max_iterations = 0
    tolerance = Float64(DEFAULT_CHARGE_CONSERVATION_TOLERANCE)
    nproc_per_node = DEFAULT_FSDP_NPROC_PER_NODE
    population_consistency_mode = DEFAULT_POPULATION_CONSISTENCY_MODE
    hydrogen_se_relaxation = DEFAULT_HYDROGEN_SE_RELAXATION
    hydrogen_se_wavelength_stride = DEFAULT_HYDROGEN_SE_WAVELENGTH_STRIDE
    enable_mpi = false
    fsdp_launcher = nothing
    diagnostics_directory = nothing
    resource_monitor_interval = 30.0
    index = 1

    while index <= length(args)
        arg = args[index]
        if arg in ("-h", "--help")
            print_usage()
            return nothing
        elseif arg in ("--charge-conservation-max-iterations", "--max-iterations")
            index == length(args) && error("Missing value after $(arg)")
            index += 1
            max_iterations = parse(Int, args[index])
        elseif startswith(arg, "--charge-conservation-max-iterations=")
            max_iterations = parse(Int, split(arg, "="; limit=2)[2])
        elseif startswith(arg, "--max-iterations=")
            max_iterations = parse(Int, split(arg, "="; limit=2)[2])
        elseif arg == "--charge-conservation-tolerance"
            index == length(args) && error("Missing value after $(arg)")
            index += 1
            tolerance = parse(Float64, args[index])
        elseif startswith(arg, "--charge-conservation-tolerance=")
            tolerance = parse(Float64, split(arg, "="; limit=2)[2])
        elseif arg in ("--population-consistency-mode", "--consistency-mode")
            index == length(args) && error("Missing value after $(arg)")
            index += 1
            population_consistency_mode = parse_population_consistency_mode(args[index])
        elseif startswith(arg, "--population-consistency-mode=") ||
               startswith(arg, "--consistency-mode=")
            population_consistency_mode = parse_population_consistency_mode(
                split(arg, "="; limit=2)[2]
            )
        elseif arg == "--hydrogen-se-relaxation"
            index == length(args) && error("Missing value after $(arg)")
            index += 1
            hydrogen_se_relaxation = parse(Float64, args[index])
        elseif startswith(arg, "--hydrogen-se-relaxation=")
            hydrogen_se_relaxation = parse(Float64, split(arg, "="; limit=2)[2])
        elseif arg == "--hydrogen-se-wavelength-stride"
            index == length(args) && error("Missing value after $(arg)")
            index += 1
            hydrogen_se_wavelength_stride = parse(Int, args[index])
        elseif startswith(arg, "--hydrogen-se-wavelength-stride=")
            hydrogen_se_wavelength_stride = parse(Int, split(arg, "="; limit=2)[2])
        elseif arg == "--fsdp-nproc-per-node"
            index == length(args) && error("Missing value after $(arg)")
            index += 1
            nproc_per_node = parse(Int, args[index])
        elseif startswith(arg, "--fsdp-nproc-per-node=")
            nproc_per_node = parse(Int, split(arg, "="; limit=2)[2])
        elseif arg == "--fsdp-launcher"
            index == length(args) && error("Missing value after $(arg)")
            index += 1
            fsdp_launcher = args[index]
        elseif startswith(arg, "--fsdp-launcher=")
            fsdp_launcher = split(arg, "="; limit=2)[2]
        elseif arg == "--mpi"
            enable_mpi = true
        elseif arg == "--diagnostics-dir"
            index == length(args) && error("Missing value after $(arg)")
            index += 1
            diagnostics_directory = args[index]
        elseif startswith(arg, "--diagnostics-dir=")
            diagnostics_directory = split(arg, "="; limit=2)[2]
        elseif arg == "--resource-monitor-interval"
            index == length(args) && error("Missing value after $(arg)")
            index += 1
            resource_monitor_interval = parse(Float64, args[index])
        elseif startswith(arg, "--resource-monitor-interval=")
            resource_monitor_interval = parse(Float64, split(arg, "="; limit=2)[2])
        elseif arg == "--no-resource-monitor"
            resource_monitor_interval = 0.0
        else
            error("Unknown argument: $(arg). Use --help for usage.")
        end
        index += 1
    end

    max_iterations >= 0 || error("Max iterations cannot be negative")
    tolerance > 0 || error("Charge-conservation tolerance must be positive")
    nproc_per_node > 0 || error("fsdppredict processes per node must be positive")
    0 < hydrogen_se_relaxation <= 1 || error("Hydrogen SE relaxation must be in (0, 1]")
    hydrogen_se_wavelength_stride > 0 || error("Hydrogen SE wavelength stride must be positive")
    resource_monitor_interval >= 0 || error("Resource-monitor interval cannot be negative")

    return (
        charge_conservation_max_iterations=max_iterations,
        charge_conservation_tolerance=tolerance,
        fsdp_nproc_per_node=nproc_per_node,
        population_consistency_mode=population_consistency_mode,
        hydrogen_se_relaxation=hydrogen_se_relaxation,
        hydrogen_se_wavelength_stride=hydrogen_se_wavelength_stride,
        enable_mpi=enable_mpi,
        fsdp_launcher=fsdp_launcher,
        diagnostics_directory=diagnostics_directory,
        resource_monitor_interval=resource_monitor_interval,
    )
end


function run_forward_cli(options)
    parallel = initialize_parallel_context(options.enable_mpi)
    diagnostics = nothing
    try
        diagnostics = initialize_diagnostics(
            parallel;
            directory=options.diagnostics_directory,
            interval=options.resource_monitor_interval,
        )
        parallel_println(parallel, "Diagnostics : ", diagnostics.directory)
        excluded = NamedTuple{
            (:enable_mpi, :diagnostics_directory, :resource_monitor_interval),
        }((
            options.enable_mpi,
            options.diagnostics_directory,
            options.resource_monitor_interval,
        ))
        run_options = Base.structdiff(options, excluded)
        run_all(; run_options..., parallel=parallel, diagnostics=diagnostics)
    catch exception
        backtrace = catch_backtrace()
        record_diagnostic_failure!(diagnostics, exception, backtrace)
        if parallel.enabled
            println(
                stderr,
                "Forward.jl MPI rank $(parallel.rank) failed during ",
                diagnostics === nothing ? "initialization" : diagnostics.phase[],
                ". Detailed failure: ",
                diagnostics === nothing ? "unavailable" : diagnostics.failure_path,
            )
            abort_parallel_context(parallel, 1)
        else
            rethrow()
        end
    finally
        if diagnostics !== nothing
            try
                stop_diagnostics!(diagnostics)
            catch exception
                println(
                    stderr,
                    "Forward.jl could not finish diagnostics cleanly: ",
                    sprint(showerror, exception, catch_backtrace()),
                )
            end
        end
        try
            finalize_parallel_context(parallel)
        catch exception
            println(stderr, "Forward.jl could not finalize MPI cleanly: ", exception)
        end
    end
    return nothing
end


if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    options = parse_forward_args(ARGS)
    if options !== nothing
        run_forward_cli(options)
    end
end
