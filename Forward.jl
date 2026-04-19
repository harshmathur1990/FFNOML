#!/usr/bin/env julia
# -----------------------------------------------------------------------------
# ForwardSynthesis.jl
#
# A scriptified version of Forward.ipynb:
#   - Reads Bifrost/Multi3D atmosphere
#   - Builds LTE populations via Saha-Boltzmann
#   - Remaps atmosphere + populations to a new column-mass (cmass) scale
#   - Two synthesis modes:
#       (A) "ml"     : NLTE populations = (predicted departure coeffs) * (LTE pops)
#       (B) "bifrost": NLTE populations read from Multi3D out_pop (and remapped)
#   - Synthesizes 1D line profiles for all columns (nx, ny) using Muspel
#   - Writes intensity + wavelength to an HDF5 file
#   - Writes two diagnostic plots (PNG)
# -----------------------------------------------------------------------------

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
# MODE 1 — ML predicted pops
# -----------------------------

model  = "FFNO3D"
snap = 385

train_dir = "training"

sim_name = "en024048_hion"

pred_h5 = joinpath(
    train_dir,
    "output_3D_sim_s5_$(sim_name)_$(snap)_$(model).hdf5"
)

out_h5 = joinpath(
    train_dir,
    "intensity_ml_$(sim_name)_$(snap)_$(model).h5"
)

const CONFIG_ML = (
    mode = :ml,

    atoms = [
        (
            name = "H",
            atom_file = "/mn/stornext/u3/harshm/Documents/WorkRepo/multi3d/input/atoms/atom.h6_tiago2.yaml",
            pops_file = "/mn/stornext/d9/data/harshm/bifrost_data/$(sim_name)/$(snap)/H/out_pop",
            nlevels = 6,
            line_index = 5,
            lower_level = 2,
            upper_level = 3
        ),
        # (
        #     name = "CA",
        #     atom_file = "/mn/stornext/u3/harshm/Documents/WorkRepo/multi3d/input/atoms/atom.ca2.yaml",
        #     pops_file = "/mn/stornext/d9/data/harshm/bifrost_data/$(sim_name)/385/CA/out_pop",
        #     nlevels = 6,
        #     line_index = 5,
        #     lower_level = 3,
        #     upper_level = 5
        # )
    ],

    mesh_file  = "/mn/stornext/d9/data/harshm/bifrost_data/$(sim_name)/$(snap)/mesh",
    atmos_file = "/mn/stornext/d9/data/harshm/bifrost_data/$(sim_name)/$(snap)/atm3d",

    model = model,

    pred_h5 = pred_h5,
    pred_key = "departure_coefficients",

    plot_diagnostics = false,

    out_h5 = out_h5,
    out_prefix = joinpath(train_dir, "diag_ml"),

    x_pick     = 33,
    y_pick     = 21,

    cmass_n      = 400,
    cmass_logmin = -6.0,
    cmass_logmax =  2.0,

    voigt = (
        a_min = 1f-4,
        a_max = 1f1,
        a_n   = 20000,
        v_min = 0f0,
        v_max = 5f2,
        v_n   = 2500
    )
)

# -----------------------------
# MODE 2 — Original Bifrost NLTE pops
# -----------------------------
const CONFIG_BIFROST = (
    mode = :bifrost,

    atoms = [
        (
            name = "H",
            atom_file = "/mn/stornext/u3/harshm/Documents/WorkRepo/multi3d/input/atoms/atom.h6_tiago2.yaml",
            pops_file = "/mn/stornext/d9/data/harshm/bifrost_data/$(sim_name)/$(snap)/H/out_pop",
            nlevels = 6,
            line_index = 5,
            lower_level = 2,
            upper_level = 3
        ),
        # (
        #     name = "CA",
        #     atom_file = "/mn/stornext/u3/harshm/Documents/WorkRepo/multi3d/input/atoms/atom.ca2.yaml",
        #     pops_file = "/mn/stornext/d9/data/harshm/bifrost_data/$(sim_name)/$(snap)/CA/out_pop",
        #     nlevels = 6,
        #     line_index = 5,
        #     lower_level = 3,
        #     upper_level = 5
        # )
    ],

    mesh_file  = "/mn/stornext/d9/data/harshm/bifrost_data/$(sim_name)/$(snap)/mesh",
    atmos_file = "/mn/stornext/d9/data/harshm/bifrost_data/$(sim_name)/$(snap)/atm3d",

    out_h5     = "IO/intensity_bifrost_$(sim_name)_$(snap).h5",
    out_prefix = "diag_bifrost",

    x_pick     = 33,
    y_pick     = 21,

    cmass_n      = 400,
    cmass_logmin = -6.0,
    cmass_logmax =  2.0,

    voigt = (
        a_min = 1f-4,
        a_max = 1f1,
        a_n   = 20000,
        v_min = 0f0,
        v_max = 5f2,
        v_n   = 2500
    )
)

# ============================================================
# USER CHOOSES WHICH ONE TO RUN
# ============================================================

# const CFG = CONFIG_ML
const CFG = CONFIG_BIFROST


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

        dep_coeff = PermutedDimsArray(raw, (3, 2, 1, 4))
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
    my_line = h_atom.lines[line_index]

    a = LinRange(Float32(voigt_cfg.a_min), Float32(voigt_cfg.a_max), voigt_cfg.a_n)
    v = LinRange(Float32(voigt_cfg.v_min), Float32(voigt_cfg.v_max), voigt_cfg.v_n)
    voigt_itp = create_voigt_itp(a, v)

    atom_files = default_background_atom_files()
    σ_itp = get_σ_itp(atms, my_line.λ0, atom_files)

    intensity = Array{Float32,3}(undef, my_line.nλ, atms.ny, atms.nx)
    p = Progress(atms.nx; desc="Synthesis columns (x)")

    n_u = nltepops_nz_nx_ny_nlev[:, :, :, upper_level]
    n_l = nltepops_nz_nx_ny_nlev[:, :, :, lower_level]

    Threads.@threads for i in 1:atms.nx
        buf = RTBuffer(atms.nz, my_line.nλ, Float32)
        for j in 1:atms.ny
            calc_line_prep!(my_line, buf, atms[:, j, i], σ_itp)
            calc_line_1D!(my_line, buf, my_line.λ, atms[:, j, i],
                          n_u[:, j, i], n_l[:, j, i], voigt_itp)
            intensity[:, j, i] = buf.intensity
        end
        next!(p)
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


# -----------------------------
# Main pipeline
# -----------------------------
function main()

    cfg = CFG

    println("Mode        : ", cfg.mode)
    println("Threads     : ", Threads.nthreads())

    println("Reading atmosphere...")
    atmos = read_atmos_multi3d(cfg.mesh_file, cfg.atmos_file)

    println("Computing LTE pops...")
    lte_atoms = Dict{String,Any}()

    for a in cfg.atoms
        atom = Muspel.read_atom(a.atom_file)
        pops = lte_pops_saha(atom, atmos)
        lte_atoms[a.name] = pops
    end

    remapped_atmos = atmos

    # ============================================================
    # ML MODE
    # ============================================================
    if cfg.mode == :ml

        dep_coeff_full = load_pred_depcoeff(cfg.pred_h5, cfg.pred_key)

        # ---------------------------------------------------
        # Split ML populations per atom
        # ---------------------------------------------------
        dep_per_atom = split_atoms(dep_coeff_full, cfg.atoms)

        # Dictionary to store final NLTE pops per atom
        nlte_atoms = Dict{String,Any}()

        for a in cfg.atoms

            h_atom = Muspel.read_atom(a.atom_file)

            dep = dep_per_atom[a.name]

            @assert size(dep,4) == h_atom.nlevels "Level mismatch for atom $(a.name)"

            nlte_pop = dep .* lte_atoms[a.name]

            nlte_atoms[a.name] = nlte_pop

        end

    # ============================================================
    # BIFROST MODE
    # ============================================================
    elseif cfg.mode == :bifrost

        println("Reading Multi3D pops...")

        nlte_atoms = Dict{String,Any}()

        for a in cfg.atoms

            atom = Muspel.read_atom(a.atom_file)

            pops_out_nlte, pops_out_lte =
                read_pops_multi3d(a.pops_file, atmos.nx, atmos.ny, atmos.nz, atom.nlevels)

            nlte_atoms[a.name] = pops_out_nlte
        end

    else
        error("Unknown mode")
    end

    for (k,v) in nlte_atoms
        println(k, " NLTE shape = ", size(v))
    end

    println("Synthesizing line profiles...")
    results = Dict{String,Any}()

    for a in cfg.atoms

        println("Synthesizing atom: ", a.name)

        h_atom = Muspel.read_atom(a.atom_file)

        syn = synthesize_intensity_3d(
            remapped_atmos,
            h_atom,
            a.line_index,
            nlte_atoms[a.name],
            a.lower_level,
            a.upper_level;
            voigt_cfg = cfg.voigt
        )

        results[a.name] = syn
    end

    println("Saving output...")
    f = h5open(cfg.out_h5, "w")

    for (name, syn) in results
        grp = create_group(f, name)
        grp["intensity"] = syn.intensity
        grp["wave"]      = syn.wave
    end

    close(f)

    println("Done.")
end

# Run
main()
