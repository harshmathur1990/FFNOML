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

train_dir = "training_lam_0.1"

sim_name = "en024048_hion_504"

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
            pops_file = "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/$(snap)/H/out_pop",
            nlevels = 6,
            line_index = 5,
            lower_level = 2,
            upper_level = 3
        ),
        # (
        #     name = "CA",
        #     atom_file = "/mn/stornext/u3/harshm/Documents/WorkRepo/multi3d/input/atoms/atom.ca2.yaml",
        #     pops_file = "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385/CA/out_pop",
        #     nlevels = 6,
        #     line_index = 5,
        #     lower_level = 3,
        #     upper_level = 5
        # )
    ],

    mesh_file  = "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/$(snap)/mesh",
    atmos_file = "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/$(snap)/atm3d",

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
            pops_file = "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385/H/out_pop",
            nlevels = 6,
            line_index = 5,
            lower_level = 2,
            upper_level = 3
        ),
        (
            name = "CA",
            atom_file = "/mn/stornext/u3/harshm/Documents/WorkRepo/multi3d/input/atoms/atom.ca2.yaml",
            pops_file = "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385/CA/out_pop",
            nlevels = 6,
            line_index = 5,
            lower_level = 3,
            upper_level = 5
        )
    ],

    mesh_file  = "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385/mesh",
    atmos_file = "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385/atm3d",

    out_h5     = "IO/intensity_bifrost.h5",
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

const CFG = CONFIG_ML
# const CFG = CONFIG_BIFROST

# -----------------------------
# Column-mass remapping helpers
# (directly from your notebook, with minor cleanup)
# -----------------------------
function cmass_from_rho(ρ::AbstractVector{T}, z::AbstractVector{T}) where {T<:Real}
    n = length(ρ)
    cm = similar(ρ)
    cm[1] = zero(T)
    @inbounds for i in 2:n
        dz = -(z[i] - z[i-1])  # z increases upward
        cm[i] = cm[i-1] + 0.5*(ρ[i] + ρ[i-1]) * dz
    end
    return cm
end

function interpolate_column(ρ, z, Y, cmass_new; logx::Bool=false, logy::Bool=false)
    cm = cmass_from_rho(ρ, z)

    x    = logx ? log10.(cm)        : cm
    xnew = logx ? log10.(cmass_new) : cmass_new

    m     = length(cmass_new)
    nvars = size(Y, 2)
    out   = Matrix{eltype(Y)}(undef, m, nvars)

    @inbounds for v in 1:nvars
        ycol = view(Y, :, v)
        y    = logy ? log10.(ycol) : ycol

        itp = LinearInterpolation(x, y, extrapolation_bc=Line())
        vals = itp.(xnew)

        out[:, v] .= logy ? 10 .^ vals : vals
    end

    return out
end

"""
ρ_super   :: (nx, ny, nz)
pops4d    :: (nx, ny, nz, nvars)
cmass_new :: (m)

returns   :: (nx, ny, m, nvars)
"""
function interpolate_everything(ρ_super, z, pops4d, cmass_new; logx::Bool=false, logy::Bool=false)
    nx, ny, nz = size(ρ_super)
    @assert size(pops4d,1) == nx
    @assert size(pops4d,2) == ny
    @assert size(pops4d,3) == nz

    nvars = size(pops4d, 4)
    m     = length(cmass_new)
    out   = Array{eltype(pops4d)}(undef, nx, ny, m, nvars)

    @threads for ix in 1:nx
        for iy in 1:ny
            ρ = view(ρ_super, ix, iy, :)
            Y = reshape(view(pops4d, ix, iy, :, :), nz, nvars)  # force (nz, nvars)
            col = interpolate_column(ρ, z, Y, cmass_new; logx=logx, logy=logy)  # (m, nvars)
            @inbounds out[ix, iy, :, :] .= col
        end
    end

    return out
end

function compute_ztop(rho3d, z, mtop)
    nx, ny, nz = size(rho3d)
    ztop = Array{eltype(rho3d)}(undef, nx, ny)
    for ix in 1:nx, iy in 1:ny
        ρ  = view(rho3d, ix, iy, :)
        cm = cmass_from_rho(ρ, z)
        itp = LinearInterpolation(cm, z, extrapolation_bc=Line())
        ztop[ix,iy] = itp(mtop)
    end
    return ztop
end

function invert_cmass(cmass, rho, ztop)
    N = length(cmass)
    z = similar(cmass)
    z[1] = ztop
    @inbounds for i = 2:N
        dm = cmass[i] - cmass[i-1]
        z[i] = z[i-1] - dm / rho[i-1]
    end
    return z
end

function compute_z_from_cmass_3d(rho_new, cmass_new, ztop)
    nx, ny, nz = size(rho_new)
    znew = Array{eltype(rho_new)}(undef, nx, ny, nz)
    for ix in 1:nx, iy in 1:ny
        znew[ix,iy,:] = invert_cmass(cmass_new, view(rho_new, ix, iy, :), ztop[ix,iy])
    end
    return znew
end

function remap_atmosphere_cmass(atmos::Atmosphere3D, new_cmass_scale)
    nx, ny, nz = atmos.nx, atmos.ny, atmos.nz

    # (nz,nx,ny) → (nx,ny,nz)
    temp = permutedims(atmos.temperature,       (2,3,1))
    vx   = permutedims(atmos.velocity_x,        (2,3,1))
    vy   = permutedims(atmos.velocity_y,        (2,3,1))
    vz   = permutedims(atmos.velocity_z,        (2,3,1))
    ne   = permutedims(atmos.electron_density,  (2,3,1))
    nh   = permutedims(atmos.hydrogen1_density, (2,3,1))
    np   = permutedims(atmos.proton_density,    (2,3,1))
    rho  = permutedims(atmos.plasma_density,    (2,3,1))

    ztop = compute_ztop(rho, atmos.z, new_cmass_scale[1])

    f4(A) = reshape(A, nx, ny, nz, 1)
    temp4, vx4, vy4, vz4 = f4(temp), f4(vx), f4(vy), f4(vz)
    ne4, nh4, np4, rho4  = f4(ne),   f4(nh), f4(np), f4(rho)

    temp_new = dropdims(interpolate_everything(rho, atmos.z, temp4, new_cmass_scale; logx=true, logy=false), dims=4)
    vx_new   = dropdims(interpolate_everything(rho, atmos.z, vx4,   new_cmass_scale; logx=true, logy=false), dims=4)
    vy_new   = dropdims(interpolate_everything(rho, atmos.z, vy4,   new_cmass_scale; logx=true, logy=false), dims=4)
    vz_new   = dropdims(interpolate_everything(rho, atmos.z, vz4,   new_cmass_scale; logx=true, logy=false), dims=4)
    ne_new   = dropdims(interpolate_everything(rho, atmos.z, ne4,   new_cmass_scale; logx=true, logy=true),  dims=4)
    nh_new   = dropdims(interpolate_everything(rho, atmos.z, nh4,   new_cmass_scale; logx=true, logy=true),  dims=4)
    np_new   = dropdims(interpolate_everything(rho, atmos.z, np4,   new_cmass_scale; logx=true, logy=true),  dims=4)
    rho_new  = dropdims(interpolate_everything(rho, atmos.z, rho4,  new_cmass_scale; logx=true, logy=true),  dims=4)

    z_new = compute_z_from_cmass_3d(rho_new, new_cmass_scale, ztop)

    # back to Atmosphere3D layout (nz,nx,ny)
    g(A) = permutedims(A, (3,1,2))

    return Atmosphere3D(
        nx, ny, length(z_new),
        atmos.x, atmos.y,
        g(z_new),
        g(temp_new),
        g(vx_new), g(vy_new), g(vz_new),
        g(ne_new), g(nh_new), g(np_new),
        g(rho_new)
    )
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

    # → (nx, ny, nz, nlevels)
    pops4d = permutedims(reint, (3,4,2,1))

    return pops4d
end

function load_pred_depcoeff(pred_h5::String, pred_key::String)

    h5open(pred_h5, "r") do f
        raw = read(f[pred_key])

        dep_coeff = PermutedDimsArray(raw, (3, 4, 2, 1))
        return dep_coeff

    end
end

function load_multi3d_pops(pops_file::String, atmos::Atmosphere3D, nlevels::Int)
    pops_out_nlte, pops_out_lte = read_pops_multi3d(pops_file, atmos.nx, atmos.ny, atmos.nz, nlevels)
    return pops_out_nlte, pops_out_lte
end

function remap_pops_to_cmass(atmos::Atmosphere3D, pops4d_nxnyznv, new_cmass_scale; logx=true, logy=true)
    # expects pops in (nx,ny,nz,nvars) OR a PermutedDimsArray that behaves like that
    rho_nxnyz = PermutedDimsArray(atmos.plasma_density, (2, 3, 1))  # (nx,ny,nz)
    pops_new  = interpolate_everything(rho_nxnyz, atmos.z, pops4d_nxnyznv, new_cmass_scale; logx=logx, logy=logy)
    return pops_new
end

# Source function at line center (same algebra you used)
function line_source_function(atom, λ0_m, nltepops, l, u)

    h = 6.62607015e-34
    c = 2.99792458e8
    ν = c / λ0_m

    n_l = nltepops[:, :, :, l]
    n_u = nltepops[:, :, :, u]

    g_l = atom.g[l]
    g_u = atom.g[u]

    prefactor = 2*h * ν^3 / c^2

    return prefactor ./ ((g_u .* n_l) ./ (g_l .* n_u) .- 1)
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
                          n_u[:, i, j], n_l[:, i, j], voigt_itp)
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

function plot_diag_depcoeff(out_png::String, cmass_new, pred_dep::AbstractVector, orig_dep::Union{Nothing,AbstractVector})
    x = log10.(cmass_new)
    plt = plot(x, log10.(pred_dep), label="pred dep coeff", color=:black)
    if orig_dep !== nothing
        plot!(plt, x, log10.(orig_dep), label="orig dep coeff", color=:red)
    end
    xlabel!(plt, "log10(cmass)")
    ylabel!(plt, "log10(dep coeff)")
    savefig(plt, out_png)
end

function plot_diag_Snu(out_png::String, cmass_new, S_pred::AbstractVector, S_orig::Union{Nothing,AbstractVector})
    x = log10.(cmass_new)
    plt = plot(x, log10.(S_pred), label="pred Sν", color=:black)
    if S_orig !== nothing
        plot!(plt, x, log10.(S_orig), label="orig Sν", color=:red)
    end
    xlabel!(plt, "log10(cmass)")
    ylabel!(plt, "log10(Sν)  [SI units]")
    savefig(plt, out_png)
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

    new_cmass_scale = Float32.(10 .^ range(cfg.cmass_logmin, cfg.cmass_logmax, length=cfg.cmass_n))

    println("Computing LTE pops...")
    lte_atoms = Dict{String,Any}()

    for a in cfg.atoms
        atom = Muspel.read_atom(a.atom_file)
        pops = lte_pops_saha(atom, atmos)
        lte_atoms[a.name] = remap_pops_to_cmass(atmos, pops, new_cmass_scale; logx=true, logy=true)
    end

    println("Remapping atmosphere...")
    remapped_atmos = remap_atmosphere_cmass(atmos, new_cmass_scale)

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

            nlte_atoms[a.name] = permutedims(nlte_pop, (3,1,2,4))

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

            pops_new = permutedims(
                remap_pops_to_cmass(
                    atmos,
                    PermutedDimsArray(pops_out_nlte, (2,3,1,4)),
                    new_cmass_scale;
                    logx=true, logy=true
                ),
                (3,1,2,4)
            )

            nlte_atoms[a.name] = pops_new
        end

    else
        error("Unknown mode")
    end

    # ---------------------------------------------------------
    # Diagnostics: compare ML vs true NLTE if pops_file exists
    # ---------------------------------------------------------
    if cfg.mode == :ml && cfg.plot_diagnostics

        println("Running diagnostics...")

        for a in cfg.atoms

            atom = Muspel.read_atom(a.atom_file)

            # -------------------------
            # TRUE NLTE pops
            # -------------------------
            pops_out_nlte, pops_out_lte =
                read_pops_multi3d(a.pops_file, atmos.nx, atmos.ny, atmos.nz, atom.nlevels)

            pops_true = permutedims(
                remap_pops_to_cmass(
                    atmos,
                    PermutedDimsArray(pops_out_nlte, (2,3,1,4)),
                    new_cmass_scale;
                    logx=true, logy=true
                ),
                (3,1,2,4)
            )

            # -------------------------
            # ML NLTE pops
            # -------------------------
            pops_ml = nlte_atoms[a.name]

            # -------------------------
            # LTE pops
            # -------------------------
            pops_lte = permutedims(
                lte_atoms[a.name],
                (3,1,2,4)
            )

            x = cfg.x_pick
            y = cfg.y_pick

            # -------------------------
            # Departure coefficient diagnostic
            # -------------------------
            l = a.lower_level

            dep_ml   = pops_ml[:,x,y,l] ./ pops_lte[:,x,y,l]
            dep_true = pops_true[:,x,y,l] ./ pops_lte[:,x,y,l]

            plot_diag_depcoeff(
                "$(cfg.out_prefix)_$(a.name)_dep.png",
                new_cmass_scale,
                dep_ml,
                dep_true
            )

            # -------------------------
            # Source function diagnostic
            # -------------------------
            λ0 = atom.lines[a.line_index].λ0

            S_ml   = line_source_function(atom, λ0, pops_ml, a.lower_level, a.upper_level)[:,x,y]
            S_true = line_source_function(atom, λ0, pops_true, a.lower_level, a.upper_level)[:,x,y]

            plot_diag_Snu(
                "$(cfg.out_prefix)_$(a.name)_Snu.png",
                new_cmass_scale,
                S_ml,
                S_true
            )
        end
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