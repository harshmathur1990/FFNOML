#!/usr/bin/env julia

using Muspel
using StaticArrays
using HDF5
using Statistics

sim_name = "en024048_hion"
snap = 385

atom_cfgs = [
    (
        name = "H",
        atom_file = "/mn/stornext/u3/harshm/Documents/WorkRepo/multi3d/input/atoms/atom.h6_tiago2.yaml",
        pops_file = "/mn/stornext/d9/data/harshm/bifrost_data/$(sim_name)/$(snap)/H/out_pop",
        nlevels = 6,
    ),
]

mesh_file = "/mn/stornext/d9/data/harshm/bifrost_data/$(sim_name)/$(snap)/mesh"
atmos_file = "/mn/stornext/d9/data/harshm/bifrost_data/$(sim_name)/$(snap)/atm3d"
pred_h5 = joinpath(@__DIR__, "training", "original_multi3d_dep_$(sim_name)_$(snap).h5")
pred_key = "departure_coefficients"


function lte_pops_saha(atom, atmos::Atmosphere3D)
    nH = atmos.hydrogen1_density .+ atmos.proton_density
    ratio = 10.0^(atom.abundance - 12.0)
    n_species = ratio .* nH

    pops = Muspel.saha_boltzmann.(
        Ref(atom),
        atmos.temperature,
        atmos.electron_density,
        n_species,
    )

    pops_s = SVector{atom.nlevels,Float32}.(pops)
    reint = reshape(reinterpret(Float32, pops_s), atom.nlevels, size(pops_s)...)
    return permutedims(reint, (2, 3, 4, 1))
end


function load_pred_depcoeff_current(pred_h5::String, pred_key::String)
    h5open(pred_h5, "r") do f
        raw = read(f[pred_key])
        return PermutedDimsArray(raw, (2, 3, 4, 1))
    end
end


function load_pred_depcoeff_fixed(pred_h5::String, pred_key::String)
    h5open(pred_h5, "r") do f
        raw = read(f[pred_key])
        return PermutedDimsArray(raw, (2, 4, 3, 1))
    end
end


function split_atoms(dep_coeff, atoms)
    offsets = cumsum([0; [a.nlevels for a in atoms]])
    out = Dict{String,Any}()

    for (i, a) in enumerate(atoms)
        s = offsets[i] + 1
        e = offsets[i + 1]
        out[a.name] = view(dep_coeff, :, :, :, s:e)
    end

    return out
end


function get_z_scale(atmos)
    if hasproperty(atmos, :z)
        return Float32.(getproperty(atmos, :z))
    elseif hasproperty(atmos, :z_scale)
        return Float32.(getproperty(atmos, :z_scale))
    elseif hasproperty(atmos, :height)
        return Float32.(getproperty(atmos, :height))
    else
        return Float32.(collect(1:atmos.nz))
    end
end


function stats(label, a, b)
    diff = Array(a .- b)
    absdiff = abs.(diff)
    reldiff = ifelse.(b .== 0, absdiff, absdiff ./ abs.(b))

    println(label)
    println("  size(a)      = ", size(a))
    println("  size(b)      = ", size(b))
    println("  max abs diff = ", maximum(absdiff))
    println("  mean abs diff = ", mean(absdiff))
    println("  max rel diff = ", maximum(reldiff))
    println("  mean rel diff = ", mean(reldiff))
end


println("Reading atmosphere...")
atmos = read_atmos_multi3d(mesh_file, atmos_file)

println("Reading Multi3D pops...")
orig_nlte_atoms = Dict{String,Any}()
orig_lte_atoms = Dict{String,Any}()

for a in atom_cfgs
    atom = Muspel.read_atom(a.atom_file)
    pops_out_nlte, pops_out_lte =
        read_pops_multi3d(a.pops_file, atmos.nx, atmos.ny, atmos.nz, atom.nlevels)

    orig_nlte_atoms[a.name] = pops_out_nlte
    orig_lte_atoms[a.name] = pops_out_lte

    println("Atom ", a.name, " direct Multi3D NLTE shape = ", size(pops_out_nlte))
    println("Atom ", a.name, " direct Multi3D LTE shape  = ", size(pops_out_lte))
end

println("Computing LTE pops...")
lte_atoms = Dict{String,Any}()

for a in atom_cfgs
    atom = Muspel.read_atom(a.atom_file)
    pops = lte_pops_saha(atom, atmos)
    lte_atoms[a.name] = pops
    println("Atom ", a.name, " Saha LTE shape          = ", size(pops))
end

dep_coeff_current = load_pred_depcoeff_current(pred_h5, pred_key)
dep_coeff_fixed = load_pred_depcoeff_fixed(pred_h5, pred_key)

println("dep_coeff_current shape = ", size(dep_coeff_current))
println("dep_coeff_fixed shape   = ", size(dep_coeff_fixed))

dep_current_per_atom = split_atoms(dep_coeff_current, atom_cfgs)
dep_fixed_per_atom = split_atoms(dep_coeff_fixed, atom_cfgs)

println("Comparing reconstructed NLTE populations...")

for a in atom_cfgs
    atom = Muspel.read_atom(a.atom_file)

    @assert size(dep_fixed_per_atom[a.name], 4) == atom.nlevels

    nlte_from_current_vs_saha = dep_current_per_atom[a.name] .* lte_atoms[a.name]
    nlte_from_fixed_vs_saha = dep_fixed_per_atom[a.name] .* lte_atoms[a.name]
    nlte_from_fixed_vs_directlte = dep_fixed_per_atom[a.name] .* orig_lte_atoms[a.name]

    println()
    println("Atom ", a.name)
    stats("  current loader vs direct Multi3D NLTE", nlte_from_current_vs_saha, orig_nlte_atoms[a.name])
    stats("  fixed loader with Saha LTE vs direct Multi3D NLTE", nlte_from_fixed_vs_saha, orig_nlte_atoms[a.name])
    stats("  fixed loader with direct Multi3D LTE vs direct Multi3D NLTE", nlte_from_fixed_vs_directlte, orig_nlte_atoms[a.name])
    stats("  Saha LTE vs direct Multi3D LTE", lte_atoms[a.name], orig_lte_atoms[a.name])
end
