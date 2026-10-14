#!/usr/bin/env julia

using Muspel
using StaticArrays
using HDF5


function atom_file(atom_name)
    atom_dir = normpath(joinpath(@__DIR__,"..","multi3d","input","atoms"))
    files = Dict(
        "H" => joinpath(atom_dir,"atom.h6_tiago2.yaml"),
        "CA" => joinpath(atom_dir,"atom.ca2.yaml"),
    )
    haskey(files, atom_name) || error("No atom file configured for $(atom_name)")
    return files[atom_name]
end


function lte_pops_saha(atom, atmos::Atmosphere3D)
    nH = atmos.hydrogen1_density .+ atmos.proton_density
    abundance_ratio = 10.0^(atom.abundance - 12.0)
    n_species = abundance_ratio .* nH

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


function main(args)
    length(args) >= 4 || error(
        "Usage: compute_muspel_lte.jl MESH ATMOS OUTPUT_H5 ATOM [ATOM ...]"
    )

    mesh_file, atmos_file, output_file = args[1:3]
    atom_names = args[4:end]

    atmos = read_atmos_multi3d(mesh_file, atmos_file)
    h5open(output_file, "w") do file
        for atom_name in atom_names
            atom = Muspel.read_atom(atom_file(atom_name))
            pops = lte_pops_saha(atom, atmos)
            group = create_group(file, atom_name)

            # Store a flat array plus its Julia shape. This avoids implicit
            # dimension reversal when the HDF5 file is read by h5py.
            group["values"] = vec(pops)
            group["shape"] = Int64.(collect(size(pops)))
        end
    end
end


main(ARGS)
