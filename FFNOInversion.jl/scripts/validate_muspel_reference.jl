#!/usr/bin/env julia

using Muspel
using AtomicData
using HDF5

length(ARGS) == 1 || error("usage: validate_muspel_reference.jl FNOML_ROOT")
root = abspath(ARGS[1])
atmos = read_atmos_multi3d(joinpath(root,"testdata/en024048_hion/385/mesh"),
                           joinpath(root,"testdata/en024048_hion/385/atm3d"))
atom_dir = normpath(joinpath(root,"..","multi3d","input","atoms"))
background = [joinpath(AtomicData.get_atom_dir(),name) for name in
    ("Al.yaml","C.yaml","Ca.yaml","Fe.yaml","H_6.yaml","He.yaml","KI.yaml",
     "Mg.yaml","N.yaml","Na.yaml","NiI.yaml","O.yaml","S.yaml","Si.yaml")]
voigt = create_voigt_itp(LinRange(1f-4,1f1,20000),LinRange(0f0,5f2,2500))

cases = (("H","atom.h6_tiago2.yaml",2,3),("CA","atom.ca2.yaml",3,5))
for (name,atom_name,lower,upper) in cases
    atom = read_atom(joinpath(atom_dir,atom_name)); line = atom.lines[5]
    sigma = get_σ_itp(atmos,line.λ0,background)
    column = atmos[:,1,1]; buffer = RTBuffer(atmos.nz,line.nλ,Float32)
    calc_line_prep!(line,buffer,column,sigma)
    population_path = joinpath(root,"training_FFNO3D_zscale_expand_lognlte",
        "output_3D_sim_s5_en024048_hion_385_FFNO3D_$(name).hdf5")
    h5open(population_path) do file
        populations = file["nlte_populations"]
        calc_line_1D!(line,buffer,line.λ,column,populations[upper,:,1,1],
                      populations[lower,:,1,1],voigt)
    end
    intensity_path = joinpath(root,"training_FFNO3D_zscale_expand_lognlte",
        "intensity_ml_en024048_hion_385_FFNO3D_$(name).h5")
    reference = h5open(intensity_path) do file
        file["$name/intensity"][:,1,1]
    end
    maxabs = maximum(abs.(buffer.intensity-reference))
    maxrel = maximum(abs.(buffer.intensity-reference)./max.(abs.(reference),eps(Float32)))
    println("$name maxabs=$maxabs maxrel=$maxrel")
    maxabs == 0 && maxrel == 0 || error("$name Muspel reference parity failed")
end
