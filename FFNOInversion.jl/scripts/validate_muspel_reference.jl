#!/usr/bin/env julia

using Muspel
using AtomicData
using HDF5

length(ARGS) == 1 || error("usage: validate_muspel_reference.jl FNOML_ROOT")
root = abspath(ARGS[1])
# abspath can retain a trailing slash for the batch script's FFNOInversion.jl/..
# argument. Use an explicit parent component: dirname(root) then names FFNOML itself.
atmosphere_dir = abspath(get(ENV,"FFNO_REFERENCE_ATMOSPHERE_DIR",
    joinpath(root,"..","bifrost_data","en024048_hion","385")))
atmos = read_atmos_multi3d(joinpath(atmosphere_dir,"mesh"),
                           joinpath(atmosphere_dir,"atm3d"))
atom_dir = abspath(get(ENV,"FFNO_ATOM_DIR",
    joinpath(root,"..","multi3d","input","atoms")))
run_root = abspath(get(ENV,"FFNOML_RUN_DIR",root))
model_dir = abspath(get(ENV,"FFNO_REFERENCE_MODEL_DIR",
    joinpath(run_root,"training_FFNO3D_zscale_expand_lognlte")))
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
    population_path = joinpath(model_dir,
        "output_3D_sim_s5_en024048_hion_385_FFNO3D_$(name).hdf5")
    h5open(population_path) do file
        populations = file["nlte_populations"]
        calc_line_1D!(line,buffer,line.λ,column,populations[upper,:,1,1],
                      populations[lower,:,1,1],voigt)
    end
    intensity_path = joinpath(model_dir,
        "intensity_ml_en024048_hion_385_FFNO3D_$(name).h5")
    reference = h5open(intensity_path) do file
        file["$name/intensity"][:,1,1]
    end
    maxabs = maximum(abs.(buffer.intensity-reference))
    maxrel = maximum(abs.(buffer.intensity-reference)./max.(abs.(reference),eps(Float32)))
    println("$name maxabs=$maxabs maxrel=$maxrel")
    maxabs == 0 && maxrel == 0 || error("$name Muspel reference parity failed")
end
println("PHASE3_MUSPEL_REFERENCE_OK cases=$(length(cases)) atmosphere=$atmosphere_dir")
