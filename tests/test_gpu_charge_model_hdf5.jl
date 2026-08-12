using Test
using HDF5

include(joinpath(@__DIR__, "..", "Forward.jl"))

@testset "GPU charge model HDF5 layout" begin
    atom = (
        χ = [0.0, 2.0e-19, 4.0e-19],
        g = [2, 4, 1],
        stage = [1, 1, 2],
        nlevels = 3,
    )
    background_atoms = [(atom=atom, abundance=2.5e-5)]
    fixed_hydrogen = reshape(Float32.(1:24), 2, 3, 4)

    mktempdir() do directory
        path = joinpath(directory, "charge-model.h5")
        h5open(path, "w") do file
            file["seed"] = [1]
        end
        write_gpu_charge_model!(
            path,
            fixed_hydrogen,
            background_atoms,
            atom,
            1:3,
        )

        h5open(path, "r") do file
            group = file["gpu_charge_model"]
            @test size(group["fixed_hydrogen_density"]) == (2, 3, 4)
            @test read(group["atom_offsets"]) == Int32[0, 3]
            @test read(group["hydrogen_prediction_indices"]) == Int32[0, 1, 2]
            @test read(group["hydrogen_stages"]) == Int32[1, 1, 2]
            @test read(attributes(group)["formula"]) == "direct_saha_boltzmann_v1"
        end
    end
end
