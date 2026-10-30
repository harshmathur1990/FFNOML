include("helpers.jl")

@testset "canonical types" begin
    a = test_atmosphere()
    @test_throws ArgumentError Grid3D([0.0,1.0,0.5],[0.0],[0.0])
    @test_throws ArgumentError Atmosphere3D(a.grid,fill(-1.0,size(a.temperature)),a.vx,a.vy,a.vz,a.vturb)
    @test StokesSet((:I,:Q,:U,:V)).components == (:I,:Q,:U,:V)
    @test_throws ArgumentError StokesSet((:I,:V,:Q))
end
