include("helpers.jl")

@testset "canonical types and force-balance selection" begin
    a = test_atmosphere()
    @test select_force_balance(a) isa HE3DMode
    am = test_atmosphere(magnetic=true)
    @test select_force_balance(am) isa MHSMode
    @test_throws ArgumentError Grid3D([0.0,1.0,0.5],[0.0],[0.0])
    @test_throws ArgumentError Atmosphere3D(a.grid,fill(-1.0,size(a.temperature)),a.vx,a.vy,a.vz,a.vturb)
    @test StokesSet((:I,:Q,:U,:V)).components == (:I,:Q,:U,:V)
    @test_throws ArgumentError StokesSet((:I,:V,:Q))
    @test HE3DBoundaryState(1e-8,1e-2,:top).boundary == :top
end
