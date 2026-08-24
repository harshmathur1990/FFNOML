@testset "future PRD and Stokes seams" begin
    a = test_atmosphere(magnetic=true)
    wave = [656.2e-9,656.3e-9]
    stokes = StokesSet((:I,:Q,:U,:V))
    ws = ForwardWorkspace(Float64,a,wave,stokes)
    future = CapabilityManifest(prd=true,stokes=(:I,:Q,:U,:V))
    model = MockForwardModel(MockPopulationModel(1e10),MockPRD(0.9),MockPolarizedSynthesizer(),IdentityObservation(),future)
    @test size(forward!(ws,model,a).data) == (2,4,2,3)

    release1 = CapabilityManifest()
    blocked = MockForwardModel(MockPopulationModel(1e10),MockPRD(0.9),MockPolarizedSynthesizer(),IdentityObservation(),release1)
    @test_throws ArgumentError forward!(ws,blocked,a)
end
