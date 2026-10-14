@testset "deterministic mock forward and residual" begin
    a = test_atmosphere()
    wave = collect(range(656.1e-9,656.5e-9,length=7))
    ws = ForwardWorkspace(Float64,a,wave,StokesSet(:I))
    model = MockForwardModel(MockPopulationModel(1e10),NonPRD(),MockIntensitySynthesizer(),IdentityObservation(),CapabilityManifest())
    result1 = copy(forward!(ws,model,a).data)
    result2 = copy(forward!(ws,model,a).data)
    @test result1 == result2
    allocation1 = @allocated forward!(ws,model,a)
    allocation2 = @allocated forward!(ws,model,a)
    @test allocation2 <= allocation1 + 1024
    obs_cube = SpectralCube(copy(result1),wave,StokesSet(:I))
    obs = ObservationCube(obs_cube,ones(size(result1)))
    r = zeros(length(result1))
    residual!(r,ResidualLayout(StokesSet(:I)),ws.output,obs)
    @test all(iszero,r)
end
