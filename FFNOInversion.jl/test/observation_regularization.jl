@testset "full-grid PSF, zero weights and regularization" begin
    a = test_atmosphere()
    wave = collect(range(656.1e-9,656.5e-9,length=9))
    observation_model = GaussianPSFObservation(0.04e-9,1.0,1.0,1.0,1.0)
    ws = ForwardWorkspace(Float64,a,wave,StokesSet(:I);observation=observation_model)
    model = MockForwardModel(MockPopulationModel(1e10),NonPRD(),MockIntensitySynthesizer(),observation_model,CapabilityManifest())
    degraded = forward!(ws,model,a)
    @test size(degraded.data) == (9,1,2,3)
    @test degraded.wavelength_m == wave
    @test all(isfinite,degraded.data)

    wavelength_weights = ones(9,1); wavelength_weights[2:2:end,1] .= 0
    spatial_weights = ones(2,3); spatial_weights[2,:] .= 0
    weights = build_inversion_weights(wavelength_weights,spatial_weights)
    @test weights[2,1,1,1] == 0
    @test weights[1,1,2,1] == 0
    @test weights[1,1,1,1] == 1
    four_stokes = ones(9,4); four_stokes[:,2:4] .= 0
    controlled = build_inversion_weights(four_stokes,spatial_weights)
    @test all(iszero,controlled[:,2:4,:,:])
    @test any(!iszero,controlled[:,1,:,:])
    observed = SpectralCube(copy(degraded.data),degraded.wavelength_m,StokesSet(:I))
    observed.data .-= 1
    obs = ObservationCube(observed,ones(size(observed.data)),weights)
    r = zeros(length(observed.data))
    residual!(r,ResidualLayout(StokesSet(:I)),degraded,obs)
    @test r == vec(weights)

    temp_vertical = VerticalRegularizationSpec((1,0,0,0,0,0,0),1.0,ntuple(_->1.0,7))
    constant_spec = RegularizationSpec(vertical=temp_vertical,horizontal=Dict(:temperature=>1.0),
        scales=Dict(:temperature=>1000.0),horizontal_order=1)
    @test regularization_penalty(a,constant_spec,1.0,1.0).total == 0
    gradient_atmosphere = test_atmosphere()
    gradient_atmosphere.temperature[:,2,:] .+= 100
    @test regularization_penalty(gradient_atmosphere,constant_spec,1.0,1.0).total > 0
    magnetic_vertical = VerticalRegularizationSpec((0,0,0,1,0,0,0),1.0,ntuple(_->1.0,7))
    magnetic_spec = RegularizationSpec(vertical=magnetic_vertical,horizontal=Dict{Symbol,Float64}(),
        scales=Dict(:B=>0.01),horizontal_order=1)
    @test_throws ArgumentError regularization_penalty(a,magnetic_spec,1.0,1.0)
    normalized = VerticalRegularizationSpec((3,1,0,0,0,0,0),1.0,ntuple(_->1.0,7))
    @test normalized.types[1] == 0
    @test normalized.types[2] == 1
    @test_throws ArgumentError VerticalRegularizationSpec((5,0,0,0,0,0,0),1.0,ntuple(_->1.0,7))

    base = test_atmosphere()
    with_pressure = Atmosphere3D(base.grid,base.temperature,base.vx,base.vy,base.vz,base.vturb;pgas=fill(0.1,size(base.temperature)))
    pressure_vertical = VerticalRegularizationSpec((0,0,0,0,0,0,1),1.0,ntuple(_->1.0,7))
    pressure_spec = RegularizationSpec(vertical=pressure_vertical,horizontal=Dict{Symbol,Float64}(),
        scales=Dict(:pgas_boundary=>0.1),horizontal_order=1)
    @test regularization_penalty(with_pressure,pressure_spec,1.0,1.0).total == 0
    with_pressure.pgas[1,:,:] .= 0.2
    @test regularization_penalty(with_pressure,pressure_spec,1.0,1.0).total > 0
end
