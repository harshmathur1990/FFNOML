using FFNOInversion

grid = Grid3D(collect(range(-6.0,1.0,length=64)),collect(0.0:31.0),collect(0.0:31.0))
shape = (64,32,32)
z = zeros(shape)
atmos = Atmosphere3D(grid,fill(5000.0,shape),copy(z),copy(z),copy(z),copy(z))
wave = collect(range(656.1e-9,656.5e-9,length=101))
workspace = ForwardWorkspace(Float64,atmos,wave,StokesSet(:I))
model = MockForwardModel(MockPopulationModel(1e10),NonPRD(),MockIntensitySynthesizer(),IdentityObservation(),CapabilityManifest())
forward!(workspace,model,atmos)
elapsed = @elapsed for _ in 1:10
    forward!(workspace,model,atmos)
end
println("mock_forward_mean_seconds=",elapsed/10)
println("output_shape=",size(workspace.output.data))
