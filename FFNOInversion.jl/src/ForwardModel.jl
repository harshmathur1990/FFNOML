mutable struct ForwardWorkspace{T<:AbstractFloat}
    populations::Array{T,4}
    intrinsic::SpectralCube{T,Array{T,4}}
    output::SpectralCube{T,Array{T,4}}
end

function ForwardWorkspace(::Type{T}, atmosphere::Atmosphere3D, wavelength_m, stokes::StokesSet;
                          observation::AbstractObservationModel=IdentityObservation()) where T<:AbstractFloat
    nz,nx,ny = size(atmosphere.temperature)
    pops = zeros(T,nz,nx,ny,1)
    intrinsic_shape = (length(wavelength_m),length(stokes.components),nx,ny)
    intrinsic = SpectralCube(zeros(T,intrinsic_shape), T.(wavelength_m), stokes)
    output = SpectralCube(zeros(T,intrinsic_shape), T.(wavelength_m), stokes)
    ForwardWorkspace{T}(pops,intrinsic,output)
end

struct MockForwardModel{P<:AbstractPopulationModel,R<:AbstractRedistributionModel,S<:AbstractSynthesizer,O<:AbstractObservationModel}
    populations::P
    redistribution::R
    synthesizer::S
    observation::O
    capabilities::CapabilityManifest
end

function forward!(workspace::ForwardWorkspace, model::MockForwardModel, atmosphere::Atmosphere3D)
    validate_capabilities(model.capabilities, model.redistribution, workspace.output.stokes)
    predict_populations!(workspace.populations, model.populations, atmosphere)
    synthesize!(workspace.intrinsic, model.synthesizer, model.redistribution, atmosphere, workspace.populations)
    apply_observation!(workspace.output, model.observation, workspace.intrinsic)
end
