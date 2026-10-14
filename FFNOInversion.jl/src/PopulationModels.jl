abstract type AbstractPopulationModel end
function predict_populations! end
"""Reverse one population inference without materializing its Jacobian.

`feature_bar` follows `FFNO_INPUT_CHANNELS`, `z_bar` follows the atmospheric
height cube, and `population_bar` has the canonical `(nz,nx,ny,nlevel)`
layout.  Production FFNO implementations execute this action on the same
persistent backend used for inference.
"""
function population_vjp! end

const FFNO_INPUT_CHANNELS=(:temperature,:vx,:vy,:vz,:log10_ne,:log10_rho)

struct PopulationMetadata
    input_channels::Tuple{Vararg{Symbol}}
    level_names::Tuple{Vararg{String}}
    checkpoint_hash::String
    output_representation::Symbol
    function PopulationMetadata(input_channels,level_names,checkpoint_hash;
                                output_representation=:linear_population_m3)
        Tuple(input_channels)==FFNO_INPUT_CHANNELS || throw(ArgumentError("FFNO input channels must be $(FFNO_INPUT_CHANNELS)"))
        isempty(level_names) && throw(ArgumentError("at least one population level is required"))
        isempty(checkpoint_hash) && throw(ArgumentError("checkpoint hash cannot be empty"))
        output_representation==:linear_population_m3 || throw(ArgumentError("FFNO backend must return linear populations in m^-3"))
        new(Tuple(input_channels),Tuple(String.(level_names)),String(checkpoint_hash),output_representation)
    end
end

"""Create FFNO channels `(channel,nz,nx,ny)` in the training order."""
function population_features(atmosphere::Atmosphere3D)
    atmosphere.ne===nothing && throw(ArgumentError("FFNO inference requires electron density"))
    atmosphere.rho===nothing && throw(ArgumentError("FFNO inference requires mass density"))
    atmosphere.z===nothing && throw(ArgumentError("FFNO inference requires geometrical height"))
    all(>(0),atmosphere.ne) || throw(ArgumentError("electron density must be positive"))
    all(>(0),atmosphere.rho) || throw(ArgumentError("mass density must be positive"))
    shape=size(atmosphere.temperature); out=Array{Float32}(undef,6,shape...)
    @views out[1,:,:,:].=atmosphere.temperature
    @views out[2,:,:,:].=atmosphere.vx
    @views out[3,:,:,:].=atmosphere.vy
    @views out[4,:,:,:].=atmosphere.vz
    @views out[5,:,:,:].=log10.(atmosphere.ne)
    @views out[6,:,:,:].=log10.(atmosphere.rho)
    all(isfinite,out) || throw(ArgumentError("FFNO features contain NaN or Inf"))
    out
end

function _spacing(values,name)
    length(values)==1 && return 1.0f0
    d=diff(values); reference=sum(d)/length(d)
    maximum(abs.(d.-reference))<=max(abs(reference)*1e-6,eps(eltype(values))*10) ||
        throw(ArgumentError("FFNO currently requires uniform $name spacing"))
    Float32(abs(reference))
end

"""In-memory recorded backend for parity tests without Python, GPU, or file I/O."""
struct RecordedPopulationModel{T<:AbstractFloat,A<:AbstractArray{T,4},Z<:AbstractArray{T,3}} <: AbstractPopulationModel
    features::A
    z::Z
    populations::A
    metadata::PopulationMetadata
    rtol::T
end

function RecordedPopulationModel(features::AbstractArray{T,4},z::AbstractArray{T,3},populations::AbstractArray{T,4},
                                 metadata::PopulationMetadata;rtol=T(1e-6)) where T<:AbstractFloat
    size(features,1)==6 || throw(DimensionMismatch("recorded feature cube needs six channels"))
    size(features)[2:4]==size(z)==size(populations)[1:3] || throw(DimensionMismatch("recorded spatial/depth shapes differ"))
    size(populations,4)==length(metadata.level_names) || throw(DimensionMismatch("recorded level count differs from metadata"))
    RecordedPopulationModel(copy(features),copy(z),copy(populations),metadata,rtol)
end

function predict_populations!(out::AbstractArray{T,4},model::RecordedPopulationModel,
                              atmosphere::Atmosphere3D,cache=nothing) where T
    features=population_features(atmosphere); z=Float32.(atmosphere.z)
    size(out)==size(model.populations) || throw(DimensionMismatch("recorded population output shape differs"))
    isapprox(features,model.features;rtol=model.rtol,atol=0) || throw(ArgumentError("atmosphere differs from recorded FFNO request"))
    isapprox(z,model.z;rtol=model.rtol,atol=0) || throw(ArgumentError("z scale differs from recorded FFNO request"))
    out.=T.(model.populations); out
end

function save_population_record(path::AbstractString,model::RecordedPopulationModel)
    open(path,"w") do io; serialize(io,model); end
    path
end

function load_population_record(path::AbstractString)
    model=open(deserialize,path)
    model isa RecordedPopulationModel || throw(ArgumentError("file does not contain a recorded population model"))
    model
end

struct MockPopulationModel{T<:AbstractFloat} <: AbstractPopulationModel
    scale::T
end
MockPopulationModel() = MockPopulationModel(1f10)

"""Deterministic Phase 0 mock: one positive population channel `(nz,nx,ny,1)`."""
function predict_populations!(out::AbstractArray{T,4}, model::MockPopulationModel, atmosphere::Atmosphere3D, cache=nothing) where T
    size(out)[1:3] == size(atmosphere.temperature) || throw(DimensionMismatch("population shape differs from atmosphere"))
    size(out,4) == 1 || throw(DimensionMismatch("mock population model has one level"))
    @views out[:,:,:,1] .= T(model.scale) .* atmosphere.temperature ./ T(5000)
    out
end
