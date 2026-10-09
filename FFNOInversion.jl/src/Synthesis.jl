abstract type AbstractSynthesizer end
function synthesize! end

struct ScalarFormalSolver <: AbstractFormalSolver end
struct OpacityContributor{M<:AbstractLineOpacityModel,P}
    model::M; populations::P
end
OpacityContributor(model::AbstractLineOpacityModel)=OpacityContributor(model,nothing)
struct MixedIntensitySynthesizer{S<:AbstractFormalSolver} <: AbstractSynthesizer
    contributors::Vector{OpacityContributor}; solver::S
end
MixedIntensitySynthesizer(contributors)=MixedIntensitySynthesizer(OpacityContributor[contributors...],ScalarFormalSolver())
MixedIntensitySynthesizer(contributors,solver::AbstractFormalSolver)=MixedIntensitySynthesizer(OpacityContributor[contributors...],solver)
mutable struct SynthesisCache{T<:AbstractFloat}
    extinction::Matrix{T}; emissivity::Matrix{T}
end
struct ThreadedSynthesisCache{T<:AbstractFloat}
    workspaces::Vector{SynthesisCache{T}}
end

struct RegionSynthesisSetup{T<:AbstractFloat,S<:AbstractSynthesizer}
    wavelength_air_m::Vector{T}
    wavelength_vacuum_m::Vector{T}
    synthesizer::S
end

"""Build and cache all line physics for configured mixed regions. This performs all file I/O."""
function build_synthesis_setup(config,atmosphere::Atmosphere3D,populations::AbstractDict,
                               eos::WittmannEOS;atom_files::AbstractDict)
    setups=RegionSynthesisSetup[]
    aliases=Dict(:halpha=>(:H,5,2,3),:h_alpha=>(:H,5,2,3),:ca_ii_8542=>(:CA,5,3,5))
    for region in config.regions
        air=wavelengths(region); vacuum=eltype(air).(_air_to_vacuum_nm.(air.*1e9).*1e-9)
        contributors=OpacityContributor[]; continuum_available=true
        for source in region.sources
            if source.mode==:ffno
                haskey(populations,:H) || throw(ArgumentError("H FFNO populations are required for Muspel continuum setup"))
                haskey(aliases,source.line) || throw(ArgumentError("unknown FFNO transition $(source.line)"))
                species,index,lower,upper=aliases[source.line]
                species==source.species || throw(ArgumentError("transition $(source.line) belongs to $species"))
                haskey(populations,species) || throw(ArgumentError("missing $species FFNO populations"))
                haskey(atom_files,species) || throw(ArgumentError("missing $species atom_file"))
                model=build_muspel_line_model(atmosphere,populations[:H],atom_files[species],index,lower,upper;
                    include_continuum=continuum_available)
                push!(contributors,OpacityContributor(model,populations[species])); continuum_available=false
            elseif source.mode==:kurucz_lte
                cache=load_kurucz_linelist(source.linelist_file)
                selected=select_kurucz_lines(cache,vacuum;margin_m=maximum(vacuum)*20*2.5e3/_C)
                isempty(selected) && throw(ArgumentError("no Kurucz lines overlap configured region: $(source.linelist_file)"))
                push!(contributors,OpacityContributor(KuruczLTEModel(selected,eos;include_continuum=continuum_available)))
                continuum_available=false
            end
        end
        isempty(contributors) && throw(ArgumentError("every region needs at least one opacity source"))
        push!(setups,RegionSynthesisSetup(collect(air),collect(vacuum),MixedIntensitySynthesizer(contributors)))
    end
    setups
end
SynthesisCache(::Type{T},nz,nλ) where T<:AbstractFloat=SynthesisCache(zeros(T,nz,nλ),zeros(T,nz,nλ))
ThreadedSynthesisCache(::Type{T},nz,nλ;threads=Threads.maxthreadid()) where T<:AbstractFloat =
    ThreadedSynthesisCache([SynthesisCache(T,nz,nλ) for _ in 1:threads])

function formal_solve!(out,::ScalarFormalSolver,χ,η,z)
    nz,nλ=size(χ); length(z)==nz || throw(DimensionMismatch("z column length mismatch"))
    @inbounds for l in 1:nλ
        source=η[nz,l]/χ[nz,l]; intensity=source
        for k in nz-1:-1:1
            next_source=η[k,l]/χ[k,l]
            dt=abs(z[k+1]-z[k])*(χ[k,l]+χ[k+1,l])/2
            if dt>40
                w1=zero(dt); w2=one(dt)/dt; w3=one(dt)-w2
            elseif dt>0.01
                w1=exp(-dt); u0=(one(dt)-w1)/dt; w2=u0-w1; w3=one(dt)-u0
            else
                w1=one(dt)-dt+dt^2/2; w2=(one(dt)/2-dt/3)*dt; w3=(one(dt)/2-dt/6)*dt
            end
            intensity=w1*intensity+w2*source+w3*next_source; source=next_source
        end
        out[l]=intensity
    end
    out
end

function synthesize!(cube::SpectralCube{T},model::MixedIntensitySynthesizer,redistribution::AbstractRedistributionModel,
                     atmosphere::Atmosphere3D,populations=nothing,cache=nothing) where T
    redistribution isa NonPRD || throw(ArgumentError("mixed intensity synthesis currently supports non-PRD only"))
    cube.stokes.components==(:I,) || throw(ArgumentError("mixed synthesizer supports only Stokes I"))
    atmosphere.z===nothing && throw(ArgumentError("formal synthesis requires geometrical z from force balance"))
    nz,nx,ny=size(atmosphere.temperature); nλ=length(cube.wavelength_m)
    size(cube.data)[3:4]==(nx,ny) || throw(DimensionMismatch("spectral spatial shape mismatch"))
    threaded=cache===nothing ? ThreadedSynthesisCache(T,nz,nλ) : cache
    threaded isa ThreadedSynthesisCache || throw(ArgumentError("mixed synthesis requires ThreadedSynthesisCache"))
    length(threaded.workspaces)>=Threads.maxthreadid() || throw(DimensionMismatch("not enough thread-local synthesis caches"))
    all(ws->size(ws.extinction)==(nz,nλ),threaded.workspaces) || throw(DimensionMismatch("synthesis cache shape mismatch"))
    Threads.@threads :static for column in 1:nx*ny
        x=(column-1)%nx+1; y=(column-1)÷nx+1
        work=threaded.workspaces[Threads.threadid()]
        fill!(work.extinction,zero(T)); fill!(work.emissivity,zero(T))
        for contributor in model.contributors
            contributor_populations=contributor.populations===nothing ? populations : contributor.populations
            add_opacity_emissivity!(work.extinction,work.emissivity,contributor.model,cube.wavelength_m,
                                    atmosphere,x,y,contributor_populations)
        end
        formal_solve!(@view(cube.data[:,1,x,y]),model.solver,work.extinction,work.emissivity,@view(atmosphere.z[:,x,y]))
    end
    cube
end

struct MockIntensitySynthesizer{T<:AbstractFloat} <: AbstractSynthesizer
    line_center_m::T; width_m::T
end
MockIntensitySynthesizer()=MockIntensitySynthesizer(656.28e-9,0.05e-9)
struct MockPolarizedSynthesizer{T<:AbstractFloat} <: AbstractSynthesizer
    polarization_fraction::T
end
MockPolarizedSynthesizer()=MockPolarizedSynthesizer(0.01)
function synthesize!(cube::SpectralCube{T},model::MockIntensitySynthesizer,redistribution::AbstractRedistributionModel,
                     atmosphere::Atmosphere3D,populations::AbstractArray{T,4},cache=nothing) where T
    cube.stokes.components==(:I,) || throw(ArgumentError("intensity synthesizer supports only Stokes I"))
    column=dropdims(sum(@view(populations[:,:,:,1]),dims=1),dims=1)
    coupling=redistribution isa MockPRD ? redistribution.coupling : one(T)
    for il in eachindex(cube.wavelength_m)
        profile=exp(-((cube.wavelength_m[il]-T(model.line_center_m))/T(model.width_m))^2)
        @views cube.data[il,1,:,:].=coupling.*profile.*column
    end
    cube
end
function synthesize!(cube::SpectralCube{T},model::MockPolarizedSynthesizer,redistribution::AbstractRedistributionModel,
                     atmosphere::Atmosphere3D,populations::AbstractArray{T,4},cache=nothing) where T
    cube.stokes.components==(:I,:Q,:U,:V) || throw(ArgumentError("polarized mock expects I,Q,U,V"))
    base=dropdims(sum(@view(populations[:,:,:,1]),dims=1),dims=1)
    for il in axes(cube.data,1)
        @views cube.data[il,1,:,:].=base
        for is in 2:4; @views cube.data[il,is,:,:].=T(model.polarization_fraction*(is-1)).*base; end
    end
    cube
end
