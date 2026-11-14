abstract type AbstractSynthesizer end
function synthesize! end

"""When true (forward! sets it on the root rank), synthesis prints progress lines to stdout."""
const SYNTHESIS_PROGRESS=Base.ScopedValues.ScopedValue(false)

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

"""Unit of every synthesized intensity (Muspel's; RH's per-Hz SI intensity is this × 1e12 λ²/c with λ in m)."""
const SYNTHESIS_INTENSITY_UNITS="kW m^-2 sr^-1 nm^-1"

"""Per-frequency output units: name and factor from W m^-2 Hz^-1 sr^-1."""
const INTENSITY_UNITS=Dict(:si=>("J s^-1 m^-2 Hz^-1 sr^-1",1.0),:cgs=>("erg s^-1 cm^-2 Hz^-1 sr^-1",1e3))

"""Factor per synthesis wavelength (the regions' grids in order) that turns the synthesized intensity
(`SYNTHESIS_INTENSITY_UNITS`) into `units` (:si or :cgs, per Hz as RH and STiC write it: I_ν = I_λ λ²/c with the
vacuum λ) divided by the region's `normalization`. It multiplies every Stokes parameter."""
function output_intensity_scale(regions,units::Symbol)
    _,factor=INTENSITY_UNITS[units]
    scale=Float64[]
    for region in regions
        vacuum=_air_to_vacuum_nm.(wavelengths(region).*1e9).*1e-9
        append!(scale,factor*1e12 .* vacuum.^2 ./ _C ./ region.normalization)  # kW -> W, nm^-1 -> m^-1, per λ -> per ν
    end
    scale
end

"""Kurucz lines of an FFNO transition's ion within this distance of its wavelength are that transition."""
const FFNO_KURUCZ_TOLERANCE_M=0.1e-9

"""Kurucz lines that are not one of the region's FFNO transitions (those would be counted twice, once in LTE)."""
function _without_ffno_transitions(lines,ffno_lines)
    duplicate(line)=any(((z,stage,λ0),)->line.atomic_number==z && line.ion_stage==stage &&
        abs(line.wavelength0_m-λ0)<=FFNO_KURUCZ_TOLERANCE_M,ffno_lines)
    kept=filter(!duplicate,lines)
    length(kept)<length(lines) && @info "Kurucz lines dropped as the region's FFNO transitions" wavelengths_nm=
        [round(line.wavelength0_m*1e9;digits=4) for line in lines if duplicate(line)]
    kept
end

"""Build and cache all line physics for configured mixed regions. This performs all file I/O. A region may combine
FFNO and Kurucz LTE sources, its FFNO lines being those `ffno_region_lines` finds in `atom_files`: their opacities and emissivities (one unit, see `source_planck`) add before a single
formal solution. Every region has one continuum, Muspel's at its central wavelength, whatever its sources: FFNO populations never
enter it. Its hydrogen, and neutral H in line broadening, are the EOS's at (T, P), or a given atmosphere's own. Kurucz lines that are the
region's FFNO transitions are dropped.

With the NLTE EOS (`[equilibrium] eos_model = "nlte"`), `nlte_hydrogen` (`synthesis_hydrogen(equilibrium)`) is the equilibrium's table: the
continuum's hydrogen and neutral H in broadening come from it, and the Kurucz lines put the metals in LTE at the atmosphere's nₑ. Without it
(the default) everything is Wittmann's LTE state at (T, P), as before."""
function build_synthesis_setup(config,atmosphere::Atmosphere3D,populations::AbstractDict,
                               eos::WittmannEOS;atom_files::AbstractDict,nlte_hydrogen::Union{Nothing,NLTEHydrogenState}=nothing)
    setups=RegionSynthesisSetup[]
    given=atmosphere.h_neutral!==nothing
    equilibrium=hasproperty(config,:equilibrium) ? config.equilibrium : nothing
    if nlte_hydrogen===nothing && !given && equilibrium!==nothing && equilibrium.backend in ("HSEMHS","given") && equilibrium.eos_model=="nlte"
        throw(ArgumentError("[equilibrium] eos_model = \"nlte\": pass build_synthesis_setup(...; nlte_hydrogen=synthesis_hydrogen(equilibrium)) so that the synthesis uses the NLTE EOS"))
    end
    hydrogen_state()=given ? nothing : nlte_hydrogen===nothing ? WittmannKuruczState(eos) : nlte_hydrogen
    line_center=hasproperty(config,:line_center) ? config.line_center : :reference
    population_limit=hasproperty(config,:ffno_population_limit) ? config.ffno_population_limit : 0.0
    no_inversion=hasproperty(config,:ffno_no_inversion) ? config.ffno_no_inversion : false
    # given populations ([population.given]) are used as they are: the FFNO limits are not theirs
    given_species=hasproperty(config,:given_populations) && config.given_populations!==nothing ? keys(config.given_populations.files) : ()
    atom_lines=any(source.mode==:ffno for region in config.regions for source in region.sources) ? ffno_atom_lines(atom_files) : FFNOLine[]
    for region in config.regions
        air=wavelengths(region); vacuum=eltype(air).(_air_to_vacuum_nm.(air.*1e9).*1e-9)
        isempty(region.sources) && throw(ArgumentError("every region needs at least one opacity source"))
        contributors=OpacityContributor[OpacityContributor(build_muspel_continuum_model(vacuum[(length(vacuum)+1)÷2]*1e9;
            eos=hydrogen_state()))]
        lines=any(source.mode==:ffno for source in region.sources) ? ffno_region_lines(region,atom_lines;line_center=line_center) : FFNOLine[]
        ffno_lines=[(line.atomic_number,line.ion_stage,ffno_line_center_nm(line)*1e-9) for line in lines]
        for line in lines
            haskey(populations,:H) || throw(ArgumentError("H FFNO populations are required for the Muspel line model"))
            species=line.species
            haskey(populations,species) || throw(ArgumentError("missing $species FFNO populations for $(describe_ffno_line(line,line_center))"))
            size(populations[species],4)==line.level_count || throw(DimensionMismatch(
                "$species FFNO populations have $(size(populations[species],4)) levels; its atom has $(line.level_count)"))
            model=build_muspel_line_model(atmosphere,populations[:H],atom_files[species],line.line_index,line.lower_level,line.upper_level;
                include_continuum=false,neutral_hydrogen=hydrogen_state(),
                center_nm=line_center===:reference ? line.reference_vacuum_nm : nothing,
                population_limit=species in given_species ? 0.0 : population_limit,no_inversion=no_inversion && !(species in given_species))
            push!(contributors,OpacityContributor(model,populations[species]))
        end
        for source in region.sources
            if source.mode==:kurucz_lte
                cache=load_kurucz_linelist(source.linelist_file)
                selected=select_kurucz_lines(cache,vacuum;margin_m=_line_margin_m(vacuum))
                isempty(selected) && throw(ArgumentError("no Kurucz lines overlap configured region: $(source.linelist_file)"))
                selected=_without_ffno_transitions(selected,ffno_lines)
                isempty(selected) && (@warn "every Kurucz line in the region is an FFNO transition; source skipped" source.linelist_file; continue)
                push!(contributors,OpacityContributor(nlte_hydrogen===nothing ? KuruczLTEModel(selected,eos;include_continuum=false) :
                    KuruczLTEModel(selected,eos,nlte_hydrogen;include_continuum=false)))
            end
        end
        push!(setups,RegionSynthesisSetup(collect(air),collect(vacuum),MixedIntensitySynthesizer(contributors)))
    end
    setups
end
SynthesisCache(::Type{T},nz,nλ) where T<:AbstractFloat=SynthesisCache(zeros(T,nz,nλ),zeros(T,nz,nλ))
ThreadedSynthesisCache(::Type{T},nz,nλ;threads=Threads.maxthreadid()) where T<:AbstractFloat =
    ThreadedSynthesisCache([SynthesisCache(T,nz,nλ) for _ in 1:threads])

const _MAX_LOG_JUMP=1.0      # largest change of ln χ or ln η within one (sub-)step
const _MAX_SUBSTEPS=256

"""Sub-steps for the step between depth points `a` and `b`: one unless ln χ or ln η changes by more than
`_MAX_LOG_JUMP`. Across a transition region on a 0.1 dex log τ grid, χ can rise 10⁸ and S fall 10⁵ in one step;
a source function linear in τ over that optically thick step gives the hot end's S a weight 1/Δτ and the
emergent intensity is set by the corona. Sub-stepping with χ and η log-linear in the integration variable puts
the opacity where the gas is cool."""
@inline function _substeps(χa,χb,ηa,ηb)
    (χa>0 && χb>0 && ηa>0 && ηb>0) || return 1
    jump=max(abs(log(χb/χa)),abs(log(ηb/ηa)))
    jump>_MAX_LOG_JUMP || return 1
    min(ceil(Int,jump/_MAX_LOG_JUMP),_MAX_SUBSTEPS)
end
"""Value at sub-point `q` of `m` between `a` (q = 0) and `b` (q = m), log-linear in between."""
@inline _substep_value(a,b,q,m)=q==0 ? a : q==m ? b : (t=oftype(a,q/m); exp((1-t)*log(a)+t*log(b)))

@inline function _formal_weights(dt)
    if dt>40
        w1=zero(dt); w2=one(dt)/dt; w3=one(dt)-w2
    elseif dt>0.01
        w1=exp(-dt); u0=(one(dt)-w1)/dt; w2=u0-w1; w3=one(dt)-u0
    else
        w1=one(dt)-dt+dt^2/2; w2=(one(dt)/2-dt/3)*dt; w3=(one(dt)/2-dt/6)*dt
    end
    w1,w2,w3
end

"""Short-characteristic formal solution, source function linear in τ per (sub-)step, from the bottom point
`nz` (I = S) to the top point 1. Steps where χ or η jump are sub-stepped (see `_substeps`)."""
function formal_solve!(out,::ScalarFormalSolver,χ,η,z)
    nz,nλ=size(χ); length(z)==nz || throw(DimensionMismatch("z column length mismatch"))
    @inbounds for l in 1:nλ
        source=η[nz,l]/χ[nz,l]; intensity=source
        for k in nz-1:-1:1
            m=_substeps(χ[k+1,l],χ[k,l],η[k+1,l],η[k,l])
            h=abs(z[k+1]-z[k])/m; chi_prev=χ[k+1,l]
            for q in 1:m
                c=_substep_value(χ[k+1,l],χ[k,l],q,m); e=_substep_value(η[k+1,l],η[k,l],q,m)
                next_source=e/c
                w1,w2,w3=_formal_weights(h*(chi_prev+c)/2)
                intensity=w1*intensity+w2*source+w3*next_source; source=next_source; chi_prev=c
            end
        end
        out[l]=intensity
    end
    out
end

"""True when synthesis integrates the optical depth on the native log10 tau500 grid.

That needs a log-tau vertical axis and the equilibrium's `chi500`. With column-HSE heights, a 0.1 dex step can
span the whole transition region (9 600 km in one column of the 504x504 run): the trapezoid `(χ[k]+χ[k+1])/2·Δz`
then charges the cool point's opacity over the hot gas's height and makes the step optically thick, so the
emergent intensity takes the coronal source function. The grid's own Δτ500 has no such error.
"""
_tau_integration(atmosphere)=vertical_coordinate(atmosphere.grid)===:log10tau500 && atmosphere.chi500!==nothing

"""True when synthesis integrates the optical depth on the native log10 column-mass grid, dτ = (χ/ρ) dm. That needs
the equilibrium's `rho`. It avoids the height trapezoid for the reason given at `_tau_integration`, and does not use
the equilibrium's τ500, which on this axis is itself an output of one trapezoid per level."""
_mass_integration(atmosphere)=vertical_coordinate(atmosphere.grid)===:log10m && atmosphere.rho!==nothing

"""True when synthesis integrates on the native axis (log10 τ500 or log10 m) rather than in height."""
_axis_integration(atmosphere)=_tau_integration(atmosphere) || _mass_integration(atmosphere)

"""Formal solution of one column. On a native log-tau grid the integration variable is s = -ln τ500
(decreasing with depth index, like z): `chi` and `eta` are scaled in place by τ500/χ500, so each step's
optical depth is the trapezoid of (χ/χ500)·τ500 in ln τ500; off the vertical the axis is τ along the line of sight,
so nothing changes. A native column-mass grid is the same with τ500 -> m and χ500 -> ρ (m along the line of sight off
the vertical). Otherwise the path is the geometric height `z`, divided by μ off the vertical (ds = dz/μ)."""
function formal_column!(out,solver,chi,eta,atmosphere,x,y)
    _axis_integration(atmosphere) || return formal_solve!(out,solver,chi,eta,_geometric_path(atmosphere,x,y))
    coordinate=_axis_coordinate(atmosphere,x,y)
    weight=_axis_weight(atmosphere,x,y)
    chi.*=weight; eta.*=weight
    formal_solve!(out,solver,chi,eta,coordinate)
end
_axis_coordinate(atmosphere,x,y)=_mass_integration(atmosphere) ? _log_mass_coordinate(atmosphere) : _log_tau_coordinate(atmosphere,x,y)
_axis_weight(atmosphere,x,y)=_mass_integration(atmosphere) ? _log_mass_weight(atmosphere,x,y) : _log_tau_weight(atmosphere,x,y)
_log_mass_coordinate(atmosphere)=-log(eltype(atmosphere.rho)(10)) .* native_depth(atmosphere.grid)
_log_mass_weight(atmosphere,x,y)=exp10.(native_depth(atmosphere.grid)) ./ @view(atmosphere.rho[:,x,y])
_log_tau_coordinate(atmosphere,x,y)=-log(eltype(atmosphere.chi500)(10)) .* @view(atmosphere.logtau500[:,x,y])
"""Geometric path coordinate of column (x, y): z, or z/μ off the vertical."""
function _geometric_path(atmosphere,x,y)
    z=@view(atmosphere.z[:,x,y]); v=atmosphere.grid.view
    is_vertical(v) ? z : z.*eltype(z)(path_factor(v))
end
_log_tau_weight(atmosphere,x,y)=exp10.(@view(atmosphere.logtau500[:,x,y])) ./ @view(atmosphere.chi500[:,x,y])

function synthesize!(cube::SpectralCube{T},model::MixedIntensitySynthesizer,redistribution::AbstractRedistributionModel,
                     atmosphere::Atmosphere3D,populations=nothing,cache=nothing) where T
    threaded=_mixed_synthesis_cache(cube,redistribution,atmosphere,cache)
    _synthesize_mixed!(cube.data,cube.wavelength_m,model,atmosphere,populations,threaded)
    cube
end

"""Synthesize every configured region into its slice of `cube`.

The cube's wavelength axis is the regions' air-wavelength grids concatenated in configuration order (`IO.jl`).
Each region is computed on its vacuum grid, because the Muspel line centres (`[physics] line_center`) and the
Kurucz line lists are vacuum wavelengths. `[outputs] wavelength_frame` only labels the written axis.
"""
function synthesize!(cube::SpectralCube{T},setups::AbstractVector{<:RegionSynthesisSetup},
                     redistribution::AbstractRedistributionModel,atmosphere::Atmosphere3D,
                     populations=nothing,cache=nothing) where T
    threaded=_mixed_synthesis_cache(cube,redistribution,atmosphere,cache)
    for (i,(setup,range)) in enumerate(zip(setups,_region_ranges(cube.wavelength_m,setups)))
        _synthesize_mixed!(@view(cube.data[range,:,:,:]),setup.wavelength_vacuum_m,setup.synthesizer,
                           atmosphere,populations,threaded;label="region=$i/$(length(setups)) wavelengths=$(length(range))")
    end
    cube
end

"""Index range of each region in the concatenated cube wavelength axis; checks the grids match."""
function _region_ranges(wavelength_m,setups)
    ranges=UnitRange{Int}[]; start=1
    for (i,setup) in enumerate(setups)
        stop=start+length(setup.wavelength_air_m)-1
        stop<=length(wavelength_m) && isapprox(wavelength_m[start:stop],setup.wavelength_air_m;rtol=1e-12) ||
            throw(DimensionMismatch("spectral region $i does not match the synthesis wavelength grid"))
        push!(ranges,start:stop); start=stop+1
    end
    start==length(wavelength_m)+1 || throw(DimensionMismatch(
        "synthesis wavelength grid has $(length(wavelength_m)) points; the regions cover $(start-1)"))
    ranges
end

"""An inclined native-z atmosphere has Cartesian columns, which are not rays: only its footprint atmosphere
(`footprint_atmosphere`) may be synthesized."""
function _check_synthesis_columns(atmosphere)
    v=atmosphere.grid.view
    vertical_coordinate(atmosphere.grid)===:z_m && is_slanted(v) && !v.footprint && throw(ArgumentError(
        "inclined view on a native height grid: synthesize the footprint atmosphere (rays), not the Cartesian columns"))
    nothing
end

function _mixed_synthesis_cache(cube::SpectralCube{T},redistribution,atmosphere,cache) where T
    redistribution isa NonPRD || throw(ArgumentError("mixed intensity synthesis currently supports non-PRD only"))
    cube.stokes.components==(:I,) || throw(ArgumentError("mixed synthesizer supports only Stokes I"))
    _axis_integration(atmosphere) || atmosphere.z!==nothing ||
        throw(ArgumentError("formal synthesis requires geometrical z, or chi500 on a log-tau grid, or rho on a log-m grid"))
    _check_synthesis_columns(atmosphere)
    nz,nx,ny=size(atmosphere.temperature); nλ=length(cube.wavelength_m)
    size(cube.data)[3:4]==(nx,ny) || throw(DimensionMismatch("spectral spatial shape mismatch"))
    threaded=cache===nothing ? ThreadedSynthesisCache(T,nz,nλ) : cache
    threaded isa ThreadedSynthesisCache || throw(ArgumentError("mixed synthesis requires ThreadedSynthesisCache"))
    length(threaded.workspaces)>=Threads.maxthreadid() || throw(DimensionMismatch("not enough thread-local synthesis caches"))
    all(ws->size(ws.extinction)==(nz,nλ),threaded.workspaces) || throw(DimensionMismatch("synthesis cache shape mismatch"))
    threaded
end

"""Opacity, emissivity and formal solution for every column; `data` is (λ, Stokes, x, y) on `wavelength`."""
function _synthesize_mixed!(data,wavelength,model::MixedIntensitySynthesizer,atmosphere,populations,threaded;
                            label="wavelengths=$(length(wavelength))")
    nx,ny=size(data,3),size(data,4); nλ=length(wavelength); ncolumns=nx*ny
    size(data,1)==nλ || throw(DimensionMismatch("spectral slice and wavelength grid differ"))
    # Rank-0 progress bar (as in Forward.jl); updates at most every 10 s to keep Slurm logs small.
    progress=Progress(ncolumns;desc="Synthesis $label (rank-0 slab) ",dt=10,output=stdout,
                      enabled=SYNTHESIS_PROGRESS[])
    Threads.@threads :static for column in 1:ncolumns
        x=(column-1)%nx+1; y=(column-1)÷nx+1
        work=threaded.workspaces[Threads.threadid()]
        extinction=@view(work.extinction[:,1:nλ]); emissivity=@view(work.emissivity[:,1:nλ])
        fill!(extinction,zero(eltype(extinction))); fill!(emissivity,zero(eltype(emissivity)))
        for contributor in model.contributors
            contributor_populations=contributor.populations===nothing ? populations : contributor.populations
            add_opacity_emissivity!(extinction,emissivity,contributor.model,wavelength,
                                    atmosphere,x,y,contributor_populations)
        end
        formal_column!(@view(data[:,1,x,y]),model.solver,extinction,emissivity,atmosphere,x,y)
        next!(progress)
    end
    finish!(progress)
    data
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
