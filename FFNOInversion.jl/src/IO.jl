struct AtmosphereInputConfig{T<:AbstractFloat}
    file::String
    logtau500_dataset::String
    temperature_dataset::String
    vx_dataset::String
    vy_dataset::String
    vz_dataset::String
    pressure_top::Union{T,String}
    magnetic_datasets::Union{Nothing,NTuple{3,String}}
end

struct ObservedDataConfig
    file::String
    intensity_dataset::String
    sigma_dataset::String
end

"""Datasets containing full-grid weights. Zero values exclude points from chi-square."""
struct WeightInputConfig
    wavelength_dataset::String
    spatial_dataset::String
end

"""The two scientific products written by one inversion run."""
struct OutputConfig
    synthesis_file::String
    atmosphere_file::String
end

struct SynthesisGridConfig{T<:AbstractFloat}
    wavelength_m::Vector{T}
    dx_m::T
    dy_m::T
end

"""One opacity source in a region. `:ffno` names a transition; `:kurucz_lte` names a cached line list."""
struct SpectralSourceConfig
    mode::Symbol
    species::Union{Nothing,Symbol}
    line::Union{Nothing,Symbol}
    linelist_file::Union{Nothing,String}
end

"""One simultaneously inverted spectral window and its instrumental profile."""
struct SpectralRegionConfig{T<:AbstractFloat}
    start_m::T
    step_m::T
    count::Int
    normalization::T
    psf_type::Symbol
    psf_file::Union{Nothing,String}
    sources::Vector{SpectralSourceConfig}
end

function wavelengths(region::SpectralRegionConfig{T}) where T
    region.start_m .+ T.(0:region.count-1) .* region.step_m
end

struct RunConfig{T<:AbstractFloat}
    atmosphere::AtmosphereInputConfig{T}
    observed::ObservedDataConfig
    weights::WeightInputConfig
    outputs::OutputConfig
    synthesis::SynthesisGridConfig{T}
    regions::Vector{SpectralRegionConfig{T}}
    observation_model::GaussianPSFObservation{T}
    stokes::StokesSet
    redistribution::Symbol
    regularization::RegularizationSpec{T}
    parallel::ParallelOptions
end

function _regions(cfg)
    raw_regions = _required(cfg,"regions")
    isempty(raw_regions) && throw(ArgumentError("at least one [[regions]] entry is required"))
    regions = SpectralRegionConfig{Float64}[]
    for (i,raw) in enumerate(raw_regions)
        start = Float64(_required(raw,"start_angstrom")) * 1e-10
        step = Float64(_required(raw,"step_angstrom")) * 1e-10
        count = Int(_required(raw,"count"))
        normalization = Float64(_required(raw,"normalization"))
        psf_type = Symbol(lowercase(String(_required(raw,"psf_type"))))
        psf_file_raw = get(raw,"psf_file",nothing)
        psf_file = psf_file_raw === nothing ? nothing : String(psf_file_raw)
        sources = SpectralSourceConfig[]
        for (j,source) in enumerate(get(raw,"sources",Any[]))
            mode = Symbol(lowercase(String(_required(source,"mode"))))
            mode in (:ffno,:kurucz_lte) || throw(ArgumentError("regions[$i].sources[$j].mode must be ffno or kurucz_lte"))
            species = haskey(source,"species") ? Symbol(uppercase(String(source["species"]))) : nothing
            line = haskey(source,"line") ? Symbol(lowercase(String(source["line"]))) : nothing
            list = haskey(source,"linelist_file") ? String(source["linelist_file"]) : nothing
            mode == :ffno && (species === nothing || line === nothing) && throw(ArgumentError("FFNO source requires species and line"))
            mode == :kurucz_lte && (list === nothing || isempty(list)) && throw(ArgumentError("Kurucz LTE source requires linelist_file"))
            push!(sources,SpectralSourceConfig(mode,species,line,list))
        end
        start > 0 || throw(ArgumentError("regions[$i].start_angstrom must be positive"))
        step > 0 || throw(ArgumentError("regions[$i].step_angstrom must be positive"))
        count > 0 || throw(ArgumentError("regions[$i].count must be positive"))
        normalization > 0 || throw(ArgumentError("regions[$i].normalization must be positive"))
        psf_type in (:none,:fpi,:file) || throw(ArgumentError("regions[$i].psf_type must be none, fpi, or file"))
        psf_type == :none && psf_file !== nothing && throw(ArgumentError("regions[$i] cannot specify psf_file with psf_type=none"))
        psf_type != :none && (psf_file === nothing || isempty(psf_file)) &&
            throw(ArgumentError("regions[$i].psf_file is required for psf_type=$psf_type"))
        push!(regions,SpectralRegionConfig(start,step,count,normalization,psf_type,psf_file,sources))
    end
    regions
end

function _required(section,key)
    haskey(section,key) || throw(ArgumentError("missing required configuration key '$key'"))
    section[key]
end

function _regularization(section)
    vertical_raw = get(section,"vertical",Dict{String,Any}())
    vertical = VerticalRegularizationSpec(Int.(get(vertical_raw,"types",fill(0,7))),
        Float64(get(vertical_raw,"regularize",0.0)),Float64.(get(vertical_raw,"weights",fill(1.0,7))))
    horizontal = Dict(Symbol(k)=>Float64(v) for (k,v) in get(section,"horizontal",Dict{String,Any}()))
    scales = Dict(Symbol(k)=>Float64(v) for (k,v) in get(section,"scales",Dict{String,Any}()))
    RegularizationSpec(vertical=vertical,horizontal=horizontal,scales=scales,
        horizontal_order=Int(get(section,"horizontal_order",1)))
end

function load_config(path::AbstractString)
    cfg = TOML.parsefile(path)
    inputs = _required(cfg,"inputs"); outputs_raw = _required(cfg,"outputs")
    atmos = _required(cfg,"atmosphere"); grid = _required(cfg,"grid")
    obs = _required(cfg,"observation")
    physics = _required(cfg,"physics")

    dx_m,dy_m = Float64(_required(grid,"dx_m")),Float64(_required(grid,"dy_m"))
    dx_m > 0 && dy_m > 0 || throw(ArgumentError("dx_m and dy_m must be positive"))
    regions = _regions(cfg)
    wavelength_m = reduce(vcat,wavelengths.(regions))
    !isempty(wavelength_m) && all(>(0),wavelength_m) || throw(ArgumentError("wavelength grid must be positive and non-empty"))
    length(unique(wavelength_m)) == length(wavelength_m) || throw(ArgumentError("spectral regions must not contain duplicate wavelengths"))

    datasets = _required(atmos,"datasets")
    magnetic = all(haskey(datasets,k) for k in ("Bx","By","Bz")) ?
        (String(datasets["Bx"]),String(datasets["By"]),String(datasets["Bz"])) : nothing
    any(haskey(datasets,k) for k in ("Bx","By","Bz")) && magnetic === nothing && throw(ArgumentError("provide all Bx, By and Bz datasets or none"))
    pressure_top = haskey(atmos,"pressure_top_pa") ? Float64(atmos["pressure_top_pa"]) : String(_required(atmos,"pressure_top_dataset"))
    pressure_top isa Float64 && pressure_top <= 0 && throw(ArgumentError("pressure_top_pa must be positive"))
    atmosphere = AtmosphereInputConfig(String(_required(inputs,"initial_atmosphere_file")),String(_required(datasets,"logtau500")),
        String(_required(datasets,"temperature")),String(_required(datasets,"vx")),String(_required(datasets,"vy")),
        String(_required(datasets,"vz")),pressure_top,magnetic)

    data = _required(obs,"datasets")
    observed = ObservedDataConfig(String(_required(inputs,"observation_file")),String(_required(data,"intensity")),String(_required(data,"sigma")))
    weights = WeightInputConfig(String(_required(data,"wavelength_weights")),String(_required(data,"spatial_weights")))
    outputs = OutputConfig(String(_required(outputs_raw,"synthesis_file")),String(_required(outputs_raw,"atmosphere_file")))
    outputs.synthesis_file == outputs.atmosphere_file && throw(ArgumentError("synthesis_file and atmosphere_file must differ"))

    stokes = StokesSet(Tuple(Symbol.(get(obs,"stokes",["I"]))))
    stokes.components == (:I,) || throw(ArgumentError("Release 1 supports Stokes I only"))
    redistribution = Symbol(get(physics,"redistribution","non_prd"))
    redistribution == :non_prd || throw(ArgumentError("Release 1 supports redistribution=non_prd only"))
    psf = get(obs,"gaussian_psf",Dict{String,Any}())
    observation_model = GaussianPSFObservation(Float64(get(psf,"spectral_fwhm_nm",0.0))*1e-9,
        Float64(get(psf,"spatial_fwhm_x_m",0.0)),Float64(get(psf,"spatial_fwhm_y_m",0.0)),dx_m,dy_m)
    regularization = _regularization(get(cfg,"regularization",Dict{String,Any}()))
    parallel_raw = get(cfg,"parallel",Dict{String,Any}())
    decomposition = Symbol(lowercase(String(get(parallel_raw,"decomposition","cartesian_2d"))))
    decomposition == :cartesian_2d || throw(ArgumentError("parallel.decomposition must be cartesian_2d"))
    threads_per_rank = Int(get(parallel_raw,"threads_per_rank",Threads.nthreads()))
    threads_per_rank > 0 || throw(ArgumentError("parallel.threads_per_rank must be positive"))
    gpu_launcher_rank = Int(get(parallel_raw,"gpu_launcher_rank",0))
    gpu_launcher_rank >= 0 || throw(ArgumentError("parallel.gpu_launcher_rank must be non-negative"))
    gpu_connect_timeout_seconds=Float64(get(parallel_raw,"gpu_connect_timeout_seconds",30.0))
    gpu_status_timeout_seconds=Float64(get(parallel_raw,"gpu_status_timeout_seconds",0.0))
    gpu_diagnostic_interval_seconds=Float64(get(parallel_raw,"gpu_diagnostic_interval_seconds",30.0))
    all(>=(0),(gpu_connect_timeout_seconds,gpu_status_timeout_seconds,gpu_diagnostic_interval_seconds)) ||
        throw(ArgumentError("parallel GPU timeout/diagnostic intervals must be non-negative"))
    gpu_diagnostics_directory=String(get(parallel_raw,"gpu_diagnostics_directory",""))
    parallel = ParallelOptions(enabled=Bool(get(parallel_raw,"enabled",false)),decomposition=decomposition,
        threads_per_rank=threads_per_rank,gpu_launcher_rank=gpu_launcher_rank,
        gpu_connect_timeout_seconds=gpu_connect_timeout_seconds,
        gpu_status_timeout_seconds=gpu_status_timeout_seconds,
        gpu_diagnostic_interval_seconds=gpu_diagnostic_interval_seconds,
        gpu_diagnostics_directory=gpu_diagnostics_directory)
    RunConfig(atmosphere,observed,weights,outputs,SynthesisGridConfig(wavelength_m,dx_m,dy_m),regions,observation_model,stokes,redistribution,regularization,parallel)
end

function dry_run_summary(config::RunConfig)
    nlambda = length(config.synthesis.wavelength_m)
    mode = isnothing(config.atmosphere.magnetic_datasets) ? "HE3D" : "MHS"
    vertical_vars = Symbol[VERTICAL_PARAMETER_ORDER[i] for i in 1:7 if config.regularization.vertical.types[i] != 0]
    regvars = sort!(collect(union(vertical_vars,keys(config.regularization.horizontal))))
    nsources=sum(length(r.sources) for r in config.regions)
    "observation_input=$(config.observed.file) atmosphere_input=$(config.atmosphere.file) synthesis_output=$(config.outputs.synthesis_file) atmosphere_output=$(config.outputs.atmosphere_file) logtau=$(config.atmosphere.logtau500_dataset) dx_m=$(config.synthesis.dx_m) dy_m=$(config.synthesis.dy_m) spectral_regions=$(length(config.regions)) spectral_sources=$nsources synthesis_wavelengths=$nlambda full_grid_psf=true zero_weight_exclusion=true stokes=$(join(config.stokes.components,',')) redistribution=$(config.redistribution) force_balance=$mode regularized=$(join(regvars,',')) mpi=$(config.parallel.enabled) decomposition=$(config.parallel.decomposition) threads_per_rank=$(config.parallel.threads_per_rank) gpu_launcher_rank=$(config.parallel.gpu_launcher_rank) gpu_connect_timeout_seconds=$(config.parallel.gpu_connect_timeout_seconds) gpu_status_timeout_seconds=$(config.parallel.gpu_status_timeout_seconds) gpu_diagnostic_interval_seconds=$(config.parallel.gpu_diagnostic_interval_seconds)"
end

function checkpoint!(path::AbstractString,state;manifest::CapabilityManifest=CapabilityManifest())
    open(path,"w") do io; serialize(io,(manifest=manifest,state=state)); end
    path
end

function restore_checkpoint(path::AbstractString;expected::Union{Nothing,CapabilityManifest}=nothing)
    payload = open(deserialize,path)
    expected !== nothing && payload.manifest != expected && throw(ArgumentError("checkpoint capability manifest differs from requested physics"))
    payload
end
