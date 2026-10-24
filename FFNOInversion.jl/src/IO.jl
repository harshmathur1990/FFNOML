"""Initial-atmosphere input. `vertical_coordinate` is the native axis (`:z_m`
or `:log10tau500`) and `vertical_dataset` the dataset holding it. `x_dataset`
and `y_dataset` name explicit horizontal axes in metres (required for native
z, optional for log tau, where `[grid] dx_m/dy_m` otherwise define x and y)."""
struct AtmosphereInputConfig{T<:AbstractFloat}
    file::String
    storage_order::Symbol
    vertical_coordinate::Symbol
    vertical_dataset::String
    x_dataset::Union{Nothing,String}
    y_dataset::Union{Nothing,String}
    temperature_dataset::String
    vx_dataset::String
    vy_dataset::String
    vz_dataset::String
    magnetic_datasets::Union{Nothing,NTuple{3,String}}
    vturb_dataset::Union{Nothing,String}
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

"""Configuration for one optimized coarse atmospheric control map. `nodes` are
vertical positions on the native coordinate `kind` (metres or dex)."""
struct ControlMapConfig{T<:AbstractFloat}
    variable::Symbol
    nodes::Vector{T}
    kind::Symbol
    control_nx::Int
    control_ny::Int
    lower::T
    upper::T
    scale::T
end

"""Legacy positional form: nodes on log10tau500."""
ControlMapConfig(variable::Symbol,nodes::Vector{T},control_nx::Int,control_ny::Int,lower::T,upper::T,scale::T) where {T<:AbstractFloat} =
    ControlMapConfig(variable,nodes,:log10tau500,control_nx,control_ny,lower,upper,scale)

function wavelengths(region::SpectralRegionConfig{T}) where T
    region.start_m .+ T.(0:region.count-1) .* region.step_m
end

struct RunConfig{T<:AbstractFloat}
    atmosphere::AtmosphereInputConfig{T}
    mode::Symbol
    observed::Union{Nothing,ObservedDataConfig}
    weights::Union{Nothing,WeightInputConfig}
    outputs::OutputConfig
    synthesis::SynthesisGridConfig{T}
    regions::Vector{SpectralRegionConfig{T}}
    observation_model::GaussianPSFObservation{T}
    stokes::StokesSet
    redistribution::Symbol
    regularization::RegularizationSpec{T}
    parallel::ParallelOptions
    controls::Vector{ControlMapConfig{T}}
    solver::Union{PrototypeSolverOptions,LBFGSSolverOptions}
    time_index::Int
    equilibrium::EquilibriumConfig
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
    length_scale_m = haskey(vertical_raw,"length_scale_m") ? Float64(vertical_raw["length_scale_m"]) : nothing
    vertical = VerticalRegularizationSpec(Int.(get(vertical_raw,"types",fill(0,7))),
        Float64(get(vertical_raw,"regularize",0.0)),Float64.(get(vertical_raw,"weights",fill(1.0,7)));
        length_scale_m=length_scale_m,length_scale_dex=Float64(get(vertical_raw,"length_scale_dex",1.0)))
    horizontal = Dict{Symbol,Float64}(Symbol(k)=>Float64(v) for (k,v) in
        get(section,"horizontal",Dict{String,Any}()))
    scales = Dict{Symbol,Float64}(Symbol(k)=>Float64(v) for (k,v) in
        get(section,"scales",Dict{String,Any}()))
    RegularizationSpec(vertical=vertical,horizontal=horizontal,scales=scales,
        horizontal_order=Int(get(section,"horizontal_order",1)))
end

const _NODE_KEYS = Dict(:z_m=>"z_nodes_m",:log10tau500=>"log_tau_nodes")

function _controls(section,vertical_coordinate::Symbol=:log10tau500)
    key=_NODE_KEYS[vertical_coordinate]; other=_NODE_KEYS[vertical_coordinate===:z_m ? :log10tau500 : :z_m]
    raw_controls=get(section,"controls",Any[])
    isempty(raw_controls) && throw(ArgumentError("at least one [[inversion.controls]] entry is required"))
    controls=ControlMapConfig{Float64}[]
    for (i,raw) in enumerate(raw_controls)
        normalized=lowercase(String(_required(raw,"variable")))
        variable=get(Dict("bx"=>:Bx,"by"=>:By,"bz"=>:Bz),normalized,Symbol(normalized))
        variable in PROTOTYPE_CONTROL_VARIABLES || throw(ArgumentError(
            "inversion.controls[$i].variable=$variable is unsupported"))
        haskey(raw,other) && throw(ArgumentError(
            "inversion.controls[$i].$other is not valid for vertical_coordinate=$(vertical_coordinate); use $key [$(vertical_unit(vertical_coordinate))]"))
        haskey(raw,key) || throw(ArgumentError(
            "inversion.controls[$i] needs $key [$(vertical_unit(vertical_coordinate))] for vertical_coordinate=$(vertical_coordinate)"))
        nodes=Float64.(raw[key])
        length(nodes)>=2 || throw(ArgumentError("inversion.controls[$i] needs at least two $key"))
        all(isfinite,nodes) || throw(ArgumentError("inversion.controls[$i].$key must be finite"))
        d=diff(nodes); (all(>(0),d)||all(<(0),d)) || throw(ArgumentError(
            "inversion.controls[$i].$key must be strictly monotonic"))
        nx=Int(get(raw,"control_nx",1)); ny=Int(get(raw,"control_ny",1))
        nx>0 && ny>0 || throw(ArgumentError("inversion control dimensions must be positive"))
        lower=Float64(_required(raw,"lower")); upper=Float64(_required(raw,"upper"))
        scale=Float64(_required(raw,"scale"))
        lower<upper || throw(ArgumentError("inversion.controls[$i] lower must be less than upper"))
        scale>0 || throw(ArgumentError("inversion.controls[$i] scale must be positive"))
        variable===:temperature && lower<=0 && throw(ArgumentError("temperature lower bound must be positive"))
        variable===:vturb && lower<0 && throw(ArgumentError("vturb lower bound must be non-negative"))
        push!(controls,ControlMapConfig(variable,nodes,vertical_coordinate,nx,ny,lower,upper,scale))
    end
    variables=getfield.(controls,:variable)
    length(unique(variables))==length(variables) || throw(ArgumentError("inversion control variables must be unique"))
    controls
end

function _prototype_solver(section)
    PrototypeSolverOptions(
        max_iterations=Int(get(section,"max_iterations",20)),
        initial_step=Float64(get(section,"initial_step",0.25)),
        step_shrink=Float64(get(section,"step_shrink",0.5)),
        minimum_step=Float64(get(section,"minimum_step",1e-3)),
        improvement_tolerance=Float64(get(section,"improvement_tolerance",1e-8)),
        maximum_evaluations=Int(get(section,"maximum_evaluations",typemax(Int))),
        checkpoint_every=Int(get(section,"checkpoint_every",1)),
        checkpoint_path=String(get(section,"checkpoint_path",""))) |> _validate
end

function _lbfgs_solver(section)
    LBFGSSolverOptions(
        max_iterations=Int(get(section,"max_iterations",50)),
        history_length=Int(get(section,"history_length",10)),
        gradient_tolerance=Float64(get(section,"gradient_tolerance",1e-6)),
        step_tolerance=Float64(get(section,"step_tolerance",1e-10)),
        objective_tolerance=Float64(get(section,"objective_tolerance",1e-12)),
        initial_step=Float64(get(section,"initial_step",1.0)),
        backtracking_factor=Float64(get(section,"backtracking_factor",0.5)),
        armijo_coefficient=Float64(get(section,"armijo_coefficient",1e-4)),
        maximum_line_search_trials=Int(get(section,"maximum_line_search_trials",20)),
        maximum_forward_evaluations=Int(get(section,"maximum_forward_evaluations",typemax(Int))),
        curvature_tolerance=Float64(get(section,"curvature_tolerance",1e-10)),
        checkpoint_every=Int(get(section,"checkpoint_every",1)),
        checkpoint_path=String(get(section,"checkpoint_path",""))) |> _validate
end

function _solver(section)
    method=Symbol(lowercase(String(get(section,"method","lbfgs"))))
    method in (:lbfgs,:bounded_lbfgs) && return _lbfgs_solver(section)
    method in (:prototype,:prototype_pattern_search) && return _prototype_solver(section)
    throw(ArgumentError("solver.method must be lbfgs or prototype_pattern_search"))
end

function _sample_control_field(atmosphere::Atmosphere3D{T},config::ControlMapConfig{T}) where T
    config.kind===vertical_coordinate(atmosphere.grid) || throw(ArgumentError(
        "control nodes for $(config.variable) are on :$(config.kind) but the atmosphere's native coordinate is :$(vertical_coordinate(atmosphere.grid))"))
    source=_control_destination(atmosphere,config.variable); depth=native_depth(atmosphere.grid)
    xsource=length(atmosphere.grid.x)==1 ? T[0] : collect(range(zero(T),one(T),length=length(atmosphere.grid.x)))
    ysource=length(atmosphere.grid.y)==1 ? T[0] : collect(range(zero(T),one(T),length=length(atmosphere.grid.y)))
    xtarget=config.control_nx==1 ? T[0] : collect(range(zero(T),one(T),length=config.control_nx))
    ytarget=config.control_ny==1 ? T[0] : collect(range(zero(T),one(T),length=config.control_ny))
    values=Array{T}(undef,length(config.nodes),config.control_nx,config.control_ny)
    for (k,node) in enumerate(config.nodes),(i,x) in enumerate(xtarget),(j,y) in enumerate(ytarget)
        z0,z1,wz=_bracket(depth,node)
        x0,x1,wx=length(xsource)==1 ? (1,1,zero(T)) : _bracket(xsource,x)
        y0,y1,wy=length(ysource)==1 ? (1,1,zero(T)) : _bracket(ysource,y)
        v0=(1-wx)*((1-wy)*source[z0,x0,y0]+wy*source[z0,x0,y1])+
            wx*((1-wy)*source[z0,x1,y0]+wy*source[z0,x1,y1])
        v1=(1-wx)*((1-wy)*source[z1,x0,y0]+wy*source[z1,x0,y1])+
            wx*((1-wy)*source[z1,x1,y0]+wy*source[z1,x1,y1])
        values[k,i,j]=(1-wz)*v0+wz*v1
    end
    NodeField(values,config.nodes,config.kind)
end

"""Initialize configured control maps by sampling the input atmosphere."""
function build_control_layout(atmosphere::Atmosphere3D{T},configs::AbstractVector{ControlMapConfig{T}}) where T
    specs=ControlMapSpec{T}[]
    for config in configs
        field=_sample_control_field(atmosphere,config)
        push!(specs,ControlMapSpec(config.variable,field;lower=config.lower,upper=config.upper,scale=config.scale))
    end
    ControlMapLayout(specs)
end

"""Native vertical coordinate, its dataset and the explicit x/y datasets.

`vertical_coordinate = "z"` requires `datasets.z_m`, `x_m` and `y_m` and forbids
`logtau500`; `"log10tau500"` requires `logtau500` and forbids `z_m`. Without the
key, a `logtau500` dataset and no `z_m` mean the legacy log-tau input."""
function _vertical_input(atmos,datasets)
    raw=get(atmos,"vertical_coordinate",nothing)
    has_z=haskey(datasets,"z_m"); has_tau=haskey(datasets,"logtau500")
    kind=if raw===nothing
        has_tau && !has_z || throw(ArgumentError(
            "atmosphere.vertical_coordinate is required unless the legacy datasets.logtau500 (and no z_m) is given"))
        :log10tau500
    else
        name=lowercase(String(raw))
        name in ("z","z_m") ? :z_m : name in ("log10tau500","logtau500") ? :log10tau500 :
            throw(ArgumentError("atmosphere.vertical_coordinate must be \"z\" or \"log10tau500\", got \"$raw\""))
    end
    if kind===:z_m
        has_tau && throw(ArgumentError("datasets.logtau500 is not allowed with vertical_coordinate=z (native height input needs no optical depth)"))
        has_z || throw(ArgumentError("vertical_coordinate=z requires datasets.z_m [m]"))
        haskey(datasets,"x_m") && haskey(datasets,"y_m") || throw(ArgumentError(
            "vertical_coordinate=z requires explicit datasets.x_m and datasets.y_m [m]"))
    else
        has_z && throw(ArgumentError("datasets.z_m is not allowed with vertical_coordinate=log10tau500"))
        has_tau || throw(ArgumentError("vertical_coordinate=log10tau500 requires datasets.logtau500"))
        haskey(datasets,"x_m") == haskey(datasets,"y_m") || throw(ArgumentError("give both datasets.x_m and datasets.y_m or neither"))
    end
    vertical=String(datasets[kind===:z_m ? "z_m" : "logtau500"])
    x=haskey(datasets,"x_m") ? String(datasets["x_m"]) : nothing
    y=haskey(datasets,"y_m") ? String(datasets["y_m"]) : nothing
    kind,vertical,x,y
end

"""Uniform spacing of a strictly increasing axis; non-uniform axes are rejected
because the FSDP population payload and the PSF kernels need one spacing."""
function _uniform_axis_spacing(values,name)
    length(values)==1 && return nothing
    all(isfinite,values) || throw(ArgumentError("$name must be finite"))
    d=diff(Float64.(values)); all(>(0),d) || throw(ArgumentError(
        "$name must be strictly increasing; decreasing horizontal axes are not normalized"))
    maximum(abs.(d.-d[1]))<=1e-9*abs(d[1]) || throw(ArgumentError(
        "$name must be uniformly spaced: the FSDP population service and the PSF kernels need one spacing"))
    d[1]
end

"""Pixel spacings: `[grid] dx_m/dy_m`, the spacing of explicit x/y datasets, or
both (which must agree to 1e-9)."""
function _horizontal_spacing(grid,file,x_dataset,y_dataset)
    configured=(haskey(grid,"dx_m") ? Float64(grid["dx_m"]) : nothing,haskey(grid,"dy_m") ? Float64(grid["dy_m"]) : nothing)
    for value in configured
        value===nothing || value>0 || throw(ArgumentError("dx_m and dy_m must be positive"))
    end
    explicit=if x_dataset===nothing
        (nothing,nothing)
    else
        isfile(file) || throw(ArgumentError("initial atmosphere file not found: $file"))
        h5open(file,"r") do f
            for name in (x_dataset,y_dataset)
                haskey(f,name) || throw(ArgumentError("HDF5 dataset not found: $name"))
            end
            (_uniform_axis_spacing(vec(read(f[x_dataset])),x_dataset),_uniform_axis_spacing(vec(read(f[y_dataset])),y_dataset))
        end
    end
    spacing=map(zip(("dx_m","dy_m"),configured,explicit)) do (name,c,e)
        c===nothing && e===nothing && throw(ArgumentError("grid.$name is required without explicit x_m/y_m datasets"))
        c!==nothing && e!==nothing && abs(c-e)>1e-9*abs(e) && throw(ArgumentError(
            "grid.$name=$c differs from the spacing $e of the explicit coordinate dataset"))
        something(c,e)
    end
    Float64(spacing[1]),Float64(spacing[2])
end

function load_config(path::AbstractString)
    cfg = TOML.parsefile(path)
    inputs = _required(cfg,"inputs"); outputs_raw = _required(cfg,"outputs")
    atmos = _required(cfg,"atmosphere"); grid = get(cfg,"grid",Dict{String,Any}())
    requested_mode=Symbol(get(cfg,"mode","inversion"))
    requested_mode in (:forward,:inversion) || throw(ArgumentError("mode must be forward or inversion"))
    solver_raw=get(cfg,"solver",Dict{String,Any}())
    forward_only=requested_mode===:forward || get(solver_raw,"max_iterations",nothing)==0
    mode=forward_only ? :forward : :inversion
    solver=forward_only ? LBFGSSolverOptions(max_iterations=0) : _solver(solver_raw)
    obs = get(cfg,"observation",Dict{String,Any}())
    physics = _required(cfg,"physics")
    datasets = _required(atmos,"datasets")
    storage_order=Symbol(lowercase(String(get(atmos,"storage_order","julia"))))
    storage_order in (:julia,:python) || throw(ArgumentError(
        "atmosphere.storage_order must be julia or python"))
    vertical_coordinate,vertical_dataset,x_dataset,y_dataset=_vertical_input(atmos,datasets)
    dx_m,dy_m = _horizontal_spacing(grid,String(_required(inputs,"initial_atmosphere_file")),x_dataset,y_dataset)
    regions = _regions(cfg)
    wavelength_m = reduce(vcat,wavelengths.(regions))
    !isempty(wavelength_m) && all(>(0),wavelength_m) || throw(ArgumentError("wavelength grid must be positive and non-empty"))
    length(unique(wavelength_m)) == length(wavelength_m) || throw(ArgumentError("spectral regions must not contain duplicate wavelengths"))

    magnetic = all(haskey(datasets,k) for k in ("Bx","By","Bz")) ?
        (String(datasets["Bx"]),String(datasets["By"]),String(datasets["Bz"])) : nothing
    any(haskey(datasets,k) for k in ("Bx","By","Bz")) && magnetic === nothing && throw(ArgumentError("provide all Bx, By and Bz datasets or none"))
    for key in ("pressure_top_pa","pressure_top_dataset")
        haskey(atmos,key) && throw(ArgumentError(
            "atmosphere.$key is no longer accepted: the pressure boundary is set by HSEMHS; configure [equilibrium.column_z] anchor/P_ref_pa (native z) or nothing (native log tau)"))
    end
    haskey(cfg,"equilibrium") || throw(ArgumentError("missing required configuration section [equilibrium] (backend = \"HSEMHS\", model, eos_table_file, ...)"))
    equilibrium = _equilibrium_config(cfg["equilibrium"],vertical_coordinate,dirname(abspath(path)))
    atmosphere = AtmosphereInputConfig{Float64}(String(_required(inputs,"initial_atmosphere_file")),storage_order,
        vertical_coordinate,vertical_dataset,x_dataset,y_dataset,
        String(_required(datasets,"temperature")),String(_required(datasets,"vx")),String(_required(datasets,"vy")),
        String(_required(datasets,"vz")),magnetic,
        haskey(datasets,"vturb") ? String(datasets["vturb"]) : nothing)

    observed=nothing; weights=nothing
    if !forward_only
        data = _required(obs,"datasets")
        observed = ObservedDataConfig(String(_required(inputs,"observation_file")),String(_required(data,"intensity")),String(_required(data,"sigma")))
        weights = WeightInputConfig(String(_required(data,"wavelength_weights")),String(_required(data,"spatial_weights")))
    end
    outputs = OutputConfig(String(_required(outputs_raw,"synthesis_file")),String(_required(outputs_raw,"atmosphere_file")))
    outputs.synthesis_file == outputs.atmosphere_file && throw(ArgumentError("synthesis_file and atmosphere_file must differ"))

    stokes = StokesSet(Tuple(Symbol.(get(obs,"stokes",["I"]))))
    stokes.components == (:I,) || throw(ArgumentError("Release 1 supports Stokes I only"))
    redistribution = Symbol(get(physics,"redistribution","non_prd"))
    redistribution == :non_prd || throw(ArgumentError("Release 1 supports redistribution=non_prd only"))
    psf = get(obs,"gaussian_psf",Dict{String,Any}())
    observation_model = GaussianPSFObservation(Float64(get(psf,"spectral_fwhm_nm",0.0))*1e-9,
        Float64(get(psf,"spatial_fwhm_x_m",0.0)),Float64(get(psf,"spatial_fwhm_y_m",0.0)),dx_m,dy_m)
    regularization = forward_only ? _regularization(Dict{String,Any}()) : _regularization(get(cfg,"regularization",Dict{String,Any}()))
    if vertical_coordinate===:z_m && regularization.vertical.length_scale_m===nothing &&
            any(t->t in (1,4),regularization.vertical.types)
        throw(ArgumentError("vertical regularization types 1/4 on a z grid require regularization.vertical.length_scale_m [m]"))
    end
    controls=forward_only ? ControlMapConfig{Float64}[] : _controls(_required(cfg,"inversion"),vertical_coordinate)
    parallel_raw = get(cfg,"parallel",Dict{String,Any}())
    decomposition = Symbol(lowercase(String(get(parallel_raw,"decomposition","cartesian_2d"))))
    decomposition in (:cartesian_2d,:x_slabs) || throw(ArgumentError("parallel.decomposition must be cartesian_2d or x_slabs"))
    min_columns_per_rank = Int(get(parallel_raw,"min_columns_per_rank",2))
    min_columns_per_rank >= 1 || throw(ArgumentError("parallel.min_columns_per_rank must be positive"))
    configured_threads = Int(get(parallel_raw,"threads_per_rank",Threads.nthreads()))
    runtime_threads = get(ENV,"FFNO_THREADS_PER_RANK","")
    threads_per_rank = if isempty(runtime_threads)
        configured_threads
    else
        try
            parse(Int,runtime_threads)
        catch
            throw(ArgumentError("FFNO_THREADS_PER_RANK must be a positive integer, got $(repr(runtime_threads))"))
        end
    end
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
        min_columns_per_rank=min_columns_per_rank,
        threads_per_rank=threads_per_rank,gpu_launcher_rank=gpu_launcher_rank,
        gpu_connect_timeout_seconds=gpu_connect_timeout_seconds,
        gpu_status_timeout_seconds=gpu_status_timeout_seconds,
        gpu_diagnostic_interval_seconds=gpu_diagnostic_interval_seconds,
        gpu_diagnostics_directory=gpu_diagnostics_directory)
    time_index=Int(get(inputs,"time_index",1)); time_index>0 || throw(ArgumentError(
        "inputs.time_index must be positive"))
    parallel.enabled && parallel.decomposition!==:x_slabs && throw(ArgumentError(
        "the HSEMHS equilibrium solves on x slabs: set parallel.decomposition = \"x_slabs\" (got $(parallel.decomposition))"))
    RunConfig(atmosphere,mode,observed,weights,outputs,SynthesisGridConfig(wavelength_m,dx_m,dy_m),regions,
        observation_model,stokes,redistribution,regularization,parallel,controls,solver,time_index,equilibrium)
end

function dry_run_summary(config::RunConfig)
    nlambda = length(config.synthesis.wavelength_m)
    vertical_vars = Symbol[VERTICAL_PARAMETER_ORDER[i] for i in 1:7 if config.regularization.vertical.types[i] != 0]
    regvars = sort!(collect(union(vertical_vars,keys(config.regularization.horizontal))))
    nsources=sum(length(r.sources) for r in config.regions)
    controlvars=join(getfield.(config.controls,:variable),',')
    solver_name=config.solver isa LBFGSSolverOptions ? "bounded_lbfgs" : "prototype_pattern_search"
    "mode=$(config.mode) observation_input=$(config.observed===nothing ? "none" : config.observed.file) atmosphere_input=$(config.atmosphere.file) atmosphere_storage_order=$(config.atmosphere.storage_order) synthesis_output=$(config.outputs.synthesis_file) atmosphere_output=$(config.outputs.atmosphere_file) time_index=$(config.time_index) vertical_coordinate=$(config.atmosphere.vertical_coordinate) vertical_dataset=$(config.atmosphere.vertical_dataset) nodes_unit=$(vertical_unit(config.atmosphere.vertical_coordinate)) dx_m=$(config.synthesis.dx_m) dy_m=$(config.synthesis.dy_m) spectral_regions=$(length(config.regions)) spectral_sources=$nsources synthesis_wavelengths=$nlambda full_grid_psf=true zero_weight_exclusion=true stokes=$(join(config.stokes.components,',')) redistribution=$(config.redistribution) equilibrium=HSEMHS:$(hsemhs_mode_name(config.equilibrium.mode)) eos_table=$(config.equilibrium.eos_table_file) controls=$controlvars solver=$solver_name max_iterations=$(config.solver.max_iterations) checkpoint=$(config.solver.checkpoint_path) regularized=$(join(regvars,',')) mpi=$(config.parallel.enabled) decomposition=$(config.parallel.decomposition) threads_per_rank=$(config.parallel.threads_per_rank) gpu_launcher_rank=$(config.parallel.gpu_launcher_rank) gpu_connect_timeout_seconds=$(config.parallel.gpu_connect_timeout_seconds) gpu_status_timeout_seconds=$(config.parallel.gpu_status_timeout_seconds) gpu_diagnostic_interval_seconds=$(config.parallel.gpu_diagnostic_interval_seconds)"
end

function checkpoint!(path::AbstractString,state;manifest::CapabilityManifest=CapabilityManifest())
    destination=abspath(path); mkpath(dirname(destination))
    temporary,io=mktemp(dirname(destination))
    try
        serialize(io,(manifest=manifest,state=state)); flush(io); close(io)
        mv(temporary,destination;force=true)
    finally
        isopen(io) && close(io)
        isfile(temporary) && rm(temporary)
    end
    path
end

function restore_checkpoint(path::AbstractString;expected::Union{Nothing,CapabilityManifest}=nothing)
    payload = open(deserialize,path)
    expected !== nothing && payload.manifest != expected && throw(ArgumentError("checkpoint capability manifest differs from requested physics"))
    payload
end
