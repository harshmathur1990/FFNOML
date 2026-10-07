using FFNOInversion
using HDF5
using Libdl

function _required_file(label::AbstractString,path::AbstractString)
    isfile(path) || throw(ArgumentError("missing $label: $path"))
    open(path,"r") do _ end
    path
end

_run_path(run_dir,path) = isabspath(path) ? normpath(path) : normpath(joinpath(run_dir,path))

function _dataset_shape(file,name::AbstractString,label::AbstractString)
    haskey(file,name) || throw(ArgumentError("$label dataset '$name' is missing"))
    Tuple(size(file[name]))
end

function _selected_shape(shape::Tuple,time_index::Int,plain_ndims::Int,label::AbstractString)
    if length(shape)==plain_ndims
        return shape
    elseif length(shape)==plain_ndims+1
        1 <= time_index <= shape[1] || throw(ArgumentError(
            "$label time_index=$time_index is outside 1:$(shape[1])"))
        return Base.tail(shape)
    end
    throw(DimensionMismatch("$label must have $plain_ndims dimensions, optionally preceded by time; got $shape"))
end

function _validate_hdf5_inputs(config,run_dir)
    atmosphere_path=_required_file("initial atmosphere",_run_path(run_dir,config.atmosphere.file))
    atmosphere_shape=h5open(atmosphere_path,"r") do file
        logtau_shape=_selected_shape(_dataset_shape(file,config.atmosphere.logtau500_dataset,"logtau500"),
            config.time_index,1,"logtau500")
        fields=(
            temperature=config.atmosphere.temperature_dataset,
            vx=config.atmosphere.vx_dataset,
            vy=config.atmosphere.vy_dataset,
            vz=config.atmosphere.vz_dataset,
        )
        selected=Dict(name=>_selected_shape(_dataset_shape(file,dataset,string(name)),
            config.time_index,3,string(name)) for (name,dataset) in pairs(fields))
        all(==(selected[:temperature]),values(selected)) || throw(DimensionMismatch(
            "atmosphere temperature/vx/vy/vz shapes differ: $selected"))
        logtau_shape[1]==selected[:temperature][1] || throw(DimensionMismatch(
            "logtau500 depth $(logtau_shape[1]) differs from atmosphere depth $(selected[:temperature][1])"))
        if config.atmosphere.vturb_dataset!==nothing
            vturb_shape=_selected_shape(_dataset_shape(file,config.atmosphere.vturb_dataset,"vturb"),
                config.time_index,3,"vturb")
            vturb_shape==selected[:temperature] || throw(DimensionMismatch("vturb shape differs from temperature"))
        end
        if config.atmosphere.magnetic_datasets!==nothing
            for name in config.atmosphere.magnetic_datasets
                magnetic_shape=_selected_shape(_dataset_shape(file,name,"magnetic field"),
                    config.time_index,3,"magnetic field")
                magnetic_shape==selected[:temperature] || throw(DimensionMismatch(
                    "magnetic field '$name' shape differs from temperature"))
            end
        end
        if config.atmosphere.pressure_top isa String
            pressure_shape=_dataset_shape(file,config.atmosphere.pressure_top,"pressure_top")
            if !isempty(pressure_shape)
                selected_pressure=_selected_shape(pressure_shape,config.time_index,2,"pressure_top")
                selected_pressure==selected[:temperature][2:3] || throw(DimensionMismatch(
                    "pressure_top shape $selected_pressure does not match atmosphere y/x shape $(selected[:temperature][2:3])"))
            end
        end
        selected[:temperature]
    end

    config.mode===:forward && return atmosphere_path,nothing,atmosphere_shape

    observation_path=_required_file("observations",_run_path(run_dir,config.observed.file))
    h5open(observation_path,"r") do file
        intensity_shape=_selected_shape(_dataset_shape(file,config.observed.intensity_dataset,"intensity"),
            config.time_index,4,"intensity")
        sigma_shape=_selected_shape(_dataset_shape(file,config.observed.sigma_dataset,"sigma"),
            config.time_index,4,"sigma")
        intensity_shape==sigma_shape || throw(DimensionMismatch("observation intensity and sigma shapes differ"))
        expected=(length(config.stokes.components),length(config.synthesis.wavelength_m),
            atmosphere_shape[2],atmosphere_shape[3])
        intensity_shape==expected || throw(DimensionMismatch(
            "observation shape $intensity_shape does not match expected $expected (Stokes,wavelength,y,x)"))
        wavelength_shape=_selected_shape(_dataset_shape(file,config.weights.wavelength_dataset,"wavelength weights"),
            config.time_index,2,"wavelength weights")
        wavelength_shape==(expected[2],expected[1]) || throw(DimensionMismatch(
            "wavelength weight shape $wavelength_shape does not match expected $((expected[2],expected[1]))"))
        spatial_shape=_selected_shape(_dataset_shape(file,config.weights.spatial_dataset,"spatial weights"),
            config.time_index,2,"spatial weights")
        spatial_shape==expected[3:4] || throw(DimensionMismatch(
            "spatial weight shape $spatial_shape does not match expected $(expected[3:4])"))
    end
    atmosphere_path,observation_path,atmosphere_shape
end

function _batch_resources(batch_script)
    values=Dict{String,String}()
    for line in eachline(batch_script)
        matched=match(r"^#SBATCH --([^=]+)=(.+)$",line)
        isnothing(matched) || (values[matched.captures[1]]=matched.captures[2])
    end
    for key in ("nodes","ntasks","ntasks-per-node","cpus-per-task","gpus-per-node")
        haskey(values,key) || throw(ArgumentError("batch script is missing #SBATCH --$key"))
    end
    nodes=parse(Int,values["nodes"])
    tasks=parse(Int,values["ntasks"])
    tasks_per_node=parse(Int,values["ntasks-per-node"])
    nodes*tasks_per_node==tasks || throw(ArgumentError(
        "invalid batch topology: nodes=$nodes × ntasks-per-node=$tasks_per_node != ntasks=$tasks"))
    values
end

function _source_description(source,run_dir)
    if source.mode===:ffno
        return "FFNO $(source.species)/$(source.line)"
    end
    line_list=_required_file("Kurucz line list",_run_path(run_dir,source.linelist_file))
    "Kurucz LTE $(line_list)"
end

function _factory_assets(run_dir,declared)
    assets=Pair{String,String}[]
    if declared!==nothing
        declared isa AbstractDict || throw(ArgumentError("FFNO_INVERSION_ASSETS must be a dictionary"))
        append!(assets,(string(label)=>_run_path(run_dir,string(path)) for (label,path) in declared))
    else
        production_assets=(
            "H checkpoint"=>joinpath(run_dir,"training_FFNO3D_zscale_expand_lognlte","3D_sim_train_H.pt"),
            "Ca checkpoint"=>joinpath(run_dir,"training_FFNO3D_zscale_expand_lognlte","3D_sim_train_CA.pt"),
            "H atom"=>joinpath(run_dir,"inputs","atoms","atom.h6_tiago2.yaml"),
            "Ca atom"=>joinpath(run_dir,"inputs","atoms","atom.ca2.yaml"),
            "EOS library"=>joinpath(run_dir,"inputs","wittmann","libwitt_ffno.$(Libdl.dlext)"),
            "partition functions"=>joinpath(run_dir,"inputs","pf_Kurucz.input"),
        )
        all(pair->isfile(last(pair)),production_assets) && append!(assets,production_assets)
    end
    for (label,path) in assets
        _required_file("model asset '$label'",path)
    end
    sort!(unique(assets);by=first)
end

function _generated_factory_assets(run_dir,declared)
    declared===nothing && return Pair{String,String}[]
    declared isa AbstractDict || throw(ArgumentError(
        "FFNO generated model assets must be a dictionary"))
    sort!(unique(string(label)=>_run_path(run_dir,string(path))
        for (label,path) in declared);by=first)
end

function run_submission_preflight(config_path::AbstractString,factory_path::AbstractString,
        run_dir::AbstractString,batch_script::AbstractString;io::IO=stdout)
    run_dir=abspath(run_dir)
    config_path=_required_file("configuration",abspath(config_path))
    factory_path=_required_file("model factory",abspath(factory_path))
    batch_script=_required_file("batch script",abspath(batch_script))
    resources=_batch_resources(batch_script)

    config=cd(run_dir) do
        load_config(config_path)
    end
    atmosphere_path,observation_path,atmosphere_shape=_validate_hdf5_inputs(config,run_dir)

    for region in config.regions
        region.psf_file===nothing || _required_file("spectral PSF",_run_path(run_dir,region.psf_file))
        for source in region.sources
            source.mode===:kurucz_lte && _required_file("Kurucz line list",_run_path(run_dir,source.linelist_file))
        end
    end

    included_factory=withenv("FFNOML_RUN_DIR"=>run_dir) do
        cd(run_dir) do
            Base.include(Main,factory_path)
        end
    end
    factory=if included_factory isa NamedTuple && hasproperty(included_factory,:factory)
        included_factory.factory
    else
        included_factory
    end
    factory isa InversionModelFactory || throw(ArgumentError("FFNO_INVERSION_FACTORY has the wrong type"))
    declared_assets=included_factory isa NamedTuple && hasproperty(included_factory,:assets) ?
        included_factory.assets : nothing
    factory_assets=_factory_assets(run_dir,declared_assets)
    declared_generated_assets=included_factory isa NamedTuple &&
        hasproperty(included_factory,:generated_assets) ? included_factory.generated_assets : nothing
    generated_factory_assets=_generated_factory_assets(run_dir,declared_generated_assets)
    if factory.population_levels isa AbstractDict
        requested=unique(source.species for region in config.regions for source in region.sources
            if source.mode===:ffno)
        missing=setdiff(collect(requested),collect(keys(factory.population_levels)))
        isempty(missing) || throw(ArgumentError("model factory has no population levels for $(join(missing,','))"))
    end

    for output in (config.outputs.synthesis_file,config.outputs.atmosphere_file)
        mkpath(dirname(_run_path(run_dir,output)))
    end

    println(io,"FFNO submission preflight")
    println(io,"  Run directory:  ",run_dir)
    println(io,"  Configuration:  ",config_path)
    println(io,"  Model factory:  ",factory_path," (load OK)")
    if !isempty(factory_assets)
        println(io,"  Model assets:")
        for (label,path) in factory_assets
            println(io,"    ",label,": ",path)
        end
    end
    if !isempty(generated_factory_assets)
        println(io,"  Generated inside allocated job:")
        for (label,path) in generated_factory_assets
            println(io,"    ",label,": ",path)
        end
    end
    println(io,"  Mode:           ",uppercase(string(config.mode)))
    println(io,"  Atmosphere:     ",atmosphere_path," shape=",atmosphere_shape," (z,y,x)")
    println(io,"  Observations:   ",isnothing(observation_path) ? "not required for forward mode" : observation_path)
    println(io,"  Spectral regions:")
    for (index,region) in enumerate(config.regions)
        wavelength=wavelengths(region).*1e10
        sources=join((_source_description(source,run_dir) for source in region.sources),"; ")
        println(io,"    [$index] ",round(first(wavelength),digits=6),"–",round(last(wavelength),digits=6),
            " Å; count=",region.count,"; step=",round(region.step_m*1e10,digits=6)," Å; ",sources)
    end
    println(io,"  Outputs:        ",_run_path(run_dir,config.outputs.synthesis_file))
    println(io,"                  ",_run_path(run_dir,config.outputs.atmosphere_file))
    println(io,"  SLURM defaults: nodes=",resources["nodes"]," ranks=",resources["ntasks"],
        " ranks/node=",resources["ntasks-per-node"]," CPUs/rank=",resources["cpus-per-task"],
        " GPUs/node=",resources["gpus-per-node"]," walltime=",get(resources,"time","unspecified"))
    println(io,"Sanity check OK")
    config
end

function main(args=ARGS)
    length(args)==4 || error("usage: preflight_olivia_inversion.jl CONFIG.toml MODEL_FACTORY.jl RUN_DIR BATCH_SCRIPT")
    try
        run_submission_preflight(args...)
    catch exception
        println(stderr,"Sanity check FAILED: ",sprint(showerror,exception))
        return 2
    end
    0
end

if abspath(PROGRAM_FILE)==@__FILE__
    exit(main())
end
