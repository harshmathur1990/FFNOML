"""
Given populations, forward only (`[population.given]`): a species' level populations read from a file instead of predicted by the FFNO, to test
the synthesis and the rest of the code on populations known to be right, such as MULTI3D's. The other species still come from the FFNO.

    [population.given]
    format = "multi3d"         # or "hdf5"
    orientation = "auto"       # or one of GIVEN_ORIENTATIONS
    tau500 = "…/tau500"        # optional: MULTI3D's tau500 cube; by default the one beside the first file, or in its directory's parent
    min_correlation = 0.5      # auto: the least correlation the chosen orientation may have
    [population.given.files]
    H = "…/H/out_pop"
    CA = "…/CA/out_pop"

`"multi3d"`: MULTI3D's `out_pop`, Float32 in Fortran order (x, y, z, level), top first, the NLTE levels [cm⁻³] first; what follows (the LTE
populations, ...) is not read. The grid is `multi3d.input`'s beside the file when there is one, otherwise the atmosphere's. `"hdf5"`: the
`populations/<species>` dataset of a forward run's atmosphere output, (level, z, y, x) [m⁻³], with its `logtau500` dataset, if any, for the check.

The populations must be on the atmosphere's own grid: the same levels (a MULTI3D file needs a height axis, and a `mesh` beside it or in its
parent must have the atmosphere's heights) and the same columns, perhaps rotated or mirrored. The preflight checks the size of each file, then
finds the orientation: at the level where the median tau500 is closest to 1 it correlates log10 tau500 with the atmosphere's temperature under
every rotation and mirror the grid allows (tau500 at a fixed height grows with the temperature, through H⁻), and takes the best; a falling
tau500 means the file is bottom first, and its levels are reversed. It prints what it found, which the outputs' provenance keeps.

Given populations skip the population mesh and `[physics] ffno_population_limit` and `ffno_no_inversion`: they are used as they are, after
a check that they are finite and not negative.
"""
const GIVEN_ORIENTATIONS=(:identity,:flip_x,:flip_y,:rotate_180,:transpose,:transpose_flip_x,:transpose_flip_y,:transpose_rotate_180)
const _GIVEN_KEYS=("format","orientation","tau500","min_correlation","files")
const _GIVEN_CORRELATION_MARGIN=0.1

struct GivenPopulationsConfig
    format::Symbol
    files::Dict{Symbol,String}
    tau500::Union{Nothing,String}
    orientation::Symbol
    min_correlation::Float64
end

_given_path(path,base)=isabspath(path) ? String(path) : normpath(joinpath(base,path))

"""`[population.given]`, or nothing; `base` is the configuration's directory, against which relative paths are taken."""
function _given_populations(raw,forward_only,base)
    raw===nothing && return nothing
    forward_only || throw(ArgumentError("[population.given] is for forward synthesis only: an inversion needs the FFNO's populations and their derivatives"))
    unknown=setdiff(collect(keys(raw)),collect(_GIVEN_KEYS))
    isempty(unknown) || throw(ArgumentError("unknown key(s) in [population.given]: $(join(sort(unknown),", "))"))
    format=Symbol(lowercase(String(get(raw,"format","multi3d"))))
    format in (:multi3d,:hdf5) || throw(ArgumentError("population.given.format must be \"multi3d\" or \"hdf5\", got \"$format\""))
    files_raw=get(raw,"files",nothing)
    files_raw isa AbstractDict && !isempty(files_raw) || throw(ArgumentError("[population.given.files] must name at least one species' file, e.g. H = \"…/H/out_pop\""))
    files=Dict{Symbol,String}(Symbol(uppercase(String(k)))=>_given_path(String(v),base) for (k,v) in files_raw)
    orientation=Symbol(lowercase(String(get(raw,"orientation","auto"))))
    orientation===:auto || orientation in GIVEN_ORIENTATIONS || throw(ArgumentError(
        "population.given.orientation must be \"auto\" or one of $(join(string.(GIVEN_ORIENTATIONS),", ")), got \"$orientation\""))
    tau500=haskey(raw,"tau500") ? _given_path(String(raw["tau500"]),base) : nothing
    min_correlation=Float64(get(raw,"min_correlation",0.5))
    -1<=min_correlation<=1 || throw(ArgumentError("population.given.min_correlation must lie in [-1, 1]"))
    GivenPopulationsConfig(format,files,tau500,orientation,min_correlation)
end

_given_transposes(o)=o in (:transpose,:transpose_flip_x,:transpose_flip_y,:transpose_rotate_180)

"""The (x, y) of the file that lands on column (i, j) of the atmosphere (nx × ny) under orientation `o`."""
function _given_source(o,i,j,nx,ny)
    o===:identity && return (i,j)
    o===:flip_x && return (nx+1-i,j)
    o===:flip_y && return (i,ny+1-j)
    o===:rotate_180 && return (nx+1-i,ny+1-j)
    o===:transpose && return (j,i)
    o===:transpose_flip_x && return (j,nx+1-i)
    o===:transpose_flip_y && return (ny+1-j,i)
    o===:transpose_rotate_180 && return (ny+1-j,nx+1-i)
    throw(ArgumentError("unknown orientation $o"))
end

"""The orientations a file of `file_shape` columns can have on an nx × ny atmosphere."""
function _given_candidates(file_shape,nx,ny)
    filter(o->(_given_transposes(o) ? (ny,nx) : (nx,ny))==Tuple(file_shape),collect(GIVEN_ORIENTATIONS))
end

function _given_oriented_map(map2,o,nx,ny)
    out=Matrix{Float64}(undef,nx,ny)
    for j in 1:ny, i in 1:nx
        a,b=_given_source(o,i,j,nx,ny); out[i,j]=map2[a,b]
    end
    out
end

function _given_correlation(a,b)
    ma=sum(a)/length(a); mb=sum(b)/length(b); sab=0.0; saa=0.0; sbb=0.0
    for k in eachindex(a,b)
        da=a[k]-ma; db=b[k]-mb; sab+=da*db; saa+=da*da; sbb+=db*db
    end
    saa>0 && sbb>0 ? sab/sqrt(saa*sbb) : 0.0
end

function _given_median(values)
    v=sort!(filter(isfinite,vec(values))); isempty(v) && return NaN
    n=length(v); isodd(n) ? v[(n+1)÷2] : (v[n÷2]+v[n÷2+1])/2
end

"""nx, ny, nz of the `multi3d.input` in `directory`, or nothing."""
function _multi3d_input_grid(directory)
    path=joinpath(directory,"multi3d.input"); isfile(path) || return nothing
    found=Dict{String,Int}()
    for line in eachline(path)
        m=match(r"^\s*(nx|ny|nz)\s*=\s*(\d+)",line); m===nothing || (found[m.captures[1]]=parse(Int,m.captures[2]))
    end
    all(haskey(found,k) for k in ("nx","ny","nz")) ? (found["nx"],found["ny"],found["nz"]) : nothing
end

"""The `mesh` beside `file` or in its directory's parent, or nothing."""
function _given_mesh(file)
    for directory in (dirname(file),dirname(dirname(file)))
        path=joinpath(directory,"mesh"); isfile(path) && return path
    end
    nothing
end

"""What a file holds before anything is read: the columns (in the file's own order), levels per cell and how they are read."""
function _given_layout(config::GivenPopulationsConfig,species,file,nx,ny,nz,levels,notes)
    isfile(file) || throw(ArgumentError("population.given.files.$species not found: $file"))
    if config.format===:multi3d
        grid=_multi3d_input_grid(dirname(file))
        fx,fy,fz=grid===nothing ? (nx,ny,nz) : grid
        grid===nothing ? push!(notes,"$species: no multi3d.input beside $file; taking the atmosphere's grid $nx x $ny x $nz") :
            push!(notes,"$species: multi3d.input gives the grid $fx x $fy x $fz")
        fz==nz || throw(DimensionMismatch("$file has $fz levels (multi3d.input); the atmosphere has $nz"))
        Set((fx,fy))==Set((nx,ny)) || throw(DimensionMismatch(
            "$file is on a $fx x $fy grid (multi3d.input); the atmosphere is $nx x $ny: the populations must be on the atmosphere's own columns"))
        cell=fx*fy*fz*sizeof(Float32); bytes=filesize(file)
        bytes%cell==0 || throw(ArgumentError("$file has $bytes bytes, not a whole number of Float32 $fx x $fy x $fz cubes ($cell bytes each)"))
        stored=bytes÷cell
        stored>=levels || throw(ArgumentError("$file holds $stored values per cell; $species has $levels levels"))
        push!(notes,"$species: $file holds $stored Float32 values per cell; the first $levels, the NLTE levels [cm^-3], are read")
        return (file=file,shape=(fx,fy),levels=levels)
    end
    h5open(file,"r") do f
        name="populations/$species"
        haskey(f,name) || throw(ArgumentError("$file has no dataset $name"))
        dims=size(f[name])
        length(dims)==4 || throw(DimensionMismatch("$file:$name needs (level, z, y, x), got $(length(dims)) dimensions"))
        dims[1]==levels || throw(DimensionMismatch("$file:$name has $(dims[1]) levels; $species has $levels"))
        dims[2]==nz || throw(DimensionMismatch("$file:$name has $(dims[2]) heights; the atmosphere has $nz"))
        Set((dims[4],dims[3]))==Set((nx,ny)) || throw(DimensionMismatch("$file:$name is on a $(dims[4]) x $(dims[3]) grid; the atmosphere is $nx x $ny"))
        push!(notes,"$species: $file:$name, (level, z, y, x) = $dims [m^-3]")
        (file=file,shape=(dims[4],dims[3]),levels=levels)
    end
end

"""The tau500 cube to check with, `(path, kind)`, or nothing: the configured file, else MULTI3D's beside the first file or in its parent,
or an HDF5 file's `logtau500`."""
function _given_tau500_source(config::GivenPopulationsConfig)
    config.tau500===nothing || return (isfile(config.tau500) ? (config.tau500,:multi3d) :
        throw(ArgumentError("population.given.tau500 not found: $(config.tau500)")))
    first_file=config.files[sort!(collect(keys(config.files));by=string)[1]]
    if config.format===:multi3d
        for directory in (dirname(first_file),dirname(dirname(first_file)))
            path=joinpath(directory,"tau500"); isfile(path) && return (path,:multi3d)
        end
        return nothing
    end
    h5open(f->haskey(f,"logtau500"),first_file,"r") ? (first_file,:hdf5) : nothing
end

"""log10 tau500 of each level, (x, y) in the file's order, through `visit(k, map)`."""
function _given_each_tau_level(visit,path,kind,shape,nz)
    fx,fy=shape
    if kind===:hdf5
        logtau=h5open(f->read(f["logtau500"]),path,"r")        # (time, z, y, x)
        size(logtau)[2:4]==(nz,fy,fx) || throw(DimensionMismatch("$path:logtau500 is $(size(logtau)); expected (time, $nz, $fy, $fx)"))
        for k in 1:nz; visit(k,permutedims(Float64.(logtau[1,k,:,:]),(2,1))); end
        return nothing
    end
    cells=fx*fy*nz; bytes=filesize(path)
    element=bytes==4cells ? Float32 : bytes==8cells ? Float64 : throw(ArgumentError(
        "$path has $bytes bytes; a $fx x $fy x $nz tau500 cube needs $(4cells) (Float32) or $(8cells) (Float64)"))
    buffer=Array{element}(undef,fx,fy)
    open(path,"r") do io
        for k in 1:nz
            read!(io,buffer); visit(k,log10.(max.(Float64.(buffer),floatmin(Float64))))
        end
    end
    nothing
end

"""The orientation and level order of the files (all share one grid), from tau500 against the atmosphere's temperature."""
function _given_orientation(config::GivenPopulationsConfig,atmosphere,shape,notes)
    nz,nx,ny=size(atmosphere.temperature)
    candidates=_given_candidates(shape,nx,ny)
    source=_given_tau500_source(config)
    if source===nothing
        config.orientation===:auto && throw(ArgumentError(
            "population.given.orientation = \"auto\" needs tau500 to check against: set population.given.tau500, or give the orientation"))
        config.orientation in candidates || throw(DimensionMismatch("orientation $(config.orientation) does not fit a $(shape[1]) x $(shape[2]) file on $nx x $ny"))
        push!(notes,"no tau500: orientation $(config.orientation) as configured, levels top first, unchecked")
        return (orientation=config.orientation,reversed=false,level=0,correlations=Dict{String,Float64}(),tau500="")
    end
    path,kind=source
    medians=fill(NaN,nz); maps=Dict{Int,Matrix{Float64}}()
    _given_each_tau_level(path,kind,shape,nz) do k,m
        medians[k]=_given_median(m)
    end
    reversed=medians[1]>medians[end]
    reversed && push!(notes,"tau500 falls with the level index: the files are bottom first, and their levels are reversed")
    level_file=argmin(k->isfinite(medians[k]) ? abs(medians[k]) : Inf,1:nz)
    level=reversed ? nz+1-level_file : level_file
    map_file=Matrix{Float64}(undef,shape...)
    _given_each_tau_level(path,kind,shape,nz) do k,m
        k==level_file && (map_file.=m)
    end
    temperature=Float64.(atmosphere.temperature[level,:,:])
    correlations=Dict{String,Float64}(string(o)=>_given_correlation(temperature,_given_oriented_map(map_file,o,nx,ny)) for o in candidates)
    ranked=sort(collect(correlations);by=last,rev=true)
    z=atmosphere.z===nothing ? "" : " (z = $(round(atmosphere.z[level,1,1]*1e-3;digits=1)) km)"
    push!(notes,"orientation check: $path, level $level$z, median log10 tau500 = $(round(medians[level_file];digits=2)); "*
        "correlation of log10 tau500 with the temperature: "*join(("$k $(round(v;digits=3))" for (k,v) in ranked),", "))
    best=Symbol(first(ranked[1])); r=last(ranked[1]); second=length(ranked)>1 ? last(ranked[2]) : -1.0
    orientation=if config.orientation===:auto
        r>=config.min_correlation && r-second>=_GIVEN_CORRELATION_MARGIN || throw(ArgumentError(
            "population.given: no orientation stands out (best $best at $(round(r;digits=3)), next $(round(second;digits=3)); "*
            "needs at least $(config.min_correlation) and a lead of $_GIVEN_CORRELATION_MARGIN): are these the populations of this atmosphere? "*
            "Give population.given.orientation to use them anyway"))
        push!(notes,"orientation $best chosen")
        best
    else
        config.orientation in candidates || throw(DimensionMismatch("orientation $(config.orientation) does not fit a $(shape[1]) x $(shape[2]) file on $nx x $ny"))
        config.orientation===best || @warn "population.given.orientation = $(config.orientation), but $best correlates best" correlations
        push!(notes,"orientation $(config.orientation) as configured$(config.orientation===best ? ", also the best" : "; $best correlates best")")
        config.orientation
    end
    (orientation=orientation,reversed=reversed,level=level,correlations=correlations,tau500=path)
end

"""MULTI3D populations are on heights: the atmosphere must be too, and a `mesh` near the file must have its heights."""
function _given_heights(config::GivenPopulationsConfig,atmosphere,notes)
    config.format===:multi3d || return nothing
    vertical_coordinate(atmosphere.grid)===:z_m || throw(ArgumentError(
        "population.given.format = \"multi3d\": MULTI3D populations are on heights, the atmosphere is on $(vertical_coordinate(atmosphere.grid))"))
    for (mesh,file) in Dict(_given_mesh(file)=>file for file in values(config.files))
        mesh===nothing && continue
        z=_read_multi3d_mesh(mesh)[3]; native=native_depth(atmosphere.grid)
        length(z)==length(native) && maximum(abs.(z.-native))<=1e-4*maximum(abs.(native)) || throw(DimensionMismatch(
            "$mesh has other heights than the atmosphere: the populations of $file are not on its levels"))
        push!(notes,"$mesh has the atmosphere's heights")
    end
    nothing
end

"""Check every file and find the orientation, without reading the populations: `(layouts, orientation, notes)`."""
function preflight_given_populations(config::GivenPopulationsConfig,atmosphere,levels::AbstractDict)
    nz,nx,ny=size(atmosphere.temperature); notes=String[]
    push!(notes,"atmosphere: $nx x $ny columns, $nz levels")
    layouts=Dict{Symbol,Any}()
    for species in sort!(collect(keys(config.files));by=string)
        haskey(levels,species) || throw(ArgumentError(
            "population.given.files.$species: the model has no species $species (it has $(join(sort!(string.(collect(keys(levels)))),", ")))"))
        layouts[species]=_given_layout(config,species,config.files[species],nx,ny,nz,Int(levels[species]),notes)
    end
    shapes=unique(l.shape for l in values(layouts))
    length(shapes)==1 || throw(DimensionMismatch("population.given: the files are on different grids, $(join(shapes,", "))"))
    _given_heights(config,atmosphere,notes)
    orientation=_given_orientation(config,atmosphere,only(shapes),notes)
    (layouts=layouts,orientation=orientation,notes=notes)
end

"""Populations of one species on the atmosphere's grid, `(nz, nx, ny, level)` [m⁻³]: a model that only hands them over."""
struct GivenPopulationModel <: AbstractPopulationModel
    species::Symbol
    populations::Array{Float32,4}
    source::String
end

function predict_populations!(out::AbstractArray{T,4},model::GivenPopulationModel,atmosphere::Atmosphere3D,cache=nothing) where T
    size(out)==size(model.populations) || throw(DimensionMismatch(
        "given $(model.species) populations are $(size(model.populations)); the workspace needs $(size(out))"))
    out.=model.populations; out
end

"""One level `(fx, fy, nz)` in the file's order onto `out` `(nz, nx, ny)`, oriented."""
function _given_place!(out,level_cube,orientation,reversed,scale)
    nz,nx,ny=size(out)
    Threads.@threads for j in 1:ny
        for i in 1:nx
            a,b=_given_source(orientation,i,j,nx,ny)
            for k in 1:nz
                out[k,i,j]=Float32(level_cube[a,b,reversed ? nz+1-k : k]*scale)
            end
        end
    end
    out
end

function _read_given_species(config::GivenPopulationsConfig,species,layout,orientation,nz,nx,ny)
    populations=Array{Float32,4}(undef,nz,nx,ny,layout.levels)
    fx,fy=layout.shape
    if config.format===:multi3d
        buffer=Array{Float32}(undef,fx,fy,nz)
        open(layout.file,"r") do io
            for level in 1:layout.levels
                read!(io,buffer)
                _given_place!(@view(populations[:,:,:,level]),buffer,orientation.orientation,orientation.reversed,1f6)
            end
        end
    else
        raw=h5open(f->read(f["populations/$species"]),layout.file,"r")       # (level, z, y, x)
        for level in 1:layout.levels
            _given_place!(@view(populations[:,:,:,level]),permutedims(raw[level,:,:,:],(3,2,1)),orientation.orientation,orientation.reversed,1f0)
        end
    end
    bad=count(v->!isfinite(v) || v<0,populations)
    bad==0 || throw(ArgumentError("$(layout.file): $bad of $(length(populations)) $species populations are not finite or negative"))
    GivenPopulationModel(species,populations,layout.file)
end

"""Preflight, then read every file: models by species, the notes and the provenance entry."""
function read_given_populations(config::GivenPopulationsConfig,atmosphere,levels::AbstractDict)
    plan=preflight_given_populations(config,atmosphere,levels)
    nz,nx,ny=size(atmosphere.temperature)
    models=Dict{Symbol,GivenPopulationModel}(species=>_read_given_species(config,species,layout,plan.orientation,nz,nx,ny)
        for (species,layout) in plan.layouts)
    for (species,model) in sort!(collect(models);by=string∘first)
        p=model.populations
        push!(plan.notes,"$species: read, levels 1-$(size(p,4)), $(minimum(p)) to $(maximum(p)) m^-3")
    end
    provenance=Dict{String,Any}("format"=>string(config.format),"orientation"=>string(plan.orientation.orientation),
        "levels_reversed"=>plan.orientation.reversed,"check_level"=>plan.orientation.level,"tau500"=>plan.orientation.tau500,
        "correlations"=>plan.orientation.correlations,"files"=>Dict(string(k)=>v for (k,v) in config.files),"notes"=>plan.notes)
    (models=models,notes=plan.notes,provenance=provenance)
end

"""Within `launch_fsdp_population_models`, the species to hand over instead of predicting: `(species, models)`, the models on rank 0 only."""
const GIVEN_POPULATIONS=Base.ScopedValues.ScopedValue{Any}(nothing)

"""Collective: on rank 0 read `[population.given]`, if any, print its notes and return what `GIVEN_POPULATIONS` holds; a failure on rank 0
is thrown on every rank."""
function prepare_given_populations(config,atmosphere,levels,context)
    given=hasproperty(config,:given_populations) ? config.given_populations : nothing
    given===nothing && return nothing
    levels isa AbstractDict || throw(ArgumentError("[population.given] needs a model factory with levels per species"))
    species=sort!(collect(keys(given.files));by=string)
    result=nothing; message=""
    if isroot(context)
        try
            result=read_given_populations(given,atmosphere,levels)
            for note in result.notes; println(stdout,"GIVEN_POPULATIONS $note"); end
            flush(stdout)
        catch exception
            message=sprint(showerror,exception)
        end
    end
    message=mpi_broadcast(isroot(context) ? message : nothing,context)
    isempty(message) || throw(ArgumentError("[population.given]: $message"))
    (species=species,models=isroot(context) ? result.models : nothing,provenance=isroot(context) ? result.provenance : nothing)
end

"""Root: true when every given species of `model` is the given model (a factory that builds its populations itself would ignore them)."""
_given_in_use(model,given)=false
function _given_in_use(model::CompositeDistributedPopulationModel,given)
    all(given.species) do species
        haskey(model.models,species) && model.models[species] isa RootDistributedPopulationModel &&
            model.models[species].root_model isa GivenPopulationModel
    end
end
