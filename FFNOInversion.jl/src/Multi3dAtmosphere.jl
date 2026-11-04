"""
The MULTI3D atmospheres: a directory with the input files of a MULTI3D run, `mesh` and `atm3d`. `mesh` is text: nx and x, ny and y, nz and z [cm], with z falling from the top. `atm3d` is
Float32 in Fortran order (x, y, z), one field after another: nₑ [cm⁻³], T [K], vx, vy, vz [km s⁻¹] and ρ [g cm⁻³], and in some snapshots the populations of six hydrogen levels [cm⁻³], the
last the protons. Index 1 of a column is already the top and (x, y) stay as they are, so results line up with MULTI3D's outputs for the snapshot and with the predictions of pipeline.py.
nₑ, ρ and any populations are given, P follows from them (`set_given_inputs!`), and HSEMHS is not used. The whole column is read: the population mesh gives the FFNO only the heights it was trained on.
"""
const _MULTI3D_FIELDS=6
const _MULTI3D_HYDROGEN_LEVELS=6
const _MULTI3D_MPS_PER_KMS=1e3
const _MULTI3D_PER_M3_PER_PER_CM3=1e6
const _MULTI3D_KG_M3_PER_G_CM3=1e3

"""x, y, z [m] of a `mesh` file."""
function _read_multi3d_mesh(path)
    values=parse.(Float64,split(read(path,String)))
    axes=Vector{Float64}[]; position=1
    for name in ("x","y","z")
        position<=length(values) || throw(ArgumentError("$path ends before the $name axis"))
        count=Int(values[position])
        count>=1 && position+count<=length(values) || throw(ArgumentError("$path: the $name axis has $count points, more than the file holds"))
        push!(axes,values[position+1:position+count].*1e-2); position+=count+1
    end
    position==length(values)+1 || throw(ArgumentError("$path has $(length(values)-position+1) numbers after the z axis"))
    Tuple(axes)
end

"""The axes [m] and the file of a snapshot directory, and whether `atm3d` holds hydrogen populations, which its size tells."""
function _multi3d_snapshot(directory)
    isdir(directory) || throw(ArgumentError("atmosphere.atmos_directory not found: $directory"))
    mesh,atm3d=joinpath(directory,"mesh"),joinpath(directory,"atm3d")
    for path in (mesh,atm3d); isfile(path) || throw(ArgumentError("not found: $path")); end
    x,y,z=_read_multi3d_mesh(mesh)
    all(<(0),diff(z)) || throw(ArgumentError("$mesh: z must fall from the top of the atmosphere"))
    cell=length(x)*length(y)*length(z)*sizeof(Float32); bytes=filesize(atm3d)
    bytes in (_MULTI3D_FIELDS*cell,(_MULTI3D_FIELDS+_MULTI3D_HYDROGEN_LEVELS)*cell) || throw(ArgumentError(
        "$atm3d has $bytes bytes: $(length(x)) x $(length(y)) x $(length(z)) needs $(_MULTI3D_FIELDS*cell), or $((_MULTI3D_FIELDS+_MULTI3D_HYDROGEN_LEVELS)*cell) with hydrogen populations"))
    (x=x,y=y,z=z,atm3d=atm3d,hydrogen=bytes>_MULTI3D_FIELDS*cell)
end

"""What the snapshot gives, under the names of `[atmosphere.datasets]`."""
function _multi3d_datasets(directory)
    datasets=Dict{String,Any}(name=>"atm3d" for name in ("ne","temperature","vx","vy","vz","rho"))
    _multi3d_snapshot(directory).hydrogen && (datasets["hydrogen_populations"]="atm3d")
    datasets
end

"""The spacing of a horizontal axis, which may fall, as y does in most snapshots: only its size is used, as in pipeline.py. `mesh` prints six digits, so uniform means within a percent of a cell."""
function _multi3d_spacing(axis,name,directory)
    length(axis)>=2 || throw(ArgumentError("$directory/mesh: the $name axis needs at least two points"))
    spacing=(axis[end]-axis[1])/(length(axis)-1)
    spacing!=0 && all(i->abs(axis[i]-axis[1]-(i-1)*spacing)<=0.01*abs(spacing),eachindex(axis)) ||
        throw(ArgumentError("$directory/mesh: the $name axis must be uniform"))
    abs(spacing)
end

"""`[grid]` with the spacing of the mesh; a value set there must agree to the digits the mesh has."""
function _multi3d_grid(directory,grid)
    snapshot=_multi3d_snapshot(directory); merged=Dict{String,Any}(grid)
    for (key,axis,name) in (("dx_m",snapshot.x,"x"),("dy_m",snapshot.y,"y"))
        spacing=_multi3d_spacing(axis,name,directory)
        haskey(grid,key) && abs(Float64(grid[key])-spacing)>1e-4*spacing && throw(ArgumentError(
            "grid.$key=$(grid[key]) differs from the spacing $spacing of $directory/mesh"))
        merged[key]=spacing
    end
    merged
end

"""Atmosphere (nz, nx, ny), top first, from `config.atmosphere.file`."""
function read_multi3d_atmosphere(config::RunConfig{T}) where T
    snapshot=_multi3d_snapshot(config.atmosphere.file); z=T.(snapshot.z)
    nx,ny,nz=length(snapshot.x),length(snapshot.y),length(z); buffer=Array{Float32}(undef,nx,ny,nz)
    function next!(stream,out,factor)
        read!(stream,buffer); scale=T(factor)
        Threads.@threads for b in 1:ny
            for a in 1:nx, k in 1:nz
                out[k,a,b]=T(buffer[a,b,k])*scale
            end
        end
        out
    end
    fields=open(snapshot.atm3d,"r") do stream
        field(factor)=next!(stream,Array{T}(undef,nz,nx,ny),factor)
        ne=field(_MULTI3D_PER_M3_PER_PER_CM3); temperature=field(1); vx=field(_MULTI3D_MPS_PER_KMS); vy=field(_MULTI3D_MPS_PER_KMS); vz=field(_MULTI3D_MPS_PER_KMS)
        rho=field(_MULTI3D_KG_M3_PER_G_CM3)
        hydrogen=snapshot.hydrogen ? Array{T}(undef,nz,nx,ny,_MULTI3D_HYDROGEN_LEVELS) : nothing
        snapshot.hydrogen && for level in 1:_MULTI3D_HYDROGEN_LEVELS
            next!(stream,@view(hydrogen[:,:,:,level]),_MULTI3D_PER_M3_PER_PER_CM3)
        end
        (;ne,temperature,vx,vy,vz,rho,hydrogen)
    end
    for (name,field) in (("nₑ",fields.ne),("temperature",fields.temperature),("ρ",fields.rho))
        all(>(0),field) && all(isfinite,field) || throw(ArgumentError("$(snapshot.atm3d): $name must be finite and positive"))
    end
    temperature=_raise_temperature_floor!(fields.temperature,config.atmosphere.temperature_floor_k)
    vz=config.vz_positive===:down ? -fields.vz : fields.vz
    grid=Grid3D(VerticalAxis(:z_m,z),T.(0:nx-1).*config.synthesis.dx_m,T.(0:ny-1).*config.synthesis.dy_m,config.view)
    is_slanted(config.view) && warn_view_travel(grid,z,"input")
    atmosphere=Atmosphere3D(grid,temperature,fields.vx,fields.vy,vz,zeros(T,nz,nx,ny);z=repeat(reshape(z,:,1,1),1,nx,ny),ne=fields.ne)
    set_given_inputs!(atmosphere;rho=fields.rho,hydrogen=fields.hydrogen)
end
