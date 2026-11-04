"""
The IRIS/MURaM FITS atmospheres: one cube per quantity in a directory, `<code>_<run>_<quantity>_<snap>.fits`, with the quantities lgtg, ux, uy, uz, lgr
and lgp, in file order (x, y, z), bottom first, and the heights [Mm] in the `z-coordinate` extension. As scripts/convert_iris_sim_fits_to_ffno_hdf5.py reads them,
so that results line up with the predictions it prepared: index 1 of a column is the top, and level by level (x, y) of the file becomes (x, reversed y), with uy kept
as it is. Rho and P are given, so nothing here derives nₑ: `GivenAtmosphereBackend` takes it from the EOS at (T, P). The whole column is read: the population mesh gives the
FFNO only the heights it was trained on.
"""
const IRIS_DATASETS=Dict{String,Any}("temperature"=>"lgtg","vx"=>"ux","vy"=>"uy","vz"=>"uz","rho"=>"lgr","pgas"=>"lgp")

# SI factor of a FITS unit string, as these files write it
const _FITS_UNITS=Dict("K"=>1.0,"m s^(-1)"=>1.0,"cm s^(-1)"=>1e-2,"km s^(-1)"=>1e3,"kg m^(-3)"=>1.0,"g cm^(-3)"=>1e3,"N m^(-2)"=>1.0,"Pa"=>1.0,
    "dyn cm^(-2)"=>0.1,"Mm"=>1e6,"km"=>1e3,"cm"=>1e-2,"m"=>1.0)
const _FITS_BLOCK=2880

_fits_unit(header,key,path)=get(_FITS_UNITS,String(get(header,key,"")),nothing)===nothing ?
    throw(ArgumentError("$path: unknown unit \"$(get(header,key,""))\" in $key")) : _FITS_UNITS[String(header[key])]

"""The cards of the header at the stream's position, as strings or numbers by key; leaves the stream at the data."""
function _read_fits_header(stream,path)
    header=Dict{String,Any}()
    while true
        block=read(stream,_FITS_BLOCK); length(block)==_FITS_BLOCK || throw(ArgumentError("$path: FITS header is truncated"))
        for start in 1:80:_FITS_BLOCK
            card=String(block[start:start+79]); key=strip(card[1:8])
            key=="END" && return header
            card[9:10]=="= " || continue
            text=card[11:end]
            if startswith(lstrip(text),"'")
                body=lstrip(text); header[key]=rstrip(body[2:something(findnext('\'',body,2),length(body)+1)-1])
            else
                field=strip(first(split(text,'/';limit=2)))
                value=tryparse(Int,field); value===nothing && (value=tryparse(Float64,field))
                header[key]=value===nothing ? String(field) : value
            end
        end
    end
end

_fits_axes(header)=Tuple(Int(header["NAXIS$i"]) for i in 1:Int(header["NAXIS"]))
_fits_data_bytes(header)=ceil(Int,prod(_fits_axes(header))*abs(Int(header["BITPIX"]))/8/_FITS_BLOCK)*_FITS_BLOCK

"""The big-endian floating-point array at the stream's position, skipping to the next header."""
function _read_fits_data(stream,header,path)
    type=get(Dict(-32=>Float32,-64=>Float64),Int(header["BITPIX"]),nothing)
    type===nothing && throw(ArgumentError("$path: BITPIX = $(header["BITPIX"]) is not supported"))
    data=Array{type}(undef,_fits_axes(header)...); read!(stream,data); data.=ntoh.(data)
    skip(stream,_fits_data_bytes(header)-sizeof(data))
    data
end

"""The cube, its header and the heights of a quantity's FITS file."""
function _read_fits_cube(path)
    open(path,"r") do stream
        header=_read_fits_header(stream,path)
        cube=_read_fits_data(stream,header,path)
        ndims(cube)==3 || throw(ArgumentError("$path: expected a 3-D cube, got $(size(cube))"))
        eof(stream) && throw(ArgumentError("$path has no height extension"))
        extension=_read_fits_header(stream,path)
        heights=_read_fits_data(stream,extension,path)
        length(heights)==size(cube,3) || throw(DimensionMismatch("$path: $(length(heights)) heights for $(size(cube,3)) levels"))
        cube,header,Float64.(vec(heights)).*_fits_unit(extension,"BUNIT",path)
    end
end

"""`<code>_<run>`, `<snap>` of the one snapshot in a directory."""
function _iris_snapshot(directory)
    isdir(directory) || throw(ArgumentError("atmosphere.atmos_directory not found: $directory"))
    found=[m for m in (match(r"^(.+)_lgtg_(.+)\.fits$",name) for name in readdir(directory)) if m!==nothing]
    length(found)==1 || throw(ArgumentError("atmosphere.atmos_directory must hold one snapshot's *_lgtg_*.fits; $directory has $(length(found))"))
    found[1][1],found[1][2]
end
_iris_path(directory,snapshot,quantity)=joinpath(directory,"$(snapshot[1])_$(quantity)_$(snapshot[2]).fits")

"""`[grid]` with the spacing of the cubes where it is not set; `load_config` checks it against nothing else, so a set value must agree here."""
function _iris_grid(directory,grid)
    path=_iris_path(directory,_iris_snapshot(directory),"lgtg")
    isfile(path) || throw(ArgumentError("not found: $path"))
    header=open(stream->_read_fits_header(stream,path),path,"r")
    merged=Dict{String,Any}(grid)
    for (key,axis) in (("dx_m",1),("dy_m",2))
        spacing=abs(Float64(header["CDELT$axis"]))*_fits_unit(header,"CUNIT$axis",path)
        haskey(grid,key) && abs(Float64(grid[key])-spacing)>1e-9*spacing && throw(ArgumentError(
            "grid.$key=$(grid[key]) differs from the spacing $spacing of $path"))
        merged[key]=spacing
    end
    merged
end

"""Atmosphere (nz, nx, ny), top first, from the FITS cubes of `config.atmosphere.file`."""
function read_iris_atmosphere(config::RunConfig{T}) where T
    directory=config.atmosphere.file; snapshot=_iris_snapshot(directory)
    function field(quantity,logarithmic)
        path=_iris_path(directory,snapshot,quantity)
        isfile(path) || throw(ArgumentError("not found: $path"))
        cube,header,heights=_read_fits_cube(path)
        levels=sortperm(heights;rev=true)
        factor=T(_fits_unit(header,"BUNIT",path)); nx,ny,_=size(cube); out=Array{T}(undef,length(levels),nx,ny)
        Threads.@threads for b in 1:ny
            for a in 1:nx, (k,level) in enumerate(levels)
                value=T(cube[a,ny+1-b,level])
                out[k,a,b]=logarithmic ? exp10(value)*factor : value*factor
            end
        end
        out,heights[levels]
    end
    temperature,z=field("lgtg",true)
    vx,_=field("ux",false); vy,_=field("uy",false); vz,_=field("uz",false)
    config.vz_positive===:down && (vz=-vz)
    rho,_=field("lgr",true); pgas,_=field("lgp",true)
    nz,nx,ny=size(temperature)
    grid=Grid3D(VerticalAxis(:z_m,T.(z)),T.(0:nx-1).*config.synthesis.dx_m,T.(0:ny-1).*config.synthesis.dy_m,config.view)
    is_slanted(config.view) && warn_view_travel(grid,T.(z),"input")
    atmosphere=Atmosphere3D(grid,temperature,vx,vy,vz,zeros(T,nz,nx,ny);z=repeat(reshape(T.(z),:,1,1),1,nx,ny))
    set_given_inputs!(atmosphere;rho=rho,pgas=pgas)
end
