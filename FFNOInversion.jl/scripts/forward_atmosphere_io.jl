# Reader for the native log tau500 forward atmosphere written by write_inversion_outputs (forward_atmosphere.h5),
# shared by column_response.jl and ffno_population_tangents.jl. Names are qualified: a model factory that loads
# Muspel would make the exported Atmosphere3D ambiguous.
"""Atmosphere (nz, nx, ny) and populations (nz, nx, ny, level) as written by write_inversion_outputs."""
function read_forward_atmosphere(path)
    h5open(path,"r") do file
        # a run with [atmosphere] frame = "z_down" writes its outputs rotated by 180° about x: rotate them back
        down=haskey(HDF5.attributes(file),"frame") && read(HDF5.attributes(file)["frame"])=="z_down"
        flip(name,a)=!down ? a : (name in ("vy","vz","z") ? -1 : 1) .* reverse(a;dims=3)
        field(name)=haskey(file,name) ? flip(name,permutedims(read(file[name])[1,:,:,:],(1,3,2))) : nothing
        view=something(FFNOInversion.read_view_attributes(HDF5.attributes(file)),FFNOInversion.ViewGeometry())
        down && (view=FFNOInversion.ViewGeometry(view.mu,-view.azimuth_deg,view.geometry))
        y=read(file["y_m"]); down && (y=reverse(-y))
        grid=FFNOInversion.Grid3D(FFNOInversion.VerticalAxis(:log10tau500,read(file["logtau_500"])),read(file["x_m"]),y,view)
        atmosphere=FFNOInversion.Atmosphere3D(grid,field("temperature"),field("vx"),field("vy"),field("vz"),field("vturb");
            pgas=field("pgas"),rho=field("rho"),ne=field("ne"),z=field("z"),logtau500=field("logtau500"),
            chi500=field("chi500"))
        populations=Dict(Symbol(name)=>flip(name,permutedims(read(file["populations"][name]),(2,4,3,1)))
            for name in keys(file["populations"]))
        atmosphere,populations
    end
end

"""Write `atmosphere` (nz, nx, ny) and `populations` (species => (nz, nx, ny, level)) in the layout that
`read_forward_atmosphere` reads (h5py order x, y, z, 1; populations x, y, z, level)."""
function write_forward_atmosphere(path,atmosphere,populations;attributes_=Dict{String,String}())
    h5open(path,"w") do file
        for name in (:temperature,:vx,:vy,:vz,:vturb,:pgas,:rho,:ne,:z,:logtau500,:chi500)
            value=getfield(atmosphere,name); value===nothing && continue
            file[string(name)]=reshape(permutedims(value,(1,3,2)),1,size(value,1),size(value,3),size(value,2))
        end
        file["logtau_500"]=collect(native_depth(atmosphere.grid))
        file["x_m"]=collect(atmosphere.grid.x); file["y_m"]=collect(atmosphere.grid.y)
        for (species,value) in populations; file["populations/$species"]=permutedims(value,(4,1,3,2)); end
        for (key,value) in FFNOInversion.view_attributes(atmosphere.grid.view,FFNOInversion.view_columns(atmosphere.grid))
            HDF5.attributes(file)[key]=value
        end
        for (key,value) in attributes_; HDF5.attributes(file)[key]=value; end
    end
    path
end

"""Every `stride`-th column of `atmosphere` in x and y, starting at the first (0-based 0, stride, ...)."""
function subsample_atmosphere(atmosphere,stride)
    stride==1 && return atmosphere
    xs=1:stride:size(atmosphere.temperature,2); ys=1:stride:size(atmosphere.temperature,3)
    pick(a)=a===nothing ? nothing : a[:,xs,ys]
    g=atmosphere.grid
    FFNOInversion.Atmosphere3D(FFNOInversion.Grid3D(g.vertical,g.x[xs],g.y[ys],g.view),pick(atmosphere.temperature),
        pick(atmosphere.vx),pick(atmosphere.vy),pick(atmosphere.vz),pick(atmosphere.vturb);pgas=pick(atmosphere.pgas),
        rho=pick(atmosphere.rho),ne=pick(atmosphere.ne),z=pick(atmosphere.z),logtau500=pick(atmosphere.logtau500),
        chi500=pick(atmosphere.chi500))
end
