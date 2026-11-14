# Column response functions of a forward run: R_v(λ, k; x, y) = ∂I(λ, x, y) / ∂v(k, x, y), the intrinsic (pre-PSF)
# Stokes I at every configured wavelength against variable v at every native log tau500 point of the same column.
#
# Everything except v is held fixed: FFNO populations, the HSE structure (pgas, rho, ne, z, chi500 and so the
# log tau500 scale) and the other variables. Each column is then independent and needs no FFNO or HSEMHS call:
# one adjoint sweep of the formal solver gives ∂I(λ)/∂χ(k, λ) and ∂I(λ)/∂η(k, λ) for every wavelength, and since every
# opacity source is local to its depth point, perturbing v by ±h at all depths at once gives ∂χ(k, λ)/∂v(k) and
# ∂η(k, λ)/∂v(k) for every k from two opacity evaluations per variable.
#
# With --population-tangents (from ffno_population_tangents.jl) the temperature response also includes the FFNO
# populations: ṅ(k; k0), their forward-mode (dual tensor) change for a 1 K rise at k0 in every column, is combined
# with G_l(k, λ) = ∂I(λ)/∂n_l(k) of the column (the same adjoint, and cell-local opacity differences one level at a
# time): R[λ, k0] += Σ_{k,l} G_l(k, λ) ṅ_l(k; k0) / Δlog τ(k0). Because the tangent was seeded in every column, this
# part also holds the neighbouring columns' influence through the FFNO's horizontal Fourier layers. pgas, rho, ne and
# z stay fixed.
#
# Output (HDF5, Julia shapes; h5py sees them reversed):
#   R/<variable>         (nλ, nz, ny', nx')  per unit log10 tau500: δI(λ) = Σ_k R[λ, k] δv(k) Δlog τ(k)
#                        temperature [I/K], vz [I/(m/s)], vturb [I/(m/s)], ne and pgas [I per unit ln v],
#                        vlos [I/(m/s)]: v moved along the line of sight n (equals vz for a vertical view; off the
#                        vertical vz is the vertical component only, docs/viewing-angle/README.md)
#   intensity            (nλ, ny', nx')      intrinsic Stokes I of the same columns
#   wavelength_air_m, wavelength_vacuum_m (nλ); logtau500, dlogtau500 (nz); x_index, y_index (0-based), x_m, y_m
#
# usage: julia --project=<FFNOInversion env> column_response.jl FORWARD.toml PATHS.jl FORWARD_ATMOSPHERE.h5
#            OUTPUT.h5 [--variables temperature,vz] [--stride 4] [--crop x0,x1,y0,y1 (0-based, inclusive)]
#            [--population-tangents TANGENTS.h5]
# PATHS.jl defines EOS_LIBRARY, PARTITION_FUNCTIONS and ATOM_FILES; the run's model_factory.jl does. No GPU.
# Under mpiexec, rank 0 hands out x slabs and writes the file and every other rank computes them with all its threads.
using FFNOInversion, HDF5, Printf
import WittBifrostEOS
import MPI
using LinearAlgebra: mul!
import Muspel                  # loads the extension that builds the H and Ca line models (import: both export Atmosphere3D)
const F = FFNOInversion

function parse_arguments(args)
    length(args)>=4 || error("usage: column_response.jl FORWARD.toml PATHS.jl FORWARD_ATMOSPHERE.h5 OUTPUT.h5 " *
        "[--variables temperature,vz] [--stride 4] [--crop x0,x1,y0,y1] [--population-tangents TANGENTS.h5]")
    options=Dict("variables"=>"temperature,vz","stride"=>"4","crop"=>"","population-tangents"=>"")
    i=5
    while i<=length(args)
        key=lstrip(args[i],'-'); haskey(options,key) && i<length(args) || error("unknown or incomplete option $(args[i])")
        options[key]=args[i+1]; i+=2
    end
    variables=Symbol.(split(options["variables"],','))
    for v in variables
        v in (:temperature,:vz,:vlos,:vturb,:ne,:pgas) || error("variable $v is not one of temperature, vz, vlos, vturb, ne, pgas")
    end
    crop=isempty(options["crop"]) ? nothing : parse.(Int,split(options["crop"],','))
    tangents=isempty(options["population-tangents"]) ? nothing : options["population-tangents"]
    tangents===nothing || :temperature in variables || error("--population-tangents needs the temperature variable")
    (config=args[1],factory=args[2],atmosphere=args[3],output=args[4],variables=variables,
     stride=parse(Int,options["stride"]),crop=crop,tangents=tangents)
end

include(joinpath(@__DIR__,"forward_atmosphere_io.jl"))

"""Width of each log tau500 cell (trapezoid weights), so that Σ_k R_k Δ_k δv_k is the intensity change."""
function cell_widths(logtau)
    n=length(logtau); d=zeros(n)
    d[1]=(logtau[2]-logtau[1])/2; d[n]=(logtau[n]-logtau[n-1])/2
    for k in 2:n-1; d[k]=(logtau[k+1]-logtau[k-1])/2; end
    d
end

"""Step and chain factor per depth: R is ∂I/∂v for T, vz, vturb and ∂I/∂ln v for ne and pgas. vz and vturb use a
fixed step because Muspel works in Float32, where a m/s Doppler shift is below the wavelength resolution."""
function perturbation(variable,value)
    variable===:temperature && return 1e-3*value,1.0
    variable in (:vz,:vlos,:vturb) && return 250.0,1.0
    1e-3*value,value
end

function opacity!(chi,eta,synthesizer,wavelength,atmosphere,x,y)
    fill!(chi,0); fill!(eta,0)
    for contributor in synthesizer.contributors
        F.add_opacity_emissivity!(chi,eta,contributor.model,wavelength,atmosphere,x,y,contributor.populations)
    end
end

"""FFNO population tangents of one x slab: species => (nz, ny', levels, nz) Float64, for the selected ys."""
function population_tangent_slab(file,x,ys)
    Dict(Symbol(name)=>Float64.(file["tangent"][name][:,:,:,:,x][:,ys,:,:]) for name in keys(file["tangent"]))
end

"""Does a region's synthesizer read population array `p` (as line populations, or as H for Muspel's continuum)?"""
uses_populations(synthesizer,p)=any(c->c.populations===p ||
    (hasproperty(c.model,:hydrogen_populations) && c.model.hydrogen_populations===p),synthesizer.contributors)

"""Population part of the temperature response of column (x, y) for one region, added to out[range, :, j]."""
function add_population_term!(out,range,j,term,synthesizer,wavelength,atmosphere,x,y,chi_bar,eta_bar,widths,buffers)
    chi_plus,eta_plus,chi_minus,eta_minus=buffers
    nz=length(widths); nl=length(wavelength); G=zeros(nz,nl); acc=zeros(nl,nz)
    for (species,tangent) in term.slab
        p=term.populations[species]; uses_populations(synthesizer,p) || continue
        for level in axes(p,4)
            column=@view p[:,x,y,level]; saved=copy(column); h=1e-3 .* saved
            column.=saved.+h; opacity!(chi_plus,eta_plus,synthesizer,wavelength,atmosphere,x,y)
            column.=saved.-h; opacity!(chi_minus,eta_minus,synthesizer,wavelength,atmosphere,x,y)
            column.=saved
            for l in 1:nl, k in 1:nz
                G[k,l]=(chi_bar[k,l]*(chi_plus[k,l]-chi_minus[k,l])+eta_bar[k,l]*(eta_plus[k,l]-eta_minus[k,l]))/(2h[k])
            end
            mul!(acc,transpose(G),@view(tangent[:,j,level,:]),1.0,1.0)       # Σ_k G_l(k, λ) ṅ_l(k; k0)
        end
    end
    for k0 in 1:nz, l in 1:nl
        out[range[l],k0,j]+=acc[l,k0]/widths[k0]
    end
end

"""Responses of column (x, y) into R[v][:, :, j] (wavelength, depth) and its intensity into I[:, j]. `term`, when
given, holds the populations and this slab's FFNO population tangents for the temperature response."""
function column_response!(R,I,j,setups,ranges,variables,atmosphere,x,y,widths,term=nothing)
    nz=length(widths)
    coordinate=F._log_tau_coordinate(atmosphere,x,y); weight=F._log_tau_weight(atmosphere,x,y)
    for (setup,range) in zip(setups,ranges)
        wavelength=setup.wavelength_vacuum_m; synthesizer=setup.synthesizer; nl=length(wavelength)
        chi=zeros(nz,nl); eta=zeros(nz,nl)
        opacity!(chi,eta,synthesizer,wavelength,atmosphere,x,y)
        chi.*=weight; eta.*=weight
        F.formal_solve!(@view(I[range,j]),synthesizer.solver,chi,eta,coordinate)
        chi_bar=zeros(nz,nl); eta_bar=zeros(nz,nl)
        F._formal_solve_vjp!(chi_bar,eta_bar,zeros(nz),synthesizer.solver,chi,eta,coordinate,ones(nl))
        chi_bar.*=weight; eta_bar.*=weight      # chi500 fixed: χ' = wχ, η' = wη
        chi_plus=similar(chi); eta_plus=similar(eta); chi_minus=similar(chi); eta_minus=similar(eta)
        for (iv,variable) in enumerate(variables)
            # vlos moves (vx, vy, vz) along n; the other variables are one stored field
            names=variable===:vlos ? (:vx,:vy,:vz) : (variable,)
            direction=variable===:vlos ? F.view_direction(atmosphere.grid.view) : (1.0,)
            columns=[@view(getfield(atmosphere,name)[:,x,y]) for name in names]; saved=copy.(columns)
            steps=perturbation.(variable,variable===:vlos ? [F.los_velocity(atmosphere,k,x,y) for k in 1:nz] : saved[1])
            h=first.(steps); chain=last.(steps)
            for (c,s,n) in zip(columns,saved,direction); c.=s.+n.*h; end
            opacity!(chi_plus,eta_plus,synthesizer,wavelength,atmosphere,x,y)
            for (c,s,n) in zip(columns,saved,direction); c.=s.-n.*h; end
            opacity!(chi_minus,eta_minus,synthesizer,wavelength,atmosphere,x,y)
            for (c,s) in zip(columns,saved); c.=s; end
            out=R[iv]
            for k in 1:nz, l in 1:nl
                d=(chi_bar[k,l]*(chi_plus[k,l]-chi_minus[k,l])+eta_bar[k,l]*(eta_plus[k,l]-eta_minus[k,l]))/(2h[k])
                out[range[l],k,j]=d*chain[k]/widths[k]
            end
        end
        term===nothing || add_population_term!(R[findfirst(==(:temperature),variables)],range,j,term,synthesizer,
            wavelength,atmosphere,x,y,chi_bar,eta_bar,widths,(chi_plus,eta_plus,chi_minus,eta_minus))
    end
end

"""Responses of every selected column at x into R[v][:, :, j] and I[:, j], threaded over the columns."""
function slab_response!(R,I,x,ys,setups,ranges,variables,atmosphere,widths,tangents=nothing,populations=nothing)
    term=tangents===nothing ? nothing : (populations=populations,slab=population_tangent_slab(tangents,x,ys))
    Threads.@threads :dynamic for j in eachindex(ys)
        column_response!(R,I,j,setups,ranges,variables,atmosphere,x,ys[j],widths,term)
    end
end

"""MPI work queue: rank 0 hands out x slabs one at a time and writes them; every other rank computes. A worker
returns slab i as one Float32 message [i, R(v1), R(v2), ..., I]; slab index 0 means stop."""
const TAG_RESULT=1
const TAG_ASSIGN=2

function serve_slabs!(write_slab,comm,nslabs,nvalues,progress)
    workers=MPI.Comm_size(comm)-1; next=1; active=0
    for worker in 1:workers
        MPI.Send(Int[next<=nslabs ? next : 0],comm;dest=worker,tag=TAG_ASSIGN)
        next<=nslabs && (active+=1); next+=1
    end
    buffer=Vector{Float32}(undef,2+nvalues); done=0
    while active>0
        MPI.Recv!(buffer,comm;source=MPI.ANY_SOURCE,tag=TAG_RESULT)
        i=Int(buffer[1]); worker=Int(buffer[2])
        MPI.Send(Int[next<=nslabs ? next : 0],comm;dest=worker,tag=TAG_ASSIGN)
        next<=nslabs || (active-=1); next+=1
        write_slab(i,@view(buffer[3:end])); done+=1; progress(done)
    end
end

function work_slabs!(compute_slab!,comm,nvalues)
    rank=MPI.Comm_rank(comm); assignment=Int[0]; buffer=Vector{Float32}(undef,2+nvalues)
    MPI.Recv!(assignment,comm;source=0,tag=TAG_ASSIGN)
    while assignment[1]>0
        buffer[1]=assignment[1]; buffer[2]=rank
        compute_slab!(assignment[1],@view(buffer[3:end]))
        MPI.Send(buffer,comm;dest=0,tag=TAG_RESULT)
        MPI.Recv!(assignment,comm;source=0,tag=TAG_ASSIGN)
    end
end

function main(a)
    MPI.Init(threadlevel=:funneled)
    comm=MPI.COMM_WORLD; rank=MPI.Comm_rank(comm); nranks=MPI.Comm_size(comm)
    config=load_config(a.config)
    atmosphere,populations=read_forward_atmosphere(a.atmosphere)
    F._tau_integration(atmosphere) || error("the forward atmosphere is not a native log tau500 grid with chi500")
    eos=WittmannEOS(Main.EOS_LIBRARY,Main.PARTITION_FUNCTIONS)
    # the NLTE EOS's hydrogen (no equilibrium runs here: the table of [equilibrium] itself)
    nlte=config.equilibrium.eos_model=="nlte" ? NLTEHydrogenState(F.WittBifrostEOS.load_table(config.equilibrium.eos_table_file;model="nlte")) : nothing
    setups=build_synthesis_setup(config,atmosphere,populations,eos;atom_files=Main.ATOM_FILES,nlte_hydrogen=nlte)
    wavelength_air=collect(config.synthesis.wavelength_m)
    ranges=F._region_ranges(wavelength_air,setups)
    wavelength_vacuum=reduce(vcat,[s.wavelength_vacuum_m for s in setups])
    nz,nx,ny=size(atmosphere.temperature); nl=length(wavelength_air)
    x0,x1,y0,y1=a.crop===nothing ? (1,nx,1,ny) : (a.crop[1]+1,a.crop[2]+1,a.crop[3]+1,a.crop[4]+1)
    xs=collect(x0:a.stride:x1); ys=collect(y0:a.stride:y1); nv=length(a.variables); nys=length(ys)
    logtau=native_depth(atmosphere.grid); widths=cell_widths(logtau)
    R=[zeros(nl,nz,nys) for _ in a.variables]; I=zeros(nl,nys)
    nvalues=(nv*nz+1)*nl*nys
    tangents=a.tangents===nothing ? nothing : h5open(a.tangents,"r")
    if tangents!==nothing
        read(tangents["depth_index"])==1:nz || error("the population tangents must cover every depth 1:$nz")
        for name in keys(tangents["tangent"])
            haskey(populations,Symbol(name)) || error("no $name populations in the atmosphere for its tangents")
            size(tangents["tangent"][name])==(nz,ny,size(populations[Symbol(name)],4),nz,nx) ||
                error("$name population tangents do not match the atmosphere grid")
        end
    end
    if rank>0
        work_slabs!(comm,nvalues) do i,out
            slab_response!(R,I,xs[i],ys,setups,ranges,a.variables,atmosphere,widths,tangents,populations)
            out.=vcat((vec(r) for r in R)...,vec(I))
        end
        MPI.Barrier(comm); return
    end
    for v in a.variables
        v===:vturb && all(iszero,atmosphere.vturb) && @warn "vturb is zero everywhere: its response is zero (line widths depend on vturb²)"
    end
    gb=4*nl*nz*length(xs)*nys*nv/1e9
    @printf("columns %d x %d (stride %d), %d wavelengths, %d depths, variables %s: %.2f GB uncompressed, %s\n",
        length(xs),nys,a.stride,nl,nz,join(a.variables,","),gb,
        nranks==1 ? "$(Threads.nthreads()) threads" : "$(nranks-1) worker ranks")
    mkpath(dirname(abspath(a.output)))
    h5open(a.output,"w") do file
        group=create_group(file,"R"); filters=[HDF5.Filters.Shuffle(),HDF5.Filters.Deflate(4)]
        datasets=[create_dataset(group,string(v),datatype(Float32),dataspace(nl,nz,nys,length(xs));
            chunk=(nl,nz,nys,1),filters=filters) for v in a.variables]
        for (dataset,v) in zip(datasets,a.variables)
            attributes(dataset)["unit"]=v in (:ne,:pgas) ? "$(FFNOInversion.SYNTHESIS_INTENSITY_UNITS) per unit ln $v per unit log10 tau500" :
                v===:temperature ? "$(FFNOInversion.SYNTHESIS_INTENSITY_UNITS) per K per unit log10 tau500" :
                "$(FFNOInversion.SYNTHESIS_INTENSITY_UNITS) per m/s per unit log10 tau500"
        end
        intensity=create_dataset(file,"intensity",datatype(Float32),dataspace(nl,nys,length(xs));
            chunk=(nl,nys,1),filters=filters)
        attributes(intensity)["units"]=FFNOInversion.SYNTHESIS_INTENSITY_UNITS
        file["wavelength_air_m"]=wavelength_air; file["wavelength_vacuum_m"]=wavelength_vacuum
        file["logtau500"]=collect(logtau); file["dlogtau500"]=widths
        file["x_index"]=xs.-1; file["y_index"]=ys.-1
        file["x_m"]=atmosphere.grid.x[xs]; file["y_m"]=atmosphere.grid.y[ys]
        attributes(file)["description"]="column response functions dI(lambda,x,y)/dv(k,x,y) of the intrinsic (pre-PSF) " *
            "intensity; populations, pgas, rho, ne, z and chi500 held fixed unless perturbed themselves"
        attributes(file)["axis_order"]="R: wavelength,logtau500,y,x; intensity: wavelength,y,x (Julia order)"
        attributes(file)["source_atmosphere"]=abspath(a.atmosphere)
        attributes(file)["config"]=abspath(a.config)
        tangents===nothing || (attributes(file)["population_tangents"]=abspath(a.tangents))
        for (key,value) in F.view_attributes(atmosphere.grid.view,"rays"); attributes(file)[key]=value; end
        tangents===nothing || (attributes(datasets[findfirst(==(:temperature),a.variables)])["populations"]=
            "FFNO populations respond (forward-mode tangents seeded in every column); pgas, rho, ne, z fixed")
        block=nl*nz*nys
        function write_slab(i,values)
            for (iv,dataset) in enumerate(datasets)
                dataset[:,:,:,i]=reshape(values[(iv-1)*block+1:iv*block],nl,nz,nys)
            end
            intensity[:,:,i]=reshape(values[nv*block+1:end],nl,nys)
        end
        start=time()
        progress(done)=begin
            elapsed=time()-start
            @printf("x %d/%d  elapsed %.1f min  remaining %.1f min\n",done,length(xs),elapsed/60,
                elapsed/60*(length(xs)-done)/done)
            flush(stdout)
        end
        if nranks==1
            for i in eachindex(xs)
                slab_response!(R,I,xs[i],ys,setups,ranges,a.variables,atmosphere,widths,tangents,populations)
                write_slab(i,Float32.(vcat((vec(r) for r in R)...,vec(I)))); progress(i)
            end
        else
            serve_slabs!(write_slab,comm,length(xs),nvalues,progress)
        end
    end
    println("wrote ",abspath(a.output))
    nranks>1 && MPI.Barrier(comm)
end

const OPTIONS=parse_arguments(ARGS)
include(abspath(OPTIONS.factory))        # run-local paths: EOS_LIBRARY, PARTITION_FUNCTIONS, ATOM_FILES
main(OPTIONS)
