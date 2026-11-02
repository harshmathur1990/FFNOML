# Temperature tangents of the FFNO populations of a forward run, for scripts/column_response.jl --population-tangents.
#
# For every log tau500 point k0, one forward-mode pass (dual tensors through the population mesh and the FFNO on the
# GPU service) gives the population change for a 1 K rise of T at k0 in every column at once:
#   ṅ(k, x, y; k0) = Σ_{x',y'} ∂n(k, x, y) / ∂T(k0, x', y')        [m^-3 per K]
# The sum keeps the 3-D coupling of the FFNO's horizontal Fourier layers; storing it per column is what makes the
# full Jacobian unnecessary. pgas, rho, ne and z (so the mesh mapping) stay fixed, as in column_response.jl.
#
# Output (HDF5, Julia shapes; h5py sees them reversed):
#   tangent/<species>    (nz, ny, levels, nk0, nx) Float32, one chunk per (k0, x)
#   populations/<species> (nz, ny, levels, nx)     the FFNO populations at the linearization point
#   logtau500 (nz), depth_index (nk0, 1-based)
# With --stride N the FFNO runs on every N-th column in x and y (its Fourier layers see the coarser spacing): the
# populations are predicted there and the subsampled atmosphere with these populations, the linearization point, is
# written to ATMOSPHERE_OUT (default OUTPUT with _atmosphere.h5) for column_response.jl --stride 1.
#
# usage (inside an Olivia GPU allocation; the FSDP service is launched by this process):
#   julia --project=<FFNOInversion env> ffno_population_tangents.jl FORWARD.toml MODEL_FACTORY.jl ATMOSPHERE.h5
#       OUTPUT.h5 [--species H,CA] [--depths 1:95] [--stride 4] [--atmosphere-out PATH] [--fd-check STEP_K]
# --fd-check also compares the first depth's JVP with central differences of two predictions, T ± STEP_K.
using FFNOInversion, HDF5, Printf
const HSEMHS = FFNOInversion.HSEMHS        # a dependency of FFNOInversion, not necessarily of the run environment
const F = FFNOInversion

function parse_arguments(args)
    length(args)>=4 || error("usage: ffno_population_tangents.jl FORWARD.toml MODEL_FACTORY.jl ATMOSPHERE.h5 " *
        "OUTPUT.h5 [--species H,CA] [--depths 1:95] [--stride 4] [--atmosphere-out PATH]")
    options=Dict("species"=>"H,CA","depths"=>"","stride"=>"1","atmosphere-out"=>"","fd-check"=>"0")
    i=5
    while i<=length(args)
        key=lstrip(args[i],'-'); haskey(options,key) && i<length(args) || error("unknown or incomplete option $(args[i])")
        options[key]=args[i+1]; i+=2
    end
    depths=isempty(options["depths"]) ? nothing : eval(Meta.parse(options["depths"]))
    stride=parse(Int,options["stride"])
    atmosphere_out=isempty(options["atmosphere-out"]) ? replace(args[4],r"\.h5$"=>"")*"_atmosphere.h5" :
        options["atmosphere-out"]
    (config=args[1],factory=args[2],atmosphere=args[3],output=args[4],species=Symbol.(split(options["species"],',')),
     depths=depths,stride=stride,atmosphere_out=atmosphere_out,fd_check=parse(Float64,options["fd-check"]))
end

include(joinpath(@__DIR__,"forward_atmosphere_io.jl"))

function main(a)
    config=load_config(a.config)
    atmosphere,stored=read_forward_atmosphere(a.atmosphere)
    if a.stride>1
        atmosphere=subsample_atmosphere(atmosphere,a.stride)
        stored=Dict(s=>p[:,1:a.stride:end,1:a.stride:end,:] for (s,p) in stored)
    end
    F._tau_integration(atmosphere) || error("the forward atmosphere is not a native log tau500 grid with chi500")
    options=population_mesh_options(config)
    options===nothing && error("the configuration has no population mesh; this script expects a native log tau run")
    nz,nx,ny=size(atmosphere.temperature); depths=a.depths===nothing ? collect(1:nz) : collect(a.depths)
    logtau=native_depth(atmosphere.grid)

    metadata(species)=PopulationMetadata(FFNO_INPUT_CHANNELS,Main.LEVEL_NAMES[species],
        get(ENV,"FFNO_$(species)_CHECKPOINT_HASH","unrecorded-example-checkpoint"))
    specs=[FSDPModelSpec(species,Main.CHECKPOINTS[species],metadata(species)) for species in a.species]
    # FFNO_FSDP_LAUNCHER: another service launcher, e.g. scripts/ffno_fsdp_service_local.sh on a host without Slurm.
    launcher=get(ENV,"FFNO_FSDP_LAUNCHER","")
    service=launch_fsdp_service(specs;(isempty(launcher) ? (;) : (;launcher=launcher))...,
        timeout_seconds=max(600.0,config.parallel.gpu_connect_timeout_seconds),
        diagnostics_directory=get(ENV,"FFNO_GPU_DIAGNOSTICS_DIR",joinpath(dirname(abspath(a.output)),"diagnostics")))
    eos=HSEMHS.load_table(options.eos_table_file)
    try
        mkpath(dirname(abspath(a.output)))
        h5open(a.output,"w") do file
            file["logtau500"]=collect(logtau); file["depth_index"]=depths
            attributes(file)["description"]="forward-mode (dual tensor) temperature tangents of the FFNO populations: " *
                "tangent[k, y, level, i, x] = sum over columns of d n(k, x, y)/d T(depth_index[i], x', y') per K; " *
                "pgas, rho, ne and z fixed"
            attributes(file)["source_atmosphere"]=abspath(a.atmosphere)
            attributes(file)["stride"]=a.stride; attributes(file)["linearization_atmosphere"]=abspath(a.atmosphere_out)
            for species in a.species
                levels=length(Main.LEVEL_NAMES[species])
                model=wrap_population_mesh(FSDPFFNOModel(service,species,metadata(species),0),options,eos)
                # The linearization point: the FFNO populations of this atmosphere (also caches the mesh populations).
                p=zeros(Float32,nz,nx,ny,levels); predict_populations!(p,model,atmosphere)
                if haskey(stored,species)
                    d=abs.(log.(p) .- log.(Float32.(stored[species])))
                    @printf("%s: |ln(predicted/stored populations)| median %.2e, max %.2e%s\n",species,
                        sort(vec(d))[length(d)÷2],maximum(d),a.stride>1 ? " (stored: full-resolution prediction)" : "")
                end
                stored[species]=p                   # the linearization point
                file["populations/$species"]=permutedims(p,(1,3,4,2))
                dataset=create_dataset(file,"tangent/$species",datatype(Float32),
                    dataspace(nz,ny,levels,length(depths),nx);chunk=(nz,ny,levels,1,1))
                attributes(dataset)["unit"]="m^-3 per K"
                feature_dot=zeros(Float32,6,nz,nx,ny); population_dot=zeros(Float32,nz,nx,ny,levels)
                start=time()
                for (i,k0) in enumerate(depths)
                    fill!(feature_dot,0); feature_dot[1,k0,:,:].=1      # 1 K at depth k0 in every column
                    population_jvp!(population_dot,model,atmosphere,feature_dot)
                    dataset[:,:,:,i,:]=permutedims(population_dot,(1,3,4,2))
                    if i==1 && a.fd_check>0
                        h=a.fd_check; plus=similar(p); minus=similar(p)
                        atmosphere.temperature[k0,:,:].+=h; predict_populations!(plus,model,atmosphere)
                        atmosphere.temperature[k0,:,:].-=2h; predict_populations!(minus,model,atmosphere)
                        atmosphere.temperature[k0,:,:].+=h; predict_populations!(p,model,atmosphere)
                        # compare relative changes ṅ/n (Float64: n ~ 1e20 m^-3 overflows Float32 sums)
                        j=Float64.(population_dot)./p; f=(Float64.(plus).-Float64.(minus))./(2h)./p
                        err=abs.(j.-f); q(v,r)=sort(vec(v))[clamp(round(Int,r*length(v)),1,length(v))]
                        @printf("%s depth %d: JVP vs central difference (T ± %g K), per unit n: ||JVP-FD||/||FD|| %.2e, |JVP-FD| median %.2e p99 %.2e max %.2e, |FD| median %.2e max %.2e, correlation %.6f\n",
                            species,k0,h,sqrt(sum(abs2,err)/sum(abs2,f)),q(err,0.5),q(err,0.99),maximum(err),
                            q(abs.(f),0.5),maximum(abs.(f)),sum(j.*f)/sqrt(sum(abs2,j)*sum(abs2,f)))
                    end
                    elapsed=time()-start
                    @printf("%s depth %d/%d (log tau %.1f)  max |ṅ/n| %.2e per K  elapsed %.1f min  remaining %.1f min\n",
                        species,i,length(depths),logtau[k0],maximum(abs.(population_dot)./p),elapsed/60,
                        elapsed/60*(length(depths)-i)/i)
                    flush(stdout)
                end
            end
        end
    finally
        close_fsdp_service!(service)
    end
    write_forward_atmosphere(a.atmosphere_out,atmosphere,stored;attributes_=Dict("description"=>
        "linearization point of $(abspath(a.output)): every $(a.stride)-th column of $(abspath(a.atmosphere)) " *
        "with the FFNO populations predicted on that grid"))
    println("wrote ",abspath(a.atmosphere_out))
    println("wrote ",abspath(a.output))
end

const OPTIONS=parse_arguments(ARGS)
include(abspath(OPTIONS.factory))        # CHECKPOINTS and LEVEL_NAMES
main(OPTIONS)
