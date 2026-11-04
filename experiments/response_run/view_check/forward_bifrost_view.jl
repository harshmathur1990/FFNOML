# V5b of docs/viewing-angle/README.md: H-alpha from the Bifrost cube on its own height grid at several views.
#
# The FFNO (FSDP service, H model) predicts the populations once on the Cartesian cube (every STRIDE-th column, Bifrost
# T, v, P, ρ, nₑ and heights; no equilibrium, so the thermodynamics are the simulation's, as in the RH and MULTI3D
# references). Each view is then synthesized through the library's own path: HybridForwardWorkspace and, for an
# inclined view, footprint_state! (state and populations shifted by +t z per level), synthesis with path dz/μ.
# Views: μ = 1; μ, φ inclined; μ, φ plane-parallel.
#
# usage (eagle7, see run_view_eagle7.sh): julia --project=<FFNOInversion.jl> forward_bifrost_view.jl CONFIG.toml
#        MODELS.jl STRIDE MU AZIMUTH_DEG OUT.h5
using FFNOInversion, HDF5, Printf
import Muspel
const F=FFNOInversion
const BIFROST="/mn/stornext/d9/data/harshm/halpha_output/BIFROST_en024048_hion_0_504_0_504.nc"

config_path,models,stride_s,mu_s,azimuth_s,out=ARGS
stride=parse(Int,stride_s); mu=parse(Float64,mu_s); azimuth=parse(Float64,azimuth_s)
include(abspath(models))                                    # CHECKPOINTS, LEVEL_NAMES, ATOM_FILES, EOS_LIBRARY, ...
config=load_config(config_path)

bf=h5open(BIFROST,"r")
z=vec(read(bf["z"])); x=read(bf["x"]); y=read(bf["y"])
xs=1:stride:length(x); ys=1:stride:length(y)
rd(name)=permutedims(Float64.(bf[name][:,ys,xs,1]),(1,3,2))  # Julia (z, y, x, t) -> (z, x, y), z top first
T=rd("temperature"); vx=rd("velocity_x"); vy=rd("velocity_y"); vz=rd("velocity_z")
pgas=rd("gas_pressure"); rho=rd("density"); ne=rd("electron_density")
close(bf)
nz,nx,ny=size(T)
gx=collect(x[xs] .- x[1]); gy=collect(y[ys] .- y[1])
@printf("Bifrost cube every %d columns: %d x %d x %d, dx %.1f km, z %.2f … %.2f Mm\n",stride,nx,ny,nz,(gx[2]-gx[1])/1e3,
    z[1]/1e6,z[end]/1e6)
atmosphere_for(view)=Atmosphere3D(Grid3D(VerticalAxis(:z_m,z),gx,gy,view),copy(T),copy(vx),copy(vy),copy(vz),zeros(nz,nx,ny);
    pgas=copy(pgas),rho=copy(rho),ne=copy(ne),z=repeat(reshape(z,nz,1,1),1,nx,ny))

# FFNO H populations on the Cartesian cube (the same for every view)
metadata=PopulationMetadata(FFNO_INPUT_CHANNELS,LEVEL_NAMES[:H],"unrecorded-example-checkpoint")
launcher=get(ENV,"FFNO_FSDP_LAUNCHER","")
service=launch_fsdp_service([FSDPModelSpec(:H,CHECKPOINTS[:H],metadata)];(isempty(launcher) ? (;) : (;launcher=launcher))...,
    timeout_seconds=900.0,diagnostics_directory=get(ENV,"FFNO_GPU_DIAGNOSTICS_DIR",joinpath(dirname(abspath(out)),"diagnostics")))
populations=zeros(nz,nx,ny,6)
try
    start=time()
    p=zeros(Float32,nz,nx,ny,6); predict_populations!(p,FSDPFFNOModel(service,:H,metadata,0),atmosphere_for(ViewGeometry()))
    populations.=p
    @printf("FFNO H populations on the Cartesian cube: %.1f s\n",time()-start)
finally
    close_fsdp_service!(service)
end

eos=WittmannEOS(EOS_LIBRARY,PARTITION_FUNCTIONS)
context=serial_context(options=ParallelOptions(threads_per_rank=Threads.nthreads()))
views=[("mu1",ViewGeometry()),("inclined",ViewGeometry(mu=mu,azimuth_deg=azimuth,geometry=:inclined)),
       ("plane_parallel",ViewGeometry(mu=mu,azimuth_deg=azimuth,geometry=:plane_parallel))]
h5open(out,"w") do file
    file["x_index"]=collect(xs.-1); file["y_index"]=collect(ys.-1); file["populations_H_z0"]=populations[argmin(abs.(z)),:,:,:]
    for (name,view) in views
        start=time()
        distributed=distribute_atmosphere(Float64,atmosphere_for(view),context)
        wavelength=reduce(vcat,wavelengths.(config.regions))
        workspace=HybridForwardWorkspace(Float64,distributed,wavelength,StokesSet(:I),Dict(:H=>6))
        F.cartesian_populations(workspace)[:H].=populations
        workspace.footprint===nothing || F.footprint_state!(workspace,distributed,context)
        atmosphere=F.synthesis_atmosphere(workspace,distributed)
        setups=build_synthesis_setup(config,atmosphere,workspace.populations,eos;atom_files=ATOM_FILES)
        cube=SpectralCube(zeros(length(wavelength),1,nx,ny),wavelength,StokesSet(:I))
        synthesize!(cube,setups,NonPRD(),atmosphere,workspace.populations)
        file["$name/intensity"]=cube.data[:,1,:,:]
        for (k,v) in view_attributes(view,F.view_columns(atmosphere.grid)); attributes(file[name])[k]=v; end
        haskey(file,"wavelength_air_m") || (file["wavelength_air_m"]=wavelength)
        @printf("%s: footprint=%s, %.1f s\n",name,workspace.footprint!==nothing,time()-start); flush(stdout)
    end
end
println("wrote $out")
