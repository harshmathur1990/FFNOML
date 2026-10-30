using FFNOInversion

context=initialize_parallel(options=ParallelOptions(enabled=true,threads_per_rank=Threads.nthreads()))
try
    mode=get(ENV,"PHASE4_RESTART_MODE",""); path=get(ENV,"PHASE4_RESTART_PATH","")
    mode in ("write","read") || error("PHASE4_RESTART_MODE must be write or read")
    isempty(path) && error("PHASE4_RESTART_PATH is required")
    nx,ny,nz=9,7,4
    function reference_atmosphere()
        grid=Grid3D([-5.0,-4,-3,-2],collect(0.0:50e3:(nx-1)*50e3),collect(0.0:50e3:(ny-1)*50e3))
        shape=(nz,nx,ny); temperature=Array{Float64}(undef,shape)
        for k in 1:nz,i in 1:nx,j in 1:ny; temperature[k,i,j]=5000+10k+2i+j; end
        vx=fill(100.0,shape); vy=fill(-50.0,shape); vz=zeros(shape); vturb=fill(1200.0,shape)
        pgas=fill(0.5,shape); rho=fill(1e-8,shape); ne=fill(1e16,shape)
        z=Array{Float64}(undef,shape); for k in 1:nz; @views z[k,:,:].=-(k-1)*50e3; end
        B=MagneticField3D(fill(1e-3,shape),fill(2e-3,shape),fill(3e-3,shape))
        Atmosphere3D(grid,temperature,vx,vy,vz,vturb;magnetic_field=B,pgas=pgas,rho=rho,ne=ne,z=z)
    end
    root_state=if isroot(context)
        mode=="write" ? reference_atmosphere() : restore_checkpoint(path;expected=CapabilityManifest()).state
    else nothing end
    expected=isroot(context) ? deepcopy(root_state) : nothing
    distributed=distribute_atmosphere(Float64,root_state,context); root_state=nothing
    restored=gather_atmosphere(distributed,context)
    if isroot(context)
        for name in (:temperature,:vx,:vy,:vz,:vturb,:pgas,:rho,:ne,:z)
            getfield(restored,name)==getfield(expected,name) || error("restart field $name differs")
        end
        restored.magnetic_field.Bx==expected.magnetic_field.Bx || error("restart Bx differs")
        restored.magnetic_field.By==expected.magnetic_field.By || error("restart By differs")
        restored.magnetic_field.Bz==expected.magnetic_field.Bz || error("restart Bz differs")
        mode=="write" && checkpoint!(path,restored)
        println("MPI_PHASE4_RESTART_$(uppercase(mode))_OK ranks=$(context.size)")
    end
    barrier(context)
finally
    finalize_parallel!(context)
end

