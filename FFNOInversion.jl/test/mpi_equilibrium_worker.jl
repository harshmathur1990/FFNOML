using Test
using FFNOInversion
using Serialization

# HSEMHS column modes across ranks: `mpiexec -n R julia --project test/mpi_equilibrium_worker.jl OUT` (R = 1, 2, 3; x
# slabs, NX = 11 gives uneven splits). Each run serializes the gathered atmosphere and spectrum to OUT_n<R>.jls on rank
# 0, so the driver can compare rank counts; it also checks collective failures and repeated calls.

const H=FFNOInversion.HSEMHS
const TABLE=get(ENV,"HSEMHS_EOS_TABLE","/mn/stornext/d9/data/harshm/MHS/HSEMHS/cache/eos_table_fullz.h5")
out=ARGS[1]

context=initialize_parallel(options=ParallelOptions(enabled=true,decomposition=:x_slabs,threads_per_rank=Threads.nthreads()))
R=context.size

function atmosphere_for(kind)
    nz,nx,ny=36,11,6
    axis=kind===:z_m ? collect(range(1.4e6,-3.5e5;length=nz)) : collect(range(-7.0,1.3;length=nz))
    grid=Grid3D(VerticalAxis(kind,axis),4.8e4 .* collect(0:nx-1),4.8e4 .* collect(0:ny-1))
    T=[(kind===:z_m ? 5000.0+2500.0*exp(-abs(axis[k])/3e5) : 4600.0+1400.0*exp10(min(axis[k],0.5)/2))+30.0*i+11.0*j
       for k in 1:nz,i in 1:nx,j in 1:ny]
    B=MagneticField3D([8e-3*cos(0.5i+0.2j) for k in 1:nz,i in 1:nx,j in 1:ny],fill(3e-3,nz,nx,ny),fill(1.5e-2,nz,nx,ny))
    z0=zeros(nz,nx,ny)
    Atmosphere3D(grid,T,copy(z0),copy(z0),copy(z0),copy(z0);magnetic_field=B)
end

results=Dict{Symbol,Any}()
ts=@testset "HSEMHS equilibrium n=$R" begin
    for kind in (:z_m,:log10tau500)
        mode=kind===:z_m ? :column_hse_z : :column_hse_ltau
        column=kind===:z_m ? H.ColumnZOptions(anchor=:top,P_ref=0.3) : H.ColumnTauOptions(align="ptot")
        config=EquilibriumConfig("HSEMHS",:column_hse,mode,274.0,TABLE,:reject_trial,column,nothing)
        root=isroot(context) ? atmosphere_for(kind) : nothing
        distributed=distribute_atmosphere(Float64,root,context)
        @test distributed.tile.yrange==1:6
        backend=prepare_equilibrium(config,distributed,context)
        model=HybridForwardModel(LocalDistributedPopulationModel(MockPopulationModel(1e10),1),NonPRD(),
            MockIntensitySynthesizer(),IdentityObservation(),backend,CapabilityManifest())
        workspace=HybridForwardWorkspace(Float64,distributed,[656.2e-9,656.25e-9,656.3e-9],StokesSet(:I),1)
        result=forward!(workspace,model,distributed,context)
        @test result.valid
        gathered=gather_atmosphere(distributed,context); spectrum=gather_spectrum(result.spectrum,distributed,context)
        isroot(context) && (results[kind]=(pgas=gathered.pgas,rho=gathered.rho,ne=gathered.ne,z=gathered.z,
            logtau500=gathered.logtau500,spectrum=spectrum.data))
        # out-of-envelope temperature on the last rank only: the same invalid verdict everywhere, no deadlock
        reference_T=copy(distributed.local_atmosphere.temperature)
        context.rank==R-1 && (distributed.local_atmosphere.temperature[:,1,1].=5.0e7)
        failed=forward!(workspace,model,distributed,context)
        @test allreduce_sum(failed.valid ? 0 : 1,context)==R && failed.equilibrium.reason===:eos_out_of_envelope
        # 100 repeated forwards alternating valid and invalid: identical verdicts on every rank, no hang
        agree=true
        for call in 1:100
            distributed.local_atmosphere.temperature.=reference_T
            isodd(call) && context.rank==R-1 && (distributed.local_atmosphere.temperature[:,1,1].=5.0e7)
            r=forward!(workspace,model,distributed,context)
            agree&=allreduce_sum(r.valid ? 1 : 0,context)==(isodd(call) ? 0 : R)
        end
        @test agree
        close_equilibrium!(backend)
    end
end
isroot(context) && open(io->serialize(io,results),"$(out)_n$(R).jls","w")
finalize_parallel!(context)
