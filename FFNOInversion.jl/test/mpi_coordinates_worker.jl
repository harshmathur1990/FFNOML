using Test
using MPI
using FFNOInversion
using SHA

# Distribute/gather round trip of native-z and native-log-tau atmospheres (with their 3-D coordinate fields) for both
# decompositions; run with `mpiexec -n R julia --project test/mpi_coordinates_worker.jl` (R = 1, 2, 3).

MPI.Init()
comm=MPI.COMM_WORLD; R=MPI.Comm_size(comm); me=MPI.Comm_rank(comm)

function coordinate_atmosphere(kind;nz=9,nx=12,ny=5)
    axis=kind===:z_m ? collect(range(1.5e6,-3.0e5;length=nz)) : collect(range(-6.0,1.0;length=nz))
    grid=Grid3D(VerticalAxis(kind,axis),1.2e6 .+ 4.8e4 .* collect(0:nx-1),4.8e4 .* collect(0:ny-1))
    f(s)=[s+100.0*k+7.0*i+0.3*j for k in 1:nz,i in 1:nx,j in 1:ny]
    native=repeat(reshape(axis,:,1,1),1,nx,ny)
    derived=[axis[k]*0.5+0.01*i-0.02*j for k in 1:nz,i in 1:nx,j in 1:ny]
    z,lt=kind===:z_m ? (native,derived) : (derived,native)
    B=MagneticField3D(f(0.1),f(0.2),f(0.3))
    Atmosphere3D(grid,f(5000.0),f(1.0),f(2.0),f(3.0),f(4.0);magnetic_field=B,pgas=f(10.0),rho=f(1e-5),ne=f(1e15),
        z=z,logtau500=lt)
end

metadata_hash(g)=bytes2hex(sha256(codeunits(string(g.vertical.kind,g.vertical.values))))

ts=@testset "coordinate round trip n=$R" begin
    for decomposition in (:cartesian_2d,:x_slabs), kind in (:z_m,:log10tau500)
        context=initialize_parallel(options=ParallelOptions(enabled=true,decomposition=decomposition))
        root=me==0 ? coordinate_atmosphere(kind) : nothing
        distributed=distribute_atmosphere(Float64,root,context)
        local_grid=distributed.local_atmosphere.grid
        @test vertical_coordinate(local_grid)===kind && vertical_coordinate(distributed.global_grid)===kind
        @test distributed.local_atmosphere.logtau500!==nothing && distributed.local_atmosphere.z!==nothing
        decomposition===:x_slabs && @test distributed.tile.yrange==1:5
        h=hash(metadata_hash(distributed.global_grid))%UInt32
        @test MPI.Allreduce(h,max,comm)==MPI.Allreduce(h,min,comm)          # identical coordinate metadata everywhere
        gathered=gather_atmosphere(distributed,context)
        if me==0
            ref=coordinate_atmosphere(kind)
            for name in (:temperature,:vx,:vy,:vz,:vturb,:pgas,:rho,:ne,:z,:logtau500)
                @test getfield(gathered,name)==getfield(ref,name)
            end
            @test gathered.magnetic_field.Bx==ref.magnetic_field.Bx
            @test native_depth(gathered.grid)==native_depth(ref.grid) && gathered.grid.x==ref.grid.x
            @test check_coordinate_consistency(gathered)
        end
    end
end
MPI.Barrier(comm)
MPI.Finalize()
